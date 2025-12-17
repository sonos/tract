use crate::context::{TractCudaStream, cuda_context};
use crate::kernels::conv::{ConvKernel, ConvKernelScratch};
use crate::kernels::{WARP_SIZE, get_cuda_view, get_cuda_view_mut};
use cudarc::cudnn::{ConvDescriptor, ConvForward, FilterDescriptor, TensorDescriptor};
use cudarc::driver::{CudaStream, LaunchArgs, LaunchConfig, PushKernelArg};
use std::any::Any;
use std::cell::RefCell;
use std::fmt::Debug;
use std::ops::Deref;
use std::sync::Weak;
use std::thread::LocalKey;
use thread_local::ThreadLocal;
use tract_core::dyn_clone::{self, DynClone};
use tract_core::internal::*;
use tract_core::ops::cnn::{Conv, KernelFormat};
use tract_core::tract_data::itertools::Itertools;
use tract_gpu::tensor::DeviceTensor;

#[derive(Debug)]
struct CudnnConvDescriptors {
    input_shape: TVec<usize>,
    conv: ConvDescriptor<f32>,
    x: TensorDescriptor<f32>,
    w: FilterDescriptor<f32>,
    y: TensorDescriptor<f32>,
    algo: cudarc::cudnn::sys::cudnnConvolutionFwdAlgo_t,
    workspace_size: usize,
}

unsafe impl Send for CudnnConvDescriptors {}

impl CudnnConvDescriptors {
    fn new(
        stream: &TractCudaStream,
        op: &Conv,
        input_shape_array: &[usize],
        weight_shape_array: &[usize],
        output_shape_array: &[usize],
    ) -> TractResult<CudnnConvDescriptors> {
        ensure!(op.pool_spec.data_format.has_n());
        ensure!(op.kernel_fmt == KernelFormat::OIHW);
        let input_shape = op.pool_spec.data_format.shape(input_shape_array)?;
        let output_shape = op.pool_spec.data_format.shape(output_shape_array)?;
        let ctx = cuda_context();
        ensure!(input_shape.hw_rank() <= 6);

        let cudnn = stream.cudnn();
        let mut pads = op
            .pool_spec
            .computed_padding(input_shape.hw_dims())
            .iter()
            .map(|p| p.pad_before as i32)
            .collect_vec();
        let mut strides = op.pool_spec.strides().iter().map(|s| *s as i32).collect_vec();
        let mut dilations = op.pool_spec.dilations().iter().map(|d| *d as i32).collect_vec();
        if input_shape.hw_rank() == 1 {
            strides.push(1);
            dilations.push(1);
            pads.push(0);
        }
        let mut conv_descriptor = cudnn
            .create_convnd::<f32>(
                &pads,
                &strides,
                &dilations,
                cudarc::cudnn::sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
            )
            .context("in create_conv2d")?;
        conv_descriptor.set_group_count(op.group as i32);
        let mut input_dims = input_shape.hw_dims().iter().map(|d| *d as i32).collect_vec();
        if input_dims.len() == 1 {
            input_dims.push(1);
        }
        input_dims.insert(0, *input_shape.n().unwrap() as i32);
        input_dims.insert(1, *input_shape.c() as i32);

        let mut input_strides = input_shape.hw_strides().iter().map(|s| *s as i32).collect_vec();
        if input_strides.len() == 1 {
            input_strides.push(*input_shape.w_stride() as i32);
        }
        input_strides.insert(0, *input_shape.n_stride().unwrap() as i32);
        input_strides.insert(1, *input_shape.c_stride() as i32);

        let input_descriptor = cudnn
            .create_nd_tensor::<f32>(&input_dims, &input_strides)
            .context("in created_4d_tensor for input")?;
        let mut filter_dims = weight_shape_array.iter().map(|d| *d as i32).collect_vec();
        if filter_dims.len() == 3 {
            filter_dims.push(1);
        }
        let filter_descriptor = cudnn
            .create_nd_filter::<f32>(
                cudarc::cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
                &filter_dims,
            )
            .context("in created_4d_tensor for filter")?;

        let mut output_dims = output_shape.hw_dims().iter().map(|d| *d as i32).collect_vec();
        if output_dims.len() == 1 {
            output_dims.push(1);
        }
        output_dims.insert(0, *output_shape.n().unwrap() as i32);
        output_dims.insert(1, *output_shape.c() as i32);

        let mut output_strides = output_shape.hw_strides().iter().map(|s| *s as i32).collect_vec();
        if output_strides.len() == 1 {
            output_strides.push(*output_shape.w_stride() as i32);
        }
        output_strides.insert(0, *output_shape.n_stride().unwrap() as i32);
        output_strides.insert(1, *output_shape.c_stride() as i32);

        let output_descriptor = cudnn
            .create_nd_tensor::<f32>(&output_dims, &output_strides)
            .context("in created_4d_tensor for output")?;

        let conv = ConvForward {
            conv: &conv_descriptor,
            x: &input_descriptor,
            w: &filter_descriptor,
            y: &output_descriptor,
        };

        let algo = cudarc::cudnn::sys::cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
        let workspace_size = conv.get_workspace_size(algo).context("in get_workspace_size()")?;

        Ok(CudnnConvDescriptors {
            input_shape: input_shape_array.into(),
            x: input_descriptor,
            w: filter_descriptor,
            y: output_descriptor,
            conv: conv_descriptor,
            algo,
            workspace_size,
        })
    }
}

impl ConvKernelScratch for Option<CudnnConvDescriptors> {}

#[derive(Debug, Default, Clone)]
pub struct ConvCudnn;

impl ConvKernel for ConvCudnn {
    fn name(&self) -> StaticName {
        "ConvCudnn".into()
    }

    fn state(&self) -> Box<dyn ConvKernelScratch> {
        Box::<Option<CudnnConvDescriptors>>::default()
    }

    fn dispatch(
        &self,
        state: &mut dyn ConvKernelScratch,
        node_id: usize,
        op: &Conv,
        stream: &TractCudaStream,
        input: &DeviceTensor,
        weights: &DeviceTensor,
        bias: Option<&DeviceTensor>,
        output: &DeviceTensor,
    ) -> TractResult<()> {
        ensure!(bias.is_none());

        let state: &mut Option<CudnnConvDescriptors> =
            state.downcast_mut().context("Wrong state")?;

        if state.as_ref().is_none_or(|desc| &*desc.input_shape != input.shape()) {
            *state = Some(CudnnConvDescriptors::new(
                stream,
                op,
                input.shape(),
                weights.shape(),
                output.shape(),
            )?)
        }

        let conv = state.as_ref().unwrap();

        let input = get_cuda_view(input);
        let weights = get_cuda_view(weights);
        let mut output = get_cuda_view_mut(output);

        let conv_forward = ConvForward { conv: &conv.conv, x: &conv.x, w: &conv.w, y: &conv.y };

        unsafe {
            let mut workspace = if conv.workspace_size > 0 {
                Some(stream.alloc::<u8>(conv.workspace_size)?)
            } else {
                None
            };
            let input_transmuted = input.transmute::<f32>(input.len() / size_of::<f32>()).unwrap();
            let weights_transmuted =
                weights.transmute::<f32>(weights.len() / size_of::<f32>()).unwrap();
            let mut output_transmuted =
                output.transmute_mut::<f32>(output.len() / size_of::<f32>()).unwrap();
            conv_forward
                .launch(
                    conv.algo,
                    workspace.as_mut().map(|w| w.as_view_mut()).as_mut(),
                    (1.0, 0.0),
                    &input_transmuted,
                    &weights_transmuted,
                    &mut output_transmuted,
                )
                .context("in launch()")?;
        }

        Ok(())
    }
}
