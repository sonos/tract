use crate::context::{cuda_context, TractCudaStream};
use crate::kernels::conv::ConvKernel;
use crate::kernels::{get_cuda_view, get_cuda_view_mut, WARP_SIZE};
use cudarc::cudnn::ConvForward;
use cudarc::driver::{LaunchArgs, LaunchConfig, PushKernelArg};
use std::fmt::Debug;
use tract_core::dyn_clone::{self, DynClone};
use tract_core::internal::*;
use tract_core::ops::cnn::{Conv, KernelFormat};
use tract_core::tract_data::itertools::Itertools;
use tract_gpu::tensor::DeviceTensor;

#[derive(Hash, Clone, Debug)]
pub struct ConvCudnn;

impl ConvKernel for ConvCudnn {
    fn name(&self) -> StaticName {
        "ConvCudnn".into()
    }

    fn dispatch(
        &self,
        op: &Conv,
        stream: &TractCudaStream,
        input: &DeviceTensor,
        weights: &DeviceTensor,
        bias: Option<&DeviceTensor>,
        output: &DeviceTensor,
    ) -> TractResult<()> {
        ensure!(bias.is_none());
        ensure!(op.pool_spec.data_format.has_n());
        ensure!(op.kernel_fmt == KernelFormat::OIHW);
        let input_shape = op.pool_spec.data_format.shape(input.shape())?;
        let output_shape = op.pool_spec.data_format.shape(output.shape())?;
        let ctx = cuda_context();
        ensure!(input_shape.hw_rank() >= 2);

        let cudnn = stream.cudnn();
        let pads = op
            .pool_spec
            .computed_padding(input_shape.hw_dims())
            .iter()
            .map(|p| p.pad_before as i32)
            .collect_vec();
        let strides = op.pool_spec.strides().iter().map(|s| *s as i32).collect_vec();
        let dilations = op.pool_spec.dilations().iter().map(|d| *d as i32).collect_vec();
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
        input_dims.insert(0, *input_shape.n().unwrap() as i32);
        input_dims.insert(1, *input_shape.c() as i32);

        let mut input_strides = input_shape.hw_strides().iter().map(|s| *s as i32).collect_vec();
        input_strides.insert(0, *input_shape.n_stride().unwrap() as i32);
        input_strides.insert(1, *input_shape.c_stride() as i32);

        // let fmt = if op.pool_spec.data_format.c_is_last() {
        //     cudarc::cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NHWC
        // } else {
        //     cudarc::cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW
        // };

        let input_descriptor = cudnn
            .create_nd_tensor::<f32>(&input_dims, &input_strides)
            .context("in created_4d_tensor for input")?;
        let filter_dims = weights.shape().iter().map(|d| *d as i32).collect_vec();
        let filter_descriptor = cudnn
            .create_nd_filter::<f32>(
                cudarc::cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
                &filter_dims,
            )
            .context("in created_4d_tensor for filter")?;

        let mut output_dims = output_shape.hw_dims().iter().map(|d| *d as i32).collect_vec();
        output_dims.insert(0, *output_shape.n().unwrap() as i32);
        output_dims.insert(1, *output_shape.c() as i32);

        let mut output_strides = output_shape.hw_strides().iter().map(|s| *s as i32).collect_vec();
        output_strides.insert(0, *output_shape.n_stride().unwrap() as i32);
        output_strides.insert(1, *output_shape.c_stride() as i32);

        let output_descriptor = cudnn
            .create_nd_tensor::<f32>(&output_dims, &output_strides)
            .context("in created_4d_tensor for output")?;

        let conv_2d = ConvForward {
            conv: &conv_descriptor,
            x: &input_descriptor,
            w: &filter_descriptor,
            y: &output_descriptor,
        };

        let input = get_cuda_view(input);
        let weights = get_cuda_view(weights);
        let mut output = get_cuda_view_mut(output);

        let algo = conv_2d.pick_algorithm().context("in pick_algorightm")?;
        let workspace_size = conv_2d.get_workspace_size(algo).context("in get_workspace_size()")?;

        unsafe {
            let mut workspace =
                if workspace_size > 0 { Some(stream.alloc::<u8>(workspace_size)?) } else { None };
            let input_transmuted = input.transmute::<f32>(input.len() / size_of::<f32>()).unwrap();
            let weights_transmuted =
                weights.transmute::<f32>(weights.len() / size_of::<f32>()).unwrap();
            let mut output_transmuted =
                output.transmute_mut::<f32>(output.len() / size_of::<f32>()).unwrap();
            conv_2d
                .launch(
                    algo,
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
