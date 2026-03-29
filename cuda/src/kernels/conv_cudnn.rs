use crate::context::{TractCudaStream, cuda_context};
use crate::kernels::conv::{ConvKernel, ConvKernelScratch};
use crate::kernels::{WARP_SIZE, get_cuda_view, get_cuda_view_mut};
use cudarc::cudnn::{
    ConvDescriptor, ConvForward, CudnnDataType, FilterDescriptor, TensorDescriptor,
};
use cudarc::driver::{CudaStream, LaunchArgs, LaunchConfig, PushKernelArg};
use std::any::Any;
use std::cell::RefCell;
use std::fmt::Debug;
use std::ops::Deref;
use std::sync::Weak;
use std::thread::LocalKey;
use tract_core::dyn_clone::{self, DynClone};
use tract_core::internal::*;
use tract_core::ops::cnn::{Conv, KernelFormat};
use tract_core::tract_data::half::f16;
use tract_core::tract_data::itertools::Itertools;
use tract_gpu::tensor::DeviceTensor;

#[derive(Debug)]
struct TypedDescriptors<T: CudnnDataType> {
    conv: ConvDescriptor<T>,
    x: TensorDescriptor<T>,
    w: FilterDescriptor<T>,
    y: TensorDescriptor<T>,
}

unsafe impl<T: CudnnDataType> Send for TypedDescriptors<T> {}

fn build_descriptors<T: CudnnDataType>(
    stream: &TractCudaStream,
    op: &Conv,
    input_shape_array: &[usize],
    weight_shape_array: &[usize],
    output_shape_array: &[usize],
) -> TractResult<TypedDescriptors<T>> {
    ensure!(op.pool_spec.data_format.has_n());
    ensure!(op.kernel_fmt == KernelFormat::OIHW);
    let input_shape = op.pool_spec.data_format.shape(input_shape_array)?;
    let output_shape = op.pool_spec.data_format.shape(output_shape_array)?;
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
        .create_convnd::<T>(
            &pads,
            &strides,
            &dilations,
            cudarc::cudnn::sys::cudnnConvolutionMode_t::CUDNN_CROSS_CORRELATION,
        )
        .context("in create_convnd")?;
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
        .create_nd_tensor::<T>(&input_dims, &input_strides)
        .context("in create_nd_tensor for input")?;

    let mut filter_dims = weight_shape_array.iter().map(|d| *d as i32).collect_vec();
    if filter_dims.len() == 3 {
        filter_dims.push(1);
    }
    let filter_descriptor = cudnn
        .create_nd_filter::<T>(
            cudarc::cudnn::sys::cudnnTensorFormat_t::CUDNN_TENSOR_NCHW,
            &filter_dims,
        )
        .context("in create_nd_filter")?;

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
        .create_nd_tensor::<T>(&output_dims, &output_strides)
        .context("in create_nd_tensor for output")?;

    Ok(TypedDescriptors {
        conv: conv_descriptor,
        x: input_descriptor,
        w: filter_descriptor,
        y: output_descriptor,
    })
}

fn get_algo_and_workspace<T: CudnnDataType>(
    desc: &TypedDescriptors<T>,
) -> TractResult<(cudarc::cudnn::sys::cudnnConvolutionFwdAlgo_t, usize)> {
    let conv_fwd = ConvForward { conv: &desc.conv, x: &desc.x, w: &desc.w, y: &desc.y };
    let algo = cudarc::cudnn::sys::cudnnConvolutionFwdAlgo_t::CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
    let workspace_size = conv_fwd.get_workspace_size(algo).context("in get_workspace_size()")?;
    Ok((algo, workspace_size))
}

#[derive(Debug)]
enum CudnnConvDescriptors {
    F32 {
        input_shape: TVec<usize>,
        desc: TypedDescriptors<f32>,
        algo: cudarc::cudnn::sys::cudnnConvolutionFwdAlgo_t,
        workspace_size: usize,
    },
    F16 {
        input_shape: TVec<usize>,
        desc: TypedDescriptors<f16>,
        algo: cudarc::cudnn::sys::cudnnConvolutionFwdAlgo_t,
        workspace_size: usize,
    },
}

impl CudnnConvDescriptors {
    fn stored_input_shape(&self) -> &[usize] {
        match self {
            CudnnConvDescriptors::F32 { input_shape, .. }
            | CudnnConvDescriptors::F16 { input_shape, .. } => input_shape,
        }
    }

    fn new(
        stream: &TractCudaStream,
        op: &Conv,
        dt: DatumType,
        input_shape_array: &[usize],
        weight_shape_array: &[usize],
        output_shape_array: &[usize],
    ) -> TractResult<Self> {
        if dt == DatumType::F16 {
            let desc = build_descriptors::<f16>(
                stream,
                op,
                input_shape_array,
                weight_shape_array,
                output_shape_array,
            )?;
            let (algo, workspace_size) = get_algo_and_workspace(&desc)?;
            Ok(CudnnConvDescriptors::F16 {
                input_shape: input_shape_array.into(),
                desc,
                algo,
                workspace_size,
            })
        } else {
            let desc = build_descriptors::<f32>(
                stream,
                op,
                input_shape_array,
                weight_shape_array,
                output_shape_array,
            )?;
            let (algo, workspace_size) = get_algo_and_workspace(&desc)?;
            Ok(CudnnConvDescriptors::F32 {
                input_shape: input_shape_array.into(),
                desc,
                algo,
                workspace_size,
            })
        }
    }
}

impl ConvKernelScratch for Option<CudnnConvDescriptors> {}

#[derive(Debug, Default, Clone, PartialEq, Eq)]
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

        if state.as_ref().is_none_or(|desc| desc.stored_input_shape() != input.shape()) {
            *state = Some(CudnnConvDescriptors::new(
                stream,
                op,
                input.datum_type(),
                input.shape(),
                weights.shape(),
                output.shape(),
            )?);
        }

        let input_view = get_cuda_view(input);
        let weights_view = get_cuda_view(weights);
        let mut output_view = get_cuda_view_mut(output);

        match state.as_ref().unwrap() {
            CudnnConvDescriptors::F32 { desc, algo, workspace_size, .. } => {
                let conv_forward =
                    ConvForward { conv: &desc.conv, x: &desc.x, w: &desc.w, y: &desc.y };
                unsafe {
                    let mut workspace = if *workspace_size > 0 {
                        Some(stream.alloc::<u8>(*workspace_size)?)
                    } else {
                        None
                    };
                    let src =
                        input_view.transmute::<f32>(input_view.len() / size_of::<f32>()).unwrap();
                    let w = weights_view
                        .transmute::<f32>(weights_view.len() / size_of::<f32>())
                        .unwrap();
                    let mut dst = output_view
                        .transmute_mut::<f32>(output_view.len() / size_of::<f32>())
                        .unwrap();
                    conv_forward
                        .launch(
                            *algo,
                            workspace.as_mut().map(|w| w.as_view_mut()).as_mut(),
                            (1.0f32, 0.0f32),
                            &src,
                            &w,
                            &mut dst,
                        )
                        .context("in launch()")?;
                }
            }
            CudnnConvDescriptors::F16 { desc, algo, workspace_size, .. } => {
                let conv_forward =
                    ConvForward { conv: &desc.conv, x: &desc.x, w: &desc.w, y: &desc.y };
                unsafe {
                    let mut workspace = if *workspace_size > 0 {
                        Some(stream.alloc::<u8>(*workspace_size)?)
                    } else {
                        None
                    };
                    let src =
                        input_view.transmute::<f16>(input_view.len() / size_of::<f16>()).unwrap();
                    let w = weights_view
                        .transmute::<f16>(weights_view.len() / size_of::<f16>())
                        .unwrap();
                    let mut dst = output_view
                        .transmute_mut::<f16>(output_view.len() / size_of::<f16>())
                        .unwrap();
                    conv_forward
                        .launch(
                            *algo,
                            workspace.as_mut().map(|w| w.as_view_mut()).as_mut(),
                            (f16::from_f32(1.0), f16::from_f32(0.0)),
                            &src,
                            &w,
                            &mut dst,
                        )
                        .context("in launch()")?;
                }
            }
        }

        Ok(())
    }
}
