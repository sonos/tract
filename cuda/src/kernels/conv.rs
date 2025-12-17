use crate::context::{TractCudaStream, cuda_context};
use crate::kernels::launch_args::TractLaunchArgs;
use crate::kernels::{WARP_SIZE, get_cuda_view};
use cudarc::driver::{LaunchArgs, LaunchConfig, PushKernelArg};
use downcast_rs::{Downcast, impl_downcast};
use std::any::Any;
use std::fmt::Debug;
use tract_core::dyn_clone::{self, DynClone};
use tract_core::internal::*;
use tract_core::ops::cnn::Conv;
use tract_gpu::tensor::DeviceTensor;

pub trait ConvKernelScratch: Debug + Downcast {}
impl_downcast!(ConvKernelScratch);

pub trait ConvKernel: 'static + Send + Sync + Debug + DynClone {
    fn name(&self) -> StaticName;
    #[allow(clippy::too_many_arguments)]
    fn state(&self) -> Box<dyn ConvKernelScratch>;
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
    ) -> TractResult<()>;
}
dyn_clone::clone_trait_object!(ConvKernel);

impl ConvKernelScratch for () {}

#[derive(Hash, Clone, Debug)]
pub struct ConvGeneric;

impl ConvKernel for ConvGeneric {
    fn name(&self) -> StaticName {
        "Generic".into()
    }

    fn state(&self) -> Box<dyn ConvKernelScratch> {
        Box::new(())
    }

    fn dispatch(
        &self,
        _state: &mut dyn ConvKernelScratch,
        _node_id: usize,
        op: &Conv,
        stream: &TractCudaStream,
        input: &DeviceTensor,
        weights: &DeviceTensor,
        bias: Option<&DeviceTensor>,
        output: &DeviceTensor,
    ) -> TractResult<()> {
        let input_shape = op.pool_spec.data_format.shape(input.shape())?;

        let ctx = cuda_context();
        let func_name = format!("conv{}d_f32_generic", input_shape.hw_rank());
        let func = ctx.load_pipeline(crate::kernels::LibraryName::Cnn, func_name)?;
        let null = stream.null::<u8>()?;
        let null_view = null.as_view();

        let mut launcher = TractLaunchArgs::new(stream, &func);

        let input = get_cuda_view(input);

        launcher.push_view(&input);
        launcher.push_i32(*input_shape.n().unwrap_or(&1));
        launcher.push_i32(*input_shape.c());
        launcher.push_slice_i32(input_shape.hw_dims());

        launcher.push_i32(*input_shape.n_stride().unwrap_or(&0));
        launcher.push_i32(*input_shape.c_stride());
        launcher.push_slice_i32(input_shape.hw_strides());

        let kfmt = op.kernel_fmt;
        let co_per_group = op.pool_spec.output_channels / op.group;
        let ci_per_group = op.pool_spec.input_channels / op.group;

        let weights_view = get_cuda_view(weights);
        launcher.push_view(&weights_view);
        // split go_i_h_w in g_o_i_h_w
        launcher.push_i32(op.group);
        launcher.push_i32(co_per_group);
        launcher.push_slice_i32(&weights.shape()[1..]);

        let group_stride = weights.strides()[0] as usize * co_per_group;
        launcher.push_i32(group_stride);
        launcher.push_slice_i32(weights.strides());

        let mut bias_view = None;
        if let Some(bias) = &bias {
            bias_view = Some(get_cuda_view(bias));
            launcher.push_view(bias_view.as_ref().unwrap());
            launcher.push_i32(if bias.rank() == 0 {
                0 // scalar bias: stride = 0 is broadcasting
            } else {
                1
            });
        } else {
            launcher.push_view(&null_view);
            launcher.push_i32(0);
        }

        let padding = op.pool_spec.computed_padding(input_shape.hw_dims());
        for d in 0..input_shape.hw_rank() {
            launcher.push_i32(padding[d].pad_before);
        }

        let strides = op.pool_spec.strides();
        launcher.push_slice_i32(&strides);

        let dilations = op.pool_spec.dilations();
        launcher.push_slice_i32(&dilations);

        let output_shape = op.pool_spec.data_format.shape(output.shape())?;
        let output = get_cuda_view(output);
        launcher.push_view(&output);
        launcher.push_i32(*output_shape.n().unwrap_or(&1));
        launcher.push_i32(*output_shape.c());
        launcher.push_slice_i32(output_shape.hw_dims());

        launcher.push_i32(*output_shape.n_stride().unwrap_or(&0));
        launcher.push_i32(*output_shape.c_stride());
        launcher.push_slice_i32(output_shape.hw_strides());

        let cfg = LaunchConfig {
            grid_dim: (
                output_shape.hw_dims().iter().product::<usize>().div_ceil(WARP_SIZE) as u32,
                *output_shape.c() as u32,
                input_shape.n().copied().unwrap_or(1) as u32,
            ),
            block_dim: (WARP_SIZE as u32, 1, 1),
            shared_mem_bytes: 0,
        };

        launcher.launch(cfg)
    }
}
