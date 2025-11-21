use crate::context::{TractCudaStream, cuda_context};
use crate::kernels::get_cuda_view;
use crate::kernels::launch_args::LaunchArgsExt;
use cudarc::driver::{LaunchArgs, LaunchConfig, PushKernelArg};
use std::fmt::Debug;
use tract_core::dyn_clone::{self, DynClone};
use tract_core::internal::*;
use tract_core::ops::cnn::Conv;
use tract_gpu::tensor::DeviceTensor;

pub trait ConvKernel: 'static + Send + Sync + Debug + DynClone {
    fn name(&self) -> StaticName;
    fn dispatch(
        &self,
        op: &Conv,
        stream: &TractCudaStream,
        input: &DeviceTensor,
        weights: &DeviceTensor,
        bias: &DeviceTensor,
        output: &DeviceTensor,
    ) -> TractResult<()>;
}
dyn_clone::clone_trait_object!(ConvKernel);

#[derive(Hash, Clone, Debug)]
pub struct Generic;

impl ConvKernel for Generic {
    fn name(&self) -> StaticName {
        "Generic".into()
    }

    fn dispatch(
        &self,
        op: &Conv,
        stream: &TractCudaStream,
        input: &DeviceTensor,
        weights: &DeviceTensor,
        bias: &DeviceTensor,
        output: &DeviceTensor,
    ) -> TractResult<()> {
        let input_shape = op.pool_spec.data_format.shape(input.shape())?;

        let ctx = cuda_context();
        let func_name = format!("conv{}d_f32_generic", input_shape.hw_rank());
        let func = ctx.load_pipeline(crate::kernels::LibraryName::Cnn, func_name)?;

        let mut launcher = stream.launch_builder(&func);

        let input = get_cuda_view(input);

        launcher.arg(&input);
        launcher.arg(input_shape.n().unwrap_or(&1));
        launcher.arg(input_shape.c());
        launcher.set_slice(input_shape.hw_dims());

        launcher.arg(input_shape.n_stride().unwrap_or(&0));
        launcher.arg(input_shape.c_stride());
        launcher.set_slice(input_shape.hw_strides());

        let kfmt = op.kernel_fmt;
        let co_per_group = op.pool_spec.output_channels / op.group;
        let ci_per_group = op.pool_spec.input_channels / op.group;

        let weights_view = get_cuda_view(weights);
        launcher.arg(&weights_view);
        // split go_i_h_w in g_o_i_h_w
        launcher.arg(&op.group);
        launcher.arg(&co_per_group);
        launcher.set_slice(&weights.shape()[1..]);

        let group_stride = weights.strides()[0] as usize * co_per_group;
        launcher.arg(&group_stride);
        launcher.set_slice(weights.strides());

        let bias_view = get_cuda_view(bias);
        launcher.arg(&bias_view);
        launcher.arg(if bias.rank() == 0 {
            &0usize // scalar bias: stride = 0 is broadcasting
        } else {
            &1usize
        });

        let padding = op.pool_spec.computed_padding(input_shape.hw_dims());
        for d in 0..input_shape.hw_rank() {
            launcher.arg(&padding[d].pad_before);
        }

        let strides = op.pool_spec.strides();
        launcher.set_slice(&strides);

        let dilations = op.pool_spec.dilations();
        launcher.set_slice(&dilations);

        let output_shape = op.pool_spec.data_format.shape(output.shape())?;
        let output = get_cuda_view(output);
        launcher.arg(&output);
        launcher.arg(output_shape.n().unwrap_or(&1));
        launcher.arg(output_shape.c());
        launcher.set_slice(output_shape.hw_dims());

        launcher.arg(output_shape.n_stride().unwrap_or(&0));
        launcher.arg(output_shape.c_stride());
        launcher.set_slice(output_shape.hw_strides());

        let cfg = LaunchConfig {
            grid_dim: (
                output_shape.hw_dims().iter().product::<usize>() as u32,
                *output_shape.c() as u32,
                input_shape.n().copied().unwrap_or(1) as u32,
            ),
            block_dim: (32, 1, 1),
            shared_mem_bytes: 0,
        };

        unsafe {
            launcher.launch(cfg);
        }
        Ok(())
    }
}
