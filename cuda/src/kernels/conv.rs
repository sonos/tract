use crate::context::{cuda_context, TractCudaStream};
use crate::kernels::get_cuda_view;
use cudarc::driver::{LaunchConfig, PushKernelArg};
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
        let ctx = cuda_context();
        let func =
            ctx.load_pipeline(crate::kernels::LibraryName::CNN, "conv2d_f32_generic".into())?;

        let mut launcher = stream.launch_builder(&func);
        let size_of = f32::datum_type().size_of();

        let input_shape = op.pool_spec.data_format.shape(input.shape())?;
        let input = get_cuda_view(input);
        launcher.arg(&input);
        launcher.arg(input_shape.n().unwrap_or(&1));
        launcher.arg(input_shape.c());
        launcher.arg(&input_shape.hw_dims()[0]);
        launcher.arg(&input_shape.hw_dims()[1]);

        let input_strides: TVec<_> = [
            input_shape.n_stride().unwrap_or(&0),
            input_shape.c_stride(),
            &input_shape.hw_strides()[0],
            &input_shape.hw_strides()[1],
        ]
        .into_iter()
        .map(|s| *s as usize * size_of)
        .collect();

        for stride in &input_strides {
            launcher.arg(stride);
        }

        let kfmt = op.kernel_fmt;
        let weights_view = get_cuda_view(weights);
        launcher.arg(&weights_view);
        launcher.arg(kfmt.o(weights.shape()));
        launcher.arg(kfmt.i(weights.shape()));
        launcher.arg(&kfmt.hw(weights.shape())[0]);
        launcher.arg(&kfmt.hw(weights.shape())[1]);

        let ker_strides: TVec<_> = [
            kfmt.o_axis(weights.shape()),
            kfmt.i_axis(weights.shape()),
            kfmt.h_axis(),
            kfmt.h_axis() + 1,
        ]
        .iter()
        .map(|axis| weights.strides()[*axis] as usize * size_of)
        .collect();

        for stride in &ker_strides {
            launcher.arg(stride);
        }

        let bias_view = get_cuda_view(bias);
        launcher.arg(&bias_view);
        if bias.rank() == 0 {
            launcher.arg(&0); // scalar bias: stride = 0 is broadcasting
        } else {
            launcher.arg(&size_of);
        }

        let ci_per_group = op.pool_spec.input_channels / op.group;
        let co_per_group = op.pool_spec.output_channels / op.group;
        launcher.arg(&ci_per_group);
        launcher.arg(&co_per_group);

        let padding = op.pool_spec.computed_padding(&input_shape.hw_dims());
        launcher.arg(&padding[0].pad_before);
        launcher.arg(&padding[1].pad_before);

        let strides = op.pool_spec.strides();
        launcher.arg(&strides[0]);
        launcher.arg(&strides[1]);

        let dilations = op.pool_spec.dilations();
        launcher.arg(&dilations[0]);
        launcher.arg(&dilations[1]);

        let output_shape = op.pool_spec.data_format.shape(output.shape())?;
        let output = get_cuda_view(output);
        launcher.arg(&output);
        launcher.arg(output_shape.n().unwrap_or(&1));
        launcher.arg(output_shape.c());
        launcher.arg(&output_shape.hw_dims()[0]);
        launcher.arg(&output_shape.hw_dims()[1]);

        let output_strides: TVec<_> = [
            output_shape.n_stride().unwrap_or(&0),
            output_shape.c_stride(),
            &output_shape.hw_strides()[0],
            &output_shape.hw_strides()[1],
        ]
        .into_iter()
        .map(|s| *s as usize * size_of)
        .collect();

        for stride in &output_strides {
            launcher.arg(stride);
        }

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
        dbg!("launched!");
        Ok(())
    }
}
