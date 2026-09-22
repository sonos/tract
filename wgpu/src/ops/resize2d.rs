use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensorExt;

use crate::kernels::resize::wgpu_resize_2d_dispatch;

/// A resize over the two trailing axes in one launch, each output gathering
/// its `window_h x window_w` taps straight from the input. The per-axis form
/// costs a launch and an intermediate per axis, and re-uploads its plans every
/// frame; here the plans are inputs 1..=4 (row indices, row weights, column
/// indices, column weights), constants that live on the device.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WgpuResize2d {
    pub window_h: usize,
    pub window_w: usize,
    pub output_shape: TVec<usize>,
}

impl Op for WgpuResize2d {
    fn name(&self) -> StaticName {
        "WgpuResize2d".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("windows {}x{} -> {:?}", self.window_h, self.window_w, self.output_shape)])
    }

    op_as_typed_op!();
}

impl EvalOp for WgpuResize2d {
    op_out_of_plan!();

    fn eval(&self, ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let inputs =
            inputs.iter().map(|it| it.to_device_tensor()).collect::<TractResult<TVec<_>>>()?;
        let input = inputs[0];
        let output = tract_gpu::turn_handler::make_tensor_for_node(
            ctx,
            input.datum_type(),
            &self.output_shape,
        )?;
        if output.len() > 0 {
            wgpu_resize_2d_dispatch(
                input,
                (inputs[1], inputs[2], self.window_h),
                (inputs[3], inputs[4], self.window_w),
                &output,
            )?;
        }
        Ok(tvec!(output.into_tensor().into_tvalue()))
    }
}

impl TypedOp for WgpuResize2d {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        tract_gpu::utils::facts_to_device_facts(inputs, |facts| {
            let shape: TVec<TDim> = self.output_shape.iter().map(|d| d.to_dim()).collect();
            Ok(tvec!(facts[0].datum_type.fact(&shape)))
        })
        .with_context(|| "Error while computing facts for WgpuResize2d")
    }
}
