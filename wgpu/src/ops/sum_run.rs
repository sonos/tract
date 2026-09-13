use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensorExt;

use crate::kernels::reduce::wgpu_sum_run_launch;

/// A sum over a contiguous run of trailing axes, one workgroup per output
/// value: the global pooling of a squeeze-excitation gate, where a per-axis
/// pass would be one launch per axis and a serial loop would starve the
/// device. The shape keeps the reduced axes as ones.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WgpuSumRun {
    pub first_axis: usize,
}

impl Op for WgpuSumRun {
    fn name(&self) -> StaticName {
        "WgpuSumRun".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("axes: {}..", self.first_axis)])
    }

    op_as_typed_op!();
}

impl EvalOp for WgpuSumRun {
    op_out_of_plan!();

    fn eval(&self, ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs);
        let input = input.to_device_tensor()?;
        let mut shape = input.shape().to_vec();
        for d in &mut shape[self.first_axis..] {
            *d = 1;
        }
        let output =
            tract_gpu::turn_handler::make_tensor_for_node(ctx, input.datum_type(), &shape)?;
        if output.len() > 0 {
            wgpu_sum_run_launch(input, self.first_axis, &output)?;
        }
        Ok(tvec!(output.into_tensor().into_tvalue()))
    }
}

impl TypedOp for WgpuSumRun {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        tract_gpu::utils::facts_to_device_facts(inputs, |facts| {
            let mut shape: TVec<_> = facts[0].shape.to_tvec();
            for d in &mut shape[self.first_axis..] {
                *d = 1.to_dim();
            }
            Ok(tvec!(facts[0].datum_type.fact(shape)))
        })
        .with_context(|| "Error while computing facts for WgpuSumRun")
    }
}
