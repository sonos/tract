use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensorExt;

use crate::kernels::chain::wgpu_chain_dispatch;
use crate::kernels::shaders::ChainStep;

/// A run of elementwise ops evaluated as one kernel. Input 0 is the head of the
/// chain and carries the output shape; the rest are the second operands of the
/// binary links, and broadcast against it.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct WgpuElementWiseChain {
    pub steps: Vec<ChainStep>,
}

impl Op for WgpuElementWiseChain {
    fn name(&self) -> StaticName {
        "WgpuElementWiseChain".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(self
            .steps
            .iter()
            .map(|s| match s {
                ChainStep::Unary(op) => op.clone(),
                ChainStep::Binary { op, rhs, swapped } => {
                    format!("{op}(#{rhs}){}", if *swapped { " swapped" } else { "" })
                }
            })
            .collect())
    }

    op_as_typed_op!();
}

impl EvalOp for WgpuElementWiseChain {
    op_out_of_plan!();

    fn eval(&self, ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let inputs =
            inputs.iter().map(|it| it.to_device_tensor()).collect::<TractResult<TVec<_>>>()?;
        let head = inputs[0];
        let output =
            tract_gpu::turn_handler::make_tensor_for_node(ctx, head.datum_type(), head.shape())?;
        if output.len() > 0 {
            wgpu_chain_dispatch(&self.steps, &inputs, &output)?;
        }
        Ok(tvec!(output.into_tensor().into_tvalue()))
    }
}

impl TypedOp for WgpuElementWiseChain {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        tract_gpu::utils::facts_to_device_facts(inputs, |facts| {
            Ok(tvec!(facts[0].datum_type.fact(facts[0].shape.clone())))
        })
        .with_context(|| "Error while computing facts for WgpuElementWiseChain")
    }
}
