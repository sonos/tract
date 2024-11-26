use crate::kernels::nn::RmsNorm;
use crate::ops::MetalEvalOp;
use crate::{MetalContext, MetalTensorExt};
use derive_new::new;
use std::sync::Arc;
use tract_core::internal::*;

#[derive(Clone, Debug, new, Hash)]
pub struct MetalRmsNorm {
    pub axis: usize,
    pub eps: Arc<Tensor>,
}

impl Op for MetalRmsNorm {
    fn name(&self) -> Cow<str> {
        "MetalRmsNorm".into()
    }
    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("axis: {:?}, eps: {:?}", self.axis, self.eps)])
    }
    op_as_typed_op!();
}

crate::impl_eval_op_for_metal_op!(MetalRmsNorm);

impl MetalEvalOp for MetalRmsNorm {
    fn metal_eval(
        &self,
        context: &MetalContext,
        node_id: usize,
        session: &mut SessionState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let opaque = args_1!(inputs);
        let input = opaque.to_metal_tensor()?;
        let output =
            crate::ops::make_tensor_for_node(session, node_id, input.datum_type(), input.shape())?;
        RmsNorm.dispatch_eval(context, input, self.axis, &self.eps, &output)?;
        Ok(tvec!(output.into_opaque_tensor().into_tvalue()))
    }
}

impl TypedOp for MetalRmsNorm {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        crate::utils::metal_facts_from_gpu(inputs, |facts| {
            let dt = facts[0].datum_type;
            let fact = dt.fact(facts[0].shape.clone());
            Ok(tvec!(fact))
        })
        .with_context(|| anyhow::anyhow!("Error while computing facts for {:?}", self.name()))
    }

    as_op!();
}
