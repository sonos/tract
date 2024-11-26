pub use crate::kernels::ElementWiseOps;
use crate::ops::MetalEvalOp;
use crate::{MetalContext, MetalTensorExt};
use tract_core::internal::*;

#[derive(Debug, Clone)]
pub struct MetalElementWiseOp(pub ElementWiseOps);

impl Op for MetalElementWiseOp {
    fn name(&self) -> Cow<str> {
        format!("Metal{}", self.0.name()).into()
    }

    fn validation(&self) -> Validation {
        self.0.validation()
    }

    fn same_as(&self, other: &dyn Op) -> bool {
        let Some(other) = other.downcast_ref::<MetalElementWiseOp>() else { return false };
        self.0 == other.0
    }

    op_as_typed_op!();
}

crate::impl_eval_op_for_metal_op!(MetalElementWiseOp);

impl MetalEvalOp for MetalElementWiseOp {
    fn metal_eval(
        &self,
        context: &MetalContext,
        node_id: usize,
        session: &mut SessionState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let opaque_a = args_1!(inputs);
        let a = opaque_a.to_metal_tensor()?;
        let output = crate::ops::make_tensor_for_node(session, node_id, a.datum_type(), a.shape())?;
        self.0.dispatch_eval(context, a, &output)?;
        Ok(tvec![output.into_opaque_tensor().into_tvalue()])
    }
}

impl TypedOp for MetalElementWiseOp {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        crate::utils::metal_facts_from_gpu(inputs, |facts| Ok(tvec!(facts[0].without_value())))
            .with_context(|| anyhow::anyhow!("Error while computing facts for {:?}", self.name()))
    }

    as_op!();
}
