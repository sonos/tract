use crate::ops::MetalEvalOp;
use crate::{kernels, MetalContext, MetalTensorExt};
use derive_new::new;
use std::fmt::Debug;
use tract_core::internal::*;

#[derive(Debug, Clone, new, Hash)]
pub struct MetalMultiBroadcastTo {
    pub shape: ShapeFact,
}

impl Op for MetalMultiBroadcastTo {
    fn name(&self) -> Cow<str> {
        "MetalMultiBroadcastTo".into()
    }

    op_as_typed_op!();
}

crate::impl_eval_op_for_metal_op!(MetalMultiBroadcastTo);

impl MetalEvalOp for MetalMultiBroadcastTo {
    fn metal_eval(
        &self,
        context: &MetalContext,
        node_id: usize,
        session: &mut SessionState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let opaque = args_1!(inputs);
        let shape = self.shape.eval_to_usize(&session.resolved_symbols)?;
        let input = opaque.to_metal_tensor()?;
        let output =
            crate::ops::make_tensor_for_node(session, node_id, input.datum_type(), &shape)?;
        kernels::array::MultiBroadcast.dispatch_eval(context, input, 0, &output)?;
        Ok(tvec![output.into_opaque_tensor().into_tvalue()])
    }
}

impl TypedOp for MetalMultiBroadcastTo {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        crate::utils::metal_facts_from_gpu(inputs, |facts| {
            let mut fact = facts[0].datum_type.fact(self.shape.clone());
            fact.uniform.clone_from(&inputs[0].uniform);
            Ok(tvec!(fact))
        })
        .with_context(|| anyhow::anyhow!("Error while computing facts for {:?}", self.name()))
    }

    as_op!();
}
