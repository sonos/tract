use crate::kernels::nn::NewGelu;
use crate::ops::MetalEvalOp;
use crate::{MetalContext, MetalTensorExt};
use derive_new::new;
use tract_core::internal::*;

#[derive(Clone, Debug, new, Hash)]
pub struct MetalNewGelu;

impl Op for MetalNewGelu {
    fn name(&self) -> Cow<str> {
        "MetalNewGelu".into()
    }

    op_as_typed_op!();
}

crate::impl_eval_op_for_metal_op!(MetalNewGelu);

impl MetalEvalOp for MetalNewGelu {
    fn metal_eval(
        &self,
        context: &MetalContext,
        node_id: usize,
        session: &mut SessionState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs);
        let input_metal = input.to_metal_tensor()?;
        let output = crate::ops::make_tensor_for_node(
            session,
            node_id,
            input_metal.datum_type(),
            input_metal.shape(),
        )?;
        NewGelu::accurate().dispatch_eval(context, input_metal, &output)?;
        Ok(tvec!(output.into_opaque_tensor().into_tvalue()))
    }
}

impl TypedOp for MetalNewGelu {
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
