use crate::internal::*;

#[derive(Debug, Clone, Default)]
pub struct Identity;

impl Op for Identity {
    fn name(&self) -> Cow<str> {
        "Identity".into()
    }

    op_as_typed_op!();
    op_as_pulsed_op!();
}

impl StatelessOp for Identity {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        Ok(inputs)
    }
}

impl TypedOp for Identity {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(inputs[0].clone()))
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        Ok(Some(TypedModelPatch::shunt_one_op(model, node)?))
    }

    fn fuse(&self, model: &TypedModel, node: &TypedNode) -> TractResult<Option<TypedModelPatch>> {
        Ok(Some(TypedModelPatch::shunt_one_op(model, node)?))
    }

    as_op!();
}

impl PulsedOp for Identity {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        Ok(tvec!(inputs[0].clone()))
    }

    as_op!();
    pulsed_op_to_typed_op!();
}
