use crate::internal::*;

#[derive(Debug, Clone, Default, Hash)]
pub struct Identity;

impl Op for Identity {
    fn name(&self) -> Cow<str> {
        "Identity".into()
    }

    op_core_mir!();
    op_as_typed_op!();
}

tract_data::impl_dyn_hash!(Identity);

impl EvalOp for Identity {
    fn is_stateless(&self) -> bool {
        true
    }

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
