use crate::internal::*;

#[derive(Debug, Clone, Default, Hash, PartialEq, Eq)]
pub struct Identity;

impl Op for Identity {
    fn name(&self) -> StaticName {
        "Identity".into()
    }

    op_as_typed_op!();
}

impl EvalOp for Identity {
    op_out_of_plan!();

    /// Evaluates the operation given the input tensors.
    fn eval(&self, _ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        Ok(inputs)
    }
}

impl TypedOp for Identity {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(inputs[0].clone()))
    }

    fn input_roi(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TVec<Option<TDim>>>> {
        crate::optim::propagate_roi::bubble_roi(model, node)
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        TypedModelPatch::shunt_one_op(model, node)
    }

    fn fuse(&self, model: &TypedModel, node: &TypedNode) -> TractResult<Option<TypedModelPatch>> {
        TypedModelPatch::shunt_one_op(model, node)
    }

    fn axes_mapping(
        &self,
        inputs: &[&TypedFact],
        outputs: &[&TypedFact],
    ) -> TractResult<AxesMapping> {
        AxesMapping::natural(inputs, outputs)
    }

    as_op!();
}
