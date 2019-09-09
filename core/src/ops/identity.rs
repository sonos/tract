use crate::internal::*;

#[derive(Debug, Clone, Default)]
pub struct Identity;

impl Op for Identity {
    fn name(&self) -> Cow<str> {
        "Identity".into()
    }

    fn fuse(&self, model: &TypedModel, node: &TypedNode) -> TractResult<Option<TypedModelPatch>> {
        Ok(Some(TypedModelPatch::shunt_one_op(model, node)?))
    }

    op_as_typed_op!();
}

impl StatelessOp for Identity {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        Ok(inputs)
    }
}

impl InferenceRulesOp for Identity {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 1)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[0].shape, &outputs[0].shape)?;
        Ok(())
    }

    inference_op_as_op!();
    to_typed!();
}

impl TypedOp for Identity {
    typed_op_as_op!();

    fn output_facts(&self, inputs: &[&TypedTensorInfo]) -> TractResult<TVec<TypedTensorInfo>> {
        Ok(tvec!(inputs[0].clone()))
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        Ok(Some(TypedModelPatch::shunt_one_op(model, node)?))
    }
}
