use crate::internal::*;

#[derive(Debug, Clone, new)]
pub struct Const {
    value: Arc<Tensor>,
}

impl Const {
    pub fn for_tensor(tensor: Tensor) -> Const {
        Const { value: tensor.into() }
    }
}

impl Op for Const {
    fn name(&self) -> Cow<str> {
        "Const".into()
    }

    to_typed!();
}

impl StatelessOp for Const {
    fn eval(&self, _inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        Ok(tvec![self.value.clone()])
    }
}

impl InferenceRulesOp for Const {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 0)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&outputs[0].value, self.value.clone().bex())?;
        Ok(())
    }

    inference_op_as_op!();
}

impl TypedOp for Const {
    stub_typed_op_as_op!();
}
