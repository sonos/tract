use crate::internal::*;

#[derive(Debug, Clone, new, Default)]
pub struct Source {}

impl Op for Source {
    fn name(&self) -> Cow<str> {
        "Source".into()
    }

    fn infer(
        &self,
        inputs: TVec<&TensorFact>,
        outputs: TVec<&TensorFact>,
    ) -> TractResult<(TVec<TensorFact>, TVec<TensorFact>)> {
        self.infer_facts(inputs, outputs)
    }
}

impl StatelessOp for Source {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, _inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        panic!("Source should not get evaluated")
    }
}

impl InferenceRulesOp for Source {
    /// Registers the inference rules of the operator.
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        _s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 0)?;
        check_output_arity(&outputs, 1)?;
        Ok(())
    }
}
