use crate::internal::*;

#[derive(Debug, Clone, new)]
pub struct Cast {
    to: DatumType,
}

impl Cast {
    /// Evaluates the operation given the input tensors.
    fn eval_t<T: Datum>(input: SharedTensor) -> TractResult<SharedTensor> {
        Ok(input.cast_to::<T>()?.into_owned().into_tensor())
    }
}

impl Op for Cast {
    fn name(&self) -> Cow<str> {
        "Cast".into()
    }
}

impl StatelessOp for Cast {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, mut inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
        let input = args_1!(inputs);
        let output = dispatch_datum!(Self::eval_t(self.to)(input))?;
        Ok(tvec!(output))
    }
}

impl InferenceRulesOp for Cast {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_input_arity(&inputs, 1)?;
        check_output_arity(&outputs, 1)?;
        s.equals(&outputs[0].datum_type, self.to)?;
        s.equals(&inputs[0].shape, &outputs[0].shape)?;
        Ok(())
    }
}
