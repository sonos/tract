use crate::ops::prelude::*;

#[derive(Debug, Clone, Default)]
pub struct Identity;

impl Op for Identity {
    fn name(&self) -> Cow<str> {
        "Identity".into()
    }
}

impl StatelessOp for Identity {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
        Ok(inputs)
    }
}

impl InferenceRulesOp for Identity {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p TensorsProxy,
        outputs: &'p TensorsProxy,
    ) -> InferenceResult {
        s.equals(&inputs.len, 1)?;
        s.equals(&outputs.len, 1)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[0].shape, &outputs[0].shape)?;
        Ok(())
    }
}
