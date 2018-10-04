use analyser::rules::prelude::*;
use ops::prelude::*;

#[derive(Debug, Clone, Default)]
pub struct Identity;

impl Op for Identity {
    fn name(&self) -> &str {
        "Identity"
    }

    /// Evaluates the operation given the input tensors.
    fn eval(&self, inputs: TVec<Value>) -> TfdResult<TVec<Value>> {
        Ok(inputs)
    }

    /// Evaluates one step of the operation on the given input tensors.
    fn step(
        &self,
        mut inputs: TVec<StepValue>,
        _: &mut Box<OpBuffer>,
    ) -> TfdResult<Option<TVec<Value>>> {
        let input = args_1!(inputs);
        match input.into_value() {
            None => Ok(None),
            Some(tv) => Ok(Some(self.eval(tvec![tv])?)),
        }
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
