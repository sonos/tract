use analyser::rules::prelude::*;
use ops::prelude::*;

#[derive(Clone, Default, Debug, new)]
struct Delay {
    axis: usize,
    delay: usize,
    count: usize,
}

impl Op for Delay {
    fn name(&self) -> &str {
        "Delay"
    }

    /// Evaluates the operation given the input tensors.
    fn eval(&self, _inputs: TVec<Value>) -> TfdResult<TVec<Value>> {
        unimplemented!();
    }
}

impl InferenceRulesOp for Delay {
    /// Registers the inference rules of the operator.
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p TensorsProxy,
        outputs: &'p TensorsProxy,
    ) -> InferenceResult {
        s.equals(&inputs.len, 0)?;
        s.equals(&outputs.len, 1)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[0].rank, &outputs[0].rank)?;
        s.given(&inputs[0].rank, move |s, rank| {
            for ax in 0..(rank as usize) {
                if ax == self.axis {
                    s.equals(&outputs[0].shape[ax], self.count.to_dim())?;
                } else {
                    s.equals(&outputs[0].shape[ax], &inputs[0].shape[ax])?;
                }
            }
            Ok(())
        })?;
        Ok(())
    }
}

