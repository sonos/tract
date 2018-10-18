use analyser::rules::prelude::*;
use ops::prelude::*;

#[derive(Debug, Clone, new)]
pub struct Sink {
    fact: TensorFact,
}

impl Op for Sink {
    fn name(&self) -> &str {
        "Sink"
    }
}

impl StatelessOp for Sink {
    fn eval(&self, _inputs: TVec<Value>) -> TfdResult<TVec<Value>> {
        Ok(tvec!())
    }
}

impl InferenceRulesOp for Sink {
    /// Registers the inference rules of the operator.
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p TensorsProxy,
        outputs: &'p TensorsProxy,
    ) -> InferenceResult {
        s.equals(&inputs.len, 1)?;
        s.equals(&outputs.len, 0)?;
        if let GenericFact::Only(dt) = self.fact.datum_type {
            s.equals(&inputs[0].datum_type, dt)?;
        }
        if let Some(shape) = self.fact.shape.concretize() {
            s.equals(&inputs[0].shape, shape)?;
        }
        Ok(())
    }
}
