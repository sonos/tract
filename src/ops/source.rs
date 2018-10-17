use analyser::rules::prelude::*;
use ops::prelude::*;

#[derive(Debug, Clone, new, Default)]
pub struct Source {
    fact: TensorFact,
}

impl Op for Source {
    fn name(&self) -> &str {
        "Source"
    }

    fn infer(
        &self,
        inputs: TVec<&TensorFact>,
        outputs: TVec<&TensorFact>,
    ) -> TfdResult<(TVec<TensorFact>, TVec<TensorFact>)> {
        self.infer_facts(inputs, outputs)
    }

    fn pulsify(&self, _inputs:TVec<&TensorFact>, _outputs:TVec<&TensorFact>, pulse: usize) -> TfdResult<PulsifiedOp> {
        if let Some(stream) = self.fact.stream_info()? {
            let mut pulse_fact = self.fact.clone();
            pulse_fact.shape.dims[stream.axis] = pulse.to_dim().into();
            return Ok(PulsifiedOp::op(Box::new(Self::new(pulse_fact.clone()))))
        }
        bail!("Could not pulsify {:?}", self)
    }
}

impl StatelessOp for Source {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, _inputs: TVec<Value>) -> TfdResult<TVec<Value>> {
        panic!("Source should not get evaluated")
    }
}

impl InferenceRulesOp for Source {
    /// Registers the inference rules of the operator.
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p TensorsProxy,
        outputs: &'p TensorsProxy,
    ) -> InferenceResult {
        s.equals(&inputs.len, 0)?;
        s.equals(&outputs.len, 1)?;
        if let GenericFact::Only(dt) = self.fact.datum_type {
            s.equals(&outputs[0].datum_type, dt)?;
        }
        if let Some(shape) = self.fact.shape.concretize() {
            s.equals(&outputs[0].shape, shape)?;
        }
        Ok(())
    }
}

