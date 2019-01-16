use crate::ops::prelude::*;

#[derive(Debug, Clone, new, Default)]
pub struct Source {
    pub fact: TensorFact,
}

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
    fn eval(&self, _inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
        panic!("Source should not get evaluated")
    }
}

impl InferenceRulesOp for Source {
    /// Registers the inference rules of the operator.
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p SharedTensorsProxy,
        outputs: &'p SharedTensorsProxy,
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
