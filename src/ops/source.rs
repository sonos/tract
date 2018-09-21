use analyser::rules::prelude::*;
use ops::prelude::*;

#[derive(Debug, Clone, new)]
pub struct Source {
    fact: TensorFact,
}

impl Op for Source {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, _inputs: TVec<Value>) -> Result<TVec<Value>> {
        panic!("Source should not get evaluated")
    }
}

impl InferenceRulesOp for Source {
    /// Registers the inference rules of the operator.
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        solver: &mut Solver<'r>,
        inputs: &'p TensorsProxy,
        outputs: &'p TensorsProxy,
    ) {
        solver
            .equals(&inputs.len, 0)
            .equals(&outputs.len, 1);
        if let GenericFact::Only(dt) = self.fact.datum_type {
            solver.equals(&outputs[0].datum_type, dt);
        }
        if let Some(shape) = self.fact.shape.concretize() {
            solver.equals(&outputs[0].shape, shape);
        }
    }
}

