use analyser::rules::prelude::*;
use ops::prelude::*;

#[derive(Debug, Clone, new)]
pub struct Sink {
    fact: TensorFact,
}

impl Sink {
    pub fn build(node: &::tf::tfpb::node_def::NodeDef) -> Result<Box<Op>> {
        let dt = node.get_attr_datum_type("dtype")?;
        Ok(Box::new(Sink {
            fact: TensorFact::dt(dt.into()),
        }))
    }
}

impl Op for Sink {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, _inputs: TVec<Value>) -> Result<TVec<Value>> {
        Ok(tvec!())
    }
}

impl InferenceRulesOp for Sink {
    /// Registers the inference rules of the operator.
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        solver: &mut Solver<'r>,
        inputs: &'p TensorsProxy,
        outputs: &'p TensorsProxy,
    ) {
        solver
            .equals(&inputs.len, 1)
            .equals(&outputs.len, 0);
        if let GenericFact::Only(dt) = self.fact.datum_type {
            solver.equals(&inputs[0].datum_type, dt);
        }
        if let Some(shape) = self.fact.shape.concretize() {
            solver.equals(&inputs[0].shape, shape);
        }
    }
}

