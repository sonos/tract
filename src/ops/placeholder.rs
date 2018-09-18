use analyser::rules::prelude::*;
use ndarray::prelude::*;
use ops::prelude::*;

#[derive(Debug, Clone, new)]
pub struct Placeholder {
    fact: TensorFact,
}

impl Placeholder {
    pub fn build(node: &::tf::tfpb::node_def::NodeDef) -> Result<Box<Op>> {
        let dt = node.get_attr_datum_type("dtype")?;
        Ok(Box::new(Placeholder {
            fact: TensorFact::dt(dt.into()),
        }))
    }
}

impl Op for Placeholder {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, _inputs: TVec<Value>) -> Result<TVec<Value>> {
        panic!("Placeholder should not get evaluated")
    }
}

impl InferenceRulesOp for Placeholder {
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

