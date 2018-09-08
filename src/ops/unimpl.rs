use analyser::rules::prelude::*;
use ops::prelude::*;

#[derive(Debug, Clone)]
pub struct UnimplementedOp(pub String, pub ::tfpb::node_def::NodeDef);

impl Op for UnimplementedOp {
    /// Evaluates the operation given the input tensors.
    fn eval(&self, _inputs: TVec<Value>) -> Result<TVec<Value>> {
        Err(format!("unimplemented operation: {}", self.0))?
    }
}

impl InferenceRulesOp for UnimplementedOp {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        _: &mut Solver<'r>,
        _: &'p TensorsProxy,
        _: &'p TensorsProxy,
    ) {
    }
}
