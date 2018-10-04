use analyser::rules::prelude::*;
use ops::prelude::*;

#[derive(Debug, Clone)]
pub struct UnimplementedOp(pub String, pub String);

impl Op for UnimplementedOp {
    fn name(&self) -> &str {
        "Unimplemented"
    }
    /// Evaluates the operation given the input tensors.
    fn eval(&self, _inputs: TVec<Value>) -> TfdResult<TVec<Value>> {
        Err(format!("unimplemented operation: {} {:?}", self.0, self.1))?
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
