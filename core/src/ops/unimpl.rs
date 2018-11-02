use ops::prelude::*;

#[derive(Debug, Clone)]
pub struct UnimplementedOp(pub String, pub String);

impl Op for UnimplementedOp {
    fn name(&self) -> &str {
        "Unimplemented"
    }
}

impl StatelessOp for UnimplementedOp {
    fn eval(&self, _inputs: TVec<Value>) -> TractResult<TVec<Value>> {
        Err(format!("unimplemented operation: {}", self.0))?
    }
}

impl InferenceRulesOp for UnimplementedOp {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        _: &mut Solver<'r>,
        _: &'p TensorsProxy,
        _: &'p TensorsProxy,
    ) -> InferenceResult {
        Ok(())
    }
}
