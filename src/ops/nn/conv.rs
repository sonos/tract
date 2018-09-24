use analyser::rules::prelude::*;
use ops::prelude::*;

#[derive(Debug, Clone, new, Default)]
pub struct Conv;

impl Op for Conv {
    fn name(&self) -> &str {
        "Conv"
    }
}

impl InferenceRulesOp for Conv {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        solver: &mut Solver<'r>,
        inputs: &'p TensorsProxy,
        outputs: &'p TensorsProxy,
    ) {
        solver
            .equals(&outputs.len, 2);
    }
}
