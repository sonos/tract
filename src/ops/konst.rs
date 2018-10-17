use analyser::rules::prelude::*;
use ops::prelude::*;

#[derive(Debug, Clone, new)]
pub struct Const {
    value: Value,
}

impl Const {
    pub fn for_tensor(tensor: Tensor) -> Const {
        let value: Value = tensor.into();
        Const {
            value: value.into_shared(),
        }
    }
}

impl Op for Const {
    fn name(&self) -> &str {
        "Const"
    }

    /// Evaluates the operation given the input tensors.
    fn eval(&self, _inputs: TVec<Value>) -> TfdResult<TVec<Value>> {
        Ok(tvec![self.value.clone()])
    }

    fn const_value(&self) -> Option<Value> {
        Some(self.value.clone())
    }

    fn pulsify(
        &self,
        _inputs: TVec<&TensorFact>,
        _outputs: TVec<&TensorFact>,
        _pulse: usize,
    ) -> TfdResult<PulsifiedOp> {
        return Ok(PulsifiedOp::op(Box::new(self.clone())));
    }
}

impl InferenceRulesOp for Const {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p TensorsProxy,
        outputs: &'p TensorsProxy,
    ) -> InferenceResult {
        s.equals(&inputs.len, 0)?;
        s.equals(&outputs.len, 1)
    }
}
