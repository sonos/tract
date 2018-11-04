use ops::prelude::*;

#[derive(Debug, Clone, new)]
pub struct Const {
    value: Tensor,
}

impl Const {
    pub fn for_tensor(tensor: DtArray) -> Const {
        Const {
            value: tensor.into()
        }
    }
}

impl Op for Const {
    fn name(&self) -> &str {
        "Const"
    }

    fn const_value(&self) -> Option<Tensor> {
        Some(self.value.clone())
    }
}

impl StatelessOp for Const {
    fn eval(&self, _inputs: TVec<Tensor>) -> TractResult<TVec<Tensor>> {
        Ok(tvec![self.value.clone()])
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
