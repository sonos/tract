use crate::ops::prelude::*;

#[derive(Debug, Clone, new)]
pub struct Const {
    value: SharedTensor,
}

impl Const {
    pub fn for_tensor(tensor: Tensor) -> Const {
        Const {
            value: tensor.into(),
        }
    }
}

impl Op for Const {
    fn name(&self) -> Cow<str> {
        "Const".into()
    }

    fn const_value(&self) -> Option<SharedTensor> {
        Some(self.value.clone())
    }
}

impl StatelessOp for Const {
    fn eval(&self, _inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
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
