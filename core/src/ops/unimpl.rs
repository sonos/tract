use crate::ops::prelude::*;

#[derive(Debug, Clone)]
pub struct UnimplementedOp {
    name: String,
    message: String,
}

impl UnimplementedOp {
    pub fn new(name: impl AsRef<str>, message: impl AsRef<str>) -> UnimplementedOp {
        UnimplementedOp {
            name: name.as_ref().to_string(),
            message: message.as_ref().to_string(),
        }
    }
}

impl Op for UnimplementedOp {
    fn name(&self) -> Cow<str> {
        format!("Unimplemented({})", self.name).into()
    }
}

impl StatelessOp for UnimplementedOp {
    fn eval(&self, _inputs: TVec<SharedTensor>) -> TractResult<TVec<SharedTensor>> {
        bail!("unimplemented operation: {}", self.name)
    }
}

impl InferenceRulesOp for UnimplementedOp {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        _: &mut Solver<'r>,
        _: &'p [TensorProxy],
        _: &'p [TensorProxy],
    ) -> InferenceResult {
        Ok(())
    }
}
