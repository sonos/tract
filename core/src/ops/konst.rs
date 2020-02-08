use crate::internal::*;

#[derive(Debug, Clone, new)]
pub struct Const {
    pub value: Arc<Tensor>,
}

impl Const {
    pub fn for_tensor(tensor: Tensor) -> Const {
        Const { value: tensor.into() }
    }
}

impl Op for Const {
    fn name(&self) -> Cow<str> {
        "Const".into()
    }

    op_as_typed_op!();
    not_a_pulsed_op!();
}

impl StatelessOp for Const {
    fn eval(&self, _inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        Ok(tvec![self.value.clone()])
    }
}

impl TypedOp for Const {
    as_op!();

    fn output_facts(&self, _inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(self.value.as_ref().into()))
    }
}
