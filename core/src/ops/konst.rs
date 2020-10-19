use crate::internal::*;

#[derive(Debug, Clone, new, Hash)]
pub struct Const(pub Arc<Tensor>);

tract_data::impl_dyn_hash!(Const);

impl Op for Const {
    fn name(&self) -> Cow<str> {
        "Const".into()
    }

    op_core_mir!();
    op_as_typed_op!();
}

impl EvalOp for Const {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, _inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        Ok(tvec![self.0.clone()])
    }
}

impl TypedOp for Const {
    as_op!();

    fn output_facts(&self, _inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(self.0.as_ref().into()))
    }
}
