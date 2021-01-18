use crate::internal::*;

#[derive(Debug, Clone, new, Hash)]
pub struct Dummy;

impl Op for Dummy {
    fn name(&self) -> Cow<str> {
        "Dummy".into()
    }

    op_core_mir!();
    op_as_typed_op!();
}

impl_dyn_hash!(Dummy);

impl EvalOp for Dummy {
    fn is_stateless(&self) -> bool {
        false
    }

    fn eval(&self, _inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        bail!("eval() called on a Dummy op. This is a bug.")
    }
}

impl TypedOp for Dummy {
    as_op!();

    fn output_facts(&self, _inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!())
    }
}
