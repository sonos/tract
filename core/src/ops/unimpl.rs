use crate::internal::*;

#[derive(Debug, Clone, Hash)]
pub struct UnimplementedOp {
    outputs: usize,
    name: String,
    message: String,
}

tract_data::impl_dyn_hash!(UnimplementedOp);

impl UnimplementedOp {
    pub fn new(outputs: usize, name: impl AsRef<str>, message: impl AsRef<str>) -> UnimplementedOp {
        UnimplementedOp {
            outputs,
            name: name.as_ref().to_string(),
            message: message.as_ref().to_string(),
        }
    }
}

impl Op for UnimplementedOp {
    fn name(&self) -> Cow<str> {
        format!("Unimplemented({})", self.name).into()
    }

    op_core!();
    not_a_typed_op!();
}

impl EvalOp for UnimplementedOp {
    fn is_stateless(&self) -> bool {
        false
    }
}
