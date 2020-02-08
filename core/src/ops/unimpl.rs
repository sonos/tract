use crate::internal::*;

#[derive(Debug, Clone)]
pub struct UnimplementedOp {
    outputs: usize,
    name: String,
    message: String,
}

impl UnimplementedOp {
    pub fn new(outputs: usize, name: impl AsRef<str>, message: impl AsRef<str>) -> UnimplementedOp {
        UnimplementedOp { outputs, name: name.as_ref().to_string(), message: message.as_ref().to_string() }
    }
}

impl Op for UnimplementedOp {
    fn name(&self) -> Cow<str> {
        format!("Unimplemented({})", self.name).into()
    }

    not_a_typed_op!();
    not_a_pulsed_op!();
}

impl StatefullOp for UnimplementedOp {
    fn state(
        &self,
        _session: &mut SessionState,
        node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        bail!("unimplemented operation: #{} {}", node_id, self.name)
    }
}
