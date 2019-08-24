use crate::internal::*;

#[derive(Debug, Clone)]
pub struct UnimplementedOp {
    name: String,
    message: String,
}

impl UnimplementedOp {
    pub fn new(name: impl AsRef<str>, message: impl AsRef<str>) -> UnimplementedOp {
        UnimplementedOp { name: name.as_ref().to_string(), message: message.as_ref().to_string() }
    }
}

impl Op for UnimplementedOp {
    fn name(&self) -> Cow<str> {
        format!("Unimplemented({})", self.name).into()
    }
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

impl InferenceRulesOp for UnimplementedOp {
    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        _: &mut Solver<'r>,
        _: &'p [TensorProxy],
        _: &'p [TensorProxy],
    ) -> InferenceResult {
        Ok(())
    }

    inference_op_as_op!();

    fn to_typed(
        &self,
        _source: &InferenceModel,
        _node: &InferenceNode,
        _target: &mut TypedModel,
        _mapping: &HashMap<OutletId, OutletId>,
    ) -> TractResult<TVec<OutletId>> {
        bail!("Operator can not be made a TypedOp.")
    }
}
