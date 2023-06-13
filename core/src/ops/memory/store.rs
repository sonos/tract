use crate::internal::*;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Store {
    pub id: String,
}

impl Store {
    pub fn new(id: &str) -> Store {
        Store { id: id.to_string() }
    }
}

impl Op for Store {
    fn name(&self) -> Cow<str> {
        "Store".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("id: {:?}", self.id)])
    }

    impl_op_same_as!();
    op_as_typed_op!();
}

impl EvalOp for Store {
    fn is_stateless(&self) -> bool {
        false
    }

    fn state(
        &self,
        _session: &mut SessionState,
        _node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(Some(Box::new(self.clone())))
    }
}

impl TypedOp for Store {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(
            inputs.len() == 2,
            "Expected two inputs (input to propagate and state to store) for Store op"
        );
        Ok(tvec![inputs[0].clone()])
    }
}

impl OpState for Store {
    fn eval(
        &mut self,
        session: &mut SessionState,
        _op: &dyn Op,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let (input, state) = args_2!(inputs);
        session.tensors.insert(self.id.clone(), state.into_tensor());
        Ok(tvec![input])
    }
}

trivial_op_state_freeeze!(Store);
