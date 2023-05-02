use crate::internal::*;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ForceEval {
    pub slots: Vec<usize>,
}

impl ForceEval {
    pub fn new(slots: Vec<usize>) -> ForceEval {
        ForceEval { slots }
    }
}

impl Op for ForceEval {
    fn name(&self) -> Cow<str> {
        "ForceEval".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("slots: {:?}", self.slots)])
    }

    impl_op_same_as!();
    op_as_typed_op!();
}

impl EvalOp for ForceEval {
    fn is_stateless(&self) -> bool {
        true
    }

    fn state(
        &self,
        _session: &mut SessionState,
        _node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(None)
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let outputs = inputs
            .into_iter()
            .enumerate()
            .filter_map(|(idx, val)| if !self.slots.contains(&idx) { Some(val) } else { None })
            .collect::<TVec<_>>();
        Ok(outputs)
    }
}

impl TypedOp for ForceEval {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let output_facts = inputs
            .iter()
            .enumerate()
            .filter_map(
                |(idx, fact)| {
                    if !self.slots.contains(&idx) {
                        Some((*fact).clone())
                    } else {
                        None
                    }
                },
            )
            .collect::<TVec<_>>();
        Ok(output_facts)
    }
}
