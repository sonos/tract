use tract_nnef::internal::*;

#[derive(Debug, Clone, Default, Hash)]
pub struct PulsedAxisSlice {
    pub axis: usize,
    pub skip: usize,
    pub take: TDim,
}



impl Op for PulsedAxisSlice {
    fn name(&self) -> Cow<str> {
        "PulsedAxisSlice".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("axis:{}, skip:{} take:{}", self.axis, self.skip, self.take)])
    }

    not_a_typed_op!();
}

impl TypedOp for PulsedAxisSlice {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(inputs[0].clone()))
    }

    as_op!();
}

impl EvalOp for PulsedAxisSlice {
    fn is_stateless(&self) -> bool {
        false
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        Ok(inputs)
    }
}
