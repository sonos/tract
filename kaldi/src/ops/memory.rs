use tract_core::internal::*;

#[derive(Clone, Debug, new)]
pub struct Memory {
    pub name: String,
    pub offset: isize,
}

impl Op for Memory {
    fn name(&self) -> std::borrow::Cow<str> {
        "kaldi.Memory".into()
    }
}

impl StatefullOp for Memory {
    fn state(&self, _session: &mut SessionState, _id: usize) -> TractResult<Option<Box<OpState>>> {
        unimplemented!()
    }
}

impl InferenceOp for Memory {
    fn infer_facts(
        &mut self,
        _inputs: TVec<&TensorFact>,
        outputs: TVec<&TensorFact>,
        observed: TVec<&TensorFact>,
    ) -> TractResult<(TVec<TensorFact>, TVec<TensorFact>, TVec<TensorFact>)> {
        let unified = outputs[0].unify(observed[0])?;
        Ok((tvec!(), tvec!(unified.clone()), tvec!(unified.clone())))
    }

    fn observe_outlets(
        &self,
        model: &InferenceModel,
        _node: &InferenceNode,
    ) -> TractResult<Vec<OutletId>> {
        Ok(vec![OutletId::new(model.node_by_name(&self.name)?.id, 0)])
    }

    inference_op_as_op!();
}
