use crate::internal::*;

#[derive(Debug, Clone, new)]
pub struct SourceState(pub usize);
trivial_op_state_freeeze!(SourceState);

impl OpState for SourceState {
    fn eval(
        &mut self,
        session: &mut SessionState,
        _op: &dyn Op,
        _inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        Ok(tvec!(session
            .inputs
            .get(&self.0)
            .with_context(|| format!("Input for node {} is missing", self.0))?
            .clone()))
    }
}

#[derive(Debug, Clone, new, Hash)]
pub struct TypedSource {
    pub fact: TypedFact,
}



impl Op for TypedSource {
    fn name(&self) -> Cow<str> {
        "Source".into()
    }
    op_as_typed_op!();
}

impl EvalOp for TypedSource {
    fn is_stateless(&self) -> bool {
        false
    }

    fn state(
        &self,
        _session: &mut SessionState,
        node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(Some(Box::new(SourceState(node_id))))
    }
}

impl TypedOp for TypedSource {
    fn output_facts(&self, _inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        Ok(tvec!(self.fact.clone()))
    }

    fn change_axes(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        _io: InOut,
        change: &AxisOp,
    ) -> TractResult<Option<AxisChangeConsequence>> {
        let mut fact = self.fact.clone();
        change.change_shape(&mut fact.shape, false)?;
        Ok(Some(AxisChangeConsequence::new(
            model,
            node,
            Some(Box::new(TypedSource::new(fact))),
            change,
        )))
    }

    fn concretize_dims(
        &self,
        _source: &TypedModel,
        node: &TypedNode,
        target: &mut TypedModel,
        _mapping: &HashMap<OutletId, OutletId>,
        values: &SymbolValues,
    ) -> TractResult<TVec<OutletId>> {
        let shape: TVec<_> = self.fact.shape.iter().map(|d| d.eval(values)).collect();
        target.wire_node(&node.name, Self { fact: self.fact.datum_type.fact(&*shape) }, &[])
    }

    as_op!();
}
