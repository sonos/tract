use crate::internal::*;

#[derive(Debug, Clone, new)]
pub struct SourceState(pub usize);

impl OpState for SourceState {
    fn eval(
        &mut self,
        session: &mut SessionState,
        _op: &dyn Op,
        _inputs: TVec<Arc<Tensor>>,
    ) -> TractResult<TVec<Arc<Tensor>>> {
        Ok(tvec!(session.inputs[&self.0].clone()))
    }
}

#[derive(Debug, Clone, new, Hash)]
pub struct TypedSource {
    pub fact: TypedFact,
}

tract_data::impl_dyn_hash!(TypedSource);

impl Op for TypedSource {
    fn name(&self) -> Cow<str> {
        "Source".into()
    }
    op_core_lir_mir!();
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
        change.change_shape(&mut fact.shape)?;
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
        target.wire_node(
            &node.name,
            Self { fact: TypedFact::dt_shape(self.fact.datum_type, &*shape)? },
            &[],
        )
    }

    as_op!();
}
