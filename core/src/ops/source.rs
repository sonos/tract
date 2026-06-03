use crate::internal::*;

#[derive(Debug, Clone, new)]
pub struct SourceState(pub usize);
trivial_op_state_freeze!(SourceState);

impl OpState for SourceState {
    fn eval(
        &mut self,
        session: &mut TurnState,
        _op: &dyn Op,
        _inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        Ok(tvec!(
            session
                .values
                .get(self.0)
                .and_then(|v| v.as_ref())
                .and_then(|vs| vs.first().cloned())
                .with_context(|| format!("Input for node {} is missing", self.0))?
        ))
    }
}

#[derive(Debug, Clone, new, Hash, PartialEq, Eq)]
pub struct TypedSource {
    pub fact: TypedFact,
}

impl Op for TypedSource {
    fn name(&self) -> StaticName {
        "Source".into()
    }
    op_as_typed_op!();
}

impl EvalOp for TypedSource {
    fn is_stateless(&self) -> bool {
        false
    }

    fn state(&self, _session: &TurnState, node_id: usize) -> TractResult<Option<Box<dyn OpState>>> {
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
        // A `Rm(ix)` request on a Source whose shape on `ix` is not
        // trivially 1 means the outer caller's view of this axis has
        // diverged from this body source fact (typical when the outer
        // post-pulse declutter folded the corresponding outer dim to 1
        // while the body source kept a non-trivially-symbolic value
        // that downstream optimization simplifies elsewhere). The
        // contract for `change_axes` is to return `Ok(None)` when the
        // change cannot be applied; bubbling an error here aborts the
        // whole declutter pass. Map the targeted failure mode to
        // `None` so ChangeAxes just skips this proposal instead.
        if let AxisOp::Rm(ix) = change.canonical().as_ref()
            && self.fact.shape.rank() > *ix
            && self.fact.shape[*ix] != 1.to_dim()
        {
            return Ok(None);
        }
        let mut fact = self.fact.clone();
        change.change_shape(&mut fact.shape, false)?;
        Ok(Some(AxisChangeConsequence::new(
            model,
            node,
            Some(Box::new(TypedSource::new(fact))),
            change,
        )))
    }

    fn set_symbols(
        &self,
        _source: &TypedModel,
        node: &TypedNode,
        target: &mut TypedModel,
        _mapping: &HashMap<OutletId, OutletId>,
        subs: &HashMap<Symbol, TDim>,
    ) -> TractResult<TVec<OutletId>> {
        let shape: TVec<_> =
            self.fact.shape.iter().map(|d| d.substitute_all(subs)).collect::<TractResult<_>>()?;
        target.wire_node(&node.name, Self { fact: self.fact.datum_type.fact(&*shape) }, &[])
    }

    as_op!();
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `TypedOp::change_axes` returns `Ok(None)` when the requested
    /// change cannot be applied; previously `TypedSource::change_axes`
    /// bubbled the underlying `change.change_shape` error, which
    /// aborts the surrounding declutter pass instead of letting
    /// `ChangeAxes` skip the proposal. Lock the contract for the
    /// `Rm(non-trivial-axis)` failure mode.
    #[test]
    fn change_axes_rm_non_trivial_returns_none() {
        let mut model = TypedModel::default();
        let out = model.add_source("x", f32::fact([1usize, 2, 4])).unwrap();
        model.select_output_outlets(&[out]).unwrap();
        let src_node = model.nodes.iter().find(|n| n.op_is::<TypedSource>()).unwrap();
        let src = src_node.op_as::<TypedSource>().unwrap();
        let result = src
            .change_axes(&model, src_node, InOut::Out(0), &AxisOp::Rm(1))
            .expect("change_axes must not bubble an error for an inapplicable change");
        assert!(
            result.is_none(),
            "Rm on a non-trivial axis must be reported as `Ok(None)`, got {result:?}",
        );
    }

    /// Sanity-check for the unchanged path: `Rm` on a trivial axis is
    /// still applied (returns `Some(consequence)`).
    #[test]
    fn change_axes_rm_trivial_still_applies() {
        let mut model = TypedModel::default();
        let out = model.add_source("x", f32::fact([1usize, 2, 4])).unwrap();
        model.select_output_outlets(&[out]).unwrap();
        let src_node = model.nodes.iter().find(|n| n.op_is::<TypedSource>()).unwrap();
        let src = src_node.op_as::<TypedSource>().unwrap();
        let result = src
            .change_axes(&model, src_node, InOut::Out(0), &AxisOp::Rm(0))
            .expect("change_axes succeeds for a trivial axis Rm");
        assert!(result.is_some(), "Rm on a trivial axis must still apply, got {result:?}");
    }
}
