use crate::internal::*;

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
    not_out_of_plan!();

    fn eval(&self, ctx: &EvalContext, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        ensure!(!inputs.is_empty(), "Input for node {} is missing", ctx.node_id);
        Ok(inputs)
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
        // Block (rather than fail) on a change this source's shape cannot
        // express: axis-change propagation reaching a scan body's input source
        // with a wire-specific `Reshape` must stop the search, not error out.
        if change.change_shape(&mut fact.shape, false).is_err() {
            return Ok(None);
        }
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

    // A change this source's shape cannot express must block the axis-change
    // search (Ok(None)) instead of erroring the whole optimization run. This
    // is how a wire-specific Reshape change stops at a scan body's input
    // source on the MMS TTS graph (#2928 follow-up diagnostics).
    #[test]
    fn blocks_inapplicable_change() -> TractResult<()> {
        let mut model = TypedModel::default();
        let s = model.add_source("s", f32::datum_type().fact([2usize, 3]))?;
        let node = model.node(s.node);
        let change =
            AxisOp::Reshape(0, tvec![4.to_dim(), 5.to_dim()].into(), tvec![20.to_dim()].into());
        let blocked = node
            .op
            .change_axes(&model, node, InOut::Out(0), &change)
            .map(|r| r.is_none())
            .unwrap_or(false);
        assert!(blocked, "expected the source to block the change");
        Ok(())
    }
}
