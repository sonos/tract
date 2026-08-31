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
