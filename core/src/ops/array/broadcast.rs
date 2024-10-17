use crate::internal::*;

#[derive(Debug, Clone, new, Hash)]
pub struct MultiBroadcastTo {
    pub shape: ShapeFact,
}

impl Op for MultiBroadcastTo {
    fn name(&self) -> Cow<str> {
        "MultiBroadcastTo".into()
    }

    op_as_typed_op!();
}

impl EvalOp for MultiBroadcastTo {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval_with_session(
        &self,
        session: &SessionState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let shape = self.shape.eval_to_usize(&session.resolved_symbols)?;
        Ok(tvec!(inputs[0].broadcast_to_shape(&shape)?.into_tvalue()))
    }
}

impl TypedOp for MultiBroadcastTo {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(inputs.len() == 1);
        let mut fact = inputs[0].datum_type.fact(self.shape.clone());
        fact.uniform.clone_from(&inputs[0].uniform);
        Ok(tvec!(fact))
    }

    fn concretize_dims(
        &self,
        _source: &TypedModel,
        node: &TypedNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
        values: &SymbolValues,
    ) -> TractResult<TVec<OutletId>> {
        let input = mapping[&node.inputs[0]];
        let op =
            Self { shape: self.shape.iter().map(|d| d.eval(values)).collect::<TVec<_>>().into() };
        target.wire_node(&node.name, op, &[input])
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let input_fact = model.outlet_fact(node.inputs[0])?;
        if input_fact.shape == self.shape {
            TypedModelPatch::shunt_one_op(model, node)
        } else {
            Ok(None)
        }
    }

    as_op!();
}
