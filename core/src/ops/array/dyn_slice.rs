use crate::internal::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash, new)]
pub struct DynSlice {
    pub axis: usize,
    pub len: TDim,
}

impl DynSlice {
    pub fn suffix(&self) -> String {
        format!("axis{}", self.axis)
    }
}

impl Op for DynSlice {
    fn name(&self) -> Cow<str> {
        "DynSlice".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("axis: {}", self.axis)])
    }

    op_as_typed_op!();

    fn same_as(&self, other: &dyn Op) -> bool {
        if let Some(other) = other.downcast_ref::<Self>() {
            other == self
        } else {
            false
        }
    }
}

impl EvalOp for DynSlice {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval_with_session(
        &self,
        session: &SessionState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let start = inputs[1]
            .cast_to::<TDim>()?
            .to_scalar::<TDim>()?
            .eval(&session.resolved_symbols)
            .to_usize()?;
        let end = inputs[2]
            .cast_to::<TDim>()?
            .to_scalar::<TDim>()?
            .eval(&session.resolved_symbols)
            .to_usize()?;
        ensure!(start <= end);
        if let Ok(len) = self.len.eval(&session.resolved_symbols).to_usize() {
            ensure!(start + len == end);
        }
        let slice = inputs[0].slice(self.axis, start, end)?;
        Ok(tvec!(slice.into()))
    }
}

impl TypedOp for DynSlice {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(inputs.len() == 3);
        let mut fact = inputs[0].without_value();
        fact.shape.set(self.axis, self.len.clone());
        Ok(tvec!(fact))
    }

    fn axes_mapping(
        &self,
        inputs: &[&TypedFact],
        _outputs: &[&TypedFact],
    ) -> TractResult<AxesMapping> {
        AxesMapping::natural_for_rank(1, 1, inputs[0].rank())?
            .with_extra_input(1)?
            .with_extra_input(2)
    }

    fn change_axes(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        io: InOut,
        change: &AxisOp,
    ) -> TractResult<Option<AxisChangeConsequence>> {
        if io == InOut::In(1) || io == InOut::In(2) {
            return Ok(None);
        }
        if let Some(axis) = change.transform_axis(self.axis) {
            if axis != self.axis {
                Ok(Some(AxisChangeConsequence::new(
                    model,
                    node,
                    Some(Box::new(DynSlice { axis, ..self.clone() }) as _),
                    change,
                )))
            } else {
                Ok(Some(AxisChangeConsequence::new(model, node, None, change)))
            }
        } else {
            Ok(None)
        }
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let inputs = model.node_input_facts(node.id)?;
        if let (Some(start), Some(end)) = (&inputs[1].konst, &inputs[2].konst) {
            let start = start.cast_to::<TDim>()?.to_scalar::<TDim>()?.clone();
            let end = end.cast_to::<TDim>()?.to_scalar::<TDim>()?.clone();

            return Ok(Some(TypedModelPatch::replace_single_op(
                model,
                node,
                &[node.inputs[0]],
                crate::ops::array::Slice { axis: self.axis, start, end },
            )?));
        }
        Ok(None)
    }

    as_op!();
}
