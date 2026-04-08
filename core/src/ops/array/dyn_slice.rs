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
    fn name(&self) -> StaticName {
        "DynSlice".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("axis: {}", self.axis)])
    }

    op_as_typed_op!();
}

impl EvalOp for DynSlice {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval_with_session(
        &self,
        _node_id: usize,
        session: &TurnState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let start = inputs[1]
            .cast_to::<TDim>()?
            .try_as_plain()?
            .to_scalar::<TDim>()?
            .eval(&session.resolved_symbols)
            .to_usize()?;
        let end = inputs[2]
            .cast_to::<TDim>()?
            .try_as_plain()?
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
        // Propagate uniform_tdim when begin is statically known to be 0.
        // With begin=0 the result coordinates are identical to the input coordinates, so
        // the uniform_tdim predicate (which is expressed in terms of coordinate symbols)
        // remains valid for the sliced output.
        let begin_is_zero = inputs[1]
            .konst
            .as_ref()
            .map(|k| {
                k.cast_to_dt(i64::datum_type())
                    .ok()
                    .and_then(|c| {
                        c.try_as_plain().ok().and_then(|p| {
                            p.as_slice::<i64>().ok().map(|s| s.iter().all(|&v| v == 0))
                        })
                    })
                    .unwrap_or(false)
            })
            .unwrap_or(false);
        if begin_is_zero {
            fact.uniform_tdim = inputs[0].uniform_tdim.clone();
        }
        Ok(tvec!(fact))
    }

    fn input_roi(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TVec<Option<TDim>>>> {
        let output_fact = model.outlet_fact(OutletId::new(node.id, 0))?;
        let Some(roi) = &output_fact.region_of_interest else { return Ok(None) };
        // Propagate output ROI to the data input only; start/end scalars don't carry ROI.
        Ok(Some(tvec![Some(roi.clone()), None, None]))
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
        rule_if!(io != InOut::In(1) && io != InOut::In(2));
        rule_if_some!(axis = change.transform_axis(self.axis));
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
    }

    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let inputs = model.node_input_facts(node.id)?;
        rule_if_some!(start = &inputs[1].konst);
        rule_if_some!(end = &inputs[2].konst);
        let start = start.cast_to::<TDim>()?.try_as_plain()?.to_scalar::<TDim>()?.clone();
        let end = end.cast_to::<TDim>()?.try_as_plain()?.to_scalar::<TDim>()?.clone();

        Ok(Some(TypedModelPatch::replace_single_op(
            model,
            node,
            &[node.inputs[0]],
            crate::ops::array::Slice { axis: self.axis, start, end },
        )?))
    }

    as_op!();
}
