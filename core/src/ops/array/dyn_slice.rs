use crate::internal::*;
use crate::num_traits::Zero;

#[derive(Debug, Clone, Default, PartialEq, Hash)]
pub struct DynSlice {
    pub axis: usize,
    pub start_input: bool,
    pub end_input: bool,
}

impl DynHash for DynSlice {
    fn dyn_hash(&self, hasher: &mut dyn std::hash::Hasher) {
        dyn_hash(&self, hasher)
    }
}

impl DynSlice {
    pub fn new(axis: usize, start_input: bool, end_input: bool) -> DynSlice {
        DynSlice { axis, start_input, end_input }
    }

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

    op_core_lir_mir!();
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

    fn eval(&self, inputs: TVec<Arc<Tensor>>) -> TractResult<TVec<Arc<Tensor>>> {
        unsafe {
            let start =
                if self.start_input { inputs[1].cast_to_scalar::<i64>()? as usize } else { 0 };
            let end = if self.end_input {
                inputs[1 + self.start_input as usize].cast_to_scalar::<i64>()? as usize
            } else {
                inputs[0].shape()[self.axis]
            };
            if start >= end {
                bail!("Invalid range {}-{}", start, end );
            }
            let mut shape: TVec<_> = inputs[0].shape().into();
            shape[self.axis] = end - start;
            let mut tensor = Tensor::uninitialized_dt(inputs[0].datum_type(), &shape)?;
            tensor.assign_slice_unchecked(.., &inputs[0], start..end, self.axis);
            Ok(tvec!(tensor.into_arc_tensor()))
        }
    }
}

impl TypedOp for DynSlice {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let mut fact = inputs[0].clone();
        fact.shape.set(self.axis, Symbol::new('l').into());
        Ok(tvec!(fact))
    }

    fn invariants(
        &self,
        inputs: &[&TypedFact],
        _outputs: &[&TypedFact],
    ) -> TractResult<Invariants> {
        let axes = (0..inputs[0].rank())
            .filter(|&ax| self.axis != ax)
            .map(|axis| AxisInfo::simple(axis))
            .collect();
        Ok(axes)
    }

    fn change_axes(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        _io: InOut,
        change: &AxisOp,
    ) -> TractResult<Option<AxisChangeConsequence>> {
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
        let start =
            if self.start_input { inputs[1].konst.clone() } else { Some(rctensor0(TDim::zero())) };
        let end = if self.end_input {
            inputs[1 + self.start_input as usize].konst.clone()
        } else {
            Some(rctensor0(inputs[0].shape[self.axis].clone()))
        };
        if let (Some(start), Some(end)) = (start, end) {
            return Ok(Some(TypedModelPatch::replace_single_op(
                model,
                node,
                &[node.inputs[0]],
                crate::ops::array::Slice {
                    axis: self.axis,
                    start: start.cast_to::<TDim>()?.to_scalar::<TDim>()?.clone(),
                    end: end.cast_to::<TDim>()?.to_scalar::<TDim>()?.clone(),
                },
            )?));
        }
        Ok(None)
    }

    fn slice_output(
        &self,
        model: &TypedModel,
        node: &TypedNode,
        patch: &mut TypedModelPatch,
        suffix: &str,
        _output_slot: usize,
        axis: usize,
        start: usize,
        end: usize,
    ) -> TractResult<Option<(OutletId, bool)>> {
        let prec = model.node(node.inputs[0].node);
        if axis != self.axis {
            let suffix = suffix.to_string() + "." + &self.suffix();
            return prec
                .op()
                .as_typed()
                .unwrap()
                .slice_output(model, &prec, patch, &suffix, node.inputs[0].slot, axis, start, end)?
                .map(|(w, no_slice_op)| {
                    Ok((patch.wire_node(format!("{}.{}", node.name, &suffix), self.clone(), &[w])?
                        [0], no_slice_op))
                })
                .transpose();
        }
        Ok(None)
    }

    as_op!();
}
