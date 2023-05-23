use crate::internal::*;
use crate::num_traits::Zero;

#[derive(Debug, Clone, PartialEq, Eq, Hash, new)]
pub struct DynSlice {
    pub axis: usize,
    pub start_input: bool,
    pub end_input: bool,
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

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        unsafe {
            let start =
                if self.start_input { inputs[1].cast_to_scalar::<i64>()? as usize } else { 0 };
            let end = if self.end_input {
                inputs[1 + self.start_input as usize].cast_to_scalar::<i64>()? as usize
            } else {
                inputs[0].shape()[self.axis]
            };
            if start >= end {
                bail!("Invalid range {}-{}", start, end);
            }
            let mut shape: TVec<_> = inputs[0].shape().into();
            shape[self.axis] = end - start;
            let mut tensor = Tensor::uninitialized_dt(inputs[0].datum_type(), &shape)?;
            tensor.assign_slice_unchecked(.., &inputs[0], start..end, self.axis);
            Ok(tvec!(tensor.into_tvalue()))
        }
    }
}

impl TypedOp for DynSlice {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let mut fact = inputs[0].without_value();
        fact.shape.set(self.axis, self.len.clone().into());
        Ok(tvec!(fact))
    }

    fn axes_mapping(
        &self,
        inputs: &[&TypedFact],
        _outputs: &[&TypedFact],
    ) -> TractResult<AxesMapping> {
        let mut axes = AxesMapping::natural_for_rank(1, 1, inputs[0].rank())?;
        if self.start_input {
            axes = axes.with_extra_input(1)?;
        }
        if self.end_input {
            axes = axes.with_extra_input(self.start_input as usize)?;
        }
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

    as_op!();
}
