use crate::kernels;
use crate::tensor::MetalTensorExt;
use tract_core::internal::*;
use tract_core::ops::array::Slice;

#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct MetalSlice {
    pub axis: usize,
    pub start: TDim,
    pub end: TDim,
}

impl MetalSlice {
    pub fn new(axis: usize, start: impl ToDim, end: impl ToDim) -> MetalSlice {
        MetalSlice { axis, start: start.to_dim(), end: end.to_dim() }
    }

    pub fn from_tract_core(op: &Slice) -> Self {
        Self { axis: op.axis, start: op.start.clone(), end: op.end.clone() }
    }

    pub fn suffix(&self, name: &str) -> String {
        format!("{}.axis{}_{}_{}", name, self.axis, self.start, self.end)
    }
}

impl Op for MetalSlice {
    fn name(&self) -> Cow<str> {
        "MetalSlice".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("axis: {}, {}..{}", self.axis, self.start, self.end)])
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

impl EvalOp for MetalSlice {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval_with_session(
        &self,
        session: &SessionState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs);
        let start = self.start.eval(&session.resolved_symbols).to_usize()?;
        let end = self.end.eval(&session.resolved_symbols).to_usize()?;
        let axis = self.axis;

        let input_shape = input.as_metal_tensor().map(|it| it.shape()).unwrap_or(input.shape());
        let input_strides =
            input.as_metal_tensor().map(|it| it.strides()).unwrap_or(input.strides());

        let input_dt =
            input.as_metal_tensor().map(|it| it.datum_type()).unwrap_or(input.datum_type());

        ensure!(
            end <= input_shape[axis] && start <= end,
            "Invalid range {}..{} for slicing {:?} on axis {}",
            start,
            end,
            input,
            axis
        );

        let mut o_shape: TVec<_> = input_shape.into();
        o_shape[axis] = end - start;
        ensure!(o_shape[axis] != 0);

        let offset = (start * input_strides[axis] as usize) * input_dt.size_of();

        if let Some(t) = input.as_metal_tensor() {
            objc::rc::autoreleasepool(|| {
                crate::METAL_CONTEXT.with_borrow(|context| {
                    Ok(tvec![kernels::array::MultiBroadcast
                        .dispatch_eval(context, t, offset, &o_shape)?
                        .into_opaque_tensor()
                        .into_tvalue()])
                })
            })
        } else {
            unsafe {
                let mut tensor = Tensor::uninitialized_dt(input.datum_type(), &o_shape)?;
                tensor.assign_slice_unchecked(.., &input, start..end, axis);
                Ok(tvec!(tensor.into_tvalue()))
            }
        }
    }
}

impl TypedOp for MetalSlice {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        anyhow::ensure!(inputs.len() == 1, "MetalSlice has one single input");
        crate::utils::metal_output_facts(inputs, |facts| {
            if let (Ok(start), Ok(end), Ok(len)) =
                (self.start.to_usize(), self.end.to_usize(), facts[0].shape[self.axis].to_usize())
            {
                ensure!(start <= end);
                ensure!(end <= len);
            }
            let mut fact = facts[0].without_value();
            fact.shape.set(self.axis, (self.end.clone() - &self.start).to_dim());
            Ok(tvec!(fact))
        })
        .with_context(|| anyhow::anyhow!("Error while computing facts for {:?}", self.name()))
    }

    fn axes_mapping(
        &self,
        inputs: &[&TypedFact],
        outputs: &[&TypedFact],
    ) -> TractResult<AxesMapping> {
        let mut mapping = AxesMapping::disconnected(inputs, outputs)?;
        for (axis, repr) in (0..inputs[0].rank()).zip('a'..) {
            if self.axis != axis {
                mapping = mapping
                    .renaming((InOut::In(0), axis), repr)?
                    .linking(repr, (InOut::Out(0), axis))?;
            }
        }
        Ok(mapping)
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
                    Some(Box::new(MetalSlice { axis, ..self.clone() }) as _),
                    change,
                )))
            } else {
                Ok(Some(AxisChangeConsequence::new(model, node, None, change)))
            }
        } else {
            Ok(None)
        }
    }

    fn concretize_dims(
        &self,
        _source: &TypedModel,
        node: &TypedNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
        values: &SymbolValues,
    ) -> TractResult<TVec<OutletId>> {
        let op = MetalSlice {
            axis: self.axis,
            start: self.start.eval(values),
            end: self.end.eval(values),
        };
        let inputs = node.inputs.iter().map(|i| mapping[i]).collect::<TVec<_>>();
        target.wire_node(&node.name, op, &inputs)
    }

    as_op!();
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::IntoMetal;
    use tract_core::internal::Tensor;

    fn run_test(shape: &[usize], slice: Slice) -> TractResult<()> {
        objc::rc::autoreleasepool(|| {
            crate::METAL_CONTEXT.with_borrow(|context| {
                let num_elements = shape.iter().product();

                let a = Tensor::from_shape(
                    &shape,
                    &(0..num_elements).map(|f| f as f32).collect::<Vec<_>>(),
                )?;
                let cpu_output = slice
                    .eval_with_session(&SessionState::default(), tvec![a.clone().into_tvalue()])?;

                let metal_slice = MetalSlice::from_tract_core(&slice);
                let a_metal = a.clone().into_metal()?.into_opaque_tensor().into_tvalue();
                let metal_output = metal_slice
                    .eval_with_session(&SessionState::default(), tvec![a_metal.clone()])?;
                context.wait_until_completed()?;

                dbg!(&cpu_output[0]);
                dbg!(metal_output[0].to_metal_tensor()?.tensor());

                cpu_output[0].close_enough(
                    metal_output[0].to_metal_tensor()?.tensor(),
                    Approximation::Approximate,
                )?;
                Ok(())
            })
        })
    }

    #[test]
    fn test_slice() -> TractResult<()> {
        run_test(&[4, 4], Slice { axis: 1, start: 0.into(), end: 4.into() })?;
        run_test(&[8, 3, 5], Slice { axis: 1, start: 1.into(), end: 3.into() })?;
        assert!(run_test(&[8, 3, 5], Slice { axis: 1, start: 1.into(), end: 7.into() }).is_err());
        assert!(run_test(&[8, 3, 5], Slice { axis: 1, start: 1.into(), end: 1.into() }).is_err());
        Ok(())
    }
}
