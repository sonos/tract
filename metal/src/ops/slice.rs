use crate::kernels;
use crate::tensor::{MetalTensor, MetalTensorExt};
use tract_core::internal::*;
use tract_core::ops::array::Slice;

#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct MetalSlice(Slice);

impl MetalSlice {
    // pub fn new(axis: usize, start: impl ToDim, end: impl ToDim) -> MetalSlice {
    //     MetalSlice { axis, start: start.to_dim(), end: end.to_dim() }
    // }

    pub fn from_tract_core(op: Slice) -> Self {
        Self(op)
    }

    pub fn suffix(&self, name: &str) -> String {
        format!("{}.axis{}_{}_{}", name, self.0.axis, self.0.start, self.0.end)
    }
}

impl Op for MetalSlice {
    fn name(&self) -> Cow<str> {
        "MetalSlice".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        self.0.info()
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
        let start = self.0.start.eval(&session.resolved_symbols).to_usize()?;
        let end = self.0.end.eval(&session.resolved_symbols).to_usize()?;
        let axis = self.0.axis;

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

        let offset = (start * input_strides[axis] as usize) * input_dt.size_of();

        let t = input.to_metal_tensor()?;

        objc::rc::autoreleasepool(|| {
            crate::METAL_CONTEXT.with_borrow(|context| {
                if o_shape[axis] == 0 {
                    Ok(tvec![unsafe {
                        MetalTensor::uninitialized_dt(t.datum_type(), &o_shape)?
                            .into_opaque_tensor()
                            .into_tvalue()
                    }])
                } else {
                    Ok(tvec![kernels::array::MultiBroadcast
                        .dispatch_eval(context, t, offset, &o_shape)?
                        .into_opaque_tensor()
                        .into_tvalue()])
                }
            })
        })
    }
}

impl TypedOp for MetalSlice {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        crate::utils::metal_output_facts(inputs, |facts| self.0.output_facts(facts))
            .with_context(|| anyhow::anyhow!("Error while computing facts for {:?}", self.name()))
    }

    fn concretize_dims(
        &self,
        _source: &TypedModel,
        node: &TypedNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
        values: &SymbolValues,
    ) -> TractResult<TVec<OutletId>> {
        let op = MetalSlice(Slice {
            axis: self.0.axis,
            start: self.0.start.eval(values),
            end: self.0.end.eval(values),
        });
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

                let metal_slice = MetalSlice::from_tract_core(slice);
                let a_metal = a.clone().into_metal()?.into_opaque_tensor().into_tvalue();
                let metal_output = metal_slice
                    .eval_with_session(&SessionState::default(), tvec![a_metal.clone()])?;
                context.wait_until_completed()?;

                cpu_output[0].close_enough(
                    &metal_output[0].to_metal_tensor()?.to_cpu(),
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
        run_test(&[8, 3, 5], Slice { axis: 1, start: 1.into(), end: 1.into() })?;
        Ok(())
    }
}
