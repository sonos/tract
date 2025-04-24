use crate::kernels;
use crate::ops::MetalEvalOp;
use crate::MetalStream;
use tract_core::internal::*;
use tract_core::ops::array::Slice;
use tract_gpu::tensor::DeviceTensorExt;

#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
pub struct MetalSlice(Slice);

impl MetalSlice {
    pub fn from_tract_core(op: Slice) -> Self {
        Self(op)
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

crate::impl_eval_op_for_metal_op!(MetalSlice);

impl MetalEvalOp for MetalSlice {
    fn metal_eval(
        &self,
        stream: &MetalStream,
        node_id: usize,
        session: &mut SessionState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let opaque = args_1!(inputs);
        let input = opaque.to_device_tensor()?;

        let start = self.0.start.eval(&session.resolved_symbols).to_usize()?;
        let end = self.0.end.eval(&session.resolved_symbols).to_usize()?;
        let axis = self.0.axis;

        let input_shape = input.shape();
        let input_strides = input.strides();
        let input_dt = input.datum_type();

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

        let output =
            crate::ops::make_tensor_for_node(session, node_id, input.datum_type(), &o_shape)?;

        // Perform slicing only if the output is not empty.
        if o_shape[axis] != 0 {
            kernels::array::MultiBroadcast.dispatch_eval(stream, input, offset, &output)?;
        }
        Ok(tvec![output.into_opaque_tensor().into_tvalue()])
    }
}

impl TypedOp for MetalSlice {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        tract_gpu::utils::facts_to_device_facts(inputs, |facts| self.0.output_facts(facts))
            .with_context(|| format!("Error while computing facts for {:?}", self.name()))
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
    use crate::utils::with_borrowed_metal_stream;
    use tract_core::internal::Tensor;
    use tract_gpu::tensor::IntoDevice;

    fn run_test(shape: &[usize], slice: Slice) -> TractResult<()> {
        with_borrowed_metal_stream(|stream| {
            let num_elements = shape.iter().product();

            let a = Tensor::from_shape(
                &shape,
                &(0..num_elements).map(|f| f as f32).collect::<Vec<_>>(),
            )?;

            let cpu_output = slice
                .eval_with_session(&SessionState::default(), tvec![a.clone().into_tvalue()])?;

            let metal_slice = MetalSlice::from_tract_core(slice);
            let a_metal = a.clone().into_device()?.into_opaque_tensor().into_tvalue();
            let mut session_state = SessionState::default();
            let mut metal_slice_state = metal_slice.state(&mut session_state, 0)?.unwrap();
            let metal_output =
                metal_slice_state.eval(&mut session_state, &metal_slice, tvec![a_metal])?;
            stream.wait_until_completed()?;

            cpu_output[0].close_enough(
                &metal_output[0].to_device_tensor()?.to_host()?.into_tensor(),
                Approximation::Approximate,
            )?;
            Ok(())
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
