use crate::tensor::DeviceTensorExt;
use crate::utils::compute_broadcast_strides;
use tract_core::internal::*;
use tract_core::ops::array::Slice;

#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct GpuSlice {
    pub inner: Slice,
}

impl GpuSlice {
    pub fn new(inner: Slice) -> Self {
        Self { inner }
    }
}

impl Op for GpuSlice {
    fn name(&self) -> StaticName {
        "GpuSlice".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        self.inner.info()
    }

    op_as_typed_op!();
}

impl EvalOp for GpuSlice {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval_with_session(
        &self,
        node_id: usize,
        session: &TurnState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let input_value = args_1!(inputs);
        let input = input_value.to_device_tensor()?;

        let start = self.inner.start.eval(&session.resolved_symbols).to_usize()?;
        let end = self.inner.end.eval(&session.resolved_symbols).to_usize()?;
        let axis = self.inner.axis;

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

        let mut o_shape: TVec<usize> = input_shape.into();
        o_shape[axis] = end - start;

        let offset = (start * input_strides[axis] as usize) * input_dt.size_of();

        let output = crate::session_handler::make_tensor_for_node(
            session,
            node_id,
            input.datum_type(),
            &o_shape,
        )?;

        if o_shape[axis] != 0 {
            // Slice uses same strides as input (broadcast strides with matching shapes)
            let broadcast_strides: TVec<isize> =
                compute_broadcast_strides(&o_shape, input_strides)?;
            let ctx = crate::device::get_context()?;
            ctx.copy_nd(
                input,
                offset,
                &broadcast_strides,
                &output,
                0,
                output.shape(),
                output.strides(),
            )?;
        }
        Ok(tvec![output.into_tensor().into_tvalue()])
    }
}

impl TypedOp for GpuSlice {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        crate::utils::facts_to_device_facts(inputs, |facts| self.inner.output_facts(facts))
            .with_context(|| format!("Error while computing facts for {:?}", self.name()))
    }

    fn substitute_symbols(
        &self,
        _source: &TypedModel,
        node: &TypedNode,
        target: &mut TypedModel,
        mapping: &HashMap<OutletId, OutletId>,
        subs: &HashMap<Symbol, TDim>,
    ) -> TractResult<TVec<OutletId>> {
        let op = GpuSlice {
            inner: Slice {
                axis: self.inner.axis,
                start: self.inner.start.substitute_all(subs)?,
                end: self.inner.end.substitute_all(subs)?,
            },
        };
        let inputs = node.inputs.iter().map(|i| mapping[i]).collect::<TVec<_>>();
        target.wire_node(&node.name, op, &inputs)
    }

    as_op!();
}
