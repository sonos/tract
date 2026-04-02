use crate::ops::broadcast::DispatchBroadcastFn;
use crate::tensor::DeviceTensorExt;
use tract_core::internal::*;
use tract_core::ops::array::Slice;

#[derive(Clone)]
pub struct GpuSlice {
    pub inner: Slice,
    pub backend_name: &'static str,
    pub dispatch: DispatchBroadcastFn,
}

impl GpuSlice {
    pub fn new(inner: Slice, backend_name: &'static str, dispatch: DispatchBroadcastFn) -> Self {
        Self { inner, backend_name, dispatch }
    }
}

impl std::fmt::Debug for GpuSlice {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "{}Slice({:?})", self.backend_name, self.inner)
    }
}

impl PartialEq for GpuSlice {
    fn eq(&self, other: &Self) -> bool {
        self.backend_name == other.backend_name && self.inner == other.inner
    }
}

impl Eq for GpuSlice {}

impl std::hash::Hash for GpuSlice {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.backend_name.hash(state);
        self.inner.hash(state);
    }
}

impl Op for GpuSlice {
    fn name(&self) -> StaticName {
        format!("{}Slice", self.backend_name).into()
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

        let mut o_shape: TVec<_> = input_shape.into();
        o_shape[axis] = end - start;

        let offset = (start * input_strides[axis] as usize) * input_dt.size_of();

        let output = crate::session_handler::make_tensor_for_node(
            session,
            node_id,
            input.datum_type(),
            &o_shape,
        )?;

        if o_shape[axis] != 0 {
            (self.dispatch)(input, offset, &output)?;
        }
        Ok(tvec![output.into_tensor().into_tvalue()])
    }
}

impl TypedOp for GpuSlice {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        crate::utils::facts_to_device_facts(inputs, |facts| self.inner.output_facts(facts))
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
        let op = GpuSlice {
            inner: Slice {
                axis: self.inner.axis,
                start: self.inner.start.eval(values),
                end: self.inner.end.eval(values),
            },
            backend_name: self.backend_name,
            dispatch: self.dispatch,
        };
        let inputs = node.inputs.iter().map(|i| mapping[i]).collect::<TVec<_>>();
        target.wire_node(&node.name, op, &inputs)
    }

    as_op!();
}
