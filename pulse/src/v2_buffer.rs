/// PulseV2Buffer: fixed-size streaming-axis history buffer.
///
/// At each pulse T, the op emits `lookback + current` samples on the buffered
/// axis: the last `lookback` samples seen plus this pulse's input. The history
/// is initialised to zeros at session start, so on T=0 the output is
/// `[zeros…lookback…, current]` — fixed shape, garbage prefix during ramp,
/// matching v1's `Delay` semantics.
///
/// Output shape on the buffered axis = input + lookback (constant). All other
/// axes are passed through unchanged. Multi-axis lookback is not supported in
/// this revision (the streaming axis is normally the only buffered one).
use crate::internal::*;
use tract_pulse_opl::tract_core::ops::{FrozenOpState, OpStateFreeze};

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct PulseV2Buffer {
    /// Per-axis lookback. lookback[i] = 0 means no buffering on axis i.
    /// Exactly one axis is expected to have non-zero lookback.
    pub lookback: TVec<usize>,
    /// Per-axis overlap: the actual data overlap the consumer needs.
    /// overlap[i] <= lookback[i]. The difference (lookback - overlap) is
    /// stride alignment padding that gets trimmed from the output.
    pub overlap: TVec<usize>,
    /// Pulse index symbol (T). Kept for compatibility with the rest of the
    /// pulsifier even though `eval` no longer needs it.
    pub pulse_id: Symbol,
    /// Pulse size symbol (P).
    pub pulse_sym: Symbol,
}

impl PulseV2Buffer {
    /// Axis with nonzero lookback. Panics if zero or more than one axis is
    /// buffered — the rewrite assumes a single streaming axis.
    pub fn buffered_axis(&self) -> Option<(usize, usize)> {
        let mut it = self.lookback.iter().copied().enumerate().filter(|(_, lb)| *lb > 0);
        let first = it.next();
        debug_assert!(it.next().is_none(), "PulseV2Buffer expects a single buffered axis");
        first
    }
}

impl Op for PulseV2Buffer {
    fn name(&self) -> StaticName {
        "PulseV2Buffer".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![match self.buffered_axis() {
            Some((ax, lb)) => format!("axis {ax}: lookback {lb}"),
            None => "passthrough".to_string(),
        }])
    }

    op_as_typed_op!();
}

impl EvalOp for PulseV2Buffer {
    fn is_stateless(&self) -> bool {
        false
    }

    fn state(
        &self,
        _session: &TurnState,
        _node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(Some(Box::new(PulseV2BufferState { history: None })))
    }
}

impl TypedOp for PulseV2Buffer {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let mut fact = inputs[0].clone();
        if let Some((axis, lookback)) = self.buffered_axis() {
            let input_dim = fact.shape[axis].clone();
            fact.shape.set(axis, input_dim + TDim::Val(lookback as i64));
        }
        Ok(tvec!(fact))
    }

    /// At declutter time, lower the trivial single-axis case to v1's `Delay`,
    /// which has the in-place memmove fast path and NNEF round-tripping. The
    /// semantic equivalence is exact: PulseV2Buffer { lookback: […N…] } with
    /// a single axis matches Delay { axis, delay: 0, overlap: N } — same
    /// state size, same per-pulse output (`input + N`), zero-initialised
    /// history on first eval.
    ///
    /// Only fires when exactly one axis has non-zero lookback. Multi-axis
    /// cases (none today, but the data structure leaves room) keep the v2
    /// op until a multi-axis Delay equivalent exists.
    fn declutter(
        &self,
        model: &TypedModel,
        node: &TypedNode,
    ) -> TractResult<Option<TypedModelPatch>> {
        let Some((axis, lookback)) = self.buffered_axis() else {
            // No buffered axis at all → identity, shunt entirely.
            return TypedModelPatch::shunt_one_op(model, node);
        };
        let input_fact = model.outlet_fact(node.inputs[0])?;
        let delay = tract_pulse_opl::ops::Delay::new_typed(input_fact, axis, 0, lookback);
        Ok(Some(TypedModelPatch::replace_single_op(model, node, &node.inputs, delay)?))
    }
}

#[derive(Debug, Clone)]
pub struct PulseV2BufferState {
    /// History on the buffered axis, shape `lookback` on that axis and
    /// matching the input on all other axes. `None` until first eval, then
    /// initialised to zeros.
    history: Option<Tensor>,
}

impl OpStateFreeze for PulseV2BufferState {
    fn freeze(&self) -> Box<dyn FrozenOpState> {
        unimplemented!("PulseV2BufferState::freeze not yet implemented")
    }
}

impl OpState for PulseV2BufferState {
    fn eval(
        &mut self,
        _session: &mut TurnState,
        op: &dyn Op,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let op = op.downcast_ref::<PulseV2Buffer>().unwrap();
        let input = args_1!(inputs);
        let input_tensor = input.into_tensor();

        let Some((axis, lookback)) = op.buffered_axis() else {
            return Ok(tvec!(input_tensor.into_tvalue()));
        };

        // Initialise history to zeros on first call. Shape matches the input's
        // shape with the buffered axis replaced by `lookback`.
        let history = match self.history.take() {
            Some(h) => h,
            None => {
                let mut shape = input_tensor.shape().to_vec();
                shape[axis] = lookback;
                Tensor::zero_dt(input_tensor.datum_type(), &shape)?
            }
        };

        // Output = concat(history, input) on the buffered axis.
        let output = Tensor::stack_tensors(axis, &[history, input_tensor])?;

        // Update history = last `lookback` samples of the output.
        let total = output.shape()[axis];
        debug_assert!(total >= lookback);
        let new_history = output.slice(axis, total - lookback, total)?.into_tensor();
        self.history = Some(new_history);

        Ok(tvec!(output.into_tvalue()))
    }
}
