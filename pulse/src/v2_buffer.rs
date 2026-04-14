/// PulseV2Buffer: stateful op that stores previous pulse tensors and
/// stitches them with the current input.
///
/// Parameterized by per-axis lookback: for each axis, how many past
/// samples to retain. Axes with lookback=0 are passed through unchanged.
///
/// At each pulse T, the op:
/// 1. Concatenates buffered tensors with the current input along each
///    axis that has nonzero lookback
/// 2. Trims to keep only (lookback + current_size) on each such axis
/// 3. Pushes the current input into the ring buffer
/// 4. Outputs the stitched tensor
///
/// At T=0 (empty buffer), output is just the current input — no lookback.
/// This means the consumer (e.g. conv) produces fewer output frames on
/// the first pulse. No garbage, no delay.
use crate::internal::*;
use tract_pulse_opl::tract_core::ops::{FrozenOpState, OpStateFreeze};

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct PulseV2Buffer {
    /// Per-axis lookback. lookback[i] = 0 means no buffering on axis i.
    /// This is the amount of history stored (may be stride-rounded).
    pub lookback: TVec<usize>,
    /// Per-axis overlap: the actual data overlap the consumer needs.
    /// overlap[i] <= lookback[i]. The difference (lookback - overlap) is
    /// stride alignment padding that gets trimmed from the output.
    pub overlap: TVec<usize>,
    /// Number of previous pulse tensors to keep in the ring buffer.
    pub depth: usize,
    /// Pulse index symbol (T).
    pub pulse_id: Symbol,
    /// Pulse size symbol (P) — used for cumulative history estimation.
    pub pulse_sym: Symbol,
}

impl PulseV2Buffer {
    /// Total lookback (sum of all axes). Used for conservative depth estimate.
    pub fn total_lookback(&self) -> usize {
        *self.lookback.iter().max().unwrap_or(&0)
    }

    /// Axes that have nonzero lookback.
    fn buffered_axes(&self) -> impl Iterator<Item = (usize, usize)> + '_ {
        self.lookback.iter().copied().enumerate().filter(|(_, lb)| *lb > 0)
    }
}

impl Op for PulseV2Buffer {
    fn name(&self) -> StaticName {
        "PulseV2Buffer".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        let axes: Vec<String> =
            self.buffered_axes().map(|(ax, lb)| format!("axis {ax}: lookback {lb}")).collect();
        Ok(vec![format!("depth={} {}", self.depth, axes.join(", "))])
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
        Ok(Some(Box::new(PulseV2BufferState {
            ring: Vec::new(),
            cumulative: tvec![0; self.lookback.len()],
        })))
    }
}

impl TypedOp for PulseV2Buffer {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let mut fact = inputs[0].clone();
        let t = TDim::Sym(self.pulse_id.clone());
        for (axis, lookback) in self.buffered_axes() {
            // Output dim = input_dim + H, where H is a symbol representing
            // the actual history provided. H is bounded: 0 <= H <= lookback.
            // At runtime, H resolves from the actual tensor size.
            // This avoids using min(T*P, lookback) which overestimates for
            // intermediate buffers (after convs that reduce the stream).
            let input_dim = fact.shape[axis].clone();
            let scope =
                fact.shape.iter().find_map(|d| d.find_scope()).unwrap_or_else(SymbolScope::default);
            let h = scope.new_with_prefix("H");
            scope.add_assertion(format!("{h} >= 0")).ok();
            scope.add_assertion(format!("{h} <= {lookback}")).ok();
            fact.shape.set(axis, input_dim + TDim::Sym(h));
        }
        Ok(tvec!(fact))
    }
}

#[derive(Debug, Clone)]
pub struct PulseV2BufferState {
    ring: Vec<Tensor>,
    /// Cumulative input elements received on each buffered axis.
    cumulative: TVec<usize>,
}

impl OpStateFreeze for PulseV2BufferState {
    fn freeze(&self) -> Box<dyn FrozenOpState> {
        unimplemented!("PulseV2BufferState::freeze not yet implemented")
    }
}

impl OpState for PulseV2BufferState {
    fn eval(
        &mut self,
        session: &mut TurnState,
        op: &dyn Op,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let op = op.downcast_ref::<PulseV2Buffer>().unwrap();
        let input = args_1!(inputs);
        let input_tensor = input.into_tensor();

        let output = if self.ring.is_empty() {
            // Track cumulative input for each buffered axis.
            for (axis, _) in op.buffered_axes() {
                self.cumulative[axis] = input_tensor.shape()[axis];
            }
            input_tensor.clone()
        } else {
            let mut result = input_tensor.clone();
            for (axis, lookback) in op.buffered_axes() {
                let current_size = input_tensor.shape()[axis];

                // Compute target history: min(T*P, lookback) per output_facts.
                let t = session.resolved_symbols.get(&op.pulse_id).unwrap_or(0) as usize;
                let p = session.resolved_symbols.get(&op.pulse_sym).unwrap_or(1) as usize;
                let target_history = lookback.min(t * p);
                // Actual history may be less if upstream produced fewer elements.
                let actual_history = target_history.min(self.cumulative[axis]);
                self.cumulative[axis] += current_size;

                let mut parts: Vec<Tensor> = Vec::with_capacity(self.ring.len() + 1);
                for prev in &self.ring {
                    parts.push(prev.clone());
                }
                parts.push(result);
                let stitched = Tensor::stack_tensors(axis, &parts)?;

                let total = stitched.shape()[axis];
                let keep = actual_history + current_size;
                result = if total > keep {
                    stitched.slice(axis, total - keep, total)?.into_tensor()
                } else {
                    stitched
                };
            }
            result
        };

        // Push current input into ring buffer, evict oldest if full.
        self.ring.push(input_tensor);
        if self.ring.len() > op.depth {
            self.ring.remove(0);
        }

        Ok(tvec!(output.into_tvalue()))
    }
}
