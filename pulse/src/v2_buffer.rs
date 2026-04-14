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
    pub lookback: TVec<usize>,
    /// Number of previous pulse tensors to keep in the ring buffer.
    pub depth: usize,
    /// Pulse index symbol (T).
    pub pulse_id: Symbol,
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
        Ok(Some(Box::new(PulseV2BufferState { ring: Vec::new() })))
    }
}

impl TypedOp for PulseV2Buffer {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let mut fact = inputs[0].clone();
        let t = TDim::Sym(self.pulse_id.clone());
        for (axis, lookback) in self.buffered_axes() {
            // Output dim = input_dim + min(T * input_dim, lookback).
            // At T=0: input_dim (no lookback). At steady state: input_dim + lookback.
            let input_dim = fact.shape[axis].clone();
            let available = t.clone() * input_dim.clone();
            let effective = TDim::Min(vec![available, TDim::Val(lookback as i64)]);
            fact.shape.set(axis, input_dim + effective);
        }
        Ok(tvec!(fact))
    }
}

#[derive(Debug, Clone)]
pub struct PulseV2BufferState {
    ring: Vec<Tensor>,
}

impl OpStateFreeze for PulseV2BufferState {
    fn freeze(&self) -> Box<dyn FrozenOpState> {
        unimplemented!("PulseV2BufferState::freeze not yet implemented")
    }
}

impl OpState for PulseV2BufferState {
    fn eval(
        &mut self,
        _state: &mut TurnState,
        op: &dyn Op,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let op = op.downcast_ref::<PulseV2Buffer>().unwrap();
        let input = args_1!(inputs);
        let input_tensor = input.into_tensor();

        let output = if self.ring.is_empty() {
            // T=0: no buffer, output is just the input.
            input_tensor.clone()
        } else {
            // Stitch along each buffered axis.
            let mut result = input_tensor.clone();
            for (axis, lookback) in op.buffered_axes() {
                // Concatenate all buffered tensors + current along this axis
                let mut parts: Vec<Tensor> = Vec::with_capacity(self.ring.len() + 1);
                for prev in &self.ring {
                    parts.push(prev.clone());
                }
                parts.push(result);
                let stitched = Tensor::stack_tensors(axis, &parts)?;

                // Trim to keep only the last (lookback + current_size)
                let total = stitched.shape()[axis];
                let current_size = input_tensor.shape()[axis];
                let keep = lookback + current_size;
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
