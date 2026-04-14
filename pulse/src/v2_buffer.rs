/// PulseV2Buffer: stateful op that stores previous pulse tensors and
/// stitches them with the current input.
///
/// At each pulse T, the op:
/// 1. Concatenates buffered tensors with the current input along `axis`
/// 2. Slices the result to keep only the last `lookback + current_size`
///    elements on `axis`
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
    /// Axis along which to concatenate.
    pub axis: usize,
    /// How many past samples to retain along `axis`.
    pub lookback: usize,
    /// Number of previous pulse tensors to keep (typically 1 for small kernels,
    /// more if lookback > pulse_size).
    pub depth: usize,
}

impl Op for PulseV2Buffer {
    fn name(&self) -> StaticName {
        "PulseV2Buffer".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("axis={} lookback={} depth={}", self.axis, self.lookback, self.depth)])
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
        // Output shape: same as input, but axis dimension = input_dim + lookback
        // (except at T=0 where lookback is 0 — but facts describe steady state)
        let mut fact = inputs[0].clone();
        fact.shape.set(self.axis, fact.shape[self.axis].clone() + self.lookback.to_dim());
        Ok(tvec!(fact))
    }
}

#[derive(Debug, Clone)]
pub struct PulseV2BufferState {
    /// Ring buffer of previous pulse tensors.
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
            // T=0: no history yet. Zero-pad the lookback region.
            let mut padded_shape = input_tensor.shape().to_vec();
            padded_shape[op.axis] += op.lookback;
            let mut padded = Tensor::zero_dt(input_tensor.datum_type(), &padded_shape)?;
            padded.assign_slice(
                op.lookback..padded_shape[op.axis],
                &input_tensor,
                0..input_tensor.shape()[op.axis],
                op.axis,
            )?;
            padded
        } else {
            // Stitch: concatenate all buffered tensors + current along axis
            let mut parts: Vec<&Tensor> = Vec::with_capacity(self.ring.len() + 1);
            parts.extend(self.ring.iter());
            parts.push(&input_tensor);
            let stitched = Tensor::stack_tensors(op.axis, &parts)?;

            // Trim to keep only the last (lookback + current_size) on axis
            let total = stitched.shape()[op.axis];
            let current_size = input_tensor.shape()[op.axis];
            let keep = op.lookback + current_size;
            if total > keep {
                stitched.slice(op.axis, total - keep, total)?.into_tensor()
            } else {
                stitched
            }
        };

        // Push current input into ring buffer, evict oldest if full
        self.ring.push(input_tensor);
        if self.ring.len() > op.depth {
            self.ring.remove(0);
        }

        Ok(tvec!(output.into_tvalue()))
    }
}
