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
    /// Pulse index symbol (T).
    pub pulse_id: Symbol,
    /// Pulse size symbol or concrete value.
    pub pulse_dim: TDim,
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
        // Output dim = input_dim + min(T * input_dim, lookback).
        // At T=0: input_dim + 0 = input_dim (no lookback yet).
        // At T≥1 (steady state): input_dim + lookback.
        let input_dim = inputs[0].shape[self.axis].clone();
        let t = TDim::Sym(self.pulse_id.clone());
        let available = t * input_dim.clone();
        let effective_lookback = TDim::Min(vec![available, TDim::Val(self.lookback as i64)]);
        let out_dim = input_dim + effective_lookback;
        let mut fact = inputs[0].clone();
        fact.shape.set(self.axis, out_dim);
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
            // T=0: no buffer, output is just the input.
            input_tensor.clone()
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
