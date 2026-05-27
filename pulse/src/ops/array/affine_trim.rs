//! `AffineChunkTrim` — pulse-aware "drop trailing `c` on the streaming
//! axis" op for affine-tail dims (`c + k·S`).  Replaces a typed
//! `Slice(0, k·S)` whose pulsifier is identity and would leave the
//! per-pulse buffer at `c + k`, breaking the chunked Reshape downstream
//! that expects `k`.  Some upstream pulsifiers absorb `c` into Delay
//! state and emit `k` per pulse; others emit `c + k`.  Handles both by
//! trimming only when the input exceeds `target_per_pulse`.

use crate::internal::*;

register_all!(AffineChunkTrim: pulsify);

fn pulsify(
    op: &AffineChunkTrim,
    _source: &TypedModel,
    node: &TypedNode,
    target: &mut PulsedModel,
    mapping: &HashMap<OutletId, OutletId>,
    _symbol: &Symbol,
    _pulse: &TDim,
) -> TractResult<Option<TVec<OutletId>>> {
    let input = mapping[&node.inputs[0]];
    target.wire_node(&*node.name, op.clone(), &[input]).map(Some)
}

#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct AffineChunkTrim {
    pub axis: usize,
    pub typed_trim: usize,
    pub target_per_pulse: usize,
}

impl Op for AffineChunkTrim {
    fn name(&self) -> StaticName {
        "AffineChunkTrim".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!(
            "axis: {} typed_trim: {} target_per_pulse: {}",
            self.axis, self.typed_trim, self.target_per_pulse
        )])
    }

    op_as_typed_op!();
}

impl EvalOp for AffineChunkTrim {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs);
        let n = input.shape()[self.axis];
        let take = if n.saturating_sub(self.typed_trim) >= self.target_per_pulse {
            n - self.typed_trim
        } else {
            n
        };
        if take == n {
            return Ok(tvec!(input));
        }
        unsafe {
            let mut shape: TVec<usize> = input.shape().into();
            shape[self.axis] = take;
            let mut tensor = Tensor::uninitialized_dt(input.datum_type(), &shape)?;
            tensor.assign_slice_unchecked(.., &input, 0..take, self.axis);
            Ok(tvec!(tensor.into_tvalue()))
        }
    }
}

impl TypedOp for AffineChunkTrim {
    as_op!();

    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let mut fact = inputs[0].without_value();
        let dim = fact.shape[self.axis].clone();
        // Mirror `eval` / `pulsed_output_facts`: trim only when the input
        // axis is concrete *and* strictly exceeds `target_per_pulse`.  Two
        // cases this handles:
        //   - pre-pulse typed model with symbolic input `c + k·S`: dim is
        //     not concretely sized → subtract `typed_trim` (`c`) as before.
        //   - pulsified model where the upstream emits `target_per_pulse`
        //     directly (e.g. Delay absorbed `c`): dim is concrete and equal
        //     to `target_per_pulse` → no trim, matching eval.  Previously
        //     this branch returned `dim - typed_trim` which lied about the
        //     output shape and broke downstream consumers (e.g. a Reshape
        //     keyed off `target_per_pulse`) — visible when --cuda translates
        //     the pulsified encoder.
        let new_dim = if let Ok(cur) = dim.to_usize() {
            if cur > self.target_per_pulse { self.target_per_pulse.to_dim() } else { cur.to_dim() }
        } else {
            dim - self.typed_trim.to_dim()
        };
        fact.shape.set(self.axis, new_dim);
        Ok(tvec!(fact))
    }
}

impl PulsedOp for AffineChunkTrim {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        let mut fact = inputs[0].clone();
        let cur = fact.shape[self.axis].to_usize()?;
        let trim_amount = cur.saturating_sub(self.target_per_pulse);
        if trim_amount > 0 {
            let new_per_pulse = cur - trim_amount;
            let mut shape: TVec<TDim> = fact.shape.iter().cloned().collect();
            shape[self.axis] = new_per_pulse.to_dim();
            fact.shape = shape.into();
        }
        if let Some(stream) = fact.stream.as_mut() {
            stream.dim = stream.dim.clone() - self.typed_trim.to_dim();
        }
        Ok(tvec!(fact))
    }

    as_op!();
    pulsed_op_to_typed_op!();
}
