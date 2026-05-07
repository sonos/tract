//! `AffineChunkTrim` — pulse-aware "drop trailing `c` on the streaming
//! axis" op, used by blockify's `wire_chunk_split` when the input dim
//! is affine in the chunk symbol (`c + k·S` for some `c > 0`).
//!
//! Why a custom op instead of a plain `Slice(0, k·S)`?  The typed
//! Slice's pulsifier produces `PulsedAxisSlice`, which is identity at
//! runtime — it only updates `stream.delay`/`stream.dim`, leaving the
//! per-pulse buffer at its original size.  Downstream the chunked
//! Reshape `[k·S] → [S, k]` then sees per-pulse `c + k` where it
//! expects `k` and bails.
//!
//! Why is the per-pulse not always `c + k` then?  Some upstream
//! pulsifiers (notably stride-conv stacks producing the encoder's
//! `1 + (T+6)/8` outputs) absorb the `c` initial frames into Delay
//! state and emit per-pulse exactly `k`.  Others (Range/LT producing
//! the padmask source) substitute `(c + k·S).substitute(S, pulse)` and
//! emit per-pulse `c + k`.  `AffineChunkTrim` handles both: it trims
//! the per-pulse buffer to `target_per_pulse = k` only when the actual
//! input exceeds `k`, and shrinks `stream.dim` by `typed_trim = c`
//! unconditionally so the chunked-Reshape's pulsifier sees a clean
//! `k·S` typed dim.  The eval mirrors the same conditional: drop
//! trailing `typed_trim` only when the result would still be at least
//! `target_per_pulse` (the "typed" eval path on a `c + k·S` fully
//! resolved input), otherwise pass through (the pulse path on an
//! already-trimmed `k`-sized buffer).

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
    /// Number of trailing elements the typed Slice would drop (= `c` in the
    /// affine `c + k·S` source dim).  Used in typed `output_facts` and to
    /// shrink `stream.dim` in the pulsed fact.
    pub typed_trim: usize,
    /// Per-pulse element count the chunked Reshape downstream expects on
    /// `axis` (= `k`).  Used to decide whether the per-pulse buffer needs
    /// physical trimming or is already at the target.
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
        // Drop trailing `typed_trim` only when doing so leaves at least
        // `target_per_pulse` elements — i.e. the input is the "raw" affine
        // size (`c + k·S` typed or `c + k` per-pulse).  Otherwise the input
        // is already trimmed by an upstream pulsifier (per-pulse exactly
        // `k`); pass through unchanged.
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
        fact.shape.set(self.axis, dim - self.typed_trim.to_dim());
        Ok(tvec!(fact))
    }
}

impl PulsedOp for AffineChunkTrim {
    fn pulsed_output_facts(&self, inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        let mut fact = inputs[0].clone();
        let cur = fact.shape[self.axis].to_usize()?;
        // Per-pulse trim: only drop the excess above `target_per_pulse`.
        // Sources whose pulsifier already delivered exactly `k` per pulse
        // (e.g. conv stacks) skip this (no shape change); sources delivering
        // `c + k` per pulse (e.g. Range/LT) get the `c` trailing dropped.
        let trim_amount = cur.saturating_sub(self.target_per_pulse);
        if trim_amount > 0 {
            let new_per_pulse = cur - trim_amount;
            let mut shape: TVec<TDim> = fact.shape.iter().cloned().collect();
            shape[self.axis] = new_per_pulse.to_dim();
            fact.shape = shape.into();
        }
        // Stream.dim always shrinks by `typed_trim` to match the typed Slice
        // it stands in for; this is what the chunked-Reshape downstream
        // matches against (`from = [k·S]`, not `[c + k·S]`).
        if let Some(stream) = fact.stream.as_mut() {
            stream.dim = stream.dim.clone() - self.typed_trim.to_dim();
        }
        Ok(tvec!(fact))
    }

    as_op!();
    pulsed_op_to_typed_op!();
}
