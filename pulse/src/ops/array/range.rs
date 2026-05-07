//! Pulsifier for `tract_core::ops::array::Range`.
//!
//! Range is a 0-streaming-input generator: its (start, end, step) inputs
//! are scalar constants; its single output is a 1-D wire whose length is
//! the symbolic `(end - start)/step`.  When `end` contains the streaming
//! symbol, `NonPulsingWrappingOp`'s konst-stripping fallback in
//! `Range::output_facts` re-evaluates the shape via `self.len`, producing
//! a fresh `Sym(range_NN)` symbol unrelated to the stream — a pulse-time
//! mismatch we sidestep by emitting `PulsedRange` instead.

use crate::fact::StreamInfo;
use crate::internal::*;
use tract_core::ops::array::Range;
use tract_pulse_opl::ops::PulsedRange;

register_all!(Range: pulsify);

fn pulsify(
    _op: &Range,
    source: &TypedModel,
    node: &TypedNode,
    target: &mut PulsedModel,
    _mapping: &HashMap<OutletId, OutletId>,
    symbol: &Symbol,
    pulse: &TDim,
) -> TractResult<Option<TVec<OutletId>>> {
    // Output shape must be a 1-D wire whose dim contains the stream symbol.
    let out_fact = &node.outputs[0].fact;
    rule_if!(out_fact.rank() == 1);
    rule_if!(out_fact.shape[0].symbols().contains(symbol));
    let stream_dim = out_fact.shape[0].clone();
    let datum_type = out_fact.datum_type;

    // Pull start/step from the source model's input facts as scalar consts.
    // (`end` only matters for the symbolic length, which we already have.)
    let input_facts = source.node_input_facts(node.id)?;
    rule_if!(input_facts.len() == 3);
    rule_if_some!(start = input_facts[0].konst.as_ref());
    rule_if_some!(step = input_facts[2].konst.as_ref());
    let start = start.clone().into_tensor();
    let step = step.clone().into_tensor();

    // Per-pulse element count on the stream axis.  Two flavours:
    //
    //   * Linear `k·S` stream dim:    per_pulse = k · pulse
    //   * Affine `c + k·S` stream dim with c > 0:
    //                                 per_pulse = k · pulse
    //     (drops the leading `c` from the per-pulse view, matching the
    //     stride-conv-pulsifier convention that absorbs the `c` initial
    //     frames into Delay state rather than emitting them per pulse —
    //     see ex16-affine-residual-then-conv for why this matters: a
    //     stride-conv-stack output with the same affine typed dim would
    //     otherwise pulsify to `k` while a parallel Range pulsified to
    //     `c + k`, and any downstream op reconciling the two paths would
    //     see a `(k)#(c+k)` Broadcast pulse fact that `pulsify_pooled_input`
    //     and friends can't divide by stride).
    //
    // Detect affine offset by extracting the constant term of the stream
    // dim (substitute symbol → 0).  Slope `k` then comes from
    // substituting symbol → 1 minus the offset.
    let zero_dim = 0i64.to_dim();
    let one_dim = 1i64.to_dim();
    let offset = stream_dim
        .clone()
        .substitute(symbol, &zero_dim)
        .ok()
        .and_then(|d| d.to_i64().ok())
        .unwrap_or(0);
    let unit_extent =
        stream_dim.clone().substitute(symbol, &one_dim).ok().and_then(|d| d.to_i64().ok());
    let pulse_concrete = pulse.to_i64()?;
    let per_pulse: usize = if offset > 0
        && let Some(unit_extent) = unit_extent
        && unit_extent > offset
    {
        let slope = unit_extent - offset;
        (slope * pulse_concrete) as usize
    } else {
        stream_dim.clone().substitute(symbol, pulse)?.to_usize()?
    };
    let pulsed =
        PulsedRange { datum_type, start, step, stream_dim: stream_dim.clone(), pulse: per_pulse };
    target.wire_node(&*node.name, pulsed, &[]).map(Some)
}

impl PulsedOp for PulsedRange {
    fn pulsed_output_facts(&self, _inputs: &[&PulsedFact]) -> TractResult<TVec<PulsedFact>> {
        let shape: TVec<TDim> = tvec!(self.pulse.to_dim());
        Ok(tvec!(PulsedFact {
            datum_type: self.datum_type,
            shape: shape.into(),
            stream: Some(StreamInfo { axis: 0, dim: self.stream_dim.clone(), delay: 0 }),
        }))
    }

    as_op!();
    pulsed_op_to_typed_op!();
}
