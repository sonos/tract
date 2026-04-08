use crate::internal::*;
use crate::model::NonPulsingWrappingOp;
use tract_core::ops::array::Range;

register_all!(Range: pulsify_range);

/// Pulsify a `Range` op whose output length contains the streaming symbol but
/// whose inputs are all non-streaming (static start / end / step).
///
/// Example: `arange = range(0, T_tokens, 1)` where `T_tokens = 1 + (S+6)/8`.
/// In the typed model the output has shape `[T_tokens]` and at runtime produces
/// `[0, 1, ..., T_tokens-1]`.  In the pulsed model we want shape `[delta]`
/// where `delta = T_tokens(pulse) - T_tokens(0)` — the incremental token count
/// per pulse.  We also wire a const `delta` as the `end` input so that
/// `Range::eval` produces exactly `delta` elements instead of `T_tokens(pulse)`.
fn pulsify_range(
    _op: &Range,
    _source: &TypedModel,
    node: &TypedNode,
    target: &mut PulsedModel,
    mapping: &HashMap<OutletId, OutletId>,
    symbol: &Symbol,
    pulse: &TDim,
) -> TractResult<Option<TVec<OutletId>>> {
    let output_shape = &node.outputs[0].fact.shape;
    // Only handle the case where the output length contains the streaming symbol.
    if output_shape.rank() != 1 || !output_shape[0].symbols().contains(symbol) {
        return Ok(None);
    }

    // All inputs must be non-streaming; streaming-input Range is not handled here.
    let any_streaming = node
        .inputs
        .iter()
        .map(|i| target.outlet_fact(mapping[i]).map(|f| f.stream.is_some()))
        .collect::<TractResult<Vec<_>>>()?
        .into_iter()
        .any(|s| s);
    if any_streaming {
        return Ok(None);
    }

    // Compute delta: len(symbol=pulse) - len(symbol=0).
    // This is the per-pulse token count used for both shape inference and runtime.
    let len_dim = &output_shape[0];
    let pulse_i64 = pulse.to_i64()?;
    let mut sv_pulse = SymbolValues::default();
    sv_pulse.set(symbol, pulse_i64);
    let mut sv_zero = SymbolValues::default();
    sv_zero.set(symbol, 0);
    let delta = len_dim.eval(&sv_pulse).to_i64()? - len_dim.eval(&sv_zero).to_i64()?;

    // Build new start=const(0), end=const(delta), step=const(1) wires so that
    // Range::eval produces exactly `delta` elements at runtime (not T_tokens(pulse)).
    // Use I64 constants: Range::make always emits I64 when inputs are TDim, so using
    // I64 inputs keeps datum_type consistent after NonPulsingWrappingOp strips konst.
    let start_wire = target.add_const(format!("{}.pulsed_start", node.name), rctensor0(0i64))?;
    let end_wire = target.add_const(format!("{}.pulsed_end", node.name), rctensor0(delta))?;
    let step_wire = target.add_const(format!("{}.pulsed_step", node.name), rctensor0(1i64))?;

    let out = target.wire_node(
        &node.name,
        NonPulsingWrappingOp(Box::new(Range::new(TDim::Val(delta)))),
        &[start_wire, end_wire, step_wire],
    )?;
    Ok(Some(out))
}
