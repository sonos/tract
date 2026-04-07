/// Pulsifier for `EinSum` in two windowed-attention cases.
///
/// **Case 1 — QK EinSum** (output has chunk-window `region_of_interest`):
///
/// For a QK einsum like `"id,jd->ij"`:
/// - axis `i` (appears in Q and output axis 0) is the streaming axis
/// - axis `j` (appears in K and output axis 1) is the key axis — needs a
///   sliding-window delay driven by the ROI window size
///
/// At pulse time (pulse = P tokens):
///   Q: [P, D]  (streaming on i-axis)
///   K: [(L+1)*P, D]  via Delay(axis=key_ax, delay=0, overlap=L*P)
///   scores: [P, (L+1)*P]  (streaming on i-axis via PulseWrappingOp)
///
/// **Case 2 — AV EinSum** (streaming attn × V with contracted streaming axis):
///
/// For an AV einsum like `"ij,jd->id"`:
/// - axis `j` is contracted and is the streaming axis of V (axis 0 of V)
/// - axis `i` is the streaming axis of attn (axis 0 of attn) and the output
///
/// At pulse time:
///   attn: [P, (L+1)*P]  (streaming on i-axis)
///   V: [(L+1)*P, D]  via Delay(axis=0, delay=0, overlap=L*P)
///   output: [P, D]  (streaming on i-axis via PulseWrappingOp)
use crate::internal::*;
use crate::model::PulseWrappingOp;
use tract_core::ops::einsum::EinSum;
use tract_core::ops::logic::classify_chunk_window;
use tract_pulse_opl::ops::Delay;

register_all!(EinSum: pulsify);

fn pulsify(
    op: &EinSum,
    source: &TypedModel,
    node: &TypedNode,
    target: &mut PulsedModel,
    mapping: &HashMap<OutletId, OutletId>,
    _symbol: &Symbol,
    _pulse: &TDim,
) -> TractResult<Option<TVec<OutletId>>> {
    // Case 1: ROI-annotated QK EinSum.
    if let Some(result) = pulsify_qk(op, source, node, target, mapping)? {
        return Ok(Some(result));
    }
    // Case 2: AV EinSum (streaming attn × streaming V with contracted streaming axis).
    pulsify_av(op, node, target, mapping)
}

/// Case 1: QK-style EinSum where the output carries a chunk-window ROI annotation.
///
/// Adds a sliding-window Delay to K on its key axis, then wires the EinSum
/// with PulseWrappingOp so the streaming dimension (Q's row axis) is propagated
/// to the output scores fact.
fn pulsify_qk(
    op: &EinSum,
    source: &TypedModel,
    node: &TypedNode,
    target: &mut PulsedModel,
    mapping: &HashMap<OutletId, OutletId>,
) -> TractResult<Option<TVec<OutletId>>> {
    let roi = match source
        .outlet_fact(OutletId::new(node.id, 0))?
        .region_of_interest
        .as_ref()
        .and_then(|r| classify_chunk_window(&r.clone().simplify()))
    {
        Some(p) => p,
        None => return Ok(None),
    };

    let chunk_size = roi.p; // P tokens per chunk
    let left_chunks = roi.left_chunks as usize; // L

    let pulsed_inputs: TVec<(usize, OutletId)> =
        node.inputs.iter().enumerate().map(|(ix, o)| (ix, mapping[o])).collect();

    // Identify Q (row) and K (col) by which AxesMapping axis connects each input's
    // streaming axis to the ROI row/col output axis.
    let (q_input_ix, k_input_ix) = {
        let mut q_ix = None;
        let mut k_ix = None;
        for (input_ix, pulsed_outlet) in &pulsed_inputs {
            let stream_axis = match target.outlet_fact(*pulsed_outlet)?.stream.as_ref() {
                Some(s) => s.axis,
                None => continue,
            };
            let out_axis = op.axes.iter_all_axes().find(|ax| {
                ax.inputs.get(*input_ix).map(|v| v.contains(&stream_axis)).unwrap_or(false)
                    && !ax.outputs[0].is_empty()
            });
            if let Some(ax) = out_axis {
                let out_pos = ax.outputs[0][0];
                if out_pos == roi.row_axis {
                    q_ix = Some(*input_ix);
                } else if out_pos == roi.col_axis {
                    k_ix = Some(*input_ix);
                }
            }
        }
        // If either Q or K couldn't be identified as streaming (e.g. Q@R^T where R is a
        // fixed position-encoding table), decline and let the generic pulsifier handle it.
        if q_ix.is_none() || k_ix.is_none() {
            return Ok(None);
        }
        (q_ix.unwrap(), k_ix.unwrap())
    };

    // The key axis in K: the K input axis that maps to the output key axis (col_axis).
    let k_axis_in_k = op
        .axes
        .iter_all_axes()
        .find(|ax| {
            ax.inputs[q_input_ix].is_empty()
                && !ax.inputs[k_input_ix].is_empty()
                && !ax.outputs[0].is_empty()
        })
        .and_then(|ax| ax.inputs[k_input_ix].first().copied())
        .with_context(|| {
            format!("ROI-aware EinSum pulsifier: cannot find key axis in K for axes {:?}", op.axes)
        })?;

    let name = &node.name;
    let q_wire = pulsed_inputs[q_input_ix].1;

    // K: Delay(axis=k_axis_in_k, delay=0, overlap=L*P).
    // output_pulse = P + overlap = (L+1)*P — exactly the context window size.
    let k_wire_in = pulsed_inputs[k_input_ix].1;
    let k_fact_typed: TypedFact = target.outlet_fact(k_wire_in)?.clone().into();
    let overlap = left_chunks * chunk_size as usize;
    let k_wire = if left_chunks > 0 {
        target.wire_node(
            format!("{name}.k_delay"),
            Delay::new_typed(&k_fact_typed, k_axis_in_k, 0, overlap),
            &[k_wire_in],
        )?[0]
    } else {
        k_wire_in
    };

    // Wire EinSum with PulseWrappingOp so that Q's streaming axis propagates
    // to the output scores fact.  We do NOT call sync_inputs here: Q intentionally
    // has delay=0 (current chunk) while K has delay=L*P (startup padding).
    let mut inputs: TVec<OutletId> = node.inputs.iter().map(|i| mapping[i]).collect();
    inputs[q_input_ix] = q_wire;
    inputs[k_input_ix] = k_wire;

    Ok(Some(target.wire_node(name, PulseWrappingOp(Box::new(op.clone())), &inputs)?))
}

/// Case 2: AV-style EinSum where one of the streaming inputs has its streaming
/// axis on a contracted dimension (V, whose token axis j maps to the key window)
/// and the other streaming input is non-contracted (attn, streaming on query axis i).
///
/// Detected when there is at least one streaming input whose stream axis maps to a
/// contracted axis (not present in the output).  A Delay is added to that input to
/// expand its key-axis from P tokens to (L+1)*P tokens (the full key window).
/// Then PulseWrappingOp is used so that the non-contracted streaming axis (i) of
/// the attn input propagates to the output.
fn pulsify_av(
    op: &EinSum,
    node: &TypedNode,
    target: &mut PulsedModel,
    mapping: &HashMap<OutletId, OutletId>,
) -> TractResult<Option<TVec<OutletId>>> {
    if node.inputs.len() != 2 {
        return Ok(None);
    }

    let pulsed_inputs: TVec<(usize, OutletId)> =
        node.inputs.iter().enumerate().map(|(ix, o)| (ix, mapping[o])).collect();

    // Among all streaming inputs, find the one whose stream axis maps to a contracted
    // axis (not in output) — this is the V-like input that needs a Delay.
    let v_info: Option<(usize, usize)> = pulsed_inputs.iter().find_map(|(ix, out)| {
        let stream = target.outlet_fact(*out).ok()?.stream.as_ref()?;
        let is_contracted = op.axes.iter_all_axes().any(|ax| {
            ax.inputs.get(*ix).map(|v| v.contains(&stream.axis)).unwrap_or(false)
                && ax.outputs[0].is_empty()
        });
        if is_contracted { Some((*ix, stream.axis)) } else { None }
    });

    let (v_input_ix, v_stream_axis) = match v_info {
        Some(info) => info,
        None => return Ok(None),
    };

    let attn_input_ix = 1 - v_input_ix;

    // Find the axis in the attn (non-V) input that corresponds to the contracted axis.
    let attn_contracted_axis = op
        .axes
        .iter_all_axes()
        .find(|ax| {
            ax.inputs.get(v_input_ix).map(|v| v.contains(&v_stream_axis)).unwrap_or(false)
                && ax.outputs[0].is_empty()
        })
        .and_then(|ax| ax.inputs.get(attn_input_ix)?.first().copied());
    let attn_contracted_axis = match attn_contracted_axis {
        Some(a) => a,
        None => return Ok(None),
    };

    // Key window = attn.shape[attn_contracted_axis] — must be concrete.
    let attn_pulsed = target.outlet_fact(pulsed_inputs[attn_input_ix].1)?.clone();
    let key_window = match attn_pulsed.shape[attn_contracted_axis].to_usize() {
        Ok(w) => w,
        Err(_) => return Ok(None),
    };

    // Pulse size = v.shape[v_stream_axis].
    let v_pulsed = target.outlet_fact(pulsed_inputs[v_input_ix].1)?.clone();
    let pulse_size = match v_pulsed.shape[v_stream_axis].to_usize() {
        Ok(p) => p,
        Err(_) => return Ok(None),
    };

    if key_window < pulse_size {
        return Ok(None);
    }
    let overlap = key_window - pulse_size;

    let name = &node.name;
    let v_wire_in = pulsed_inputs[v_input_ix].1;
    let v_fact_typed: TypedFact = v_pulsed.into();
    let v_wire = if overlap > 0 {
        target.wire_node(
            format!("{name}.v_delay"),
            Delay::new_typed(&v_fact_typed, v_stream_axis, 0, overlap),
            &[v_wire_in],
        )?[0]
    } else {
        v_wire_in
    };

    // Wire the EinSum with PulseWrappingOp so the non-contracted streaming axis
    // (attn's query axis) propagates to the output.  We place attn first so
    // PulseWrappingOp finds its non-contracted stream axis before V's contracted one.
    let mut inputs: TVec<OutletId> = node.inputs.iter().map(|i| mapping[i]).collect();
    inputs[v_input_ix] = v_wire;

    Ok(Some(target.wire_node(name, PulseWrappingOp(Box::new(op.clone())), &inputs)?))
}
