use crate::internal::*;
use crate::model::NonPulsingWrappingOp;
use tract_core::ops::array::DynSlice;
use tract_core::ops::logic::classify_chunk_window;
use tract_core::ops::math;

register_all!(DynSlice: pulsify_dyn_slice);

fn pulsify_dyn_slice(
    op: &DynSlice,
    source: &TypedModel,
    node: &TypedNode,
    target: &mut PulsedModel,
    mapping: &HashMap<OutletId, OutletId>,
    _symbol: &Symbol,
    pulse: &TDim,
) -> TractResult<Option<TVec<OutletId>>> {
    let data_input = mapping[&node.inputs[0]];
    let data_fact = target.outlet_fact(data_input)?.clone();

    // Check output ROI for chunk-window annotation.
    let out_fact = source.outlet_fact(OutletId::new(node.id, 0))?;
    let cw = match out_fact
        .region_of_interest
        .as_ref()
        .and_then(|r| classify_chunk_window(&r.clone().simplify()))
    {
        Some(cw) => cw,
        None => return Ok(None),
    };

    // Only fire when slicing on the col_axis of the ROI.
    if op.axis != cw.col_axis {
        return Ok(None);
    }

    let pulse_i64 = pulse.to_i64()?;
    let w = (cw.left_chunks as i64 + 1) * pulse_i64; // key window size

    if data_fact.stream.is_some() {
        // Case A: streaming input, non-streaming axis = col_axis.
        // Replace the dynamic end with start + W; read start from the source wire.
        let start_i64 = source
            .outlet_fact(node.inputs[1])?
            .konst
            .as_ref()
            .and_then(|k| k.cast_to_scalar::<i64>().ok())
            .unwrap_or(0i64);
        // Only apply if the input has enough room on this axis.
        let input_axis_len = data_fact.shape[op.axis].to_i64().ok();
        if input_axis_len.is_some_and(|l| start_i64 + w > l) {
            return Ok(None);
        }
        use crate::model::PulseWrappingOp;
        use tract_core::ops::array::Slice;
        let out = target.wire_node(
            &node.name,
            PulseWrappingOp(Box::new(Slice::new(
                op.axis,
                start_i64 as usize,
                (start_i64 + w) as usize,
            ))),
            &[data_input],
        )?;
        return Ok(Some(out));
    }

    // Case B: non-streaming input (e.g. R extraction from r_full).
    // Adjust start and end to extract W+P-1 rows for the windowed RPE slice.
    //
    // Original formula (batch mode, axis=0):
    //   start = center - T   (→ center - P at pulse time)
    //   end   = center + T - 1  (→ center + P - 1 at pulse time)
    //   len   = 2T - 1
    //
    // Windowed pulse mode (left_chunks = L):
    //   start = center - (P-1)  = original_start + 1
    //   end   = center + W      = original_end + L*P + 1
    //   len   = W + P - 1
    let new_len = w + pulse_i64 - 1;
    if new_len <= 0 {
        return Ok(None);
    }

    // Build adjusted start and end wires by adding integer constants.
    let start_wire = mapping[&node.inputs[1]];
    let end_wire = mapping[&node.inputs[2]];

    let adj_start = add_i64_const(target, &node.name, "start_adj", start_wire, 1)?;
    let lp1 = (cw.left_chunks as i64) * pulse_i64 + 1;
    let adj_end = add_i64_const(target, &node.name, "end_adj", end_wire, lp1)?;

    let out = target.wire_node(
        &node.name,
        NonPulsingWrappingOp(Box::new(DynSlice::new(op.axis, TDim::Val(new_len)))),
        &[data_input, adj_start, adj_end],
    )?;
    Ok(Some(out))
}

/// Wire `wire + delta` as a NonPulsing scalar I64 addition in the pulsed model.
fn add_i64_const(
    target: &mut PulsedModel,
    node_name: &str,
    suffix: &str,
    wire: OutletId,
    delta: i64,
) -> TractResult<OutletId> {
    if delta == 0 {
        return Ok(wire);
    }
    let const_wire = target.add_const(format!("{node_name}.{suffix}_c"), rctensor0(delta))?;
    Ok(target.wire_node(
        format!("{node_name}.{suffix}"),
        NonPulsingWrappingOp(Box::new(math::add())),
        &[wire, const_wire],
    )?[0])
}
