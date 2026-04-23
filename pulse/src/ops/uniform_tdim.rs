/// Pulsifier for `UniformTDim`.
///
/// Two regimes depending on `left_chunks`:
///
/// **`left_chunks == 0` (no lookback):** the mask is permanently all-True
/// (standard) or all-False (inverted) — no startup transient.  Emit a
/// constant tensor of that value with shape `[..., P, (L+1)*P]`.
///
/// **`left_chunks > 0` (lookback):** the mask has a startup transient: for
/// the first L chunks the K Delay buffer is zero-padded, and those L*P
/// positions should be masked False (out-of-window).  Emit a `ChunkWindowMask`
/// stateful op (shape `[P, (L+1)*P]`) followed by `AxisOp::Reshape` to
/// restore any leading singleton dimensions.  Inverted expressions
/// (`1 + -1*cw`) emit constant all-False (no startup issue: attention is
/// always fully masked at steady state, which is handled by Iff's own
/// pulsifier; this path is a safe fallback).
use crate::internal::*;
use crate::model::NonPulsingWrappingOp;
use crate::ops::mask::ChunkWindowMask;
use tract_core::ops::change_axes::AxisOp;
use tract_core::ops::konst::Const;
use tract_core::ops::logic::{
    ChunkWindowParams, classify_chunk_window, classify_negated_chunk_window,
};
use tract_core::ops::uniform_tdim::UniformTDim;

register_all!(UniformTDim: pulsify);

fn pulsify(
    op: &UniformTDim,
    _source: &TypedModel,
    node: &TypedNode,
    target: &mut PulsedModel,
    _mapping: &HashMap<OutletId, OutletId>,
    symbol: &Symbol,
    pulse: &TDim,
) -> TractResult<Option<TVec<OutletId>>> {
    let expr = op.expr.clone().simplify();
    let (ChunkWindowParams { p, left_chunks, row_axis, col_axis }, fill_value) =
        if let Some(cw) = classify_chunk_window(&expr) {
            (cw, true) // standard: in-window → True
        } else if let Some(cw) = classify_negated_chunk_window(&expr) {
            (cw, false) // inverted: in-window → False (was masked out)
        } else {
            return Ok(None);
        };

    // The raw pulse is in the streaming symbol's units (e.g. audio frames).
    // The token-axis pulse may differ when the output has a downsampling factor,
    // and the output dim may include a constant offset (e.g. 1+(T+6)/8).
    // Compute the per-pulse token count as shape(symbol=pulse) - shape(symbol=0).
    let pulse_i64 = pulse.to_i64()?;
    let pulse_size = if let Some(dim) = op.shape.iter().find(|d| d.symbols().contains(symbol)) {
        let mut sv_at = SymbolValues::default();
        sv_at.set(symbol, pulse_i64);
        let mut sv_zero = SymbolValues::default();
        sv_zero.set(symbol, 0);
        let at_pulse = dim.eval(&sv_at).to_i64()?;
        let at_zero = dim.eval(&sv_zero).to_i64()?;
        (at_pulse - at_zero) as usize
    } else {
        pulse_i64 as usize
    };

    ensure!(
        pulse_size == p as usize,
        "UniformTDim pulsifier: pulse size {pulse_size} != expr chunk size {p}"
    );

    let left_chunks = left_chunks as usize;
    let key_window = (left_chunks + 1) * pulse_size;
    let rank = op.shape.len();

    // Build the output shape: leading dims (before row_axis) stay as-is (evaluated
    // at symbol=0, since they don't depend on the streaming symbol or are batch dims),
    // row_axis → pulse_size, col_axis → key_window.
    let mut sv_zero = SymbolValues::default();
    sv_zero.set(symbol, 0);
    let mut shape: Vec<usize> = Vec::with_capacity(rank);
    for (ax, dim) in op.shape.iter().enumerate() {
        if ax == row_axis {
            shape.push(pulse_size);
        } else if ax == col_axis {
            shape.push(key_window);
        } else {
            // Non-streaming dim: evaluate at symbol=0.
            shape.push(dim.eval(&sv_zero).to_usize()?);
        }
    }

    // For left_chunks > 0 and standard convention (fill_value=true): emit a
    // ChunkWindowMask stateful op so that zero-padded K positions during
    // startup are correctly masked False.  A constant all-True mask would
    // incorrectly treat the zero-padded lookback keys as valid.
    if left_chunks > 0 && fill_value {
        let cwm_wire = target.wire_node(
            format!("{}.chunk_window_mask", node.name),
            ChunkWindowMask { left_chunks, pulse_size, key_window },
            &[],
        )?[0];

        // ChunkWindowMask produces [pulse_size, key_window] (rank 2).
        // Reshape to the full target shape (restores leading singleton dims).
        let wire = if rank == 2 {
            cwm_wire
        } else {
            target.wire_node(
                format!("{}.reshape", node.name),
                NonPulsingWrappingOp(Box::new(AxisOp::Reshape(
                    0,
                    tvec![pulse_size.to_dim(), key_window.to_dim()],
                    shape.iter().map(|&s| TDim::Val(s as i64)).collect(),
                ))),
                &[cwm_wire],
            )?[0]
        };
        return Ok(Some(tvec![wire]));
    }

    // Default: constant tensor (all-True for standard L=0, all-False for inverted).
    let total: usize = shape.iter().product();
    let data = vec![fill_value; total];
    let tensor = Tensor::from_shape(&shape, &data)?;

    Ok(Some(target.wire_node(
        &node.name,
        NonPulsingWrappingOp(Box::new(Const::new(tensor.into_arc_tensor())?)),
        &[],
    )?))
}
