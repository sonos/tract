/// Pulsifier for `UniformTDim`.
///
/// At pulse time the chunk-window mask is always all-true within the key
/// window `[(c-L)*P, (c+1)*P)` that the ROI-aware EinSum already enforces
/// via a Delay on K.  So the pulsified `UniformTDim` is a constant all-true
/// tensor.
///
/// The output shape respects the original node's shape: leading singleton
/// dims are kept, and the two streaming dims become `[P, (L+1)*P]`.
use crate::internal::*;
use crate::model::NonPulsingWrappingOp;
use tract_core::ops::konst::Const;
use tract_core::ops::logic::{ChunkWindowParams, classify_chunk_window};
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
    let ChunkWindowParams { p, left_chunks, row_axis, col_axis } =
        match classify_chunk_window(&op.expr.clone().simplify()) {
            Some(cw) => cw,
            None => return Ok(None),
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

    let key_window = (left_chunks as usize + 1) * pulse_size;
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

    let total: usize = shape.iter().product();
    let data = vec![true; total];
    let tensor = Tensor::from_shape(&shape, &data)?;

    Ok(Some(target.wire_node(
        &node.name,
        NonPulsingWrappingOp(Box::new(Const::new(tensor.into_arc_tensor())?)),
        &[],
    )?))
}
