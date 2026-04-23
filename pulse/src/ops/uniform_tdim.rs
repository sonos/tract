/// Pulsifier for `UniformTDim`.
///
/// At pulse time the chunk-window mask is always all-true within the key
/// window `[(c-L)*P, (c+1)*P)` that the ROI-aware EinSum already enforces
/// via a Delay on K.  So the pulsified `UniformTDim` is a constant all-true
/// tensor of shape `[P, (L+1)*P]`.
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
    _symbol: &Symbol,
    pulse: &TDim,
) -> TractResult<Option<TVec<OutletId>>> {
    let ChunkWindowParams { p, left_chunks, .. } =
        match classify_chunk_window(&op.expr.clone().simplify()) {
            Some(cw) => cw,
            None => return Ok(None),
        };

    let pulse_size = pulse.to_i64()? as usize;
    ensure!(
        pulse_size == p as usize,
        "UniformTDim pulsifier: pulse size {pulse_size} != expr chunk size {p}"
    );

    let key_window = (left_chunks as usize + 1) * pulse_size;

    let all_true = tract_core::ndarray::Array2::<bool>::from_elem((pulse_size, key_window), true);
    let tensor = tract_core::internal::Tensor::from(all_true);

    Ok(Some(target.wire_node(
        &node.name,
        NonPulsingWrappingOp(Box::new(Const::new(tensor.into_arc_tensor())?)),
        &[],
    )?))
}
