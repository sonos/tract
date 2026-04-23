/// Pulsifier for `Iff` when the condition wire carries a chunk-window
/// `uniform_tdim` expression.
///
/// At pulse time the ROI-aware EinSum + Delay guarantees that every key
/// position presented to the window falls within the causal window
/// `[c-L, c]`, so the mask is always all-true.  The false branch (fill
/// value, typically -inf broadcast to `[S, S]`) is therefore never
/// selected.  We elide the Iff entirely and wire the true branch directly.
///
/// This avoids a shape-inference problem: the fill's `MultiBroadcastTo`
/// retains the symbolic `[S, S]` shape in the pulsed model, which would
/// cause Iff to broadcast its output to `[S, S]` — giving the Softmax a
/// symbolic dimension that cannot be evaluated at runtime.
use crate::internal::*;
use tract_core::ops::logic::{Iff, classify_chunk_window};

register_all!(Iff: pulsify);

fn pulsify(
    _op: &Iff,
    source: &TypedModel,
    node: &TypedNode,
    _target: &mut PulsedModel,
    mapping: &HashMap<OutletId, OutletId>,
    _symbol: &Symbol,
    _pulse: &TDim,
) -> TractResult<Option<TVec<OutletId>>> {
    // inputs[0] = condition, inputs[1] = true branch, inputs[2] = false branch
    let cond_outlet = node.inputs[0];
    let cond_fact = source.outlet_fact(cond_outlet)?;

    // Only fire when the condition carries a chunk-window uniform_tdim.
    // In that case the mask is always all-true within the key window that the
    // ROI-aware EinSum + Delay already enforces, so we can elide the Iff.
    let expr = match cond_fact.uniform_tdim.as_ref() {
        Some(e) => e.clone().simplify(),
        None => return Ok(None),
    };
    if classify_chunk_window(&expr).is_none() {
        return Ok(None);
    }

    // The Iff is always true: just pass through the true branch.
    Ok(Some(tvec![mapping[&node.inputs[1]]]))
}
