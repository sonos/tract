//! Pulsifier for [`tract_transformers::ops::DiagGather`].
//!
//! The op itself and the `Pad → Reshape → Slice → Reshape → Slice` skew-trick
//! detect pass live in the `tract-transformers` crate (alongside ApplyRope,
//! ScaledMaskedSoftmax, etc.).  Only the pulse-specific pulsifier — plus its
//! local chunk-window mask classifier — lives here.

use crate::internal::*;
use crate::model::PulseWrappingOp;
use tract_core::ops::logic::sym_to_coord_axis;
use tract_transformers::ops::DiagGather;

// Re-export the detect pass at this path for in-pulse callers.  New code
// should import directly from `tract_transformers::ops`.
pub use tract_transformers::ops::detect_diag_gather;

register_all!(DiagGather: pulsify_diag_gather);

/// If `expr` is a 2-D chunk-window `uniform_tdim` mask, return the live
/// window width `W = (L+1)·P` — the per-row output length DiagGather should
/// produce in pulsed form.
///
/// The mask shape is
/// `Ge(Val(L), diff) * Ge(diff, Val(0))` with
/// `diff = floor((🎯row + r_off) / P) - floor((🎯col + c_off) / P) + constant`.
///
/// Semantically: a query at chunk-index `c = floor(i/P)` attends to keys in
/// chunks `c-L..=c`, i.e. `(L+1)·P` consecutive columns.  That width is the
/// only signal the DiagGather pulsifier needs from the ROI — it does not
/// care how the window decomposes into `L` and `P` individually.
fn chunk_window_width(expr: &TDim) -> Option<u64> {
    let TDim::Mul(factors) = expr else { return None };
    let n = factors.len();
    if n < 2 {
        return None;
    }
    for f0 in 0..n {
        for f1 in 0..n {
            if f0 == f1 {
                continue;
            }
            let TDim::Ge(lhs0, rhs0) = &factors[f0] else { continue };
            let TDim::Ge(lhs1, rhs1) = &factors[f1] else { continue };
            let TDim::Val(l) = lhs0.as_ref() else { continue };
            let TDim::Val(0) = rhs1.as_ref() else { continue };
            let Some((row, col, p)) = extract_div_diff_axes(rhs0) else { continue };
            let Some((row2, col2, p2)) = extract_div_diff_axes(lhs1) else { continue };
            if row != row2 || col != col2 || p != p2 {
                continue;
            }
            if *l < 0 {
                continue;
            }
            return Some((*l as u64 + 1) * p);
        }
    }
    None
}

/// Try to decompose `expr` as a chunk-index difference:
///   `floor((🎯row + r_off) / P) - floor((🎯col + c_off) / P) + constant`
fn extract_div_diff_axes(expr: &TDim) -> Option<(usize, usize, u64)> {
    let TDim::Add(terms) = expr else { return None };
    let mut pos: Option<(usize, u64)> = None;
    let mut neg: Option<(usize, u64)> = None;
    for term in terms {
        match term {
            TDim::Div(inner, p) => {
                if let Some(axis) = extract_coord_sym_from_div_arg(inner) {
                    pos = Some((axis, *p));
                }
            }
            TDim::MulInt(-1, inner) => {
                if let TDim::Div(inner2, p) = inner.as_ref() {
                    if let Some(axis) = extract_coord_sym_from_div_arg(inner2) {
                        neg = Some((axis, *p));
                    }
                }
            }
            TDim::Val(_) => {}
            _ => return None,
        }
    }
    let (row_axis, p_row) = pos?;
    let (col_axis, p_col) = neg?;
    if p_row != p_col {
        return None;
    }
    Some((row_axis, col_axis, p_row))
}

fn extract_coord_sym_from_div_arg(inner: &TDim) -> Option<usize> {
    match inner {
        TDim::Sym(sym) => sym_to_coord_axis(sym),
        TDim::Add(terms) => {
            let mut axis = None;
            for t in terms {
                match t {
                    TDim::Sym(sym) => {
                        if axis.is_some() {
                            return None;
                        }
                        axis = sym_to_coord_axis(sym);
                    }
                    TDim::Val(_) => {}
                    _ => return None,
                }
            }
            axis
        }
        _ => None,
    }
}

fn pulsify_diag_gather(
    _op: &DiagGather,
    source: &TypedModel,
    node: &TypedNode,
    target: &mut PulsedModel,
    mapping: &HashMap<OutletId, OutletId>,
    _symbol: &Symbol,
    _pulse: &TDim,
) -> TractResult<Option<TVec<OutletId>>> {
    // Pulse-time `out_len` is the live-window width W from the output's
    // chunk-window ROI.  If the ROI is missing or doesn't match the
    // chunk-window pattern, defer to the regular fallback.
    let roi_raw = source.outlet_fact(OutletId::new(node.id, 0))?.region_of_interest.clone();
    rule_if_some!(w = roi_raw.as_ref().and_then(|r| chunk_window_width(&r.clone().simplify())));

    let input_wire = mapping[&node.inputs[0]];
    let input_fact = target.outlet_fact(input_wire)?.clone();
    let stream = input_fact.stream.as_ref().context("DiagGather input must be streaming")?;

    // P_local: the pulse size at this level (after any subsampling).  In the
    // windowed input the relative-position axis has `W + P_local − 1`
    // entries; distance 0 sits at position `P_local − 1`.
    let p_local = input_fact.shape[stream.axis].to_i64()?;

    let pulsed_op = DiagGather { offset: (p_local - 1).to_dim(), out_len: (w as i64).to_dim() };

    let out = target.wire_node(&node.name, PulseWrappingOp(Box::new(pulsed_op)), &[input_wire])?;
    Ok(Some(out))
}
