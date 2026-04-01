/// `FoldWindowAttention` optimizer pass.
///
/// Detects the pattern:
///
/// ```text
/// q [T, D]  ──┐
///              ├─ EinSum("id,jd->ij") ──→ scores [T,T] ──→ Iff(chunk_window_mask) ──→ Softmax ──→ EinSum("ij,jd->id") ──→ output [T,D]
/// k [T, D]  ──┘                                                                                                   /
/// v [T, D]  ──────────────────────────────────────────────────────────────────────────────────────────────────────
/// ```
///
/// where the `Iff` condition wire has a `uniform_tdim` expressing a 2-D chunk-window
/// mask: `0 ≤ floor(i/P) − floor(j/P) ≤ L`.
///
/// It rewrites the attention block to a bounded-window form (identical to
/// `block-left-1`/`block-left-L`):
///
/// ```text
/// q_c [C,P,D] = reshape(q)
/// k_ctx [C,(L+1)P,D] = concat([k_lag_L, …, k_lag_1, reshape(k)], axis=1)
///   where k_lag_d = slice(pad(reshape(k), before=d, axis=0), end=C, axis=0)
/// v_ctx [C,(L+1)P,D] = same for v
/// scores_c = einsum("cpd,cld->cpl", q_c, k_ctx)
/// attn_c   = softmax(scores_c, axes=[2])
/// out_c    = einsum("cpl,cld->cpd", attn_c, v_ctx)
/// output   = reshape(out_c)   # [T, D]
/// ```
///
/// This bounded form can be pulsified by the existing Delay-op machinery.
use crate::internal::*;
use crate::ops::array::{Pad, PadMode, Slice, TypedConcat};
use crate::ops::einsum::EinSum;
use crate::ops::logic::{Iff, classify_chunk_window};
use crate::ops::nn::Softmax;
use crate::optim::OptimizerSession;
use std::str::FromStr;

#[derive(Clone, Debug, Default)]
pub struct FoldWindowAttention(usize);

impl super::TypedPass for FoldWindowAttention {
    fn reset(&mut self) -> TractResult<()> {
        self.0 = 0;
        Ok(())
    }

    fn next(
        &mut self,
        _session: &mut OptimizerSession,
        model: &TypedModel,
    ) -> TractResult<Option<TypedModelPatch>> {
        for node in &model.nodes[self.0..] {
            self.0 = node.id + 1;
            if let Some(patch) = try_fold(model, node)? {
                return Ok(Some(patch));
            }
        }
        Ok(None)
    }
}

// ── Pattern detection ─────────────────────────────────────────────────────────

/// Given an `EinSum` op for `scores = q·kᵀ` and the mask axes (row, col) from the
/// chunk-window params, returns `(q_input_idx, q_token_axis, k_input_idx, k_token_axis)`.
fn find_qk_inputs(
    einsum: &EinSum,
    row_axis: usize, // which axis of the *output* is the query/row axis
    col_axis: usize, // which axis of the *output* is the key/col axis
) -> Option<(usize, usize, usize, usize)> {
    let axes = &einsum.axes;
    // Find the axis descriptor that appears at output position `row_axis`.
    let row_desc =
        axes.iter_all_axes().find(|a| !a.outputs.is_empty() && a.outputs[0].contains(&row_axis))?;
    // Find the axis descriptor that appears at output position `col_axis`.
    let col_desc =
        axes.iter_all_axes().find(|a| !a.outputs.is_empty() && a.outputs[0].contains(&col_axis))?;

    // Each free axis must come from exactly one input.
    let q_input_idx = row_desc.inputs.iter().position(|inp| !inp.is_empty())?;
    let q_token_axis = *row_desc.inputs[q_input_idx].first()?;

    let k_input_idx = col_desc.inputs.iter().position(|inp| !inp.is_empty())?;
    let k_token_axis = *col_desc.inputs[k_input_idx].first()?;

    if q_input_idx == k_input_idx {
        return None; // Q and K must be distinct inputs
    }
    Some((q_input_idx, q_token_axis, k_input_idx, k_token_axis))
}

/// In the output EinSum (`attn · V → output`), find which input is V.
/// `attn_outlet` is the `OutletId` of the softmax output.
/// Returns `(v_input_idx, v_token_axis)`.
fn find_v_input(out_einsum_node: &TypedNode, attn_outlet: OutletId) -> Option<(usize, usize)> {
    let out_einsum = out_einsum_node.op_as::<EinSum>()?;
    if out_einsum_node.inputs.len() != 2 {
        return None;
    }
    let attn_idx = out_einsum_node.inputs.iter().position(|&inp| inp == attn_outlet)?;
    let v_idx = 1 - attn_idx;

    // The contracted axis is the one that appears in both attn and V but not in the output.
    let contracted = out_einsum
        .axes
        .iter_all_axes()
        .filter(|a| a.outputs[0].is_empty())
        .filter(|a| !a.inputs[attn_idx].is_empty() && !a.inputs[v_idx].is_empty())
        .collect::<Vec<_>>();
    if contracted.len() != 1 {
        return None;
    }
    let v_token_axis = *contracted[0].inputs[v_idx].first()?;
    Some((v_idx, v_token_axis))
}

// ── Graph transformation ──────────────────────────────────────────────────────

/// Build `k_ctx [C, (L+1)*P, D]` from `k_c [C, P, D]` by prepending L lagged copies.
fn build_kv_context(
    patch: &mut TypedModelPatch,
    wire: OutletId, // k_c or v_c: [C, P, D]
    chunk_count: &TDim,
    rank: usize,      // rank after chunk reshape (= original_rank + 1)
    left_chunks: i64, // L
    dt: DatumType,
    name_prefix: &str,
) -> TractResult<OutletId> {
    let zero_val = Tensor::zero_dt(dt, &[])?;
    let pad_mode = PadMode::Constant(zero_val.into_arc_tensor());

    let mut parts: TVec<OutletId> = tvec![];
    for d in (1..=(left_chunks as usize)).rev() {
        // Pad `d` zero-rows before axis 0.
        let mut pads = vec![(0usize, 0usize); rank];
        pads[0] = (d, 0);
        let padded = patch.wire_node(
            format!("{name_prefix}.pad{d}"),
            Pad { pads, mode: pad_mode.clone() },
            &[wire],
        )?[0];
        // Slice back to C rows.
        let lagged = patch.wire_node(
            format!("{name_prefix}.lag{d}"),
            Slice::new(0, TDim::Val(0), chunk_count.clone()),
            &[padded],
        )?[0];
        parts.push(lagged);
    }
    parts.push(wire);

    if parts.len() == 1 {
        Ok(parts[0])
    } else {
        Ok(patch.wire_node(
            format!("{name_prefix}.ctx"),
            TypedConcat::new(1), // concat along the P axis (axis 1 in [C,P,D])
            &parts,
        )?[0])
    }
}

// ── Main entry point ──────────────────────────────────────────────────────────

fn try_fold(model: &TypedModel, node: &TypedNode) -> TractResult<Option<TypedModelPatch>> {
    // 1. Must be an Iff (select/mask) node.
    if !node.op_is::<Iff>() {
        return Ok(None);
    }

    // 2. The condition wire must carry a chunk-window uniform_tdim.
    let cond_fact = model.outlet_fact(node.inputs[0])?;
    if cond_fact.konst.is_some() {
        return Ok(None);
    }
    let expr = match &cond_fact.uniform_tdim {
        Some(e) => e,
        None => return Ok(None),
    };
    let cw = match classify_chunk_window(expr) {
        Some(cw) => cw,
        None => return Ok(None),
    };

    // 3. The true branch (scores) must come from a 2-input EinSum.
    let scores_outlet = node.inputs[1];
    let scores_node = model.node(scores_outlet.node);
    let scores_einsum = match scores_node.op_as::<EinSum>() {
        Some(e) if scores_node.inputs.len() == 2 => e,
        _ => return Ok(None),
    };

    // 4. Both inputs to the score EinSum must be rank-2 (token × head).
    let (q_input_idx, q_token_ax, k_input_idx, k_token_ax) =
        match find_qk_inputs(scores_einsum, cw.row_axis, cw.col_axis) {
            Some(v) => v,
            None => return Ok(None),
        };
    let q_outlet = scores_node.inputs[q_input_idx];
    let k_outlet = scores_node.inputs[k_input_idx];
    let q_fact = model.outlet_fact(q_outlet)?;
    let k_fact = model.outlet_fact(k_outlet)?;

    // Only handle rank-2 tensors for now (token × head_dim).
    if q_fact.rank() != 2 || k_fact.rank() != 2 {
        return Ok(None);
    }
    if q_token_ax != 0 || k_token_ax != 0 {
        return Ok(None);
    }

    let token_dim = q_fact.shape[0].clone(); // T
    if token_dim != k_fact.shape[0] {
        return Ok(None);
    }

    let k_dt = k_fact.datum_type;

    // 5. Single successor of Iff must be Softmax over col_axis.
    let softmax_node = match model.single_succ(node.id)? {
        Some(n) => n,
        None => return Ok(None),
    };
    let softmax = match softmax_node.op_as::<Softmax>() {
        Some(s) => s,
        None => return Ok(None),
    };
    if softmax.axes != tvec![cw.col_axis] {
        return Ok(None);
    }

    // 6. Single successor of Softmax must be EinSum (output weighted-sum).
    let out_node = match model.single_succ(softmax_node.id)? {
        Some(n) => n,
        None => return Ok(None),
    };
    if out_node.op_as::<EinSum>().is_none() {
        return Ok(None);
    }
    let out_einsum = out_node.op_as::<EinSum>().unwrap();

    // 7. Find V in the output EinSum.
    let softmax_out = OutletId::new(softmax_node.id, 0);
    let (v_idx, v_token_ax) = match find_v_input(out_node, softmax_out) {
        Some(v) => v,
        None => return Ok(None),
    };
    let v_outlet = out_node.inputs[v_idx];
    let v_fact = model.outlet_fact(v_outlet)?;
    if v_fact.rank() != 2 || v_token_ax != 0 {
        return Ok(None);
    }
    if v_fact.shape[0] != token_dim {
        return Ok(None);
    }
    let v_dt = v_fact.datum_type;

    // ── All preconditions met – build the transformation ──────────────────────

    let chunk_count = TDim::Div(Box::new(token_dim.clone()), cw.chunk_size); // C = T/P
    let p_tdim = TDim::Val(cw.chunk_size as i64);
    // Use chunk_count * p_tdim instead of token_dim as the reshape `from` spec so that
    // both sides of the volume check reduce to the same symbolic form (MulInt(P, Div(T, P))).
    let token_dim_as_product = chunk_count.clone() * p_tdim.clone();

    let prefix = &node.name;

    let mut patch = TypedModelPatch::default();

    // Tap Q, K, V.
    let q_tap = patch.tap_model(model, q_outlet)?;
    let k_tap = patch.tap_model(model, k_outlet)?;
    let v_tap = patch.tap_model(model, v_outlet)?;

    // Reshape Q, K, V: [T, D] → [C, P, D].
    // Use token_dim_as_product (= P*(T/P)) as the from-spec so both sides simplify identically.
    let reshape_token = AxisOp::Reshape(
        0,
        tvec![token_dim_as_product.clone()],
        tvec![chunk_count.clone(), p_tdim.clone()],
    );

    let q_c = patch.wire_node(format!("{prefix}.q_reshape"), reshape_token.clone(), &[q_tap])?[0];
    let k_c = patch.wire_node(format!("{prefix}.k_reshape"), reshape_token.clone(), &[k_tap])?[0];
    let v_c = patch.wire_node(format!("{prefix}.v_reshape"), reshape_token.clone(), &[v_tap])?[0];

    // Build K and V context windows: [C, (L+1)*P, D].
    let k_ctx = build_kv_context(
        &mut patch,
        k_c,
        &chunk_count,
        3,
        cw.left_chunks,
        k_dt,
        &format!("{prefix}.k"),
    )?;
    let v_ctx = build_kv_context(
        &mut patch,
        v_c,
        &chunk_count,
        3,
        cw.left_chunks,
        v_dt,
        &format!("{prefix}.v"),
    )?;

    // New score EinSum: "cpd,cld->cpl"  ([C,P,D] × [C,(L+1)P,D] → [C,P,(L+1)P]).
    let score_axes = AxesMapping::from_str("cpd,cld->cpl")?;
    let scores_c = patch.wire_node(
        format!("{prefix}.scores_c"),
        EinSum::new(score_axes, scores_einsum.operating_dt),
        &[q_c, k_ctx],
    )?[0];

    // New Softmax over axis 2 (the `l` context axis).
    let new_softmax = Softmax { axes: tvec![2], ..softmax.clone() };
    let attn_c = patch.wire_node(format!("{prefix}.attn_c"), new_softmax, &[scores_c])?[0];

    // New output EinSum: "cpl,cld->cpd"  ([C,P,(L+1)P] × [C,(L+1)P,D] → [C,P,D]).
    let out_axes = AxesMapping::from_str("cpl,cld->cpd")?;
    let out_c = patch.wire_node(
        format!("{prefix}.out_c"),
        EinSum::new(out_axes, out_einsum.operating_dt),
        &[attn_c, v_ctx],
    )?[0];

    // Reshape output back: [C, P, D] → [T, D].
    let reshape_back = AxisOp::Reshape(
        0,
        tvec![chunk_count.clone(), p_tdim.clone()],
        tvec![token_dim_as_product.clone()],
    );
    let output = patch.wire_node(format!("{prefix}.out_reshape"), reshape_back, &[out_c])?[0];

    // Shunt the output of the original output EinSum node.
    patch.shunt_outside(model, out_node.id.into(), output)?;
    Ok(Some(patch))
}
