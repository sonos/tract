//! tract `EinSum` (general matmul-shaped) → MIL `matmul` translator.
//!
//! Handles **runtime × runtime** (and runtime × const) matmul-shaped
//! einsums that arise from transformer attention + Linear layers,
//! including the many decorated forms tract emits after declutter.
//!
//! ## Evolution (Phase 4)
//!
//! - **v1** (transformer-prep): single M / single N / single K, runtime-runtime,
//!   batch dims at same position in both inputs + output. Covered ~20% of
//!   SAM 2 Hiera-Tiny einsums.
//! - **v2** (multi-axis): allow multiple M / N / K axes, flatten via reshape
//!   pre/post matmul. Plus strip-A / strip-B / expand-out unit-dim axes.
//!   Absorbed all Linear-layer einsums (50/78 stranded).
//! - **v3** (batch-reorder): allow batch axes at differing positions in
//!   inputs / output via pre/post-Transpose canonicalisation. Absorbed
//!   Q @ K and attn @ V attention patterns (28 → 9 stranded).
//! - **v4** (input external boundary): MLPackage external input shape
//!   strips strip-A/B unit dims so rank > 5 tract tensors can flow in
//!   under MIL's rank-5 cap. CoremlOp does the squeeze at the
//!   tract↔MLPackage boundary. Absorbed 8 of 9 stranded `dcbamk,kn->cabmn`
//!   QKV-projection einsums (9 → 1 stranded).
//! - **v5** (output external boundary): symmetric — MLPackage external
//!   output shape strips expand-out unit dims so rank > 5 outputs work.
//!   CoremlOp re-inserts the unit dims at the consumer side. Absorbed
//!   the last `NHWI,OI->NOHbWa` neck einsum (0 stranded EinSums on SAM 2).
//! - Rank-padding bridge: explicit reshape from upstream-declared rank
//!   to expected `a_external_shape` rank, so downstream MIL ops see
//!   consistent ranks even when upstream pads to rank 4 and we expect 5.
//!
//! ## Axis classification
//!
//! Walk the `AxesMapping` and bucket each axis by where it appears:
//!
//!   * **K (reduced)**: in input 0 AND input 1, NOT in output. May be
//!     multiple — flattened into a single K' for the matmul.
//!   * **M-axes**: in input 0 AND output, NOT in input 1. May be multiple.
//!   * **N-axes**: in input 1 AND output, NOT in input 0. May be multiple.
//!   * **Batch**: in input 0 AND input 1 AND output. May be at differing
//!     positions in any of the three slots — we pre/post-transpose to
//!     canonicalise.
//!   * **Strip-A**: in input 0 only — must be size 1, reshape removes it.
//!   * **Strip-B**: in input 1 only — must be size 1.
//!   * **Expand-Out**: in output only — must be size 1, post-reshape adds.
//!
//! ## Pipeline
//!
//! ```text
//!   input A → pre-transpose (canonical = [batch_in_out_order..., M_axes..., K_axes...])
//!           → reshape to [batch_flat, M_flat, K_flat]
//!   input B → pre-transpose (canonical = [batch..., K_axes..., N_axes_in_out_order...])
//!           → reshape to [batch_flat, K_flat, N_flat]
//!   matmul → [batch_flat, M_flat, N_flat]
//!   → reshape to canonical layout [batch_in_out_order..., m_axes..., n_axes...]
//!   → post-transpose to actual output axis positions (mapping canonical → output)
//!   → final reshape inserts expand-out unit dims
//! ```
//!
//! For most patterns the pre/post-transpose is the identity (when input
//! axes are already in canonical order); we elide the no-op transposes in
//! the emit phase.
//!
//! ## Scope
//!
//!   * Multi-M, multi-N, multi-K axes (each side's axes must be a
//!     consistent set across slots).
//!   * Strip and expand unit-dim axes.
//!   * Arbitrary batch-axis positions in input 0, input 1, and output.
//!   * Both runtime and Const operands. Const = absorbed into MILBlob v2.
//!
//! ## Deferred
//!
//!   * True non-unit strip axes (would need a Reduce<Sum> before matmul).
//!   * MIL `matmul` only supports rank ≥ 2 — we always reshape to rank 3.

use std::collections::HashMap;

use anyhow::Result;

use tract_core::internal::*;
use tract_core::ops::einsum::EinSum;

use crate::mil::blob::{BlobBuilder, BlobDataType};
use crate::mil::op::{arg_name, op_const_blob, op_const_immediate};
use crate::mil::value::{DataType, tensor_type, tensor_type_scalar, tv_ints};
use crate::mlpackage;
use crate::proto::core_ml::specification::mil_spec as mil;

use super::{const_tensor, shape_to_concrete_i64};

#[allow(clippy::large_enum_variant)]
pub enum GeneralMatMulAnalysis {
    Translatable(GeneralMatMulPlan),
    Skip(String),
}

pub enum MatMulOperand {
    Runtime { shape: Vec<i64> },
    Const { tensor: Tensor },
}

pub struct GeneralMatMulPlan {
    pub a: MatMulOperand,
    pub b: MatMulOperand,
    /// Original A shape — tract's view (may include strip-A unit dims).
    pub a_shape: Vec<i64>,
    /// Original B shape.
    pub b_shape: Vec<i64>,
    /// **MLPackage-boundary A shape**: `a_shape` with strip-A positions
    /// removed (rank can drop from 6 → 5 etc.). This is what the MLPackage
    /// declares as its external input — CoremlOp does the squeeze at the
    /// boundary between tract-side and MLPackage-side. Equal to `a_shape`
    /// when there are no strip-A axes.
    pub a_external_shape: Vec<i64>,
    /// Same for B.
    pub b_external_shape: Vec<i64>,
    /// Permutation to apply to A (as it flows in at `a_external_shape`)
    /// bringing it into canonical `[batch_in_out_order..., m_axes..., k_axes...]`
    /// layout. `perm_a[i] = a_external_shape position at canonical position i`.
    pub perm_a: Vec<i32>,
    /// A's shape after pre-transpose (= a_external_shape permuted by perm_a).
    pub a_canon_shape: Vec<i64>,
    /// Same for B: canonical layout `[batch_in_out_order..., k_axes..., n_axes...]`.
    pub perm_b: Vec<i32>,
    pub b_canon_shape: Vec<i64>,
    /// A reshaped to rank 3 `[batch_flat, M_flat, K_flat]`.
    pub a_matmul_shape: Vec<i64>,
    /// B reshaped to rank 3 `[batch_flat, K_flat, N_flat]`.
    pub b_matmul_shape: Vec<i64>,
    /// matmul output `[batch_flat, M_flat, N_flat]`.
    pub matmul_out_shape: Vec<i64>,
    /// matmul output un-flattened to canonical layout
    /// `[batch_in_out_order..., m_axes..., n_axes...]`.
    pub canon_out_shape: Vec<i64>,
    /// Permutation from canonical-out layout → actual output positions of
    /// the (non-expand) output axes. Identity → no transpose needed.
    pub perm_out: Vec<i32>,
    /// Shape after post-transpose (= canon_out_shape permuted by perm_out).
    pub post_transpose_shape: Vec<i64>,
    /// Tract-view output shape (with expand-out unit dims inserted at
    /// their tract positions). Used downstream to know what tract expects.
    pub output_shape: Vec<i64>,
    /// **MLPackage-boundary output shape**: `output_shape` with expand-out
    /// unit-dim positions removed. CoremlOp re-inserts those dims at the
    /// consumer-side boundary. Equal to `output_shape` when there are no
    /// expand-out axes. Used to avoid declaring rank > 5 outputs.
    pub output_external_shape: Vec<i64>,
    pub output_fact: TypedFact,
}

pub fn analyse_general_matmul(
    model: &TypedModel,
    node: &TypedNode,
) -> Result<GeneralMatMulAnalysis> {
    let Some(es) = node.op_as::<EinSum>() else {
        return Ok(GeneralMatMulAnalysis::Skip("not an EinSum".into()));
    };
    if es.q_params.is_some() {
        return Ok(GeneralMatMulAnalysis::Skip("quantized EinSum".into()));
    }
    if es.operating_dt != DatumType::F16 {
        return Ok(GeneralMatMulAnalysis::Skip(format!(
            "EinSum operating_dt {:?} (need F16)",
            es.operating_dt
        )));
    }
    if node.inputs.len() != 2 {
        return Ok(GeneralMatMulAnalysis::Skip(format!(
            "EinSum has {} inputs (need 2)",
            node.inputs.len()
        )));
    }

    // ---- Classify each axis.
    let mut m_axes: Vec<&tract_core::axes::Axis> = Vec::new();
    let mut n_axes: Vec<&tract_core::axes::Axis> = Vec::new();
    let mut k_axes: Vec<&tract_core::axes::Axis> = Vec::new();
    let mut batch_axes: Vec<&tract_core::axes::Axis> = Vec::new();
    let mut strip_a: Vec<&tract_core::axes::Axis> = Vec::new();
    let mut strip_b: Vec<&tract_core::axes::Axis> = Vec::new();
    let mut expand_out: Vec<&tract_core::axes::Axis> = Vec::new();
    for ax in es.axes.iter_all_axes() {
        let in_a = !ax.inputs[0].is_empty();
        let in_b = !ax.inputs[1].is_empty();
        let in_out = !ax.outputs[0].is_empty();
        match (in_a, in_b, in_out) {
            (true, false, true) => m_axes.push(ax),
            (false, true, true) => n_axes.push(ax),
            (true, true, false) => k_axes.push(ax),
            (true, true, true) => batch_axes.push(ax),
            (true, false, false) => strip_a.push(ax),
            (false, true, false) => strip_b.push(ax),
            (false, false, true) => expand_out.push(ax),
            (false, false, false) => {
                return Ok(GeneralMatMulAnalysis::Skip(format!(
                    "EinSum axis {:?} has no presence anywhere — malformed",
                    ax.repr
                )));
            }
        }
    }

    if m_axes.is_empty() {
        return Ok(GeneralMatMulAnalysis::Skip(format!("EinSum {} missing M axes", es.axes)));
    }
    if n_axes.is_empty() {
        return Ok(GeneralMatMulAnalysis::Skip(format!("EinSum {} missing N axes", es.axes)));
    }
    if k_axes.is_empty() {
        // Pure outer-product / broadcast-multiply einsum (no K contraction).
        // E.g. RoPE's `m,an->abnm` = `output[a,b,n,m] = A[m] * B[a,n]`. A
        // separate translator (`einsum_outer.rs`) handles these via reshape-
        // to-output-rank + broadcast `mb.mul`. Defer here.
        return Ok(GeneralMatMulAnalysis::Skip(format!(
            "EinSum {} missing K axes (defer to einsum_outer.rs)",
            es.axes
        )));
    }

    let pos = |ax: &tract_core::axes::Axis, slot: AxisSlot| -> Result<usize, String> {
        let positions = match slot {
            AxisSlot::InputA => &ax.inputs[0],
            AxisSlot::InputB => &ax.inputs[1],
            AxisSlot::Output => &ax.outputs[0],
        };
        if positions.len() != 1 {
            return Err(format!("axis {} has {} positions in {slot:?}", ax.repr, positions.len()));
        }
        Ok(positions[0])
    };

    let rank_a = es.axes.rank(InOut::In(0));
    let rank_b = es.axes.rank(InOut::In(1));
    let rank_out = es.axes.rank(InOut::Out(0));

    // ---- Resolve operands + concrete shapes.
    let (a, a_shape) = resolve_operand(model, node.inputs[0])?;
    let (b, b_shape) = resolve_operand(model, node.inputs[1])?;
    if matches!((&a, &b), (MatMulOperand::Const { .. }, MatMulOperand::Const { .. })) {
        return Ok(GeneralMatMulAnalysis::Skip(
            "EinSum has both inputs Const (should have been const-folded)".into(),
        ));
    }
    // Note: there used to be a "rank-≤2 × rank-2 const weight: defer to
    // specific matmul.rs" guard here. Removed because matmul.rs only
    // handles a fixed table of einsum signatures (`mkba,kn->n`,
    // `k,kn->mnab`, `IHW,OI->O`, `I,OI->OHW`) and rejects everything else
    // — including ViT's FFN `mk,kn->amn`. The fusion driver already tries
    // matmul.rs FIRST (line ~1115 in fusion.rs); we only get here if
    // matmul.rs returned Skip. Deferring back was creating a hole that
    // stranded 13 EinSums per ViT.

    // ---- Verify strip axes are size 1.
    for ax in &strip_a {
        let p = pos(ax, AxisSlot::InputA).map_err(|e| anyhow::anyhow!(e))?;
        if a_shape[p] != 1 {
            return Ok(GeneralMatMulAnalysis::Skip(format!(
                "{} strip-A axis '{}' at position {p} has size {} (need 1)",
                es.axes, ax.repr, a_shape[p]
            )));
        }
    }
    for ax in &strip_b {
        let p = pos(ax, AxisSlot::InputB).map_err(|e| anyhow::anyhow!(e))?;
        if b_shape[p] != 1 {
            return Ok(GeneralMatMulAnalysis::Skip(format!(
                "{} strip-B axis '{}' at position {p} has size {} (need 1)",
                es.axes, ax.repr, b_shape[p]
            )));
        }
    }

    // ---- Order batch axes by their OUTPUT position (canonical = output order).
    let mut batch_sorted: Vec<&tract_core::axes::Axis> = batch_axes.clone();
    batch_sorted.sort_by_key(|ax| pos(ax, AxisSlot::Output).unwrap_or(usize::MAX));

    // ---- Order M and N axes by their OUTPUT position (canonical layout has
    // these in output order).
    let mut m_sorted: Vec<&tract_core::axes::Axis> = m_axes.clone();
    m_sorted.sort_by_key(|ax| pos(ax, AxisSlot::Output).unwrap_or(usize::MAX));
    let mut n_sorted: Vec<&tract_core::axes::Axis> = n_axes.clone();
    n_sorted.sort_by_key(|ax| pos(ax, AxisSlot::Output).unwrap_or(usize::MAX));

    // ---- Canonical layout for input A: [batch_sorted..., m_sorted..., k_axes...]
    // The k_axes order doesn't matter as long as input A and input B agree —
    // pick A's input-position order.
    let mut k_sorted_by_a: Vec<&tract_core::axes::Axis> = k_axes.clone();
    k_sorted_by_a.sort_by_key(|ax| pos(ax, AxisSlot::InputA).unwrap_or(usize::MAX));

    let canonical_a_axes: Vec<&tract_core::axes::Axis> = batch_sorted
        .iter()
        .copied()
        .chain(m_sorted.iter().copied())
        .chain(k_sorted_by_a.iter().copied())
        .collect();
    let canonical_b_axes: Vec<&tract_core::axes::Axis> = batch_sorted
        .iter()
        .copied()
        .chain(k_sorted_by_a.iter().copied())
        .chain(n_sorted.iter().copied())
        .collect();

    // ---- Build "external" shapes: the rank-reduced shapes the MLPackage
    // declares as its inputs (= a_shape / b_shape with strip-A / strip-B
    // unit-dim positions removed). CoremlOp does the squeeze at the
    // tract-side ↔ MLPackage-side boundary.
    let strip_a_pos: std::collections::HashSet<usize> = strip_a
        .iter()
        .map(|ax| pos(ax, AxisSlot::InputA))
        .collect::<Result<_, _>>()
        .map_err(|e| anyhow::anyhow!(e))?;
    let strip_b_pos: std::collections::HashSet<usize> = strip_b
        .iter()
        .map(|ax| pos(ax, AxisSlot::InputB))
        .collect::<Result<_, _>>()
        .map_err(|e| anyhow::anyhow!(e))?;
    let a_external_shape: Vec<i64> = a_shape
        .iter()
        .enumerate()
        .filter(|(i, _)| !strip_a_pos.contains(i))
        .map(|(_, &v)| v)
        .collect();
    let b_external_shape: Vec<i64> = b_shape
        .iter()
        .enumerate()
        .filter(|(i, _)| !strip_b_pos.contains(i))
        .map(|(_, &v)| v)
        .collect();
    // Build maps: original input position → external position.
    // Iterates non-strip positions in order; their index in the output is
    // their external position.
    let map_pos =
        |strip_pos: &std::collections::HashSet<usize>, rank: usize| -> Vec<Option<usize>> {
            let mut next_external = 0;
            (0..rank)
                .map(|p| {
                    if strip_pos.contains(&p) {
                        None
                    } else {
                        let ext = next_external;
                        next_external += 1;
                        Some(ext)
                    }
                })
                .collect()
        };
    let a_orig_to_ext = map_pos(&strip_a_pos, rank_a);
    let b_orig_to_ext = map_pos(&strip_b_pos, rank_b);

    // perm_a: for each canonical position, which EXTERNAL A position holds it?
    // (NOT original — since strip axes are gone at the MLPackage boundary.)
    let perm_a: Vec<usize> = canonical_a_axes
        .iter()
        .map(|ax| -> Result<usize, String> {
            let orig = pos(ax, AxisSlot::InputA)?;
            a_orig_to_ext[orig].ok_or_else(|| {
                format!("axis '{}' at original A position {orig} unexpectedly stripped", ax.repr)
            })
        })
        .collect::<Result<_, _>>()
        .map_err(|e| anyhow::anyhow!(e))?;
    let perm_b: Vec<usize> = canonical_b_axes
        .iter()
        .map(|ax| -> Result<usize, String> {
            let orig = pos(ax, AxisSlot::InputB)?;
            b_orig_to_ext[orig].ok_or_else(|| {
                format!("axis '{}' at original B position {orig} unexpectedly stripped", ax.repr)
            })
        })
        .collect::<Result<_, _>>()
        .map_err(|e| anyhow::anyhow!(e))?;

    // a_canon_shape = a_external_shape permuted by perm_a.
    let a_canon_shape: Vec<i64> = perm_a.iter().map(|&p| a_external_shape[p]).collect();
    let b_canon_shape: Vec<i64> = perm_b.iter().map(|&p| b_external_shape[p]).collect();

    // Sanity: a_canon_shape + strip_a positions must account for all of a_shape.
    if a_canon_shape.len() + strip_a.len() != rank_a {
        return Ok(GeneralMatMulAnalysis::Skip(format!(
            "{} canonical-A rank ({}) + strip-A ({}) != input A rank ({rank_a})",
            es.axes,
            a_canon_shape.len(),
            strip_a.len()
        )));
    }
    if b_canon_shape.len() + strip_b.len() != rank_b {
        return Ok(GeneralMatMulAnalysis::Skip(format!(
            "{} canonical-B rank ({}) + strip-B ({}) != input B rank ({rank_b})",
            es.axes,
            b_canon_shape.len(),
            strip_b.len()
        )));
    }

    // ---- Compute flattened M, N, K dimensions.
    let m_flat: i64 = m_sorted.iter().try_fold(1i64, |acc, ax| {
        Ok::<_, anyhow::Error>(
            acc * a_shape[pos(ax, AxisSlot::InputA).map_err(|e| anyhow::anyhow!(e))?],
        )
    })?;
    let n_flat: i64 = n_sorted.iter().try_fold(1i64, |acc, ax| {
        Ok::<_, anyhow::Error>(
            acc * b_shape[pos(ax, AxisSlot::InputB).map_err(|e| anyhow::anyhow!(e))?],
        )
    })?;
    let k_flat_a: i64 = k_sorted_by_a.iter().try_fold(1i64, |acc, ax| {
        Ok::<_, anyhow::Error>(
            acc * a_shape[pos(ax, AxisSlot::InputA).map_err(|e| anyhow::anyhow!(e))?],
        )
    })?;
    let k_flat_b: i64 = k_sorted_by_a.iter().try_fold(1i64, |acc, ax| {
        Ok::<_, anyhow::Error>(
            acc * b_shape[pos(ax, AxisSlot::InputB).map_err(|e| anyhow::anyhow!(e))?],
        )
    })?;
    if k_flat_a != k_flat_b {
        return Ok(GeneralMatMulAnalysis::Skip(format!(
            "K-dim mismatch: input 0 = {k_flat_a}, input 1 = {k_flat_b}"
        )));
    }
    let batch_flat: i64 = batch_sorted.iter().try_fold(1i64, |acc, ax| {
        Ok::<_, anyhow::Error>(
            acc * a_shape[pos(ax, AxisSlot::InputA).map_err(|e| anyhow::anyhow!(e))?],
        )
    })?;

    // matmul shapes: rank 3, [batch_flat, M_flat, K_flat] for A; [batch_flat, K_flat, N_flat] for B.
    let a_matmul_shape: Vec<i64> = vec![batch_flat, m_flat, k_flat_a];
    let b_matmul_shape: Vec<i64> = vec![batch_flat, k_flat_b, n_flat];
    let matmul_out_shape: Vec<i64> = vec![batch_flat, m_flat, n_flat];

    // ---- Canonical output layout (un-flattening matmul output):
    // [batch_axes_sorted..., m_axes_sorted..., n_axes_sorted...]
    let mut canon_out_shape: Vec<i64> = Vec::with_capacity(rank_out);
    for ax in &batch_sorted {
        canon_out_shape.push(a_shape[pos(ax, AxisSlot::InputA).map_err(|e| anyhow::anyhow!(e))?]);
    }
    for ax in &m_sorted {
        canon_out_shape.push(a_shape[pos(ax, AxisSlot::InputA).map_err(|e| anyhow::anyhow!(e))?]);
    }
    for ax in &n_sorted {
        canon_out_shape.push(b_shape[pos(ax, AxisSlot::InputB).map_err(|e| anyhow::anyhow!(e))?]);
    }
    // canon_out_shape rank = batch + m + n = rank_out - expand_out.len()
    let expected_canon_rank = rank_out - expand_out.len();
    if canon_out_shape.len() != expected_canon_rank {
        return Ok(GeneralMatMulAnalysis::Skip(format!(
            "{} canonical-out rank {} != expected {expected_canon_rank}",
            es.axes,
            canon_out_shape.len()
        )));
    }

    // ---- post_transpose: from canonical positions [0..batch_n, batch_n..batch_n+m_n, ...]
    // to actual output positions of these axes (excluding expand_out).
    // canonical position i has the axis from canonical_out_axes[i].
    let canonical_out_axes: Vec<&tract_core::axes::Axis> = batch_sorted
        .iter()
        .copied()
        .chain(m_sorted.iter().copied())
        .chain(n_sorted.iter().copied())
        .collect();
    // For each canonical position, what's its actual output position?
    let target_pos: Vec<usize> = canonical_out_axes
        .iter()
        .map(|ax| pos(ax, AxisSlot::Output))
        .collect::<Result<_, _>>()
        .map_err(|e| anyhow::anyhow!(e))?;
    // Build the permutation perm_out such that:
    //   post_transpose_output[target_pos[i]_filtered_for_non_expand] = canon_input[i]
    // Concretely: among the non-expand output positions, sort canonical
    // positions by their target_pos to get perm_out.
    //
    // perm_out[output_position_among_non_expand] = canonical_position_to_pull_from
    let n_non_expand = canonical_out_axes.len();
    let mut indexed_targets: Vec<(usize, usize)> =
        target_pos.iter().enumerate().map(|(i, &t)| (t, i)).collect();
    indexed_targets.sort_by_key(|&(t, _)| t);
    let perm_out: Vec<usize> = indexed_targets.iter().map(|&(_, i)| i).collect();
    // post_transpose_shape: canon_out_shape permuted by perm_out.
    let post_transpose_shape: Vec<i64> = perm_out.iter().map(|&i| canon_out_shape[i]).collect();
    debug_assert_eq!(post_transpose_shape.len(), n_non_expand);

    // ---- Output shape at tract rank: insert size-1 dims at expand_out positions.
    let mut tract_output_shape: Vec<i64> = vec![1; rank_out];
    let expand_positions: std::collections::HashSet<usize> = expand_out
        .iter()
        .map(|ax| pos(ax, AxisSlot::Output))
        .collect::<Result<_, _>>()
        .map_err(|e| anyhow::anyhow!(e))?;
    let mut post_idx = 0;
    for (i, slot) in tract_output_shape.iter_mut().enumerate() {
        if !expand_positions.contains(&i) {
            *slot = post_transpose_shape[post_idx];
            post_idx += 1;
        }
    }

    // ---- Final MLPackage output shape: rank-pad to 4 (or higher if tract rank > 4)
    // to match the in-MLPackage rank-4-padding convention used by binop / reduce /
    // activation / etc. Within an MLPackage we always emit rank-4 tensors when
    // tract's view is rank ≤ 4. The CoremlOp boundary handles the reshape back to
    // tract's rank when feeding consumers.
    let target_rank = tract_output_shape.len().max(4);
    let pad_extra = target_rank.saturating_sub(tract_output_shape.len());
    let output_shape: Vec<i64> = if pad_extra > 0 {
        std::iter::repeat_n(1i64, pad_extra).chain(tract_output_shape.iter().copied()).collect()
    } else {
        tract_output_shape
    };

    // ---- MLPackage-boundary output shape: strip the expand-out unit dims.
    // For tract output rank > 5 (e.g. NHWI,OI->NOHbWa where output is rank
    // 6), we'd otherwise overflow MIL's rank-5 cap on outputs. CoremlOp at
    // the consumer side re-inserts the unit dims at their tract positions.
    let output_external_shape: Vec<i64> = if expand_positions.is_empty() {
        output_shape.clone()
    } else {
        // Strip expand-out positions from the FULL output_shape (not the
        // rank-pad-extended portion — pad-extra is leading-1s that aren't
        // expand axes). Compute expand positions in the rank-padded shape:
        // they shift right by `pad_extra`.
        let shifted: std::collections::HashSet<usize> =
            expand_positions.iter().map(|&p| p + pad_extra).collect();
        output_shape
            .iter()
            .enumerate()
            .filter(|(i, _)| !shifted.contains(i))
            .map(|(_, &v)| v)
            .collect()
    };

    // ---- Dtype + rank checks.
    let a_dt = match &a {
        MatMulOperand::Runtime { .. } => model.outlet_fact(node.inputs[0])?.datum_type,
        MatMulOperand::Const { tensor } => tensor.datum_type(),
    };
    let b_dt = match &b {
        MatMulOperand::Runtime { .. } => model.outlet_fact(node.inputs[1])?.datum_type,
        MatMulOperand::Const { tensor } => tensor.datum_type(),
    };
    if a_dt != DatumType::F16 || b_dt != DatumType::F16 {
        return Ok(GeneralMatMulAnalysis::Skip(format!(
            "matmul operand dtypes ({a_dt:?}, {b_dt:?}) — need F16/F16"
        )));
    }
    // Rank check — measure the EFFECTIVE ranks at the MLPackage boundary
    // (= post-strip / post-strip-expand ranks). The MLPackage boundary is
    // rank 5; what tract surfaces in the original a_shape/b_shape/output
    // may be rank > 5 due to strip-A/B + expand-out unit dims, but those
    // don't count toward the boundary rank — CoremlOp does the squeeze
    // (input) and unsqueeze (output) at the boundary.
    let ext_rank_a = a_external_shape.len();
    let ext_rank_b = b_external_shape.len();
    let ext_rank_out = output_external_shape.len();
    if ext_rank_a > 5 || ext_rank_b > 5 || ext_rank_out > 5 {
        return Ok(GeneralMatMulAnalysis::Skip(format!(
            "matmul effective rank exceeds MIL limit (ext_a={ext_rank_a}, ext_b={ext_rank_b}, \
             ext_out={ext_rank_out}, max 5)"
        )));
    }

    Ok(GeneralMatMulAnalysis::Translatable(GeneralMatMulPlan {
        a,
        b,
        a_shape,
        b_shape,
        a_external_shape,
        b_external_shape,
        output_external_shape,
        perm_a: perm_a.iter().map(|&v| v as i32).collect(),
        a_canon_shape,
        perm_b: perm_b.iter().map(|&v| v as i32).collect(),
        b_canon_shape,
        a_matmul_shape,
        b_matmul_shape,
        matmul_out_shape,
        canon_out_shape,
        perm_out: perm_out.iter().map(|&v| v as i32).collect(),
        post_transpose_shape,
        output_shape,
        output_fact: node.outputs[0].fact.clone(),
    }))
}

#[derive(Clone, Copy, Debug)]
enum AxisSlot {
    InputA,
    InputB,
    Output,
}

fn resolve_operand(model: &TypedModel, outlet: OutletId) -> Result<(MatMulOperand, Vec<i64>)> {
    if let Some(t) = const_tensor(model, outlet)? {
        let shape: Vec<i64> = t.shape().iter().map(|&s| s as i64).collect();
        return Ok((MatMulOperand::Const { tensor: t.into_owned() }, shape));
    }
    let fact = model.outlet_fact(outlet)?;
    let shape = match shape_to_concrete_i64(&fact.shape) {
        Some(s) => s,
        None => anyhow::bail!("matmul operand has symbolic shape: {:?}", fact.shape),
    };
    Ok((MatMulOperand::Runtime { shape: shape.clone() }, shape))
}

fn is_identity_perm(perm: &[i32]) -> bool {
    perm.iter().enumerate().all(|(i, &p)| (p as usize) == i)
}

pub fn emit_general_matmul_mil(
    plan: &GeneralMatMulPlan,
    blob: &mut BlobBuilder,
    input_a_name: Option<&str>,
    input_b_name: Option<&str>,
    output_name: &str,
) -> Result<Vec<mil::Operation>> {
    let p = output_name;
    let mut ops: Vec<mil::Operation> = Vec::new();

    // Resolve A/B's MIL names.
    let a_raw = match (&plan.a, input_a_name) {
        (MatMulOperand::Runtime { .. }, Some(n)) => n.to_string(),
        (MatMulOperand::Const { tensor }, _) => {
            let off = blob.add(BlobDataType::Float16, tensor.as_bytes());
            let name = format!("{p}_a_const");
            let ty = tensor_type(DataType::Float16, &plan.a_shape);
            ops.push(op_const_blob(&name, ty, mlpackage::WEIGHT_BLOB_PATH, off));
            name
        }
        (MatMulOperand::Runtime { .. }, None) => {
            anyhow::bail!("A is Runtime but emit got input_a_name=None")
        }
    };
    let b_raw = match (&plan.b, input_b_name) {
        (MatMulOperand::Runtime { .. }, Some(n)) => n.to_string(),
        (MatMulOperand::Const { tensor }, _) => {
            let off = blob.add(BlobDataType::Float16, tensor.as_bytes());
            let name = format!("{p}_b_const");
            let ty = tensor_type(DataType::Float16, &plan.b_shape);
            ops.push(op_const_blob(&name, ty, mlpackage::WEIGHT_BLOB_PATH, off));
            name
        }
        (MatMulOperand::Runtime { .. }, None) => {
            anyhow::bail!("B is Runtime but emit got input_b_name=None")
        }
    };

    // Helper: emit a transpose if the perm isn't identity OR if the rank
    // changes (when stripping axes via reshape — perm.len() < input rank).
    // For strip-via-perm, we use reshape directly since transpose preserves rank.
    fn emit_transpose_or_reshape(
        ops: &mut Vec<mil::Operation>,
        input_name: &str,
        input_shape: &[i64],
        perm: &[i32],
        out_name: &str,
        out_shape: &[i64],
    ) {
        // If perm is identity AND the rank matches, no-op (alias the name).
        if perm.len() == input_shape.len() && is_identity_perm(perm) {
            // Caller should have handled aliasing; if we got here, emit an
            // identity transpose anyway. (Cheaper alternative: skip emit and
            // have caller use input_name directly — done at callsites.)
            return;
        }
        // If perm has same rank as input, emit a transpose.
        if perm.len() == input_shape.len() {
            let perm_n = format!("{out_name}_perm");
            let i32t = tensor_type(DataType::Int32, &[perm.len() as i64]);
            ops.push(op_const_immediate(&perm_n, i32t, tv_ints(perm.to_vec())));
            ops.push(mil::Operation {
                r#type: "transpose".into(),
                inputs: HashMap::from([
                    ("x".into(), arg_name(input_name)),
                    ("perm".into(), arg_name(&perm_n)),
                ]),
                outputs: vec![mil::NamedValueType {
                    name: out_name.to_string(),
                    r#type: Some(tensor_type(DataType::Float16, out_shape)),
                }],
                blocks: vec![],
                attributes: HashMap::new(),
            });
        } else {
            // Rank changed (strip axes) — use a reshape.
            let shape_n = format!("{out_name}_shape");
            let i32t = tensor_type(DataType::Int32, &[out_shape.len() as i64]);
            ops.push(op_const_immediate(
                &shape_n,
                i32t,
                tv_ints(out_shape.iter().map(|&v| v as i32).collect()),
            ));
            ops.push(mil::Operation {
                r#type: "reshape".into(),
                inputs: HashMap::from([
                    ("x".into(), arg_name(input_name)),
                    ("shape".into(), arg_name(&shape_n)),
                ]),
                outputs: vec![mil::NamedValueType {
                    name: out_name.to_string(),
                    r#type: Some(tensor_type(DataType::Float16, out_shape)),
                }],
                blocks: vec![],
                attributes: HashMap::new(),
            });
        }
    }

    fn emit_reshape(
        ops: &mut Vec<mil::Operation>,
        input_name: &str,
        out_name: &str,
        out_shape: &[i64],
    ) {
        let shape_n = format!("{out_name}_shape");
        let i32t = tensor_type(DataType::Int32, &[out_shape.len() as i64]);
        ops.push(op_const_immediate(
            &shape_n,
            i32t,
            tv_ints(out_shape.iter().map(|&v| v as i32).collect()),
        ));
        ops.push(mil::Operation {
            r#type: "reshape".into(),
            inputs: HashMap::from([
                ("x".into(), arg_name(input_name)),
                ("shape".into(), arg_name(&shape_n)),
            ]),
            outputs: vec![mil::NamedValueType {
                name: out_name.to_string(),
                r#type: Some(tensor_type(DataType::Float16, out_shape)),
            }],
            blocks: vec![],
            attributes: HashMap::new(),
        });
    }

    // ---- Stage 1: pre-transpose A to canonical layout (or reshape if strip).
    //
    // The MLPackage value flowing in (`a_raw`) is at the EXTERNAL shape
    // (= `a_external_shape`), since CoremlOp does the strip-axis squeeze
    // at the boundary AND the in-MLPackage rank-4 padding convention
    // (when external rank < 4) is also handled by the upstream's declared
    // shape. We bridge from the actual upstream MIL declared rank to the
    // external shape via a reshape if they differ.
    let a_tract_rank_name = {
        let target = super::rank::pad_to_rank_4(&plan.a_external_shape);
        if target == plan.a_external_shape {
            a_raw.clone()
        } else {
            // Reshape from the rank-4-padded MLPackage shape down to the
            // external shape (which can be < rank 4). Same element count.
            let n = format!("{p}_a_in");
            emit_reshape(&mut ops, &a_raw, &n, &plan.a_external_shape);
            n
        }
    };
    let a_canon_name =
        if plan.perm_a.len() == plan.a_external_shape.len() && is_identity_perm(&plan.perm_a) {
            a_tract_rank_name
        } else {
            let n = format!("{p}_a_canon");
            emit_transpose_or_reshape(
                &mut ops,
                &a_tract_rank_name,
                &plan.a_external_shape,
                &plan.perm_a,
                &n,
                &plan.a_canon_shape,
            );
            n
        };
    let b_tract_rank_name = {
        let target = super::rank::pad_to_rank_4(&plan.b_external_shape);
        if target == plan.b_external_shape {
            b_raw.clone()
        } else {
            let n = format!("{p}_b_in");
            emit_reshape(&mut ops, &b_raw, &n, &plan.b_external_shape);
            n
        }
    };
    let b_canon_name =
        if plan.perm_b.len() == plan.b_external_shape.len() && is_identity_perm(&plan.perm_b) {
            b_tract_rank_name
        } else {
            let n = format!("{p}_b_canon");
            emit_transpose_or_reshape(
                &mut ops,
                &b_tract_rank_name,
                &plan.b_external_shape,
                &plan.perm_b,
                &n,
                &plan.b_canon_shape,
            );
            n
        };

    // ---- Stage 2: reshape A and B to rank-3 [B, M, K] / [B, K, N].
    let a_mm_name = if plan.a_canon_shape == plan.a_matmul_shape {
        a_canon_name
    } else {
        let n = format!("{p}_a_mm_in");
        emit_reshape(&mut ops, &a_canon_name, &n, &plan.a_matmul_shape);
        n
    };
    let b_mm_name = if plan.b_canon_shape == plan.b_matmul_shape {
        b_canon_name
    } else {
        let n = format!("{p}_b_mm_in");
        emit_reshape(&mut ops, &b_canon_name, &n, &plan.b_matmul_shape);
        n
    };

    // ---- Stage 3: matmul (transpose flags always false — we canonicalised).
    let bool_t = tensor_type_scalar(DataType::Bool);
    let bool_val = |v: bool| mil::TensorValue {
        value: Some(mil::tensor_value::Value::Bools(mil::tensor_value::RepeatedBools {
            values: vec![v],
        })),
    };
    let tx_n = format!("{p}_tx");
    let ty_n = format!("{p}_ty");
    ops.push(op_const_immediate(&tx_n, bool_t.clone(), bool_val(false)));
    ops.push(op_const_immediate(&ty_n, bool_t, bool_val(false)));

    let mm_out_name = format!("{p}_mm");
    ops.push(mil::Operation {
        r#type: "matmul".into(),
        inputs: HashMap::from([
            ("x".into(), arg_name(&a_mm_name)),
            ("y".into(), arg_name(&b_mm_name)),
            ("transpose_x".into(), arg_name(&tx_n)),
            ("transpose_y".into(), arg_name(&ty_n)),
        ]),
        outputs: vec![mil::NamedValueType {
            name: mm_out_name.clone(),
            r#type: Some(tensor_type(DataType::Float16, &plan.matmul_out_shape)),
        }],
        blocks: vec![],
        attributes: HashMap::new(),
    });

    // ---- Stage 4: reshape matmul output back to canonical [batch..., m..., n...] layout.
    let canon_out_name = if plan.matmul_out_shape == plan.canon_out_shape {
        mm_out_name
    } else {
        let n = format!("{p}_canon_out");
        emit_reshape(&mut ops, &mm_out_name, &n, &plan.canon_out_shape);
        n
    };

    // ---- Stage 5: post-transpose to put each axis at its actual output position.
    let transposed_name =
        if plan.canon_out_shape == plan.post_transpose_shape && is_identity_perm(&plan.perm_out) {
            canon_out_name
        } else {
            let n = format!("{p}_post_t");
            emit_transpose_or_reshape(
                &mut ops,
                &canon_out_name,
                &plan.canon_out_shape,
                &plan.perm_out,
                &n,
                &plan.post_transpose_shape,
            );
            n
        };

    // ---- Stage 6: final reshape to MLPackage-boundary output shape.
    // We declare output_external_shape (with expand-out unit dims removed)
    // — the consumer-side CoremlOp re-inserts them at tract positions.
    // This keeps the MLPackage output rank ≤ 5.
    if plan.post_transpose_shape != plan.output_external_shape || transposed_name != output_name {
        emit_reshape(&mut ops, &transposed_name, output_name, &plan.output_external_shape);
    }
    Ok(ops)
}
