//! Subgraph identification + materialisation for `tract-coreml`.
//!
//! The Coreml `Transform` partitions the typed model into maximal connected
//! subgraphs of translatable nodes and replaces each subgraph with a single
//! `CoremlOp` that owns one MLPackage covering the whole region. Why fuse:
//!
//! - **Amortise Core ML setup**: one `MLModel.predict()` per subgraph instead
//!   of per op (each prediction has ~10–40 ms cross-domain dispatch overhead).
//! - **Engage ANE**: Apple's scheduler only routes MLPackages above a
//!   size/op-count threshold to the Neural Engine. Per-Conv MLPackages are
//!   too small (see `notes/phase-1-mvp.md` §4 for evidence). MobileNet at
//!   533 ops → 2 CoremlOps engages ANE at 185 mW.
//!
//! ## Two-stage pipeline
//!
//! ### Stage 1: `identify_subgraphs` — union-find with cycle avoidance
//!
//! Partition translatable nodes into maximal connected components. Two rules
//! prevent cross-subgraph cycles in the post-fusion model:
//!
//! 1. **All-or-nothing**: a node only joins a subgraph when *all* its non-
//!    const, non-source data inputs are translatable. Prevents the case where
//!    a binop has one translatable side and one CPU-only side — uniting just
//!    the translatable side puts the node in a subgraph whose external_input
//!    on the other side depends transitively on the node's own output (via
//!    the CPU chain), creating a wiring cycle. Source predecessors are
//!    excluded from the check (they're external boundaries; can't cycle).
//!
//! 2. **Convex-region**: when a node's translatable predecessors sit in
//!    *different* existing subgraphs, the union is allowed iff the merged
//!    region's quotient graph is acyclic. Checked by `merger_has_cycle`:
//!    forward BFS from the candidate-merger set through CPU nodes; if any
//!    path loops back into the merged set, reject the union. Relaxes the
//!    original "single-root" rule which rejected ALL multi-root unions —
//!    this is what lets us collapse MODNet from 9 → 2 CoremlOps and gives
//!    SAM 2 its 39 CoremlOps instead of more.
//!
//! ### Stage 2: `build_subgraph_mlpackage` — emit one MLPackage per subgraph
//!
//! For each subgraph:
//!
//! 1. **Pre-scan for multi-node pattern detectors** (InstanceNorm, LayerNorm).
//!    These walk the subgraph anchored at a final-Mul/Add and verify the
//!    chain matches the lowered ONNX `InstanceNormalization` /
//!    `LayerNormalization` shape. If matched, the anchor's translator emits
//!    a single `mb.instance_norm` / `mb.layer_norm` op and the chain members
//!    are marked `absorbed` (skipped in step 2).
//!
//! 2. **Per-node MIL emission** in topological order. Each translatable node
//!    is dispatched to its per-op translator (`conv`, `binop`, `softmax`,
//!    `general_matmul`, etc.). Translators emit MIL ops + (optional) weight
//!    blob entries. Const-fed inputs are absorbed into the MLPackage as
//!    constants via `crate::ops::const_tensor`.
//!
//! 3. **External I/O wiring**. External inputs become MLPackage feature
//!    inputs; external outputs become feature outputs. Boundary shapes use
//!    each translator's `*_external_shape` (post-strip if rank > 5) so the
//!    MLPackage stays under MIL's rank-5 cap. CoremlOp's `eval` reshapes
//!    between tract-side and MLPackage-side ranks at the boundary.
//!
//! 4. **MIL Program → .mlpackage on disk**. The serialised MIL bytes + weight
//!    blob bytes are written via [`mlpackage::write`]. The
//!    [`compile_cache`](crate::compile_cache) layer keys an `Arc<MLModel>`
//!    handle to the SHA-256 of `(MIL bytes, weight bytes)` so identical
//!    subgraphs across runs hit the cache and skip both write + compile.
//!
//! ## In-MLPackage rank-4-padding convention
//!
//! Tracks rank conventions vary: tract surfaces CHW (rank 3) or NCHW (rank 4)
//! depending on op. Inside a single MLPackage, every translator pads to
//! rank 4 (with leading 1s) so chain consistency is maintained. The
//! `*_external_shape` mechanism (see `general_matmul.rs`, `reshape.rs`,
//! `move_axis.rs`) handles tensors with rank > 5 by stripping leading unit
//! dims at the MLPackage boundary; CoremlOp restores them at the consumer
//! side. See module-level rustdoc on `lib.rs` for the convention summary.

use std::collections::{HashMap, HashSet};

use anyhow::{Result, anyhow, bail};
use prost::Message;

use tract_core::internal::*;
use tract_core::ops::cast::Cast;
use tract_core::ops::konst::Const;
use tract_core::ops::source::TypedSource;

use crate::mil::blob::BlobBuilder;
use crate::mil::program::single_function_program;
use crate::mil::value::{DataType, tensor_type};
use crate::mlpackage;
use crate::ops::{
    activation, add_axis, avgpool, binop, broadcast, cast, concat, conv, einsum, einsum_outer,
    gather, general_matmul, iff, instance_norm, layer_norm, matmul, maxpool, move_axis, pad,
    reduce, reshape, resize, rm_axis, rms_norm, scatter_nd, slice, softmax,
};
use crate::proto::core_ml::specification as spec;
use crate::proto::core_ml::specification::mil_spec as mil;

/// A connected component of translatable nodes inside a `TypedModel`.
pub struct Subgraph {
    /// Member node IDs in topological (eval) order.
    pub nodes: Vec<usize>,
    /// External inputs: outlets coming from outside the subgraph; become the
    /// MLPackage's named inputs (and the `CoremlOp`'s tract-side inputs).
    pub external_inputs: Vec<OutletId>,
    /// External outputs: outlets within the subgraph that are consumed
    /// outside (by non-subgraph nodes, OR are model outputs); become the
    /// MLPackage's named outputs.
    pub external_outputs: Vec<OutletId>,
}

/// Ask each per-op translator: can this node be lowered to MIL?
///
/// Const-like nodes (`Const`, `Cast(Const)`) are explicitly excluded — they're
/// absorbed into consumer MLPackages via `crate::ops::const_tensor` rather
/// than appearing as MIL ops in their own right. Without this guard, a
/// `Cast(Const)` would be flagged translatable by `analyse_cast` AND
/// const-like by `is_const_like`, breaking the boundary computation.
fn is_translatable(model: &TypedModel, node: &TypedNode) -> bool {
    if is_const_like(model, node) {
        return false;
    }
    matches!(conv::analyse_conv(model, node).ok(), Some(conv::ConvAnalysis::Translatable(_)))
        || matches!(
            binop::analyse_binop(model, node).ok(),
            Some(binop::BinOpAnalysis::Translatable(_))
        )
        || matches!(cast::analyse_cast(model, node).ok(), Some(cast::CastAnalysis::Translatable(_)))
        || matches!(
            einsum::analyse_einsum(model, node).ok(),
            Some(conv::ConvAnalysis::Translatable(_))
        )
        || matches!(
            concat::analyse_concat(model, node).ok(),
            Some(concat::ConcatAnalysis::Translatable(_))
        )
        || matches!(
            maxpool::analyse_maxpool(model, node).ok(),
            Some(maxpool::MaxPoolAnalysis::Translatable(_))
        )
        || matches!(
            reduce::analyse_reduce(model, node).ok(),
            Some(reduce::ReduceAnalysis::Translatable(_))
        )
        || matches!(
            rm_axis::analyse_rm_axis(model, node).ok(),
            Some(rm_axis::RmAxisAnalysis::Translatable(_))
        )
        || matches!(
            activation::analyse_activation(model, node).ok(),
            Some(activation::ActivationAnalysis::Translatable(_))
        )
        || matches!(
            add_axis::analyse_add_axis(model, node).ok(),
            Some(add_axis::AddAxisAnalysis::Translatable(_))
        )
        || matches!(
            resize::analyse_resize(model, node).ok(),
            Some(resize::ResizeAnalysis::Translatable(_))
        )
        || matches!(
            slice::analyse_slice(model, node).ok(),
            Some(slice::SliceAnalysis::Translatable(_))
        )
        || matches!(
            matmul::analyse_matmul(model, node).ok(),
            Some(matmul::MatMulAnalysis::Translatable(_))
        )
        || matches!(
            avgpool::analyse_sumpool(model, node).ok(),
            Some(avgpool::AvgPoolAnalysis::Translatable(_))
        )
        || matches!(
            broadcast::analyse_broadcast(model, node).ok(),
            Some(broadcast::BroadcastAnalysis::Translatable(_))
        )
        || matches!(
            softmax::analyse_softmax(model, node).ok(),
            Some(softmax::SoftmaxAnalysis::Translatable(_))
        )
        || matches!(
            general_matmul::analyse_general_matmul(model, node).ok(),
            Some(general_matmul::GeneralMatMulAnalysis::Translatable(_))
        )
        || matches!(
            rms_norm::analyse_rms_norm(model, node).ok(),
            Some(rms_norm::RmsNormAnalysis::Translatable(_))
        )
        || matches!(
            reshape::analyse_reshape(model, node).ok(),
            Some(reshape::ReshapeAnalysis::Translatable(_))
        )
        || matches!(
            move_axis::analyse_move_axis(model, node).ok(),
            Some(move_axis::MoveAxisAnalysis::Translatable(_))
        )
        || matches!(pad::analyse_pad(model, node).ok(), Some(pad::PadAnalysis::Translatable(_)))
        || matches!(
            gather::analyse_gather(model, node).ok(),
            Some(gather::GatherAnalysis::Translatable(_))
        )
        || matches!(iff::analyse_iff(model, node).ok(), Some(iff::IffAnalysis::Translatable(_)))
        || matches!(
            scatter_nd::analyse_scatter_nd(model, node).ok(),
            Some(scatter_nd::ScatterNdAnalysis::Translatable(_))
        )
        || matches!(
            einsum_outer::analyse_einsum_outer(model, node).ok(),
            Some(einsum_outer::EinsumOuterAnalysis::Translatable(_))
        )
}

/// MLPackage-side input shape that a `node` expects at input slot `slot`.
/// Used by the subgraph builder to derive external-input MLPackage shapes
/// (which may differ from the tract-side fact shape — Conv on CHW data
/// prepends N=1, etc.).
fn input_shape_at_slot(model: &TypedModel, node: &TypedNode, slot: usize) -> Result<Vec<i64>> {
    if let conv::ConvAnalysis::Translatable(plan) = conv::analyse_conv(model, node)? {
        // Conv has only one data input slot (slot 0); slot 1+2 are absorbed.
        return Ok(plan.input_shape);
    }
    if let conv::ConvAnalysis::Translatable(plan) = einsum::analyse_einsum(model, node)? {
        // EinSum-as-conv: data input slot 0; weight slot 1 is absorbed (Const).
        if slot != 0 {
            bail!("EinSum::input_shape_at_slot: slot {slot} is not a Data input");
        }
        return Ok(plan.input_shape);
    }
    if let binop::BinOpAnalysis::Translatable(plan) = binop::analyse_binop(model, node)? {
        let shape = match (slot, &plan.a, &plan.b) {
            (0, binop::BinOpInput::Data { shape }, _) => shape.clone(),
            (1, _, binop::BinOpInput::Data { shape }) => shape.clone(),
            _ => bail!(
                "BinOp::input_shape_at_slot: slot {slot} is not a Data input on node {} ({})",
                node.id,
                node.name
            ),
        };
        return Ok(shape);
    }
    if let cast::CastAnalysis::Translatable(plan) = cast::analyse_cast(model, node)? {
        // Cast has 1 data input (slot 0).
        if slot == 0 {
            return Ok(plan.input_shape);
        }
        bail!("Cast::input_shape_at_slot: slot {slot} is invalid (Cast has 1 input)");
    }
    if let concat::ConcatAnalysis::Translatable(plan) = concat::analyse_concat(model, node)? {
        // Concat is variadic; each slot has its own MLPackage-side shape.
        let shapes = plan.input_shapes();
        if slot >= shapes.len() {
            bail!("Concat::input_shape_at_slot: slot {slot} >= input count {}", shapes.len());
        }
        // Const slots are absorbed inline at emit; the boundary shape is
        // only meaningful for Data inputs, but we return the rank-4 shape
        // anyway in case the caller is interested.
        return Ok(shapes[slot].clone());
    }
    if let maxpool::MaxPoolAnalysis::Translatable(plan) = maxpool::analyse_maxpool(model, node)? {
        if slot != 0 {
            bail!("MaxPool::input_shape_at_slot: slot {slot} invalid (1 input)");
        }
        return Ok(plan.input_shape);
    }
    if let reduce::ReduceAnalysis::Translatable(plan) = reduce::analyse_reduce(model, node)? {
        if slot != 0 {
            bail!("Reduce::input_shape_at_slot: slot {slot} invalid (1 input)");
        }
        return Ok(plan.input_shape);
    }
    if let rm_axis::RmAxisAnalysis::Translatable(plan) = rm_axis::analyse_rm_axis(model, node)? {
        if slot != 0 {
            bail!("RmAxis::input_shape_at_slot: slot {slot} invalid (1 input)");
        }
        return Ok(plan.input_shape);
    }
    if let activation::ActivationAnalysis::Translatable(plan) =
        activation::analyse_activation(model, node)?
    {
        if slot != 0 {
            bail!("Activation::input_shape_at_slot: slot {slot} invalid (1 input)");
        }
        return Ok(plan.shape);
    }
    if let add_axis::AddAxisAnalysis::Translatable(plan) = add_axis::analyse_add_axis(model, node)?
    {
        if slot != 0 {
            bail!("AddAxis::input_shape_at_slot: slot {slot} invalid (1 input)");
        }
        return Ok(plan.input_shape);
    }
    if let resize::ResizeAnalysis::Translatable(plan) = resize::analyse_resize(model, node)? {
        // Resize has 1 data input (slot 0); scales/sizes inputs are absorbed as Const.
        if slot != 0 {
            bail!("Resize::input_shape_at_slot: slot {slot} is not the Data input");
        }
        return Ok(plan.input_shape);
    }
    if let slice::SliceAnalysis::Translatable(plan) = slice::analyse_slice(model, node)? {
        if slot != 0 {
            bail!("Slice::input_shape_at_slot: slot {slot} invalid (1 input)");
        }
        return Ok(plan.input_shape);
    }
    if let matmul::MatMulAnalysis::Translatable(plan) = matmul::analyse_matmul(model, node)? {
        if slot != 0 {
            bail!(
                "MatMul::input_shape_at_slot: slot {slot} is not the Data input (weight \
                 absorbed as Const)"
            );
        }
        return Ok(plan.input_shape);
    }
    if let avgpool::AvgPoolAnalysis::Translatable(plan) = avgpool::analyse_sumpool(model, node)? {
        if slot != 0 {
            bail!("SumPool::input_shape_at_slot: slot {slot} invalid (1 input)");
        }
        return Ok(plan.input_shape);
    }
    if let broadcast::BroadcastAnalysis::Translatable(plan) =
        broadcast::analyse_broadcast(model, node)?
    {
        if slot != 0 {
            bail!("MultiBroadcastTo::input_shape_at_slot: slot {slot} invalid (1 input)");
        }
        return Ok(plan.input_shape);
    }
    if let softmax::SoftmaxAnalysis::Translatable(plan) = softmax::analyse_softmax(model, node)? {
        if slot != 0 {
            bail!("Softmax::input_shape_at_slot: slot {slot} invalid (1 input)");
        }
        return Ok(plan.input_shape);
    }
    if let general_matmul::GeneralMatMulAnalysis::Translatable(plan) =
        general_matmul::analyse_general_matmul(model, node)?
    {
        // Return the EXTERNAL shape (= tract shape with strip-A/B unit dims
        // removed) so the MLPackage boundary stays under the rank-5 cap and
        // CoremlOp does the strip-axis squeeze at the boundary.
        match (slot, &plan.a, &plan.b) {
            (0, general_matmul::MatMulOperand::Runtime { .. }, _) => {
                return Ok(plan.a_external_shape);
            }
            (1, _, general_matmul::MatMulOperand::Runtime { .. }) => {
                return Ok(plan.b_external_shape);
            }
            _ => bail!(
                "GeneralMatMul::input_shape_at_slot: slot {slot} is a Const operand on node {} \
                 ({})",
                node.id,
                node.name
            ),
        }
    }
    if let rms_norm::RmsNormAnalysis::Translatable(plan) = rms_norm::analyse_rms_norm(model, node)?
    {
        if slot != 0 {
            bail!("RmsNorm::input_shape_at_slot: slot {slot} invalid (1 input)");
        }
        return Ok(plan.input_shape);
    }
    if let reshape::ReshapeAnalysis::Translatable(plan) = reshape::analyse_reshape(model, node)? {
        if slot != 0 {
            bail!("Reshape::input_shape_at_slot: slot {slot} invalid (1 input)");
        }
        // Return EXTERNAL input shape (post-strip if rank > 5).
        return Ok(plan.input_external_shape);
    }
    if let move_axis::MoveAxisAnalysis::Translatable(plan) =
        move_axis::analyse_move_axis(model, node)?
    {
        if slot != 0 {
            bail!("MoveAxis::input_shape_at_slot: slot {slot} invalid (1 input)");
        }
        return Ok(plan.input_external_shape);
    }
    if let pad::PadAnalysis::Translatable(plan) = pad::analyse_pad(model, node)? {
        if slot != 0 {
            bail!("Pad::input_shape_at_slot: slot {slot} invalid (1 input)");
        }
        return Ok(plan.input_shape);
    }
    if let gather::GatherAnalysis::Translatable(plan) = gather::analyse_gather(model, node)? {
        // Gather has 2 input slots (data + indices). Either may be Const
        // (absorbed at emit). When the slot is Data, return its rank-N shape
        // (no rank-padding — Gather operates at the natural data rank).
        match (slot, &plan.data, &plan.indices) {
            (0, gather::GatherInput::Data { shape, .. }, _) => return Ok(shape.clone()),
            (1, _, gather::GatherInput::Data { shape, .. }) => return Ok(shape.clone()),
            (0, gather::GatherInput::Const { .. }, _)
            | (1, _, gather::GatherInput::Const { .. }) => {
                bail!(
                    "Gather::input_shape_at_slot: slot {slot} is a Const operand on node {} \
                     (caller should not request the boundary shape for a Const)",
                    node.name
                );
            }
            _ => bail!("Gather::input_shape_at_slot: slot {slot} invalid (Gather has 2 inputs)"),
        }
    }
    if let iff::IffAnalysis::Translatable(plan) = iff::analyse_iff(model, node)? {
        // Iff has 3 slots: cond + t + f. Any may be Const.
        let pick = match slot {
            0 => &plan.cond,
            1 => &plan.t,
            2 => &plan.f,
            _ => bail!("Iff::input_shape_at_slot: slot {slot} invalid (Iff has 3 inputs)"),
        };
        return match pick {
            iff::IffInput::Data { shape, .. } => Ok(shape.clone()),
            iff::IffInput::Const { .. } => bail!(
                "Iff::input_shape_at_slot: slot {slot} is a Const operand on node {} \
                 (caller should not request the boundary shape for a Const)",
                node.name
            ),
        };
    }
    if let scatter_nd::ScatterNdAnalysis::Translatable(plan) =
        scatter_nd::analyse_scatter_nd(model, node)?
    {
        let pick = match slot {
            0 => &plan.data,
            1 => &plan.indices,
            2 => &plan.updates,
            _ => bail!("ScatterNd::input_shape_at_slot: slot {slot} invalid (3 inputs)"),
        };
        return match pick {
            scatter_nd::ScatterNdInput::Data { shape, .. } => Ok(shape.clone()),
            scatter_nd::ScatterNdInput::Const { .. } => bail!(
                "ScatterNd::input_shape_at_slot: slot {slot} is a Const operand on node {} \
                 (caller should not request the boundary shape for a Const)",
                node.name
            ),
        };
    }
    if let einsum_outer::EinsumOuterAnalysis::Translatable(plan) =
        einsum_outer::analyse_einsum_outer(model, node)?
    {
        let pick = match slot {
            0 => &plan.a,
            1 => &plan.b,
            _ => bail!("EinsumOuter::input_shape_at_slot: slot {slot} invalid (2 inputs)"),
        };
        return match pick {
            einsum_outer::EinsumOuterOperand::Runtime { shape } => Ok(shape.clone()),
            einsum_outer::EinsumOuterOperand::Const { .. } => {
                bail!("EinsumOuter::input_shape_at_slot: slot {slot} is a Const operand")
            }
        };
    }
    bail!("input_shape_at_slot: node {} ({}) is not translatable", node.id, node.name)
}

/// MLPackage-side output shape + tract-side output fact for a translatable
/// node. Used for external-output MLPackage shape derivation.
fn output_shape_and_fact(model: &TypedModel, node: &TypedNode) -> Result<(Vec<i64>, TypedFact)> {
    if let conv::ConvAnalysis::Translatable(plan) = conv::analyse_conv(model, node)? {
        return Ok((plan.output_shape, plan.output_fact));
    }
    if let conv::ConvAnalysis::Translatable(plan) = einsum::analyse_einsum(model, node)? {
        return Ok((plan.output_shape, plan.output_fact));
    }
    if let binop::BinOpAnalysis::Translatable(plan) = binop::analyse_binop(model, node)? {
        return Ok((plan.output_shape, plan.output_fact));
    }
    if let cast::CastAnalysis::Translatable(plan) = cast::analyse_cast(model, node)? {
        // Cast preserves shape, only changes dtype. Output shape == input shape.
        return Ok((plan.input_shape, plan.output_fact));
    }
    if let concat::ConcatAnalysis::Translatable(plan) = concat::analyse_concat(model, node)? {
        return Ok((plan.output_shape, plan.output_fact));
    }
    if let maxpool::MaxPoolAnalysis::Translatable(plan) = maxpool::analyse_maxpool(model, node)? {
        return Ok((plan.output_shape, plan.output_fact));
    }
    if let reduce::ReduceAnalysis::Translatable(plan) = reduce::analyse_reduce(model, node)? {
        return Ok((plan.output_shape, plan.output_fact));
    }
    if let rm_axis::RmAxisAnalysis::Translatable(plan) = rm_axis::analyse_rm_axis(model, node)? {
        return Ok((plan.output_shape, plan.output_fact));
    }
    if let activation::ActivationAnalysis::Translatable(plan) =
        activation::analyse_activation(model, node)?
    {
        // Pointwise: output shape == input shape.
        return Ok((plan.shape, plan.output_fact));
    }
    if let add_axis::AddAxisAnalysis::Translatable(plan) = add_axis::analyse_add_axis(model, node)?
    {
        return Ok((plan.output_shape, plan.output_fact));
    }
    if let resize::ResizeAnalysis::Translatable(plan) = resize::analyse_resize(model, node)? {
        return Ok((plan.output_shape, plan.output_fact));
    }
    if let slice::SliceAnalysis::Translatable(plan) = slice::analyse_slice(model, node)? {
        return Ok((plan.output_shape, plan.output_fact));
    }
    if let matmul::MatMulAnalysis::Translatable(plan) = matmul::analyse_matmul(model, node)? {
        return Ok((plan.output_shape, plan.output_fact));
    }
    if let avgpool::AvgPoolAnalysis::Translatable(plan) = avgpool::analyse_sumpool(model, node)? {
        return Ok((plan.output_shape, plan.output_fact));
    }
    if let broadcast::BroadcastAnalysis::Translatable(plan) =
        broadcast::analyse_broadcast(model, node)?
    {
        return Ok((plan.output_shape, plan.output_fact));
    }
    if let softmax::SoftmaxAnalysis::Translatable(plan) = softmax::analyse_softmax(model, node)? {
        return Ok((plan.input_shape, plan.output_fact));
    }
    if let general_matmul::GeneralMatMulAnalysis::Translatable(plan) =
        general_matmul::analyse_general_matmul(model, node)?
    {
        // Return EXTERNAL output shape (= output_shape with expand-out unit
        // dims removed) so MLPackage outputs stay under rank-5. CoremlOp at
        // the consumer side re-inserts the unit dims.
        return Ok((plan.output_external_shape, plan.output_fact));
    }
    if let rms_norm::RmsNormAnalysis::Translatable(plan) = rms_norm::analyse_rms_norm(model, node)?
    {
        return Ok((plan.input_shape, plan.output_fact));
    }
    if let reshape::ReshapeAnalysis::Translatable(plan) = reshape::analyse_reshape(model, node)? {
        // Return EXTERNAL output shape (post-strip if rank > 5).
        return Ok((plan.output_external_shape, plan.output_fact));
    }
    if let move_axis::MoveAxisAnalysis::Translatable(plan) =
        move_axis::analyse_move_axis(model, node)?
    {
        return Ok((plan.output_external_shape, plan.output_fact));
    }
    if let pad::PadAnalysis::Translatable(plan) = pad::analyse_pad(model, node)? {
        return Ok((plan.output_shape, plan.output_fact));
    }
    if let gather::GatherAnalysis::Translatable(plan) = gather::analyse_gather(model, node)? {
        return Ok((plan.output_shape, plan.output_fact));
    }
    if let iff::IffAnalysis::Translatable(plan) = iff::analyse_iff(model, node)? {
        return Ok((plan.output_shape, plan.output_fact));
    }
    if let scatter_nd::ScatterNdAnalysis::Translatable(plan) =
        scatter_nd::analyse_scatter_nd(model, node)?
    {
        return Ok((plan.output_shape, plan.output_fact));
    }
    if let einsum_outer::EinsumOuterAnalysis::Translatable(plan) =
        einsum_outer::analyse_einsum_outer(model, node)?
    {
        return Ok((plan.output_shape, plan.output_fact));
    }
    bail!("output_shape_and_fact: node {} ({}) is not translatable", node.id, node.name)
}

/// True if the dependency graph would have a cycle in the quotient where
/// `m_set` is collapsed into a single node. Equivalently: there exists a
/// path from some node in `m_set` to another node in `m_set` that goes
/// through nodes NOT in `m_set`. Used by [`identify_subgraphs`] to decide
/// whether merging two existing subgraphs through a common consumer is
/// safe (convex-region union check).
fn merger_has_cycle(model: &TypedModel, m_set: &HashSet<usize>) -> bool {
    // Forward BFS from `m_set` through non-`m_set` nodes; if we ever step
    // back into `m_set`, the merger creates a cycle.
    let mut visited: HashSet<usize> = HashSet::new();
    let mut frontier: Vec<usize> = Vec::new();
    // Seed with direct successors of m_set that are NOT in m_set.
    for other in &model.nodes {
        if m_set.contains(&other.id) {
            continue;
        }
        if other.inputs.iter().any(|i| m_set.contains(&i.node)) && visited.insert(other.id) {
            frontier.push(other.id);
        }
    }
    while let Some(cur) = frontier.pop() {
        for other in &model.nodes {
            if !other.inputs.iter().any(|i| i.node == cur) {
                continue;
            }
            if m_set.contains(&other.id) {
                return true; // m → CPU → m: cycle
            }
            if visited.insert(other.id) {
                frontier.push(other.id);
            }
        }
    }
    false
}

/// True for `TypedSource` (model input boundary). Sources do no compute
/// and can never participate in a wiring cycle, so they're treated as
/// non-blocking by the union-find's all-or-nothing rule.
fn is_source(node: &TypedNode) -> bool {
    node.op_as::<TypedSource>().is_some()
}

/// True for nodes whose value is computed at translate time and absorbed
/// into the MLPackage (Const, plus `Cast(Const)` which `analyse_conv` walks
/// through). These don't appear as MIL ops in the Program — their value is
/// either inlined or written to the weight blob — so they don't count as
/// "data inputs" when computing subgraph boundaries.
fn is_const_like(model: &TypedModel, node: &TypedNode) -> bool {
    if node.op_as::<Const>().is_some() {
        return true;
    }
    if node.op_as::<Cast>().is_some() && node.inputs.len() == 1 {
        let pred = &model.nodes[node.inputs[0].node];
        return pred.op_as::<Const>().is_some();
    }
    false
}

/// Identify maximal connected subgraphs of translatable nodes in `model`.
///
/// Two translatable nodes are unioned when they're connected by a data edge
/// (i.e. one is the input of the other through a non-const-like predecessor).
/// Const-like predecessors don't propagate the union — they're absorbed into
/// the consumer's MLPackage as constants.
pub fn identify_subgraphs(model: &TypedModel) -> Result<Vec<Subgraph>> {
    let n = model.nodes.len();
    let translatable: Vec<bool> =
        (0..n).map(|id| is_translatable(model, &model.nodes[id])).collect();

    // Union-find with path compression.
    let mut parent: Vec<usize> = (0..n).collect();
    fn find(parent: &mut [usize], x: usize) -> usize {
        let mut root = x;
        while parent[root] != root {
            root = parent[root];
        }
        let mut cur = x;
        while parent[cur] != root {
            let next = parent[cur];
            parent[cur] = root;
            cur = next;
        }
        root
    }
    fn union(parent: &mut [usize], x: usize, y: usize) {
        let rx = find(parent, x);
        let ry = find(parent, y);
        if rx != ry {
            parent[rx] = ry;
        }
    }

    // Union: for each translatable node n, union it with its translatable
    // predecessors. Two cumulative rules avoid cross-subgraph cycles:
    //
    // 1. **All-or-nothing**: only union when *all* of n's "real" CPU-side
    //    data inputs are translatable. A predecessor is "real CPU" if it's
    //    NOT a const-like (already absorbed into MLPackage as a constant)
    //    AND NOT a Source (a pure input boundary that does no compute and
    //    can never participate in a wiring cycle). Prevents the case where
    //    a binop has one translatable side and one CPU-COMPUTE side —
    //    uniting just the translatable side puts n in a subgraph whose
    //    external_input on the other side depends transitively on n's own
    //    output (via the CPU chain), creating a wiring cycle.
    //
    //    Source predecessors are treated as "external boundary" rather than
    //    "blocking CPU residual". Without that exclusion, the trivial
    //    `LayerNorm(source_input)` chain (Reduce + Sub + RmsNorm + Mul + Add
    //    where Sub takes both source and Reduce-output) wouldn't fuse,
    //    because Sub's all-or-nothing check sees source as non-translatable
    //    and blocks the union.
    //
    // 2. **Convex-region** (Phase 3 capstone — supersedes the original
    //    "single-root" rule): when n's translatable predecessors P1, P2 sit
    //    in DIFFERENT existing subgraphs S1, S2, unioning them through n is
    //    safe IF AND ONLY IF the merged region M = S1 ∪ S2 ∪ {n} is convex
    //    in the dependency graph — i.e. no CPU-path from M's outputs ever
    //    loops back into M. The strict "single-root" rule rejected ALL
    //    multi-root unions, which is correctness-conservative but leaves
    //    boundary trips on the table. The convex check `merger_has_cycle`
    //    rejects only the genuinely-cyclic mergers.
    for node in &model.nodes {
        if !translatable[node.id] {
            continue;
        }
        // "Translatable predecessors" = non-const, non-source data inputs
        // that are themselves translatable. These are the only ones we
        // actually union with. Source inputs are external boundaries — they
        // don't block the union and don't get unioned with `node`.
        let data_inputs: Vec<OutletId> = node
            .inputs
            .iter()
            .filter(|i| !is_const_like(model, &model.nodes[i.node]))
            .filter(|i| !is_source(&model.nodes[i.node]))
            .copied()
            .collect();
        if data_inputs.is_empty() {
            continue;
        }
        // Rule 1: all-or-nothing (over the real CPU-compute predecessors).
        if !data_inputs.iter().all(|i| translatable[i.node]) {
            continue;
        }
        // Rule 2: convex-region check (relaxed single-root).
        let pred_roots: HashSet<usize> =
            data_inputs.iter().map(|i| find(&mut parent, i.node)).collect();
        if pred_roots.len() > 1 {
            // Compute the would-be merged set: all translatable nodes whose
            // current root is in pred_roots, plus `node` itself.
            let mut m_set: HashSet<usize> = HashSet::with_capacity(64);
            for (id, &is_t) in translatable.iter().enumerate() {
                if is_t && pred_roots.contains(&find(&mut parent, id)) {
                    m_set.insert(id);
                }
            }
            m_set.insert(node.id);
            if merger_has_cycle(model, &m_set) {
                continue;
            }
        }
        for input in &data_inputs {
            union(&mut parent, node.id, input.node);
        }
    }

    // Group by root, ordered by topo position.
    let topo = model.eval_order()?;
    let mut groups: HashMap<usize, Vec<usize>> = HashMap::new();
    for &node_id in &topo {
        if !translatable[node_id] {
            continue;
        }
        let root = find(&mut parent, node_id);
        groups.entry(root).or_default().push(node_id);
    }

    // For each group, compute external I/O.
    let mut subgraphs = Vec::with_capacity(groups.len());
    for (_root, nodes) in groups {
        let internal: HashSet<usize> = nodes.iter().copied().collect();
        let mut external_inputs: Vec<OutletId> = Vec::new();
        let mut seen_inputs: HashSet<OutletId> = HashSet::new();
        let mut external_outputs: Vec<OutletId> = Vec::new();
        let mut seen_outputs: HashSet<OutletId> = HashSet::new();

        // External inputs: any input outlet pointing outside the subgraph,
        // EXCLUDING const-like predecessors (those are absorbed into the MLPackage
        // as constants by the per-op emitter, not wired as MLPackage inputs).
        for &n_id in &nodes {
            let n = &model.nodes[n_id];
            for input in &n.inputs {
                if internal.contains(&input.node) {
                    continue;
                }
                let pred = &model.nodes[input.node];
                if is_const_like(model, pred) {
                    continue;
                }
                if seen_inputs.insert(*input) {
                    external_inputs.push(*input);
                }
            }
        }

        // External outputs: any output of a subgraph node consumed externally
        // (either by a non-subgraph node, or by the model's overall outputs).
        for &n_id in &nodes {
            let n = &model.nodes[n_id];
            for slot in 0..n.outputs.len() {
                let outlet = OutletId::new(n_id, slot);
                let is_model_output = model.outputs.contains(&outlet);
                let is_consumed_externally = is_model_output
                    || model
                        .nodes
                        .iter()
                        .filter(|other| !internal.contains(&other.id))
                        .any(|other| other.inputs.contains(&outlet));
                if is_consumed_externally && seen_outputs.insert(outlet) {
                    external_outputs.push(outlet);
                }
            }
        }

        subgraphs.push(Subgraph { nodes, external_inputs, external_outputs });
    }

    Ok(subgraphs)
}

/// What the subgraph MLPackage exposes to the surrounding tract graph. The
/// `CoremlOp` we emit needs this to wire inputs/outputs in eval.
pub struct SubgraphIO {
    /// Per-input feature names (positional).
    pub input_names: Vec<String>,
    /// Per-input shape the MLPackage expects (rank 4 for Conv-class subgraphs;
    /// `CoremlOp` reshapes between this and the tract-side fact shape).
    pub coreml_input_shapes: Vec<Vec<usize>>,
    /// Per-input MLPackage-side dtype (e.g. F16 for floats, I32 for indices).
    /// Diverges from the tract input fact dtype only when MIL feature inputs
    /// don't natively support the tract dtype (notably I64 → I32).
    pub coreml_input_dtypes: Vec<DatumType>,
    /// Per-input tract source outlet (so the transform knows where to wire
    /// the data input from in the *target* model).
    pub source_outlets: Vec<OutletId>,
    /// Per-output feature names (positional).
    pub output_names: Vec<String>,
    /// Per-output shape the MLPackage produces.
    pub coreml_output_shapes: Vec<Vec<usize>>,
    /// Per-output tract fact (the shape `CoremlOp` reshapes back to).
    pub output_facts: TVec<TypedFact>,
    /// Per-output the original tract outlet inside the source model — used
    /// by the transform to remap consumers from the old outlet to the new
    /// `CoremlOp` output slot.
    pub source_outputs: Vec<OutletId>,
}

/// Materialise a `Subgraph` to disk as a single MLPackage. Returns the I/O
/// wiring info needed to build a `CoremlOp` around it.
pub fn build_subgraph_mlpackage(
    out_path: &std::path::Path,
    model: &TypedModel,
    subgraph: &Subgraph,
) -> Result<SubgraphIO> {
    let mut blob = BlobBuilder::new();
    let mut mil_ops: Vec<mil::Operation> = Vec::new();
    // outlet -> name in the MIL program
    let mut name_for: HashMap<OutletId, String> = HashMap::new();

    // 1. Name + declare external inputs.
    //
    // The MLPackage's declared input shape must match what the consumer op
    // expects — for Conv on CHW data, rank-4 (N=1 prepended), per
    // `analyse_conv`'s shape synthesis. Find the first subgraph consumer of
    // each external input and derive the shape via `input_shape_at_slot`.
    // `CoremlOp` reshapes between tract-side and MLPackage-side at eval time.
    let mut input_names: Vec<String> = Vec::with_capacity(subgraph.external_inputs.len());
    let mut coreml_input_shapes: Vec<Vec<usize>> =
        Vec::with_capacity(subgraph.external_inputs.len());
    let mut coreml_input_dtypes: Vec<DatumType> =
        Vec::with_capacity(subgraph.external_inputs.len());
    let mut mil_inputs: Vec<mil::NamedValueType> =
        Vec::with_capacity(subgraph.external_inputs.len());
    for (i, outlet) in subgraph.external_inputs.iter().enumerate() {
        let name = format!("in{i}");
        let fact = model.outlet_fact(*outlet)?;
        // F16 is the standard floating-point dtype for our pipeline; I32 is
        // accepted directly for index inputs (Gather's indices slot, etc.).
        // I64 is mapped to I32 at the MIL boundary because MLMultiArray
        // doesn't natively support Int64 — `CoremlOp::eval` does the
        // narrowing cast at predict time before constructing the MLMultiArray.
        // (ONNX exports of LLMs typically declare token-id inputs as I64
        // matching ONNX's INT64 semantics, but our MIL gather op consumes I32
        // — so the cast pipeline is: tract I64 input → eval-side cast →
        // MLMultiArray I32 → MIL gather indices.)
        let (mil_input_dtype, mil_input_datum) = match fact.datum_type {
            DatumType::F16 => (DataType::Float16, DatumType::F16),
            DatumType::I32 => (DataType::Int32, DatumType::I32),
            DatumType::I64 => (DataType::Int32, DatumType::I32),
            other => bail!(
                "subgraph external input {outlet:?} has dtype {other:?} \
                 (supported: F16, I32, I64)"
            ),
        };

        // Find a consumer of this outlet inside the subgraph + the slot it
        // connects to.
        let mut consumer_slot: Option<(usize, usize)> = None;
        for &n_id in &subgraph.nodes {
            let n = &model.nodes[n_id];
            for (slot, input) in n.inputs.iter().enumerate() {
                if input == outlet {
                    consumer_slot = Some((n_id, slot));
                    break;
                }
            }
            if consumer_slot.is_some() {
                break;
            }
        }
        let (consumer_id, slot) = consumer_slot.ok_or_else(|| {
            anyhow!("subgraph external input {outlet:?} has no consumer inside the subgraph")
        })?;
        let ml_shape_i64 = input_shape_at_slot(model, &model.nodes[consumer_id], slot)?;
        let ml_shape: Vec<usize> = ml_shape_i64.iter().map(|&s| s as usize).collect();

        coreml_input_shapes.push(ml_shape);
        coreml_input_dtypes.push(mil_input_datum);
        mil_inputs.push(mil::NamedValueType {
            name: name.clone(),
            r#type: Some(tensor_type(mil_input_dtype, &ml_shape_i64)),
        });
        name_for.insert(*outlet, name.clone());
        input_names.push(name);
    }

    // 2a. Pre-scan: detect multi-node InstanceNorm + LayerNorm patterns.
    // Each anchor (final Mul for InstanceNorm; final Add or Mul for
    // LayerNorm) walks back to verify the chain. If matched, the anchor
    // emits one consolidated MIL op and the chain members get marked as
    // `absorbed` (skipped in the per-node emit loop below).
    //
    // Build `subgraph_set` first so the detector can verify all chain
    // members are in this subgraph. LayerNorm is checked first because its
    // chain (~10 ops) is a strict superset of InstanceNorm's pattern shape
    // — running LayerNorm first claims the deepest matches and we fall
    // back to InstanceNorm only for residual Mul-anchored chains.
    let subgraph_set: HashSet<usize> = subgraph.nodes.iter().copied().collect();
    let in_subgraph = |id: usize| subgraph_set.contains(&id);
    let mut absorbed_set: HashSet<usize> = HashSet::new();
    let mut layer_norm_at: HashMap<usize, layer_norm::LayerNormPlan> = HashMap::new();
    let mut instance_norm_at: HashMap<usize, instance_norm::InstanceNormPlan> = HashMap::new();
    for &n_id in &subgraph.nodes {
        let n = &model.nodes[n_id];
        if let Some(plan) = layer_norm::detect_layer_norm(model, n, &in_subgraph) {
            if plan.absorbed.iter().any(|m| absorbed_set.contains(m)) {
                continue;
            }
            for &m in &plan.absorbed {
                absorbed_set.insert(m);
            }
            layer_norm_at.insert(n_id, plan);
        }
    }
    for &n_id in &subgraph.nodes {
        if absorbed_set.contains(&n_id) || layer_norm_at.contains_key(&n_id) {
            continue;
        }
        let n = &model.nodes[n_id];
        if let Some(plan) = instance_norm::detect_instance_norm(model, n, &in_subgraph) {
            if plan.absorbed.iter().any(|m| absorbed_set.contains(m)) {
                continue;
            }
            for &m in &plan.absorbed {
                absorbed_set.insert(m);
            }
            instance_norm_at.insert(n_id, plan);
        }
    }

    // 2b. Walk subgraph nodes in topo order, emit per-node MIL ops.
    //
    // Node names use the within-subgraph index (`n0`, `n1`, ...) rather than
    // the source-model node ID, so identical subgraph content produces
    // identical MIL bytes regardless of how the surrounding model was built.
    // This is what makes the persistent compile cache (compile_cache.rs) hit
    // when the same model is loaded twice.
    for (sg_idx, &n_id) in subgraph.nodes.iter().enumerate() {
        let n = &model.nodes[n_id];
        // Skip absorbed-into-an-instance-norm-chain nodes; the anchor emits.
        if absorbed_set.contains(&n_id) {
            continue;
        }
        // Each translatable op currently has 1 output; generalise when multi-
        // output ops join (e.g. Split).
        let output_name = format!("n{sg_idx}");
        name_for.insert(OutletId::new(n_id, 0), output_name.clone());

        // InstanceNorm anchor: emit the consolidated MIL op.
        if let Some(plan) = instance_norm_at.get(&n_id) {
            let in_name = name_for
                .get(&plan.data_input)
                .ok_or_else(|| {
                    anyhow!(
                        "InstanceNorm anchor {} (id {}) data input {:?} not yet mapped",
                        n.name,
                        n_id,
                        plan.data_input
                    )
                })?
                .clone();
            mil_ops.extend(instance_norm::emit_instance_norm_mil(
                plan,
                &mut blob,
                &in_name,
                &output_name,
            )?);
            continue;
        }
        // LayerNorm anchor: emit the consolidated MIL op.
        if let Some(plan) = layer_norm_at.get(&n_id) {
            let in_name = name_for
                .get(&plan.data_input)
                .ok_or_else(|| {
                    anyhow!(
                        "LayerNorm anchor {} (id {}) data input {:?} not yet mapped",
                        n.name,
                        n_id,
                        plan.data_input
                    )
                })?
                .clone();
            mil_ops.extend(layer_norm::emit_layer_norm_mil(
                plan,
                &mut blob,
                &in_name,
                &output_name,
            )?);
            continue;
        }

        // Per-op dispatch.
        if let conv::ConvAnalysis::Translatable(plan) = conv::analyse_conv(model, n)? {
            let in_name = name_for
                .get(&n.inputs[0])
                .ok_or_else(|| {
                    anyhow!(
                        "Conv node {} (id {}) has unmapped data input {:?}",
                        n.name,
                        n_id,
                        n.inputs[0]
                    )
                })?
                .clone();
            mil_ops.extend(conv::emit_conv_mil(&plan, &mut blob, &in_name, &output_name)?);
            continue;
        }
        if let conv::ConvAnalysis::Translatable(plan) = einsum::analyse_einsum(model, n)? {
            // EinSum-as-1×1-conv: data input slot 0; weight slot 1 absorbed.
            let in_name = name_for
                .get(&n.inputs[0])
                .ok_or_else(|| {
                    anyhow!(
                        "EinSum node {} (id {}) has unmapped data input {:?}",
                        n.name,
                        n_id,
                        n.inputs[0]
                    )
                })?
                .clone();
            mil_ops.extend(conv::emit_conv_mil(&plan, &mut blob, &in_name, &output_name)?);
            continue;
        }
        if let binop::BinOpAnalysis::Translatable(plan) = binop::analyse_binop(model, n)? {
            // For Data inputs we look up the producer's name; for Const inputs
            // (already absorbed by `analyse_binop` into `plan.a`/`plan.b`) we
            // pass `None` and `emit_binop_mil` writes the const blob inline.
            let in_a_name = match plan.a {
                binop::BinOpInput::Data { .. } => Some(
                    name_for
                        .get(&n.inputs[0])
                        .ok_or_else(|| {
                            anyhow!(
                                "{:?} node {} (id {}) has unmapped Data input 0 {:?}",
                                plan.kind,
                                n.name,
                                n_id,
                                n.inputs[0]
                            )
                        })?
                        .clone(),
                ),
                binop::BinOpInput::Const { .. } => None,
            };
            let in_b_name = match plan.b {
                binop::BinOpInput::Data { .. } => Some(
                    name_for
                        .get(&n.inputs[1])
                        .ok_or_else(|| {
                            anyhow!(
                                "{:?} node {} (id {}) has unmapped Data input 1 {:?}",
                                plan.kind,
                                n.name,
                                n_id,
                                n.inputs[1]
                            )
                        })?
                        .clone(),
                ),
                binop::BinOpInput::Const { .. } => None,
            };
            mil_ops.extend(binop::emit_binop_mil(
                &plan,
                &mut blob,
                in_a_name.as_deref(),
                in_b_name.as_deref(),
                &output_name,
            )?);
            continue;
        }
        if let cast::CastAnalysis::Translatable(plan) = cast::analyse_cast(model, n)? {
            let in_name = name_for
                .get(&n.inputs[0])
                .ok_or_else(|| {
                    anyhow!(
                        "Cast node {} (id {}) has unmapped data input {:?}",
                        n.name,
                        n_id,
                        n.inputs[0]
                    )
                })?
                .clone();
            mil_ops.extend(cast::emit_cast_mil(&plan, &mut blob, &in_name, &output_name)?);
            continue;
        }
        if let concat::ConcatAnalysis::Translatable(plan) = concat::analyse_concat(model, n)? {
            // For each tract input slot: if the plan classified it as a
            // runtime Data input, look up the upstream-emitted MIL name and
            // pass it through; if it's a Const, pass `None` and the emit
            // writes the const blob inline. Same pattern as BinOp.
            let in_owned: Vec<Option<String>> = n
                .inputs
                .iter()
                .enumerate()
                .map(|(slot, outlet)| match &plan.inputs[slot] {
                    concat::ConcatInput::Data { .. } => name_for
                        .get(outlet)
                        .cloned()
                        .map(Some)
                        .ok_or_else(|| {
                            anyhow!(
                                "Concat node {} (id {}) has unmapped input slot {slot} = {outlet:?}",
                                n.name,
                                n_id
                            )
                        }),
                    concat::ConcatInput::Const { .. } => Ok(None),
                })
                .collect::<Result<_>>()?;
            let in_refs: Vec<Option<&str>> = in_owned.iter().map(|o| o.as_deref()).collect();
            mil_ops.extend(concat::emit_concat_mil(&plan, &mut blob, &in_refs, &output_name)?);
            continue;
        }
        if let maxpool::MaxPoolAnalysis::Translatable(plan) = maxpool::analyse_maxpool(model, n)? {
            let in_name = name_for
                .get(&n.inputs[0])
                .ok_or_else(|| {
                    anyhow!(
                        "MaxPool node {} (id {}) has unmapped data input {:?}",
                        n.name,
                        n_id,
                        n.inputs[0]
                    )
                })?
                .clone();
            mil_ops.extend(maxpool::emit_maxpool_mil(&plan, &mut blob, &in_name, &output_name)?);
            continue;
        }
        if let reduce::ReduceAnalysis::Translatable(plan) = reduce::analyse_reduce(model, n)? {
            let in_name = name_for
                .get(&n.inputs[0])
                .ok_or_else(|| {
                    anyhow!(
                        "Reduce node {} (id {}) has unmapped data input {:?}",
                        n.name,
                        n_id,
                        n.inputs[0]
                    )
                })?
                .clone();
            mil_ops.extend(reduce::emit_reduce_mil(&plan, &mut blob, &in_name, &output_name)?);
            continue;
        }
        if let rm_axis::RmAxisAnalysis::Translatable(plan) = rm_axis::analyse_rm_axis(model, n)? {
            let in_name = name_for
                .get(&n.inputs[0])
                .ok_or_else(|| {
                    anyhow!(
                        "RmAxis node {} (id {}) has unmapped data input {:?}",
                        n.name,
                        n_id,
                        n.inputs[0]
                    )
                })?
                .clone();
            mil_ops.extend(rm_axis::emit_rm_axis_mil(&plan, &mut blob, &in_name, &output_name)?);
            continue;
        }
        if let activation::ActivationAnalysis::Translatable(plan) =
            activation::analyse_activation(model, n)?
        {
            let in_name = name_for
                .get(&n.inputs[0])
                .ok_or_else(|| {
                    anyhow!(
                        "Activation node {} (id {}) has unmapped data input {:?}",
                        n.name,
                        n_id,
                        n.inputs[0]
                    )
                })?
                .clone();
            mil_ops.extend(activation::emit_activation_mil(
                &plan,
                &mut blob,
                &in_name,
                &output_name,
            )?);
            continue;
        }
        if let add_axis::AddAxisAnalysis::Translatable(plan) = add_axis::analyse_add_axis(model, n)?
        {
            let in_name = name_for
                .get(&n.inputs[0])
                .ok_or_else(|| {
                    anyhow!(
                        "AddAxis node {} (id {}) has unmapped data input {:?}",
                        n.name,
                        n_id,
                        n.inputs[0]
                    )
                })?
                .clone();
            mil_ops.extend(add_axis::emit_add_axis_mil(&plan, &mut blob, &in_name, &output_name)?);
            continue;
        }
        if let resize::ResizeAnalysis::Translatable(plan) = resize::analyse_resize(model, n)? {
            let in_name = name_for
                .get(&n.inputs[0])
                .ok_or_else(|| {
                    anyhow!(
                        "Resize node {} (id {}) has unmapped data input {:?}",
                        n.name,
                        n_id,
                        n.inputs[0]
                    )
                })?
                .clone();
            mil_ops.extend(resize::emit_resize_mil(&plan, &mut blob, &in_name, &output_name)?);
            continue;
        }
        if let slice::SliceAnalysis::Translatable(plan) = slice::analyse_slice(model, n)? {
            let in_name = name_for
                .get(&n.inputs[0])
                .ok_or_else(|| {
                    anyhow!(
                        "Slice node {} (id {}) has unmapped data input {:?}",
                        n.name,
                        n_id,
                        n.inputs[0]
                    )
                })?
                .clone();
            mil_ops.extend(slice::emit_slice_mil(&plan, &mut blob, &in_name, &output_name)?);
            continue;
        }
        if let matmul::MatMulAnalysis::Translatable(plan) = matmul::analyse_matmul(model, n)? {
            let in_name = name_for
                .get(&n.inputs[0])
                .ok_or_else(|| {
                    anyhow!(
                        "MatMul node {} (id {}) has unmapped data input {:?}",
                        n.name,
                        n_id,
                        n.inputs[0]
                    )
                })?
                .clone();
            mil_ops.extend(matmul::emit_matmul_mil(&plan, &mut blob, &in_name, &output_name)?);
            continue;
        }
        if let avgpool::AvgPoolAnalysis::Translatable(plan) = avgpool::analyse_sumpool(model, n)? {
            let in_name = name_for
                .get(&n.inputs[0])
                .ok_or_else(|| {
                    anyhow!(
                        "SumPool node {} (id {}) has unmapped data input {:?}",
                        n.name,
                        n_id,
                        n.inputs[0]
                    )
                })?
                .clone();
            mil_ops.extend(avgpool::emit_sumpool_mil(&plan, &mut blob, &in_name, &output_name)?);
            continue;
        }
        if let broadcast::BroadcastAnalysis::Translatable(plan) =
            broadcast::analyse_broadcast(model, n)?
        {
            let in_name = name_for
                .get(&n.inputs[0])
                .ok_or_else(|| {
                    anyhow!(
                        "MultiBroadcastTo node {} (id {}) has unmapped data input {:?}",
                        n.name,
                        n_id,
                        n.inputs[0]
                    )
                })?
                .clone();
            mil_ops.extend(broadcast::emit_broadcast_mil(
                &plan,
                &mut blob,
                &in_name,
                &output_name,
            )?);
            continue;
        }
        if let softmax::SoftmaxAnalysis::Translatable(plan) = softmax::analyse_softmax(model, n)? {
            let in_name = name_for
                .get(&n.inputs[0])
                .ok_or_else(|| {
                    anyhow!(
                        "Softmax node {} (id {}) has unmapped data input {:?}",
                        n.name,
                        n_id,
                        n.inputs[0]
                    )
                })?
                .clone();
            mil_ops.extend(softmax::emit_softmax_mil(&plan, &mut blob, &in_name, &output_name)?);
            continue;
        }
        if let rms_norm::RmsNormAnalysis::Translatable(plan) = rms_norm::analyse_rms_norm(model, n)?
        {
            let in_name = name_for
                .get(&n.inputs[0])
                .ok_or_else(|| {
                    anyhow!(
                        "RmsNorm node {} (id {}) has unmapped data input {:?}",
                        n.name,
                        n_id,
                        n.inputs[0]
                    )
                })?
                .clone();
            mil_ops.extend(rms_norm::emit_rms_norm_mil(&plan, &mut blob, &in_name, &output_name)?);
            continue;
        }
        if let reshape::ReshapeAnalysis::Translatable(plan) = reshape::analyse_reshape(model, n)? {
            let in_name = name_for
                .get(&n.inputs[0])
                .ok_or_else(|| {
                    anyhow!(
                        "Reshape node {} (id {}) has unmapped data input {:?}",
                        n.name,
                        n_id,
                        n.inputs[0]
                    )
                })?
                .clone();
            mil_ops.extend(reshape::emit_reshape_mil(&plan, &mut blob, &in_name, &output_name)?);
            continue;
        }
        if let move_axis::MoveAxisAnalysis::Translatable(plan) =
            move_axis::analyse_move_axis(model, n)?
        {
            let in_name = name_for
                .get(&n.inputs[0])
                .ok_or_else(|| {
                    anyhow!(
                        "MoveAxis node {} (id {}) has unmapped data input {:?}",
                        n.name,
                        n_id,
                        n.inputs[0]
                    )
                })?
                .clone();
            mil_ops.extend(move_axis::emit_move_axis_mil(
                &plan,
                &mut blob,
                &in_name,
                &output_name,
            )?);
            continue;
        }
        if let pad::PadAnalysis::Translatable(plan) = pad::analyse_pad(model, n)? {
            let in_name = name_for
                .get(&n.inputs[0])
                .ok_or_else(|| {
                    anyhow!(
                        "Pad node {} (id {}) has unmapped data input {:?}",
                        n.name,
                        n_id,
                        n.inputs[0]
                    )
                })?
                .clone();
            mil_ops.extend(pad::emit_pad_mil(&plan, &mut blob, &in_name, &output_name)?);
            continue;
        }
        if let general_matmul::GeneralMatMulAnalysis::Translatable(plan) =
            general_matmul::analyse_general_matmul(model, n)?
        {
            let a_name = match plan.a {
                general_matmul::MatMulOperand::Runtime { .. } => Some(
                    name_for
                        .get(&n.inputs[0])
                        .ok_or_else(|| {
                            anyhow!(
                                "GeneralMatMul node {} (id {}) has unmapped Runtime input 0 {:?}",
                                n.name,
                                n_id,
                                n.inputs[0]
                            )
                        })?
                        .clone(),
                ),
                general_matmul::MatMulOperand::Const { .. } => None,
            };
            let b_name = match plan.b {
                general_matmul::MatMulOperand::Runtime { .. } => Some(
                    name_for
                        .get(&n.inputs[1])
                        .ok_or_else(|| {
                            anyhow!(
                                "GeneralMatMul node {} (id {}) has unmapped Runtime input 1 {:?}",
                                n.name,
                                n_id,
                                n.inputs[1]
                            )
                        })?
                        .clone(),
                ),
                general_matmul::MatMulOperand::Const { .. } => None,
            };
            mil_ops.extend(general_matmul::emit_general_matmul_mil(
                &plan,
                &mut blob,
                a_name.as_deref(),
                b_name.as_deref(),
                &output_name,
            )?);
            continue;
        }
        if let gather::GatherAnalysis::Translatable(plan) = gather::analyse_gather(model, n)? {
            // Two slots: data (slot 0) + indices (slot 1). For Const slots
            // pass `None`; emit_gather_mil writes the const blob inline.
            let data_name = match &plan.data {
                gather::GatherInput::Data { .. } => Some(
                    name_for
                        .get(&n.inputs[0])
                        .ok_or_else(|| {
                            anyhow!(
                                "Gather node {} (id {}) has unmapped Data input 0 {:?}",
                                n.name,
                                n_id,
                                n.inputs[0]
                            )
                        })?
                        .clone(),
                ),
                gather::GatherInput::Const { .. } => None,
            };
            let indices_name = match &plan.indices {
                gather::GatherInput::Data { .. } => Some(
                    name_for
                        .get(&n.inputs[1])
                        .ok_or_else(|| {
                            anyhow!(
                                "Gather node {} (id {}) has unmapped Data input 1 {:?}",
                                n.name,
                                n_id,
                                n.inputs[1]
                            )
                        })?
                        .clone(),
                ),
                gather::GatherInput::Const { .. } => None,
            };
            mil_ops.extend(gather::emit_gather_mil(
                &plan,
                &mut blob,
                data_name.as_deref(),
                indices_name.as_deref(),
                &output_name,
            )?);
            continue;
        }
        if let iff::IffAnalysis::Translatable(plan) = iff::analyse_iff(model, n)? {
            // Three slots: cond + t + f. For Const slots pass `None`;
            // emit_iff_mil writes the const blob/immediate inline.
            let cond_name = match &plan.cond {
                iff::IffInput::Data { .. } => Some(
                    name_for
                        .get(&n.inputs[0])
                        .ok_or_else(|| {
                            anyhow!(
                                "Iff node {} (id {}) has unmapped Data input 0 (cond) {:?}",
                                n.name,
                                n_id,
                                n.inputs[0]
                            )
                        })?
                        .clone(),
                ),
                iff::IffInput::Const { .. } => None,
            };
            let t_name = match &plan.t {
                iff::IffInput::Data { .. } => Some(
                    name_for
                        .get(&n.inputs[1])
                        .ok_or_else(|| {
                            anyhow!(
                                "Iff node {} (id {}) has unmapped Data input 1 (t) {:?}",
                                n.name,
                                n_id,
                                n.inputs[1]
                            )
                        })?
                        .clone(),
                ),
                iff::IffInput::Const { .. } => None,
            };
            let f_name = match &plan.f {
                iff::IffInput::Data { .. } => Some(
                    name_for
                        .get(&n.inputs[2])
                        .ok_or_else(|| {
                            anyhow!(
                                "Iff node {} (id {}) has unmapped Data input 2 (f) {:?}",
                                n.name,
                                n_id,
                                n.inputs[2]
                            )
                        })?
                        .clone(),
                ),
                iff::IffInput::Const { .. } => None,
            };
            mil_ops.extend(iff::emit_iff_mil(
                &plan,
                &mut blob,
                cond_name.as_deref(),
                t_name.as_deref(),
                f_name.as_deref(),
                &output_name,
            )?);
            continue;
        }
        if let scatter_nd::ScatterNdAnalysis::Translatable(plan) =
            scatter_nd::analyse_scatter_nd(model, n)?
        {
            // Three slots: data + indices + updates. Any may be Const.
            let resolve =
                |slot: usize, kind: &scatter_nd::ScatterNdInput| -> Result<Option<String>> {
                    match kind {
                    scatter_nd::ScatterNdInput::Data { .. } => Ok(Some(
                        name_for
                            .get(&n.inputs[slot])
                            .ok_or_else(|| {
                                anyhow!(
                                    "ScatterNd node {} (id {}) has unmapped Data input {slot} {:?}",
                                    n.name,
                                    n_id,
                                    n.inputs[slot]
                                )
                            })?
                            .clone(),
                    )),
                    scatter_nd::ScatterNdInput::Const { .. } => Ok(None),
                }
                };
            let data_name = resolve(0, &plan.data)?;
            let indices_name = resolve(1, &plan.indices)?;
            let updates_name = resolve(2, &plan.updates)?;
            mil_ops.extend(scatter_nd::emit_scatter_nd_mil(
                &plan,
                &mut blob,
                data_name.as_deref(),
                indices_name.as_deref(),
                updates_name.as_deref(),
                &output_name,
            )?);
            continue;
        }
        if let einsum_outer::EinsumOuterAnalysis::Translatable(plan) =
            einsum_outer::analyse_einsum_outer(model, n)?
        {
            let a_name = match &plan.a {
                einsum_outer::EinsumOuterOperand::Runtime { .. } => Some(
                    name_for
                        .get(&n.inputs[0])
                        .ok_or_else(|| {
                            anyhow!(
                                "EinsumOuter node {} (id {}) has unmapped Runtime input 0 {:?}",
                                n.name,
                                n_id,
                                n.inputs[0]
                            )
                        })?
                        .clone(),
                ),
                einsum_outer::EinsumOuterOperand::Const { .. } => None,
            };
            let b_name = match &plan.b {
                einsum_outer::EinsumOuterOperand::Runtime { .. } => Some(
                    name_for
                        .get(&n.inputs[1])
                        .ok_or_else(|| {
                            anyhow!(
                                "EinsumOuter node {} (id {}) has unmapped Runtime input 1 {:?}",
                                n.name,
                                n_id,
                                n.inputs[1]
                            )
                        })?
                        .clone(),
                ),
                einsum_outer::EinsumOuterOperand::Const { .. } => None,
            };
            mil_ops.extend(einsum_outer::emit_einsum_outer_mil(
                &plan,
                &mut blob,
                a_name.as_deref(),
                b_name.as_deref(),
                &output_name,
            )?);
            continue;
        }
        bail!(
            "subgraph contains node {} (id {}) flagged as translatable but no per-op \
             emitter handled it (dispatch table out of sync with is_translatable?)",
            n.name,
            n_id
        );
    }

    // 3. Resolve external_outputs to their MIL names; collect output facts +
    //    coreml_output_shapes so the transform can build the CoremlOp.
    let mut output_names: Vec<String> = Vec::with_capacity(subgraph.external_outputs.len());
    let mut coreml_output_shapes: Vec<Vec<usize>> =
        Vec::with_capacity(subgraph.external_outputs.len());
    let mut output_facts: TVec<TypedFact> = tvec![];
    for outlet in &subgraph.external_outputs {
        let name = name_for
            .get(outlet)
            .ok_or_else(|| anyhow!("external output {outlet:?} has no MIL name"))?
            .clone();
        let n = &model.nodes[outlet.node];
        let (ml_shape_i64, output_fact) = output_shape_and_fact(model, n)?;
        let ml_shape: Vec<usize> = ml_shape_i64.iter().map(|&s| s as usize).collect();
        coreml_output_shapes.push(ml_shape);
        output_facts.push(output_fact);
        output_names.push(name);
    }

    // 4. Build the MIL Program + Model proto.
    let program = single_function_program(mil_inputs, output_names.clone(), mil_ops);

    // The Model description needs concrete I/O shapes. Use the MLPackage shapes
    // (the rank-4 ones) for both inputs and outputs.
    let in_shapes_i64: Vec<Vec<i64>> =
        coreml_input_shapes.iter().map(|s| s.iter().map(|&x| x as i64).collect()).collect();
    let out_shapes_i64: Vec<Vec<i64>> =
        coreml_output_shapes.iter().map(|s| s.iter().map(|&x| x as i64).collect()).collect();
    let model_proto = build_model(
        program,
        &input_names,
        &in_shapes_i64,
        &coreml_input_dtypes,
        &output_names,
        &out_shapes_i64,
    );

    let model_bytes = model_proto.encode_to_vec();
    let weight_bytes = blob.finish();
    mlpackage::write(out_path, &model_bytes, &weight_bytes)?;

    Ok(SubgraphIO {
        input_names,
        coreml_input_shapes,
        coreml_input_dtypes,
        source_outlets: subgraph.external_inputs.clone(),
        output_names,
        coreml_output_shapes,
        output_facts,
        source_outputs: subgraph.external_outputs.clone(),
    })
}

/// Pure-in-memory variant of [`build_subgraph_mlpackage`]: returns the
/// `(model.mlmodel, weight.bin)` byte pair *plus* the I/O wiring info,
/// without writing anything to disk. The persistent compile cache uses this
/// to compute the cache key from the bytes before deciding whether to write
/// the `.mlpackage` to disk and call `compileModelAtURL`.
pub fn build_subgraph_artifacts(
    model: &TypedModel,
    subgraph: &Subgraph,
) -> Result<(SubgraphIO, Vec<u8>, Vec<u8>)> {
    // Reuse the disk-writing path through a temp dir, then read the bytes
    // back. Building a clean in-memory variant would require duplicating the
    // walk logic; the temp-write-then-read shortcut avoids that and is fast
    // because the bytes are small (single-digit MB at most for our targets).
    //
    // The temp dir is removed at the end; the cache layer writes the canonical
    // copy elsewhere on a miss.
    let tmp = std::env::temp_dir().join(format!(
        "tract-coreml-artifacts-{}-{}.mlpackage",
        std::process::id(),
        uuid::Uuid::new_v4().simple()
    ));
    let io = build_subgraph_mlpackage(&tmp, model, subgraph)?;
    let model_bytes = std::fs::read(tmp.join("Data/com.apple.CoreML/model.mlmodel"))?;
    let weight_bytes = std::fs::read(tmp.join("Data/com.apple.CoreML/weights/weight.bin"))?;
    let _ = std::fs::remove_dir_all(&tmp);
    Ok((io, model_bytes, weight_bytes))
}

/// Build a `Model` proto wrapping a multi-input multi-output MIL Program.
fn build_model(
    program: mil::Program,
    input_names: &[String],
    in_shapes: &[Vec<i64>],
    in_dtypes: &[DatumType],
    output_names: &[String],
    out_shapes: &[Vec<i64>],
) -> spec::Model {
    // Map a tract DatumType to the corresponding MLPackage feature dtype.
    // Float16 is the default for floats (rest of the pipeline is FP16);
    // Int32 covers index inputs (Gather etc.) — including I64 inputs that
    // CoremlOp::eval narrows to I32 at predict time.
    fn array_data_type(dt: DatumType) -> i32 {
        use spec::array_feature_type::ArrayDataType as A;
        match dt {
            DatumType::F16 => A::Float16 as i32,
            DatumType::F32 => A::Float32 as i32,
            DatumType::I32 => A::Int32 as i32,
            // I64 isn't a native MLMultiArray dtype — eval-side cast brings
            // it down to I32 at the boundary (see CoremlOp::eval).
            DatumType::I64 => A::Int32 as i32,
            other => panic!("build_model: unsupported feature dtype {other:?}"),
        }
    }

    let inputs: Vec<spec::FeatureDescription> = input_names
        .iter()
        .zip(in_shapes)
        .zip(in_dtypes)
        .map(|((name, shape), dtype)| spec::FeatureDescription {
            name: name.clone(),
            short_description: String::new(),
            r#type: Some(spec::FeatureType {
                is_optional: false,
                r#type: Some(spec::feature_type::Type::MultiArrayType(spec::ArrayFeatureType {
                    shape: shape.clone(),
                    data_type: array_data_type(*dtype),
                    shape_flexibility: None,
                    default_optional_value: None,
                })),
            }),
        })
        .collect();
    let outputs: Vec<spec::FeatureDescription> = output_names
        .iter()
        .zip(out_shapes)
        .map(|(name, shape)| spec::FeatureDescription {
            name: name.clone(),
            short_description: String::new(),
            r#type: Some(spec::FeatureType {
                is_optional: false,
                r#type: Some(spec::feature_type::Type::MultiArrayType(spec::ArrayFeatureType {
                    shape: shape.clone(),
                    data_type: spec::array_feature_type::ArrayDataType::Float16 as i32,
                    shape_flexibility: None,
                    default_optional_value: None,
                })),
            }),
        })
        .collect();

    let description = spec::ModelDescription {
        input: inputs,
        output: outputs,
        functions: vec![],
        default_function_name: String::new(),
        predicted_feature_name: String::new(),
        predicted_probabilities_name: String::new(),
        training_input: vec![],
        metadata: None,
        state: vec![],
    };
    spec::Model {
        specification_version: 8,
        description: Some(description),
        is_updatable: false,
        r#type: Some(spec::model::Type::MlProgram(program)),
    }
}
