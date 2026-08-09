use tract_core::internal::*;

use crate::ops::change_axes::GpuAxisOp;
use crate::ops::fused_view_copy::{GpuFusedViewCopy, ViewStep};
use crate::ops::slice::GpuSlice;
use crate::rule_ensure;

fn view_step(node: &TypedNode) -> Option<ViewStep> {
    if let Some(op) = node.op_as::<GpuSlice>() {
        return Some(ViewStep::Slice(op.inner.clone()));
    }
    if let Some(op) = node.op_as::<GpuAxisOp>() {
        return Some(ViewStep::Axis(op.inner.clone()));
    }
    None
}

/// True when `node` feeds exactly one other layout op, which will fold it
/// into its own chain.
fn absorbed_by_consumer(model: &TypedModel, node: &TypedNode) -> bool {
    let outlet = OutletId::new(node.id, 0);
    if model.outputs.contains(&outlet) {
        return false;
    }
    let succs = model.outlet_successors(outlet);
    succs.len() == 1 && view_step(&model.nodes()[succs[0].node]).is_some()
}

/// Collapses a maximal chain of layout ops (GpuSlice / GpuAxisOp) ending at
/// `node` into one `GpuFusedViewCopy` (a single strided copy). Chains of
/// pure Add/Rm/Reshape are left alone unless they cannot be folded for free
/// into their consumer: `fuse_axis_op` handles those without any copy when
/// the consumer is a single-output device op.
pub fn fuse_view_copy_chain_at(
    model: &TypedModel,
    node: &TypedNode,
) -> TractResult<Option<TypedModelPatch>> {
    rule_ensure!(std::env::var_os("TRACT_GPU_DISABLE_FUSED_VIEW_COPY").is_none());
    rule_ensure!(view_step(node).is_some());
    rule_ensure!(node.outputs.len() == 1);
    rule_ensure!(!absorbed_by_consumer(model, node));

    let mut steps: TVec<ViewStep> = tvec![view_step(node).unwrap()];
    let mut head = node;
    while let Some(prev) = model.single_prec(head.id)? {
        if prev.outputs.len() != 1
            || model.outputs.contains(&OutletId::new(prev.id, 0))
            || model.outlet_successors(OutletId::new(prev.id, 0)).len() != 1
        {
            break;
        }
        let Some(step) = view_step(prev) else { break };
        steps.push(step);
        head = prev;
    }
    rule_ensure!(steps.len() >= 2);
    steps.reverse();

    // A chain of pure Add/Rm/Reshape is free once folded into a
    // single-output consumer; only take it when it contains a real copy
    // (slice or move) or when no consumer can fold it.
    let has_real_copy = steps.iter().any(|s| {
        matches!(s, ViewStep::Slice(_)) || matches!(s, ViewStep::Axis(AxisOp::Move(..)))
    });
    if !has_real_copy {
        let outlet = OutletId::new(node.id, 0);
        let succs = model.outlet_successors(outlet);
        let foldable = succs.len() == 1
            && model.nodes()[succs[0].node].outputs.len() == 1
            && !model.outputs.contains(&outlet);
        rule_ensure!(!foldable);
    }

    let mut patch = TypedModelPatch::default();
    let input = patch.tap_model(model, head.inputs[0])?;
    let out = patch.wire_node(
        format!("{}.fused_view_copy", node.name),
        GpuFusedViewCopy { steps },
        &[input],
    )?;
    patch.shunt_outside(model, node.id.into(), out[0])?;
    Ok(Some(patch))
}

pub fn fuse_view_copy_slice(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    _name: &str,
    _op: &GpuSlice,
) -> TractResult<Option<TypedModelPatch>> {
    fuse_view_copy_chain_at(model, node)
}

pub fn fuse_view_copy_axis(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    _name: &str,
    _op: &GpuAxisOp,
) -> TractResult<Option<TypedModelPatch>> {
    fuse_view_copy_chain_at(model, node)
}
