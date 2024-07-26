use crate::ops::{MetalSync, MetalSyncKind};
use crate::rewrite_rules::previous_node;
use crate::rule_ensure;
use tract_core::internal::*;

pub fn rewire_metal_sync(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    _node_name: &str,
    op: &MetalSync,
) -> TractResult<Option<TypedModelPatch>> {
    // Search pattern => ToCPU => ToGPU

    rule_ensure!(op.kind == MetalSyncKind::ToGpu);

    // Identify precessor ToCpu
    let Some(sync_cpu_prec) = previous_node(model, node) else { return Ok(None) };
    let Some(sync_cpu_prec_op) = sync_cpu_prec.op_as::<MetalSync>() else { return Ok(None) };
    rule_ensure!(sync_cpu_prec_op.kind == MetalSyncKind::ToCpu);

    let patch =
        TypedModelPatch::rewire(model, &sync_cpu_prec.inputs, &[node.id.into()], &|_p, xs| {
            Ok(xs.into())
        })?;
    Ok(Some(patch))
}
