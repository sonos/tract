use crate::ops::{MetalSync, MetalSyncKind};
use crate::rewrite_rules::{next_node, previous_node};
use crate::rule_ensure;
use crate::tensor::MetalTensorExt;
use tract_core::internal::*;
use tract_core::ops::konst::Const;
use tract_core::tract_linalg::block_quant::BlockQuantValue;

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
    let Some(sync_cpu_prec) = previous_node(model, node) else {
        return Ok(None);
    };
    let Some(sync_cpu_prec_op) = sync_cpu_prec.op_as::<MetalSync>() else {
        return Ok(None);
    };
    rule_ensure!(sync_cpu_prec_op.kind == MetalSyncKind::ToCpu);

    let patch =
        TypedModelPatch::rewire(model, &sync_cpu_prec.inputs, &[node.id.into()], &|_p, xs| {
            Ok(xs.into())
        })?;
    Ok(Some(patch))
}

pub fn rewire_metal_sync_after_const(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    node_name: &str,
    op: &Const,
) -> TractResult<Option<TypedModelPatch>> {
    // Search pattern => Const => ToCPU

    let Some(gpu_const) = op.val().as_metal_tensor() else {
        return Ok(None);
    };
    let cpu_const = gpu_const.to_cpu()?;

    // Identify successor ToCpu
    let Some(sync_cpu) = next_node(model, node) else {
        return Ok(None);
    };
    let Some(sync_cpu_op) = sync_cpu.op_as::<MetalSync>() else {
        return Ok(None);
    };
    rule_ensure!(sync_cpu_op.kind == MetalSyncKind::ToCpu);

    let mut opaque_fact: Option<Box<dyn OpaqueFact>> = None;
    if let Some(of) = cpu_const
        .to_scalar::<Opaque>()
        .ok()
        .and_then(|od| od.downcast_ref::<BlockQuantValue>())
        .map(|bqv| bqv.fact.clone())
    {
        opaque_fact = Some(Box::new(of));
    }

    let mut patch = TypedModelPatch::default();
    let out = patch.wire_node(
        node_name.to_string(),
        Const::new_with_opt_opaque_fact(cpu_const, opaque_fact)?,
        &[],
    )?;
    patch.shunt_outside(model, sync_cpu.id.into(), out[0])?;
    Ok(Some(patch))
}
