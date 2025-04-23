use crate::rewrite_rules::{next_node, previous_node};
use crate::rule_ensure;
use crate::sync::{DeviceSync, DeviceSyncKind};
use crate::tensor::DeviceTensorExt;
use tract_core::internal::*;
use tract_core::ops::konst::Const;
use tract_core::tract_linalg::block_quant::BlockQuantValue;

pub fn rewire_syncs(model: &mut TypedModel) -> TractResult<()> {
    Rewriter::default()
        .with_rule_for("remove-back-and-forth-sync", rewire_back_and_forth_sync)
        .with_rule_for("remove-sync-after-const", rewire_sync_after_const)
        .rewrite(&(), model)
}

pub fn rewire_back_and_forth_sync(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    _node_name: &str,
    op: &DeviceSync,
) -> TractResult<Option<TypedModelPatch>> {
    // Search pattern => ToHost => ToDevice
    rule_ensure!(op.kind == DeviceSyncKind::ToDevice);

    // Identify precessor ToHost
    let Some(sync_to_host_prec) = previous_node(model, node) else {
        return Ok(None);
    };
    let Some(sync_to_host_prec_op) = sync_to_host_prec.op_as::<DeviceSync>() else {
        return Ok(None);
    };
    rule_ensure!(sync_to_host_prec_op.kind == DeviceSyncKind::ToHost);

    let patch =
        TypedModelPatch::rewire(model, &sync_to_host_prec.inputs, &[node.id.into()], &|_p, xs| {
            Ok(xs.into())
        })?;
    Ok(Some(patch))
}

pub fn rewire_sync_after_const(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    node_name: &str,
    op: &Const,
) -> TractResult<Option<TypedModelPatch>> {
    // Search pattern => Const => ToHost

    let Some(device_const) = op.val().as_device_tensor() else {
        return Ok(None);
    };
    let host_const = device_const.to_host()?;

    // Identify successor ToHost
    let Some(sync_to_host) = next_node(model, node) else {
        return Ok(None);
    };
    let Some(sync_to_host_op) = sync_to_host.op_as::<DeviceSync>() else {
        return Ok(None);
    };
    rule_ensure!(sync_to_host_op.kind == DeviceSyncKind::ToHost);

    let mut opaque_fact: Option<Box<dyn OpaqueFact>> = None;
    if let Some(of) = host_const
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
        Const::new_with_opt_opaque_fact(host_const, opaque_fact)?,
        &[],
    )?;
    patch.shunt_outside(model, sync_to_host.id.into(), out[0])?;
    Ok(Some(patch))
}
