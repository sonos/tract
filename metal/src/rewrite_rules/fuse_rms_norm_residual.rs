use tract_core::internal::*;
use tract_core::ops::binary::TypedBinOp;
use tract_core::ops::math::Add;
use tract_gpu::fact::DeviceTypedFactExt;
use tract_gpu::ops::binary::GpuBinOp;
use tract_gpu::ops::rms_norm::GpuRmsNorm;
use tract_gpu::rule_ensure;

/// Fold a standalone elementwise Add feeding a RmsNorm into the norm kernel
/// (`GpuRmsNorm::has_residual`): the pre-norm residual add of transformer
/// blocks otherwise costs one singleton dispatch per layer. The fused kernel
/// computes the sum in the input dtype (bit-identical to the Add dispatch)
/// and returns it as a second output, so every other consumer of the add
/// (the residual stream) rewires to that output.
///
/// Only same-shape, same-dtype adds qualify (no broadcast in the kernel).
pub fn fuse_rms_norm_residual(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    node_name: &str,
    op: &GpuRmsNorm,
) -> TractResult<Option<TypedModelPatch>> {
    rule_ensure!(!op.has_residual);
    rule_ensure!(op.backend_name == "Metal");
    rule_ensure!(std::env::var_os("TRACT_METAL_DISABLE_RMS_NORM_RESIDUAL").is_none());
    let add_outlet = node.inputs[0];
    let add_node = model.node(add_outlet.node);
    let is_add = add_node
        .op_as::<GpuBinOp>()
        .map(|bin| bin.mini_op.is::<Add>())
        .or_else(|| add_node.op_as::<TypedBinOp>().map(|bin| bin.0.is::<Add>()))
        .unwrap_or(false);
    rule_ensure!(is_add);
    let facts = model.node_input_facts(add_node.id)?;
    rule_ensure!(facts.len() == 2);
    let (a, b) = (facts[0], facts[1]);
    let (Some(da), Some(db)) = (a.as_device_fact(), b.as_device_fact()) else {
        return Ok(None);
    };
    rule_ensure!(da.shape == db.shape);
    rule_ensure!(da.datum_type == db.datum_type);
    rule_ensure!(matches!(da.datum_type, DatumType::F16 | DatumType::F32));
    // The kernel reads both operands with the norm input's natural layout.
    let Some(dnorm) = model.node_input_facts(node.id)?[0].as_device_fact() else {
        return Ok(None);
    };
    rule_ensure!(da.datum_type == dnorm.datum_type);

    let mut patch = TypedModelPatch::new(format!("fuse residual add into {node_name}"));
    let mut inputs = tvec![
        patch.tap_model(model, add_node.inputs[0])?,
        patch.tap_model(model, add_node.inputs[1])?,
    ];
    for input in &node.inputs[1..] {
        inputs.push(patch.tap_model(model, *input)?);
    }
    let fused = GpuRmsNorm { has_residual: true, ..op.clone() };
    let out = patch.wire_node(format!("{node_name}.residual"), fused, &inputs)?;
    patch.shunt_outside(model, node.id.into(), out[0])?;
    patch.shunt_outside(model, add_outlet, out[1])?;
    Ok(Some(patch))
}
