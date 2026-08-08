use tract_core::internal::*;

use crate::fact::DeviceTypedFactExt;
use crate::ops::cast::GpuCast;

/// Bypasses a float precision round-trip at the device level:
/// `GpuCast(GpuCast(x, narrow), wide)` where `x` is already `wide`. These
/// pairs appear when a backend conversion (e.g. the MoE lowering) casts an
/// activation back up right after the source graph cast it down. Skipping the
/// intermediate rounding only gains precision and removes one or two kernel
/// dispatches per occurrence (the narrow cast dies too when unused).
pub fn bypass_device_downcast_roundtrip(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    _node_name: &str,
    op: &GpuCast,
) -> TractResult<Option<TypedModelPatch>> {
    rule_if!(op.to.is_float());
    let prec = &model.nodes()[node.inputs[0].node];
    rule_if_some!(down_op = prec.op_as::<GpuCast>());
    rule_if!(down_op.to.is_float());
    rule_if!(op.to.size_of() > down_op.to.size_of());
    let src_outlet = prec.inputs[0];
    let src_fact = model.outlet_fact(src_outlet)?;
    let src_dt = src_fact
        .as_device_fact()
        .map(|f| f.fact.datum_type)
        .unwrap_or(src_fact.datum_type);
    rule_if!(src_dt.is_float());
    rule_if!(src_dt == op.to);

    let mut patch = TypedModelPatch::default();
    let src = patch.tap_model(model, src_outlet)?;
    patch.shunt_outside(model, node.id.into(), src)?;
    Ok(Some(patch))
}
