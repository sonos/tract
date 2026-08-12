use tract_core::internal::*;

use crate::fact::DeviceTypedFactExt;
use crate::ops::cast::GpuCast;

/// Bypasses a float precision round-trip at the device level:
/// `GpuCast(GpuCast(x, narrow), wide)` where `x` is already `wide`. These
/// pairs appear when one lowering casts an activation down (e.g. the MoE
/// lowering restoring the layer's declared f16 dtype) and the next lowering
/// casts it right back up. Skipping the intermediate rounding only gains
/// precision and removes one or two kernel dispatches per occurrence (the
/// narrow cast dies too when unused). Restricted to SYNTHETIC downcasts
/// (inserted by our own lowering): a downcast converted from the source
/// graph carries model semantics and must round.
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
    rule_if!(down_op.synthetic);
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tensor::DeviceTensor;

    fn noop_dispatch(_i: &DeviceTensor, _o: &DeviceTensor) -> TractResult<()> {
        Ok(())
    }

    fn any_dt(_dt: DatumType) -> bool {
        true
    }

    fn cast(to: DatumType, synthetic: bool) -> GpuCast {
        let c = GpuCast::new(to, "Test", noop_dispatch, any_dt).unwrap();
        if synthetic { c.into_synthetic() } else { c }
    }

    fn roundtrip_patch(down_synthetic: bool) -> TractResult<Option<TypedModelPatch>> {
        let mut model = TypedModel::default();
        let x = model.add_source("x", f32::datum_type().fact([4, 4]))?;
        let down = model.wire_node("down", cast(DatumType::F16, down_synthetic), &[x])?[0];
        let up = model.wire_node("up", cast(DatumType::F32, true), &[down])?[0];
        model.select_output_outlets(&[up])?;
        let node = model.node(up.node);
        let op = node.op_as::<GpuCast>().unwrap().clone();
        bypass_device_downcast_roundtrip(&(), &model, node, "up", &op)
    }

    #[test]
    fn bypasses_lowering_inserted_downcast_roundtrip() -> TractResult<()> {
        assert!(roundtrip_patch(true)?.is_some());
        Ok(())
    }

    /// A downcast converted from the source graph carries model semantics:
    /// the rule must leave its rounding in place.
    #[test]
    fn keeps_source_graph_downcast_roundtrip() -> TractResult<()> {
        assert!(roundtrip_patch(false)?.is_none());
        Ok(())
    }
}
