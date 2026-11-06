use tract_core::internal::*;
use tract_gpu::ops::concat::GpuConcat;
use tract_gpu::rule_ensure;

use crate::kernels::resize::RESIZE_2D_WG;
use crate::ops::resize2d::WgpuResize2d;

/// Bilinear resize reads independent channels, so it can read both inputs of a
/// channel concat directly.
pub fn fuse_concat_resize(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    name: &str,
    resize: &WgpuResize2d,
) -> TractResult<Option<TypedModelPatch>> {
    rule_ensure!(!resize.concat_inputs && resize.window_h == 2 && resize.window_w == 2);
    let concat = model.node(node.inputs[0].node);
    let Some(op) = concat.op_as::<GpuConcat>() else { return Ok(None) };
    rule_ensure!(op.axis == 1 && concat.inputs.len() == 2);
    rule_ensure!(model.single_succ(concat.id)?.is_some_and(|succ| succ.id == node.id));
    rule_ensure!(!model.output_outlets()?.contains(&node.inputs[0]));
    let [1, c, h, w] = resize.output_shape.as_slice() else { return Ok(None) };
    let [wx, wy, wz] = RESIZE_2D_WG.map(|n| n as usize);
    rule_ensure!(w.div_ceil(wx) <= 65535 && h.div_ceil(wy) <= 65535 && c.div_ceil(wz) <= 65535);
    for &outlet in &concat.inputs {
        let fact = model.outlet_fact(outlet)?;
        rule_ensure!(fact.datum_type == DatumType::F16);
        let Some([1, c, h, w]) = fact.shape.as_concrete() else { return Ok(None) };
        rule_ensure!(*c != 0 && *h != 0 && *w != 0);
    }

    let mut patch = TypedModelPatch::default();
    let inputs = std::iter::once(concat.inputs[0])
        .chain(node.inputs.iter().copied().skip(1))
        .chain(std::iter::once(concat.inputs[1]))
        .map(|outlet| patch.tap_model(model, outlet))
        .collect::<TractResult<Vec<_>>>()?;
    let op = WgpuResize2d { concat_inputs: true, ..resize.clone() };
    let output = patch.wire_node(format!("{name}.concat"), op, &inputs)?[0];
    patch.shunt_outside(model, node.id.into(), output)?;
    Ok(Some(patch))
}
