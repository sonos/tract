use tract_core::internal::*;
use tract_gpu::ops::concat::GpuConcat;
use tract_gpu::ops::slice::GpuSlice;
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

/// Write a decoder resize and its same-size trailing channel groups into the
/// final destination. Retain the original interpolation tables and arithmetic.
pub fn fuse_decoder_resize_output(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    name: &str,
    resize: &WgpuResize2d,
) -> TractResult<Option<TypedModelPatch>> {
    rule_ensure!(
        resize.concat_inputs
            && resize.append_inputs == 0
            && resize.window_h == 2
            && resize.window_w == 2
    );
    let [1, channels, height, width] = resize.output_shape.as_slice() else { return Ok(None) };
    rule_ensure!(matches!(
        (*channels, *height, *width),
        (128, 46, 80) | (80, 90, 160) | (40, 180, 320) | (32, 360, 640)
    ));
    rule_ensure!(model.outlet_fact(node.inputs[0])?.datum_type == DatumType::F16);
    let mut terminal = node;
    let out = model.output_outlets()?;
    let mut h = *height;
    if *height == 46
        && !out.contains(&terminal.id.into())
        && let Some(crop) = model.single_succ(terminal.id)?
        && crop.op_as::<GpuSlice>().is_some_and(|s| {
            s.inner.axis == 2 && s.inner.start == 0.to_dim() && s.inner.end == 45.to_dim()
        })
    {
        h = 45;
        terminal = crop;
    }
    rule_ensure!(!out.contains(&terminal.id.into()));
    let Some(concat) = model.single_succ(terminal.id)? else { return Ok(None) };
    rule_ensure!(
        concat.op_as::<GpuConcat>().is_some_and(|op| op.axis == 1)
            && matches!(concat.inputs.len(), 2 | 3)
            && concat.inputs[0] == OutletId::from(terminal.id)
    );
    let append_inputs = concat.inputs.len() - 1;
    let mut extra = 0;
    for &input in &concat.inputs[1..] {
        let f = model.outlet_fact(input)?;
        let Some([1, n, ih, iw]) = f.shape.as_concrete() else { return Ok(None) };
        rule_ensure!(f.datum_type == DatumType::F16 && *n > 0 && *ih == h && iw == width);
        extra += n;
    }
    rule_ensure!(matches!((h, extra), (45, 43) | (90, 27) | (180, 19) | (360, 3)));
    let mut patch = TypedModelPatch::default();
    let inputs = node
        .inputs
        .iter()
        .copied()
        .chain(concat.inputs.iter().copied().skip(1))
        .map(|o| patch.tap_model(model, o))
        .collect::<TractResult<TVec<_>>>()?;
    let op = WgpuResize2d {
        output_shape: tvec![1, channels + extra, h, *width],
        append_inputs,
        ..resize.clone()
    };
    let output = patch.wire_node(format!("{name}.decoder-resize"), op, &inputs)?[0];
    patch.shunt_outside(model, concat.id.into(), output)?;
    Ok(Some(patch))
}
