use tract_core::internal::*;
use tract_gpu::ops::concat::GpuConcat;
use tract_gpu::ops::slice::GpuSlice;
use tract_gpu::rule_ensure;

use crate::kernels::resize::RESIZE_2D_WG;
use crate::ops::resize2d::OutputConcat;
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
        .chain(node.inputs[1..5].iter().copied())
        .chain(std::iter::once(concat.inputs[1]))
        .chain(node.inputs[5..].iter().copied())
        .map(|outlet| patch.tap_model(model, outlet))
        .collect::<TractResult<Vec<_>>>()?;
    let op = WgpuResize2d { concat_inputs: true, ..resize.clone() };
    let output = patch.wire_node(format!("{name}.concat"), op, &inputs)?[0];
    patch.shunt_outside(model, node.id.into(), output)?;
    Ok(Some(patch))
}

/// A resize whose result only feeds a concat along an axis ahead of the two
/// resized ones, with unit axes before it, writes its slab of the concat output
/// in place and copies the other concat inputs beside it. Slices from 0 on the
/// resized axes between the two are absorbed: the resize computes fewer rows or
/// columns with its original plans. Composes with [`fuse_concat_resize`].
pub fn fuse_resize_concat_output(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    name: &str,
    resize: &WgpuResize2d,
) -> TractResult<Option<TypedModelPatch>> {
    rule_ensure!(resize.output_concat.is_none());
    let rank = resize.output_shape.len();
    let outputs = model.output_outlets()?;
    let mut terminal = node;
    let concat = loop {
        rule_ensure!(!outputs.contains(&terminal.id.into()));
        let Some(succ) = model.single_succ(terminal.id)? else { return Ok(None) };
        let Some(slice) = succ.op_as::<GpuSlice>() else { break succ };
        rule_ensure!(slice.inner.axis >= rank - 2 && slice.inner.start == 0.to_dim());
        terminal = succ;
    };
    let Some(op) = concat.op_as::<GpuConcat>() else { return Ok(None) };
    let Some(shape) = model.outlet_fact(concat.id.into())?.shape.as_concrete() else {
        return Ok(None);
    };
    rule_ensure!(op.axis < rank - 2 && shape[..op.axis].iter().all(|&d| d == 1));
    let Some(position) = concat.inputs.iter().position(|i| i.node == terminal.id) else {
        return Ok(None);
    };

    let mut patch = TypedModelPatch::default();
    let siblings = concat.inputs.iter().enumerate().filter(|(ix, _)| *ix != position);
    let inputs = node
        .inputs
        .iter()
        .chain(siblings.map(|(_, outlet)| outlet))
        .map(|&outlet| patch.tap_model(model, outlet))
        .collect::<TractResult<TVec<_>>>()?;
    let op = WgpuResize2d {
        output_shape: shape.into(),
        output_concat: Some(OutputConcat {
            axis: op.axis,
            position,
            siblings: concat.inputs.len() - 1,
        }),
        ..resize.clone()
    };
    let output = patch.wire_node(format!("{name}.into-concat"), op, &inputs)?[0];
    patch.shunt_outside(model, concat.id.into(), output)?;
    Ok(Some(patch))
}
