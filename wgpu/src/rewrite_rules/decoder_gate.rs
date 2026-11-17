use tract_core::internal::*;
use tract_core::ops::math::Mul;
use tract_core::ops::nn::Sigmoid;
use tract_gpu::fact::DeviceTypedFactExt;
use tract_gpu::ops::binary::GpuBinOp;
use tract_gpu::ops::concat::GpuConcat;
use tract_gpu::ops::element_wise::GpuElementWise;
use tract_gpu::ops::slice::GpuSlice;
use tract_gpu::rule_ensure;

use crate::ops::conv::WgpuConv;
use crate::ops::decoder_gate::WgpuDecoderSigmoid;
use crate::ops::decoder_gate::WgpuPackedDecoderGate;

fn f16_channels(model: &TypedModel, outlet: OutletId) -> TractResult<Option<[usize; 3]>> {
    let fact = model.outlet_fact(outlet)?;
    let Some(device) = fact.as_device_fact() else { return Ok(None) };
    if fact.datum_type != DatumType::F16
        || !device.is_from_device()
        || device.is_state_owned()
        || device.fact.is_exotic()
    {
        return Ok(None);
    }
    let Some([1, c, h, w]) = fact.shape.as_concrete() else { return Ok(None) };
    Ok((*c > 0 && *h > 0 && *w > 0).then_some([*c, *h, *w]))
}

fn is_channel_slice(node: &TypedNode, src: OutletId, start: usize, end: usize) -> bool {
    node.inputs.as_slice() == [src]
        && node.op_as::<GpuSlice>().is_some_and(|op| {
            op.inner.axis == 1 && op.inner.start == start.to_dim() && op.inner.end == end.to_dim()
        })
}

fn is_wgpu_sigmoid(node: &TypedNode) -> bool {
    node.op_as::<GpuElementWise>()
        .is_some_and(|op| op.backend_name == "Wgpu" && op.mini_op.is::<Sigmoid>())
}

fn split_channels(model: &TypedModel, source: OutletId, half: usize) -> Option<[&TypedNode; 2]> {
    let [a, b] = model.outlet_successors(source) else { return None };
    let (a, b) = (model.node(a.node), model.node(b.node));
    let end = half.checked_mul(2)?;
    if is_channel_slice(a, source, 0, half) && is_channel_slice(b, source, half, end) {
        Some([a, b])
    } else if is_channel_slice(b, source, 0, half) && is_channel_slice(a, source, half, end) {
        Some([b, a])
    } else {
        None
    }
}

/// Fuses a split FP16 decoder gate, its reset product, and the following
/// feature concat. Both branches share one sigmoid on the rounded logits.
pub fn fuse_decoder_gate(
    shader_f16: &bool,
    model: &TypedModel,
    node: &TypedNode,
    name: &str,
    concat: &GpuConcat,
) -> TractResult<Option<TypedModelPatch>> {
    rule_ensure!(*shader_f16 && concat.axis == 1);
    let [feature, gated] = node.inputs.as_slice() else { return Ok(None) };
    let mul = model.node(gated.node);
    rule_ensure!(
        *gated == mul.id.into()
            && mul
                .op_as::<GpuBinOp>()
                .is_some_and(|op| op.backend_name == "Wgpu" && op.mini_op.is::<Mul>())
    );
    let [reset, previous] = mul.inputs.as_slice() else { return Ok(None) };
    let reset_sigmoid = model.node(reset.node);
    let Some(sigmoid) = reset_sigmoid.op_as::<GpuElementWise>() else { return Ok(None) };
    rule_ensure!(is_wgpu_sigmoid(reset_sigmoid) && *reset == reset_sigmoid.id.into());
    let [lower] = reset_sigmoid.inputs.as_slice() else { return Ok(None) };
    let lower_slice = model.node(lower.node);
    let [logits] = lower_slice.inputs.as_slice() else { return Ok(None) };
    let Some(shape @ [c, h, w]) = f16_channels(model, *feature)? else { return Ok(None) };
    let Some(full) = c.checked_mul(2) else { return Ok(None) };
    rule_ensure!(
        f16_channels(model, *previous)? == Some(shape)
            && f16_channels(model, *logits)? == Some([full, h, w])
            && f16_channels(model, *reset)? == Some(shape)
            && f16_channels(model, *gated)? == Some(shape)
    );
    let Some(len) = c.checked_mul(h).and_then(|n| n.checked_mul(w)) else { return Ok(None) };
    rule_ensure!(len.is_multiple_of(4) && len <= u32::MAX as usize / 2);
    let Some([lo, hi]) = split_channels(model, *logits, c) else { return Ok(None) };
    rule_ensure!(*lower == lo.id.into());
    let Some(update_sigmoid) = model.single_succ(hi.id)? else { return Ok(None) };
    rule_ensure!(
        is_wgpu_sigmoid(update_sigmoid) && update_sigmoid.inputs.as_slice() == [hi.id.into()]
    );
    let producer = model.node(logits.node);
    // The shared sigmoid needs packed logits, unlike the original slice copies.
    rule_ensure!(
        producer.op_is::<WgpuConv>()
            || producer.op_is::<GpuSlice>()
            || producer.op_as::<GpuBinOp>().is_some_and(|op| op.backend_name == "Wgpu")
            || producer.op_as::<GpuElementWise>().is_some_and(|op| op.backend_name == "Wgpu")
    );
    let outputs = model.output_outlets()?;
    rule_ensure!(
        [*lower, hi.id.into(), *reset, *gated].iter().all(|o| !outputs.contains(o))
            && (outputs.contains(&update_sigmoid.id.into())
                || !model.outlet_successors(update_sigmoid.id.into()).is_empty())
    );
    rule_ensure!(
        model.single_succ(lo.id)?.is_some_and(|n| n.id == reset_sigmoid.id)
            && model.single_succ(reset_sigmoid.id)?.is_some_and(|n| n.id == mul.id)
            && model.single_succ(mul.id)?.is_some_and(|n| n.id == node.id)
    );
    let mut patch = TypedModelPatch::default();
    let x = patch.tap_model(model, *logits)?;
    let shared = patch.wire_node(
        format!("{}.before-channel-split", reset_sigmoid.name),
        WgpuDecoderSigmoid(sigmoid.clone()),
        &[x],
    )?[0];
    let update = patch.wire_node(update_sigmoid.name.clone(), hi.op.clone(), &[shared])?[0];
    let feature = patch.tap_model(model, *feature)?;
    let previous = patch.tap_model(model, *previous)?;
    let out = patch.wire_node(
        format!("{name}.packed-reset-gate"),
        WgpuPackedDecoderGate,
        &[feature, shared, previous],
    )?[0];
    patch.shunt_outside(model, update_sigmoid.id.into(), update)?;
    patch.shunt_outside(model, node.id.into(), out)?;
    Ok(Some(patch))
}
