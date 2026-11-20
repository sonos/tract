use tract_core::internal::*;
use tract_core::ops::konst::Const;
use tract_gpu::fact::DeviceTypedFactExt;
use tract_gpu::ops::binary::GpuBinOp;
use tract_gpu::ops::element_wise::GpuElementWise;
use tract_gpu::ops::slice::GpuSlice;
use tract_gpu::rule_ensure;

use crate::kernels::bin_ops::BINARY_OPS;
use crate::kernels::element_wise::ELEMENT_WISE_OPS;
use crate::ops::conv::WgpuConv;

/// The elementwise op a slice feeds: `slot` and `other` describe a binary op's
/// second operand, both `None` for a unary one.
struct Tail {
    node: usize,
    name: String,
    slot: Option<usize>,
    other: Option<OutletId>,
}

fn fact_len(model: &TypedModel, outlet: OutletId) -> TractResult<TDim> {
    let fact = model.outlet_fact(outlet)?;
    let shape: TVec<TDim> = fact
        .as_device_fact()
        .map(|f| f.shape.dims().into())
        .unwrap_or_else(|| fact.shape.dims().into());
    Ok(shape.iter().product::<TDim>())
}

/// The same operand outlet, or two distinct constants holding the same tensor.
fn same_operand(model: &TypedModel, a: OutletId, b: OutletId) -> TractResult<bool> {
    if a == b {
        return Ok(true);
    }
    let (Some(ka), Some(kb)) =
        (model.node(a.node).op_as::<Const>(), model.node(b.node).op_as::<Const>())
    else {
        return Ok(false);
    };
    Ok(ka.val() == kb.val())
}

/// The elementwise op a slice's single consumer applies to it, if the consumer
/// is a Wgpu binary or unary op over a scalar broadcast.
fn tail_of(model: &TypedModel, slice: &TypedNode) -> TractResult<Option<Tail>> {
    let Some(succ) = model.single_succ(slice.id)? else { return Ok(None) };
    if let Some(op) = succ.op_as::<GpuBinOp>() {
        rule_ensure!(op.backend_name == "Wgpu");
        let name = op.mini_op.name().to_lowercase();
        rule_ensure!(BINARY_OPS.contains(&name.as_str()));
        rule_ensure!(succ.inputs.len() == 2);
        let slot = succ.inputs.iter().position(|i| i.node == slice.id).unwrap();
        let other = succ.inputs[1 - slot];
        rule_ensure!(fact_len(model, other)? == 1.to_dim());
        return Ok(Some(Tail { node: succ.id, name, slot: Some(slot), other: Some(other) }));
    }
    if let Some(op) = succ.op_as::<GpuElementWise>() {
        rule_ensure!(op.backend_name == "Wgpu");
        let name = op.mini_op.name().to_lowercase();
        rule_ensure!(ELEMENT_WISE_OPS.contains(&name.as_str()));
        rule_ensure!(succ.inputs.len() == 1 && succ.inputs[0].node == slice.id);
        return Ok(Some(Tail { node: succ.id, name, slot: None, other: None }));
    }
    Ok(None)
}

/// A conv whose output splits into channel slices that all feed the same scalar
/// elementwise op can apply it to the whole output first: the slices then read
/// the already-activated tensor, and the op rides the conv's epilogue instead of
/// running once per half.
pub fn lift_elementwise_above_slices(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    node_name: &str,
    _op: &WgpuConv,
) -> TractResult<Option<TypedModelPatch>> {
    rule_ensure!(node.outputs.len() == 1);
    let conv_out = OutletId::new(node.id, 0);
    rule_ensure!(!model.outputs.contains(&conv_out));
    rule_ensure!(node.outputs[0].successors.len() >= 2);

    let mut axis = None;
    let mut slices = vec![];
    for inlet in node.outputs[0].successors.iter() {
        let slice = model.node(inlet.node);
        let Some(op) = slice.op_as::<GpuSlice>() else { return Ok(None) };
        rule_ensure!(slice.inputs.len() == 1 && slice.inputs[0] == conv_out);
        rule_ensure!(axis.is_none_or(|a| a == op.inner.axis));
        axis = Some(op.inner.axis);
        rule_ensure!(!model.outputs.contains(&OutletId::new(slice.id, 0)));
        slices.push(slice);
    }

    let mut tails = vec![];
    for slice in &slices {
        let Some(tail) = tail_of(model, slice)? else { return Ok(None) };
        tails.push(tail);
    }
    let first = &tails[0];
    for tail in &tails[1..] {
        rule_ensure!(tail.name == first.name && tail.slot == first.slot);
        if let (Some(_), Some(_)) = (first.slot, tail.slot) {
            rule_ensure!(same_operand(model, first.other.unwrap(), tail.other.unwrap())?)
        }
    }

    let mut patch = TypedModelPatch::default();
    let conv_tap = patch.tap_model(model, conv_out)?;
    let lifted_inputs = match first.slot {
        Some(slot) => {
            let scalar = patch.tap_model(model, first.other.unwrap())?;
            if slot == 0 { vec![conv_tap, scalar] } else { vec![scalar, conv_tap] }
        }
        None => vec![conv_tap],
    };
    let lifted = patch.wire_node(
        format!("{node_name}.lifted"),
        model.node(first.node).op.clone(),
        &lifted_inputs,
    )?;
    for (slice, tail) in slices.iter().zip(tails.iter()) {
        let new_slice =
            patch.wire_node(format!("{}.lifted", slice.name), slice.op.clone(), &[lifted[0]])?;
        patch.shunt_outside(model, OutletId::new(tail.node, 0), new_slice[0])?;
    }
    Ok(Some(patch))
}
