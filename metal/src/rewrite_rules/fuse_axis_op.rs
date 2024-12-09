use crate::kernels::matmul::MlxGemm;
use crate::ops::{MetalAxisOp, MetalFusedAxisOp};
use crate::rewrite_rules::{next_node, previous_node, previous_nodes};
use crate::rule_ensure;
use tract_core::internal::*;

fn is_suppored_axis_op(op: &MetalAxisOp) -> bool {
    matches!(op.0, AxisOp::Add(_) | AxisOp::Rm(_) | AxisOp::Reshape(..))
}

pub fn collect_chain_of_axis_ops<'a>(
    model: &'a TypedModel,
    node: &'a TypedNode,
) -> TractResult<Option<(TVec<MetalAxisOp>, &'a TypedNode)>> {
    let mut cursor = node;
    let mut head_of_chain = node;
    let mut acc_axis_ops = tvec![];
    loop {
        let Some(axis_op) = cursor.op_as::<MetalAxisOp>().filter(|o| is_suppored_axis_op(o)) else {
            break;
        };

        head_of_chain = cursor;

        let Some(prev_node) = previous_node(model, cursor) else {
            break;
        };

        acc_axis_ops.push(axis_op.clone());
        cursor = prev_node;
    }

    if acc_axis_ops.is_empty() {
        Ok(None)
    } else {
        Ok(Some((acc_axis_ops.into_iter().rev().collect(), head_of_chain)))
    }
}

pub fn fuse_axis_op(
    _ctx: &(),
    model: &TypedModel,
    axis_node: &TypedNode,
    _axis_node_name: &str,
    axis_op: &MetalAxisOp,
) -> TractResult<Option<TypedModelPatch>> {
    rule_ensure!(is_suppored_axis_op(axis_op));

    let Some(node) = next_node(model, axis_node) else { return Ok(None) };

    let node_name = &node.name;

    let in_nodes = previous_nodes(model, node);

    let mut grouped_axis_ops = tvec![];
    let mut tap_inputs = tvec![];

    let mut patch = TypedModelPatch::default();

    for (in_idx, in_node) in in_nodes.into_iter().enumerate() {
        match collect_chain_of_axis_ops(model, in_node)? {
            Some((acc_axis_ops, in_node)) => {
                grouped_axis_ops.push(acc_axis_ops);
                tap_inputs.push(patch.tap_model(model, in_node.inputs[0])?);
            }
            None => {
                grouped_axis_ops.push(tvec![]);
                tap_inputs.push(patch.tap_model(model, node.inputs[in_idx])?);
            }
        }
    }

    let out = if let Some(op) = node.op_as::<crate::ops::MetalBinOp>() {
        patch.wire_node(
            format!("{node_name}.fused_axis_op"),
            MetalFusedAxisOp { grouped_axis_ops, op: op.clone() },
            &tap_inputs,
        )?
    } else if let Some(op) = node.op_as::<crate::ops::MetalGemm<MlxGemm>>() {
        patch.wire_node(
            format!("{node_name}.fused_axis_op"),
            MetalFusedAxisOp { grouped_axis_ops, op: op.clone() },
            &tap_inputs,
        )?
    } else if let Some(op) = node.op_as::<crate::ops::MetalMultiBroadcastTo>() {
        patch.wire_node(
            format!("{node_name}.fused_axis_op"),
            MetalFusedAxisOp { grouped_axis_ops, op: op.clone() },
            &tap_inputs,
        )?
    } else if let Some(op) = node.op_as::<crate::ops::MetalElementWiseOp>() {
        patch.wire_node(
            format!("{node_name}.fused_axis_op"),
            MetalFusedAxisOp { grouped_axis_ops, op: op.clone() },
            &tap_inputs,
        )?
    } else if let Some(op) = node.op_as::<crate::ops::MetalRmsNorm>() {
        patch.wire_node(
            format!("{node_name}.fused_axis_op"),
            MetalFusedAxisOp { grouped_axis_ops, op: op.clone() },
            &tap_inputs,
        )?
    } else if let Some(op) = node.op_as::<crate::ops::MetalSilu>() {
        patch.wire_node(
            format!("{node_name}.fused_axis_op"),
            MetalFusedAxisOp { grouped_axis_ops, op: op.clone() },
            &tap_inputs,
        )?
    } else if let Some(op) = node.op_as::<crate::ops::MetalNewGelu>() {
        patch.wire_node(
            format!("{node_name}.fused_axis_op"),
            MetalFusedAxisOp { grouped_axis_ops, op: op.clone() },
            &tap_inputs,
        )?
    } else if let Some(op) = node.op_as::<crate::ops::MetalSoftmax>() {
        patch.wire_node(
            format!("{node_name}.fused_axis_op"),
            MetalFusedAxisOp { grouped_axis_ops, op: op.clone() },
            &tap_inputs,
        )?
    } else if let Some(op) = node.op_as::<crate::ops::MetalAxisOp>() {
        rule_ensure!(matches!(op.0, AxisOp::Move(..)));
        patch.wire_node(
            format!("{node_name}.fused_axis_op"),
            MetalFusedAxisOp { grouped_axis_ops, op: op.clone() },
            &tap_inputs,
        )?
    } else {
        return Ok(None);
    };

    patch.shunt_outside(model, node.id.into(), out[0])?;

    Ok(Some(patch))
}
