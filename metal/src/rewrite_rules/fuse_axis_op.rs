use crate::kernels::matmul::{MfaGemm, MlxGemm, MpsMatMul};
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

#[macro_export]
macro_rules! dispatch_metal_op {
    ($node: expr, $body:expr, $($op:path),+,) => {
        $(
            if let Some(op) = $node.op_as::<$op>() {
                return $body(op.clone());
            }
        )*
    };
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
            Some((acc_axis_ops, head_of_chain)) => {
                grouped_axis_ops.push(acc_axis_ops);
                tap_inputs.push(patch.tap_model(model, head_of_chain.inputs[0])?);
            }
            None => {
                grouped_axis_ops.push(tvec![]);
                tap_inputs.push(patch.tap_model(model, node.inputs[in_idx])?);
            }
        }
    }

    // Handle all compatible ops.
    dispatch_metal_op!(
        node,
        |op| {
            let out = patch.wire_node(
                format!("{node_name}.fused_axis_op"),
                MetalFusedAxisOp { grouped_axis_ops, op },
                &tap_inputs,
            )?;
            patch.shunt_outside(model, node.id.into(), out[0])?;
            Ok(Some(patch))
        },
        crate::ops::MetalBinOp,
        crate::ops::MetalGemm<MlxGemm>,
        crate::ops::MetalGemm<MpsMatMul>,
        crate::ops::MetalGemm<MfaGemm>,
        crate::ops::MetalMultiBroadcastTo,
        crate::ops::MetalElementWiseOp,
        crate::ops::MetalRmsNorm,
        crate::ops::MetalSilu,
        crate::ops::MetalNewGelu,
        crate::ops::MetalSoftmax,
        crate::ops::MetalRotateHalf,
        crate::ops::MetalApplyRope,
        crate::ops::MetalReduce,
        crate::ops::MetalSlice,
        crate::ops::MetalConcat,
        crate::ops::MetalCast,
        crate::ops::MetalScaledMaskedSoftmax,
    );

    // Handle AxisOp::Move operator.
    if let Some(op) = node.op_as::<crate::ops::MetalAxisOp>() {
        rule_ensure!(matches!(op.0, AxisOp::Move(..)));
        let out = patch.wire_node(
            format!("{node_name}.fused_axis_op"),
            MetalFusedAxisOp { grouped_axis_ops, op: op.clone() },
            &tap_inputs,
        )?;
        patch.shunt_outside(model, node.id.into(), out[0])?;
        Ok(Some(patch))
    } else {
        Ok(None)
    }
}
