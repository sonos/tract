use crate::kernels::matmul::{GgmlGemm, MfaGemm, MlxGemm};
use crate::ops::{MetalAxisOp, MetalFusedAxisOp};
use crate::rewrite_rules::{next_node, previous_node, previous_nodes};
use crate::rule_ensure;
use tract_core::internal::*;

fn is_supported_axis_op(op: &MetalAxisOp) -> bool {
    matches!(op.0, AxisOp::Add(_) | AxisOp::Rm(_) | AxisOp::Reshape(..) | AxisOp::Move(..))
}

pub fn collect_chain_of_axis_ops<'a>(
    model: &'a TypedModel,
    node: &'a TypedNode,
) -> TractResult<Option<(TVec<MetalAxisOp>, &'a TypedNode)>> {
    let mut cursor = node;
    let mut head_of_chain = node;
    let mut acc_axis_ops = tvec![];
    loop {
        let Some(axis_op) = cursor.op_as::<MetalAxisOp>().filter(|o| is_supported_axis_op(o)) else {
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
    rule_ensure!(is_supported_axis_op(axis_op));

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
        crate::ops::MetalGemm<MfaGemm>,
        crate::ops::MetalGemm<GgmlGemm>,
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

fn can_fuse_with_move(op: &MetalAxisOp) -> bool {
    matches!(op.0, AxisOp::Add(_) | AxisOp::Rm(_))
}

pub fn fuse_move_axis(
    _ctx: &(),
    model: &TypedModel,
    axis_node: &TypedNode,
    axis_node_name: &str,
    axis_op: &MetalAxisOp,
) -> TractResult<Option<TypedModelPatch>> {
    rule_ensure!(matches!(axis_op.0, AxisOp::Move(..)) && !axis_op.1);
    
    let Some(mut cursor) = next_node(model, axis_node) else { return Ok(None) };

    // Fuse consecutive MoveAxis if possible 
    if let (AxisOp::Move(from_1, to_1), AxisOp::Move(from_2, to_2)) = (axis_op.0.clone(),
        cursor.op_as::<MetalAxisOp>().map(|ax_op| ax_op.0.clone()).unwrap_or(AxisOp::Add(0))) {
        let mut patch = TypedModelPatch::default();

        let new_op = if to_1 == to_2 {
            MetalAxisOp(AxisOp::Move(from_1, from_2), false)
        }
        else if to_1 == from_2 {
            MetalAxisOp(AxisOp::Move(from_1, to_2), false)
        }
        else {
            println!("Can't fuse MoveAxis {:?} because of other MoveAxis {:?}", axis_op.0, cursor.op());
            return Ok(None)
        };

        let input = patch.taps(model, &axis_node.inputs)?;
        let out = patch.wire_node(
            format!("{axis_node_name}.fused_move_axis"), new_op, &input)?;
        patch.shunt_outside(model, cursor.id.into(), out[0])?;
        return Ok(Some(patch))
    }

    let mut prev_node = axis_node;
    loop {
        let Some(_) = cursor.op_as::<MetalAxisOp>().filter(|o| can_fuse_with_move(o)) else {
            break;
        };
        let Some(node) = next_node(model, cursor) else { break; };
        prev_node = cursor;
        cursor = node;
    }

    if cursor.op_is::<crate::ops::MetalGemm<MlxGemm>>() || 
        cursor.op_is::<crate::ops::MetalGemm<MfaGemm>>() ||
        cursor.op_is::<crate::ops::MetalSync>() || 
        cursor.op_is::<crate::ops::MetalElementWiseOp>() || 
        cursor.op_is::<crate::ops::MetalBinOp>() ||
        // Op reshaping to Dim3 
        cursor.op_is::<crate::ops::MetalSoftmax>() ||
        cursor.op_is::<crate::ops::MetalReduce>() ||
        cursor.op_is::<crate::ops::MetalRmsNorm>() ||
        cursor.op_is::<MetalAxisOp>() {
            println!("Can't fuse MoveAxis {:?} because of {:?}", axis_op.0, cursor.op());
            return Ok(None)
    }

    // GGML MM only supports discontiguous data on activations
    if cursor.op_is::<crate::ops::MetalGemm<GgmlGemm>>() && 
    (model.node_output_facts(prev_node.id)?[0] == model.node_input_facts(cursor.id)?[1]){
        println!("GGML MoveAxis is {:?}. With input: {:?}", axis_op, model.node_input_facts(axis_node.id)?);
        println!("GGML inputs: {:?}", model.node_input_facts(cursor.id)?);
        return Ok(None)
    }

    let new_op = MetalAxisOp(axis_op.0.clone(), true);
    let patch = TypedModelPatch::replace_single_op(model, axis_node, &axis_node.inputs, new_op)?;
    Ok(Some(patch))
}