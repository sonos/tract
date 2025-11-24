use crate::ops::{CudaAxisOp, CudaFusedAxisOp};
use tract_core::internal::*;
use tract_core::tract_data::itertools::Itertools;
use tract_gpu::fact::DeviceTypedFactExt;
use tract_gpu::rule_ensure;

fn is_supported_axis_op(op: &CudaAxisOp) -> bool {
    matches!(op.0, AxisOp::Add(_) | AxisOp::Rm(_) | AxisOp::Reshape(..))
}

// these are operators that can handle arbitrarty strides
fn can_fuse_move(model: &TypedModel, axis_node: &TypedNode) -> bool {
    // List of operators that supports non-contiguous inputs
    model.single_succ(axis_node.id).unwrap().is_some_and(|node| {
        node.op_is::<crate::ops::CudaConcat>()
            || node.op_is::<crate::ops::CudaApplyRope>()
            || node.op_is::<crate::ops::CudaScaledMaskedSoftmax>()
            || node.op_is::<crate::ops::CudaSlice>()
            || node.op_is::<crate::ops::CudaMultiBroadcastTo>()
            || node.op_is::<crate::ops::CudaDynKVCache>()
            || node.op_is::<crate::ops::CudaGgmlQuantQ81>()
            || node.op_is::<crate::ops::CudaPad>()
            || node.op_is::<crate::ops::CudaDelay>()
            || node.op_is::<crate::ops::CudaPulsePad>()
    })
}

pub fn collect_chain_of_axis_ops<'a>(
    model: &'a TypedModel,
    mut cursor: &'a TypedNode,
) -> TractResult<Option<(TVec<CudaAxisOp>, &'a TypedNode)>> {
    let mut acc_axis_ops = tvec![];
    let mut head_of_chain = cursor;

    while let Some(axis_op) = cursor.op_as::<CudaAxisOp>().filter(|o| {
        is_supported_axis_op(o) || (matches!(o.0, AxisOp::Move(..)) && can_fuse_move(model, cursor))
    }) {
        acc_axis_ops.push(axis_op.clone());
        head_of_chain = cursor;

        if let Some(prev) = model.single_prec(cursor.id)? {
            cursor = prev;
        } else {
            break;
        }
    }

    Ok(if acc_axis_ops.is_empty() {
        None
    } else {
        Some((acc_axis_ops.into_iter().rev().collect(), head_of_chain))
    })
}

pub fn fuse_axis_op(
    _ctx: &(),
    model: &TypedModel,
    axis_node: &TypedNode,
    _axis_node_name: &str,
    axis_op: &CudaAxisOp,
) -> TractResult<Option<TypedModelPatch>> {
    // Only support certain axis ops (or a Move, which is handled specially below)
    rule_ensure!(is_supported_axis_op(axis_op) || matches!(axis_op.0, AxisOp::Move(..)));

    let Some(node) = model.single_succ(axis_node.id)? else {
        return Ok(None);
    };

    // Disallow fusing when the successor is already an axis/fused op or a sync,
    // *unless* it's a Move AxisOp (we allow that via the early-quit branch).
    let is_axis_like = node.op_is::<CudaAxisOp>() || node.op_is::<CudaFusedAxisOp>();
    let is_allowed_move =
        node.op_as::<CudaAxisOp>().is_some_and(|op| matches!(op.0, AxisOp::Move(..)));

    rule_ensure!(!is_axis_like || is_allowed_move);

    let node_name = &node.name;

    let Some(in_nodes) = model.all_prec(node.id)? else {
        return Ok(None);
    };

    let mut grouped_axis_ops: TVec<TVec<CudaAxisOp>> = tvec![];
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

    // If the successor is a Move, we may fuse it now or defer.
    if let Some(op) = node.op_as::<crate::ops::CudaAxisOp>() {
        if matches!(op.0, AxisOp::Move(..)) {
            let should_defer_move = !grouped_axis_ops[0].is_empty() && !can_fuse_move(model, node);
            if should_defer_move {
                let out = patch.wire_node(
                    format!("{node_name}.fused_axis_op"),
                    CudaFusedAxisOp { grouped_axis_ops, op: Box::new(op.clone()) },
                    &tap_inputs,
                )?;
                patch.shunt_outside(model, node.id.into(), out[0])?;
                return Ok(Some(patch));
            } else {
                // Nothing to do right now; we’ll fuse on a later pass.
                return Ok(None);
            }
        }
    }

    // General case: fuse using the successor's op.
    let out = patch.wire_node(
        format!("{node_name}.fused_axis_op"),
        CudaFusedAxisOp { grouped_axis_ops, op: node.op.clone() },
        &tap_inputs,
    )?;
    patch.shunt_outside(model, node.id.into(), out[0])?;
    Ok(Some(patch))
}

pub fn fuse_move_axis(
    _ctx: &(),
    model: &TypedModel,
    axis_node: &TypedNode,
    axis_node_name: &str,
    axis_op: &CudaAxisOp,
) -> TractResult<Option<TypedModelPatch>> {
    rule_ensure!(matches!(axis_op.0, AxisOp::Move(..)));

    let in_fact = model.node_input_facts(axis_node.id)?[0];
    let in_shape =
        in_fact.as_device_fact().map(|mf| mf.shape.clone()).unwrap_or(in_fact.shape.clone());

    let out_fact = model.node_output_facts(axis_node.id)?[0];
    let out_shape =
        out_fact.as_device_fact().map(|mf| mf.shape.clone()).unwrap_or(out_fact.shape.clone());

    // Checks if MoveAxis has no impact on shape + layout
    if in_shape == out_shape {
        if let (Some(in_strides), AxisOp::Move(from, to)) =
            (in_shape.as_concrete().map(Tensor::natural_strides), axis_op.0.clone())
        {
            let mut out_strides = in_strides.clone();
            let remove_stride = out_strides.remove(from);
            out_strides.insert(to, remove_stride);
            if in_strides == out_strides {
                return TypedModelPatch::shunt_one_op(model, axis_node);
            }
        }
    }

    // Reshape are always fusable. Change Move by Reshape if possible
    let simpl_op = CudaAxisOp::simplify_axis_op(axis_op.0.clone(), in_shape.dims());
    if simpl_op != *axis_op {
        return Ok(Some(TypedModelPatch::replace_single_op(
            model,
            axis_node,
            &[axis_node.inputs[0]],
            simpl_op,
        )?));
    }

    // Fuse consecutive MoveAxis if possible
    let Some(cursor) = model.single_succ(axis_node.id)? else { return Ok(None) };
    if let (AxisOp::Move(from_1, to_1), AxisOp::Move(from_2, to_2)) = (
        axis_op.0.clone(),
        cursor.op_as::<CudaAxisOp>().map(|ax_op| ax_op.0.clone()).unwrap_or(AxisOp::Add(0)),
    ) {
        let max_rank = [from_1, from_2, to_1, to_2].iter().max().unwrap() + 1;
        let mut perm: TVec<usize> = (0..max_rank).collect_vec().into();

        AxisOp::Move(from_1, to_1).change_shape_array(&mut perm, false)?;
        AxisOp::Move(from_2, to_2).change_shape_array(&mut perm, false)?;
        let new_axis_ops = perm_to_ops(&perm);
        if new_axis_ops.len() == 1 {
            let mut patch = TypedModelPatch::default();
            let inputs = patch.taps(model, &axis_node.inputs)?;
            let out = patch.wire_node(
                format!("{axis_node_name}.fused_move_axis"),
                CudaAxisOp(new_axis_ops[0].clone()),
                &inputs,
            )?;
            patch.shunt_outside(model, cursor.id.into(), out[0])?;
            return Ok(Some(patch));
        }
    }

    // Add(x) -> Move(x, y)
    let Some(cursor) = model.single_prec(axis_node.id)? else { return Ok(None) };
    if let (AxisOp::Move(from_1, to_1), AxisOp::Add(ax)) = (
        axis_op.0.clone(),
        cursor.op_as::<CudaAxisOp>().map(|ax_op| ax_op.0.clone()).unwrap_or(AxisOp::Rm(0)),
    ) {
        if ax == from_1 {
            let mut patch = TypedModelPatch::default();
            let inputs = patch.taps(model, &cursor.inputs)?;
            let out =
                patch.wire_node(cursor.name.clone(), CudaAxisOp(AxisOp::Add(to_1)), &inputs)?;
            patch.shunt_outside(model, axis_node.id.into(), out[0])?;
            return Ok(Some(patch));
        }
    }
    Ok(None)
}
