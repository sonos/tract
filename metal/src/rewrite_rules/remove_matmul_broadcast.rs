use crate::rewrite_rules::next_node;
use crate::{rule_ensure, MetalGemmImplKind, MetalTransform};
use tract_core::internal::*;
use tract_core::ops::array::MultiBroadcastTo;
use tract_core::ops::einsum::BasicMatMul;

use super::previous_node;

pub fn remove_ggml_broadcast_pre_matmul(
    ctx: &MetalTransform,
    model: &TypedModel,
    node: &TypedNode,
    node_name: &str,
    _op: &MultiBroadcastTo,
) -> TractResult<Option<TypedModelPatch>> {
    // Search Pattern: AddAxis(2) -> MultiBroadcastTo -> Reshape(1, (a, b), (a * b)) -> Move(0, 1) (optional) -> BasicMatmul
    let Some(add_axis_node) =
        previous_node(model, node).filter(|n| n.op_as::<AxisOp>() == Some(&AxisOp::Add(2)))
    else {
        return Ok(None);
    };
    rule_ensure!(add_axis_node.outputs[0].successors.len() == 1);
    rule_ensure!(node.outputs[0].successors.len() == 1);

    let node_out_shape = node.outputs[0].fact.shape.dims();
    let reshape_expected = AxisOp::Reshape(
        1,
        tvec![node_out_shape[1].clone(), node_out_shape[2].clone()],
        tvec![node_out_shape[1].clone() * node_out_shape[2].clone()],
    );

    let Some(reshape_node) =
        next_node(model, node).filter(|n| n.op_as::<AxisOp>() == Some(&reshape_expected))
    else {
        return Ok(None);
    };

    let mut patch = TypedModelPatch::default();
    let inputs = patch.taps(model, &add_axis_node.inputs)?;

    // Check if optional Move before matmul is present
    let reshape_out_shape = reshape_node.outputs[0].fact.shape.dims();
    let (mm_node, new_mm_input, prev_node_id) = match next_node(model, reshape_node) {
        Some(n) if n.op_is::<BasicMatMul>() => (n, inputs[0], reshape_node.id),
        Some(n)
            if n.op_as::<AxisOp>() == Some(&AxisOp::Move(0, 1))
                && reshape_out_shape[0] == TDim::Val(1)
                && next_node(model, n).is_some_and(|m| m.op_is::<BasicMatMul>()) =>
        {
            let swap_input =
                patch.wire_node(format!("{node_name}.reshape"), AxisOp::Move(0, 1), &inputs)?[0];
            (next_node(model, n).unwrap(), swap_input, n.id)
        }
        _ => return Ok(None),
    };

    // Only Ggml kernels have internal broadcasting
    let in_facts = model.node_input_facts(mm_node.id)?;
    match ctx.gemm_impl {
        Some(MetalGemmImplKind::Ggml) => {}
        None if in_facts[0].datum_type != DatumType::F32 => return Ok(None),
        _ => {}
    }

    let mut matmul_inputs = patch.taps(model, &mm_node.inputs)?;
    for (idx, o) in mm_node.inputs.iter().enumerate() {
        if o.node == prev_node_id {
            matmul_inputs[idx] = new_mm_input;
        }
    }

    let out = patch.wire_node(&mm_node.name, &mm_node.op, &matmul_inputs)?;
    patch.shunt_outside(model, mm_node.id.into(), out[0])?;
    Ok(Some(patch))
}
