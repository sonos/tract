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

    let mut axis_op_chain: TVec<&AxisOp> = tvec![];
    let mut running_node = reshape_node;
    while let Some(node) = next_node(model, running_node).filter(|n| n.op_is::<AxisOp>()) {
        axis_op_chain.push(node.op_as::<AxisOp>().unwrap());
        running_node = node;
    }

    rule_ensure!(next_node(model, running_node).is_some_and(|node| node.op_is::<BasicMatMul>()));

    if let Some(mm_node) = next_node(model, running_node) {
        let mut new_mm_input = inputs[0];
        if (axis_op_chain == tvec![&AxisOp::Move(0, 1)] && reshape_out_shape[0] == TDim::Val(1))
            || axis_op_chain == tvec![&AxisOp::Rm(0), &AxisOp::Add(1)]
        {
            new_mm_input =
                patch.wire_node(format!("{node_name}.reshape"), AxisOp::Move(0, 1), &inputs)?[0];
        }

        // Only Ggml kernels have internal broadcasting. If no kernel impl is specified,
        // Ggml kernels will be used if and only activations are F32
        let in_facts = model.node_input_facts(mm_node.id)?;
        if ctx.gemm_impl != Some(MetalGemmImplKind::Ggml)
            && (ctx.gemm_impl.is_some() || in_facts[0].datum_type != DatumType::F32)
        {
            return Ok(None);
        }

        let mut matmul_inputs = patch.taps(model, &mm_node.inputs)?;
        for (idx, o) in mm_node.inputs.iter().enumerate() {
            if o.node == running_node.id {
                matmul_inputs[idx] = new_mm_input;
            }
        }

        let out = patch.wire_node(&mm_node.name, &mm_node.op, &matmul_inputs)?;
        patch.shunt_outside(model, mm_node.id.into(), out[0])?;
        return Ok(Some(patch));
    }
    Ok(None)
}
