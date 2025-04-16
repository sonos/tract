pub mod apply_rope;
pub mod gelu_approximate;
pub mod rms_norm;
pub mod scaled_masked_softmax;
pub mod silu;

use tract_core::internal::*;
use tract_core::ops::konst::Const;
use tract_nnef::tract_core;

//pub use apply_rope::{as_apply_rope_rule, ApplyRope};
pub use apply_rope::{as_apply_rope_rule, as_rotate_half_rule};
pub use gelu_approximate::as_gelu_approx_rule;
pub use rms_norm::{as_rms_norm_rule, remove_rms_norm_cast};
pub use scaled_masked_softmax::as_scaled_masked_softmax_rule;
pub use silu::as_silu_rule;
//pub use untranspose_matmul_output::untranspose_matmul_output;

use tract_core::ops::binary::TypedBinOp;
use tract_core::ops::math::{Add, Mul};

#[macro_export]
macro_rules! rule_ensure {
    ($cond:expr) => {
        if !$cond {
            return Ok(None);
        }
    };
}

fn next_node<'a>(model: &'a TypedModel, node: &TypedNode) -> Option<&'a TypedNode> {
    if node.outputs.iter().map(|of| of.successors.len()).sum::<usize>() != 1 {
        return None;
    }
    let succ = node.outputs[0].successors[0];
    Some(&model.nodes()[succ.node])
}

fn previous_node<'a>(model: &'a TypedModel, node: &TypedNode) -> Option<&'a TypedNode> {
    if node.inputs.len() != 1 {
        return None;
    }
    Some(&model.nodes()[node.inputs[0].node])
}

fn previous_nodes<'a>(model: &'a TypedModel, node: &TypedNode) -> TVec<&'a TypedNode> {
    node.inputs.iter().map(|n| &model.nodes()[n.node]).collect()
}

fn collect_node_const_inputs<'a>(model: &'a TypedModel, node: &TypedNode) -> TVec<&'a Const> {
    node.inputs
        .iter()
        .filter_map(|i| {
            let prec = &model.nodes()[i.node];
            prec.op_as::<Const>()
        })
        .collect::<TVec<_>>()
}

fn single_prev_node_as<'a, O: TypedOp>(
    model: &'a TypedModel,
    node: &TypedNode,
) -> Option<(usize, &'a TypedNode)> {
    let prev_nodes = node
        .inputs
        .iter()
        .enumerate()
        .filter_map(|(in_idx, i)| {
            let prec = &model.nodes()[i.node];
            prec.op_is::<O>().then_some((in_idx, prec))
        })
        .collect::<TVec<_>>();

    if prev_nodes.len() != 1 {
        None
    } else {
        Some(prev_nodes[0])
    }
}

fn find_succ_mul_with_const<'a>(
    model: &'a TypedModel,
    node: &'a TypedNode,
    konst: f32,
) -> Option<&'a TypedNode> {
    let mul_coef_a = next_node(model, node)?;
    let mul_coef_a_op = mul_coef_a.op_as::<TypedBinOp>()?;
    (mul_coef_a_op.0.is::<Mul>() && matches_single_input_const(model, mul_coef_a, konst))
        .then_some(mul_coef_a)
}

fn find_succ_add_with<'a>(
    model: &'a TypedModel,
    node: &'a TypedNode,
    outled_id: &OutletId,
) -> Option<&'a TypedNode> {
    let add_succ = next_node(model, node)?;
    let add_succ_op = add_succ.op_as::<TypedBinOp>()?;
    (add_succ_op.0.is::<Add>() && add_succ.inputs.contains(outled_id)).then_some(add_succ)
}

fn matches_single_input_const(model: &TypedModel, node: &TypedNode, konst: f32) -> bool {
    let consts = collect_node_const_inputs(model, node);
    if consts.len() != 1 {
        return false;
    }
    let Ok(in_const) = consts[0].val().cast_to_dt(DatumType::F32) else {
        return false;
    };
    let Ok(in_const) = in_const.to_scalar_tensor() else {
        return false;
    };

    in_const.close_enough(&tensor0(konst), Approximation::Approximate).is_ok()
}

fn find_succ_add_with_const<'a>(
    model: &'a TypedModel,
    node: &'a TypedNode,
    konst: f32,
) -> Option<&'a TypedNode> {
    let add_coef_a = next_node(model, node)?;
    let add_coef_a_op = add_coef_a.op_as::<TypedBinOp>()?;
    if !add_coef_a_op.0.is::<Add>() {
        return None;
    }
    (add_coef_a_op.0.is::<Add>() && matches_single_input_const(model, add_coef_a, konst))
        .then_some(add_coef_a)
}
