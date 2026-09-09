use tract_core::internal::*;
use tract_gpu::fact::DeviceTypedFactExt;
use tract_gpu::ops::change_axes::GpuAxisOp;
use tract_gpu::ops::element_wise::GpuElementWise;
use tract_gpu::rule_ensure;

use crate::kernels::gemv_pair::{MAX_ACT_OPERANDS, MAX_WIDTH};
use crate::kernels::matmul::mkn;
use crate::kernels::shaders::{ChainStep, ELEMENT_WISE_OPS};
use crate::ops::chain::WgpuElementWiseChain;
use crate::ops::gemv_pair::WgpuGemvPair;
use crate::ops::matmul::WgpuGemm;

/// Steps over the ops that only respell a vector, since how the hidden vector
/// is shaped between the two products is invisible once they are one kernel.
fn skip_reshapes(model: &TypedModel, from: usize) -> TractResult<Option<&TypedNode>> {
    let mut cur = from;
    loop {
        let Some(succ) = model.single_succ(cur)? else { return Ok(None) };
        let respell = succ.op_as::<GpuAxisOp>().is_some_and(|a| {
            matches!(a.inner, AxisOp::Rm(_) | AxisOp::Add(_) | AxisOp::Reshape(..))
        });
        if respell {
            cur = succ.id;
        } else {
            return Ok(Some(succ));
        }
    }
}

/// The activation between the two products, and how many operands of its own it
/// carries.
fn activation(node: &TypedNode) -> Option<(Vec<ChainStep>, usize)> {
    if let Some(chain) = node.op_as::<WgpuElementWiseChain>() {
        return Some((chain.steps.clone(), node.inputs.len() - 1));
    }
    let op = node.op_as::<GpuElementWise>()?;
    let name = op.mini_op.name().to_lowercase();
    ELEMENT_WISE_OPS.contains(&name.as_str()).then(|| (vec![ChainStep::Unary(name)], 0))
}

/// A device fact keeps the tensor's shape inside it, not on the outer fact.
fn concrete_shape(model: &TypedModel, outlet: OutletId) -> TractResult<Option<TVec<usize>>> {
    let fact = model.outlet_fact(outlet)?;
    let shape = match fact.as_device_fact() {
        Some(f) => f.shape.clone(),
        None => fact.shape.clone(),
    };
    Ok(shape.as_concrete().map(|s| s.into()))
}

/// The number of columns a single-row product produces, or `None` if it has
/// more than one row.
fn single_row(model: &TypedModel, node: &TypedNode, op: &WgpuGemm) -> TractResult<Option<usize>> {
    let (Some(a), Some(b)) =
        (concrete_shape(model, node.inputs[0])?, concrete_shape(model, node.inputs[1])?)
    else {
        return Ok(None);
    };
    let (m, _, n) = mkn(&a, &b, op.op.transpose_a, op.op.transpose_b)?;
    Ok((m == 1).then_some(n))
}

/// Merges a squeeze-excitation gate's two dense layers, and the activation
/// between them, into one kernel. Each product has a single row and a handful of
/// columns, so on their own they are three dispatches that barely occupy the
/// device and cost more to record than to run.
pub fn fuse_gemv_pair(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    node_name: &str,
    op: &WgpuGemm,
) -> TractResult<Option<TypedModelPatch>> {
    rule_ensure!(op.epilogue.is_empty());
    let Some(hidden) = single_row(model, node, op)? else { return Ok(None) };
    rule_ensure!(hidden <= MAX_WIDTH);

    let Some(act_node) = skip_reshapes(model, node.id)? else { return Ok(None) };
    let Some((act, act_extras)) = activation(act_node) else { return Ok(None) };
    rule_ensure!(act_extras <= MAX_ACT_OPERANDS);

    let Some(second) = skip_reshapes(model, act_node.id)? else { return Ok(None) };
    let Some(second_op) = second.op_as::<WgpuGemm>() else { return Ok(None) };
    rule_ensure!(second_op.epilogue.is_empty());
    let Some(hidden_shape) = concrete_shape(model, second.inputs[0])? else { return Ok(None) };
    let Some(out_width) = single_row(model, second, second_op)? else { return Ok(None) };
    rule_ensure!(out_width <= MAX_WIDTH);

    let mut patch = TypedModelPatch::default();
    let mut wires = vec![node.inputs[0], node.inputs[1], second.inputs[1]];
    wires.extend(act_node.inputs.iter().skip(1).copied());
    let inputs =
        wires.iter().map(|i| patch.tap_model(model, *i)).collect::<TractResult<Vec<_>>>()?;
    let out = patch.wire_node(
        format!("{node_name}.gemv_pair"),
        WgpuGemvPair { first: op.op, second: second_op.op, hidden: hidden_shape, act },
        &inputs,
    )?;
    patch.shunt_outside(model, second.id.into(), out[0])?;
    Ok(Some(patch))
}
