use tract_core::internal::*;
use tract_gpu::fact::DeviceTypedFactExt;
use tract_gpu::ops::binary::GpuBinOp;
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
        WgpuGemvPair {
            first: op.op,
            second: second_op.op,
            hidden: hidden_shape,
            act,
            act_extras,
            post: vec![],
            scaled: false,
            out_shape: None,
        },
        &inputs,
    )?;
    patch.shunt_outside(model, second.id.into(), out[0])?;
    Ok(Some(patch))
}

/// A value with at most one axis longer than one: any reshaping or moving of
/// its axes leaves the data where it is.
fn is_vector(model: &TypedModel, outlet: OutletId) -> TractResult<bool> {
    Ok(concrete_shape(model, outlet)?.is_some_and(|s| s.iter().filter(|d| **d != 1).count() <= 1))
}

/// Steps over the axis ops that only respell a vector, returning the node
/// that finally consumes it and which of its inputs the vector arrives on.
fn skip_vector_respellings(
    model: &TypedModel,
    from: usize,
) -> TractResult<Option<(&TypedNode, usize)>> {
    let mut cur = from;
    loop {
        let Some(succ) = model.single_succ(cur)? else { return Ok(None) };
        if succ.op_is::<GpuAxisOp>() && is_vector(model, succ.id.into())? {
            cur = succ.id;
        } else {
            let slot = succ.inputs.iter().position(|i| i.node == cur).unwrap_or(usize::MAX);
            return Ok(Some((succ, slot)));
        }
    }
}

/// The scalar a multiplication scales its other operand by, when it has one.
fn scalar_factor(
    model: &TypedModel,
    node: &TypedNode,
) -> TractResult<Option<(OutletId, OutletId)>> {
    let Some(op) = node.op_as::<GpuBinOp>() else { return Ok(None) };
    if op.mini_op.name().to_lowercase() != "mul" {
        return Ok(None);
    }
    for slot in 0..2 {
        let konst = node.inputs[slot];
        let fact = model.outlet_fact(konst)?;
        let scalar = fact.konst.is_some()
            && concrete_shape(model, konst)?.is_some_and(|s| s.iter().product::<usize>() == 1);
        if scalar {
            return Ok(Some((node.inputs[1 - slot], konst)));
        }
    }
    Ok(None)
}

/// Folds what surrounds a gate's pair of products into the same kernel: the
/// scalar the pooled input is normalised by, read on load, and the elementwise
/// steps that shape the gate afterwards, run once per output value. Together
/// with the axis ops between them these are dispatches over a few dozen values.
pub fn fuse_gemv_pair_tail(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    node_name: &str,
    op: &WgpuGemvPair,
) -> TractResult<Option<TypedModelPatch>> {
    rule_ensure!(op.post.is_empty() && !op.scaled);
    let Some(w2) = concrete_shape(model, node.inputs[2])? else { return Ok(None) };
    let (_, _, n) = mkn(&op.hidden, &w2, op.second.transpose_a, op.second.transpose_b)?;

    let mut x = node.inputs[0];
    let mut scale = None;
    let mut respellings: Vec<&TypedNode> = vec![];
    let mut cur = model.node(x.node);
    while cur.op_is::<GpuAxisOp>()
        && is_vector(model, cur.id.into())?
        && model.single_succ(cur.id)?.is_some()
    {
        respellings.push(cur);
        cur = model.node(cur.inputs[0].node);
    }
    if model.single_succ(cur.id)?.is_some() {
        if let Some((input, factor)) = scalar_factor(model, cur)? {
            x = input;
            scale = Some(factor);
        } else {
            respellings.clear();
        }
    } else {
        respellings.clear();
    }

    let mut post = vec![];
    let mut post_extras = vec![];
    let mut last = node.id;
    let mut out_shape = None;
    if let Some((consumer, 0)) = skip_vector_respellings(model, node.id)? {
        let operands = consumer.inputs.iter().skip(1).copied().collect::<Vec<_>>();
        let fits = is_vector(model, consumer.inputs[0])?
            && is_vector(model, consumer.id.into())?
            && operands.iter().all(|o| {
                concrete_shape(model, *o).ok().flatten().is_some_and(|s| {
                    let len = s.iter().product::<usize>();
                    len == 1 || len == n
                })
            });
        let steps = activation(consumer).filter(|(_, extras)| fits && *extras == operands.len());
        if let Some((steps, _)) = steps {
            post = steps
                .into_iter()
                .map(|s| match s {
                    ChainStep::Binary { op: name, rhs, swapped } => {
                        ChainStep::Binary { op: name, rhs: rhs + op.act_extras, swapped }
                    }
                    other => other,
                })
                .collect();
            post_extras = operands;
            last = consumer.id;
            out_shape = concrete_shape(model, consumer.id.into())?;
        }
    }
    rule_ensure!(scale.is_some() || !post.is_empty());
    rule_ensure!(op.act_extras + post_extras.len() + scale.is_some() as usize <= MAX_ACT_OPERANDS);

    let mut patch = TypedModelPatch::default();
    let mut x = patch.tap_model(model, x)?;
    for respelling in respellings.iter().rev() {
        x = patch.wire_node(format!("{}.tail", respelling.name), respelling.op.clone(), &[x])?[0];
    }
    let mut wires = vec![node.inputs[1], node.inputs[2]];
    wires.extend(node.inputs[3..].iter().copied());
    wires.extend(post_extras);
    wires.extend(scale);
    let mut inputs =
        wires.iter().map(|i| patch.tap_model(model, *i)).collect::<TractResult<Vec<_>>>()?;
    inputs.insert(0, x);
    let out = patch.wire_node(
        format!("{node_name}.tail"),
        WgpuGemvPair {
            post,
            scaled: scale.is_some(),
            out_shape: out_shape.or_else(|| op.out_shape.clone()),
            ..op.clone()
        },
        &inputs,
    )?;
    patch.shunt_outside(model, last.into(), out[0])?;
    Ok(Some(patch))
}
