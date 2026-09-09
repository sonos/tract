use tract_core::internal::*;
use tract_gpu::ops::binary::GpuBinOp;
use tract_gpu::ops::element_wise::GpuElementWise;
use tract_gpu::rule_ensure;

use crate::kernels::chain::MAX_EXTRA_INPUTS;
use crate::kernels::shaders::{BINARY_OPS, ChainStep, ELEMENT_WISE_OPS};
use crate::ops::chain::WgpuElementWiseChain;

/// The op's name as the generated WGSL spells it, if there is one.
fn unary_name(node: &TypedNode) -> Option<String> {
    let op = node.op_as::<GpuElementWise>()?;
    let name = op.mini_op.name().to_lowercase();
    ELEMENT_WISE_OPS.contains(&name.as_str()).then_some(name)
}

fn binary_name(node: &TypedNode) -> Option<String> {
    let op = node.op_as::<GpuBinOp>()?;
    let name = op.mini_op.name().to_lowercase();
    BINARY_OPS.contains(&name.as_str()).then_some(name)
}

/// A single elementwise op, as the one step it becomes. `slot` is the input the
/// running value arrives on, so the other one is the step's extra operand.
fn one_step(node: &TypedNode, slot: usize, rhs: usize) -> Option<(ChainStep, Option<OutletId>)> {
    if let Some(op) = unary_name(node) {
        return Some((ChainStep::Unary(op), None));
    }
    let op = binary_name(node)?;
    let other = node.inputs[1 - slot];
    Some((ChainStep::Binary { op, rhs, swapped: slot == 1 }, Some(other)))
}

pub fn grow_elementwise_chain(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    node_name: &str,
    op: &WgpuElementWiseChain,
) -> TractResult<Option<TypedModelPatch>> {
    fuse_into_successor(model, node, node_name, op.steps.clone(), node.inputs.len() - 1)
}

pub fn start_elementwise_chain(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    node_name: &str,
    _op: &GpuElementWise,
) -> TractResult<Option<TypedModelPatch>> {
    let Some(name) = unary_name(node) else { return Ok(None) };
    fuse_into_successor(model, node, node_name, vec![ChainStep::Unary(name)], 0)
}

pub fn start_binary_chain(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    node_name: &str,
    _op: &GpuBinOp,
) -> TractResult<Option<TypedModelPatch>> {
    // The head of a chain carries its shape, so it has to be the operand that
    // is not broadcast.
    let facts = model.node_input_facts(node.id)?;
    let out_shape = model.node_output_facts(node.id)?[0].shape.clone();
    rule_ensure!(facts[0].shape == out_shape);
    let Some(name) = binary_name(node) else { return Ok(None) };
    fuse_into_successor(
        model,
        node,
        node_name,
        vec![ChainStep::Binary { op: name, rhs: 1, swapped: false }],
        1,
    )
}

/// Merges `node` and its single consumer into one chain kernel. Growth stops at
/// a value anyone else reads, at a step that would change the running shape,
/// and at the uniform slot's operand budget.
fn fuse_into_successor(
    model: &TypedModel,
    node: &TypedNode,
    node_name: &str,
    mut steps: Vec<ChainStep>,
    extras: usize,
) -> TractResult<Option<TypedModelPatch>> {
    let Some(succ) = model.single_succ(node.id)? else { return Ok(None) };
    rule_ensure!(!succ.op_is::<WgpuElementWiseChain>());
    rule_ensure!(unary_name(succ).is_some() || binary_name(succ).is_some());
    rule_ensure!(succ.inputs.iter().filter(|i| i.node == node.id).count() == 1);
    rule_ensure!(
        model.node_output_facts(succ.id)?[0].shape == model.node_output_facts(node.id)?[0].shape
    );
    rule_ensure!(extras < MAX_EXTRA_INPUTS);

    let slot = succ.inputs.iter().position(|i| i.node == node.id).unwrap();
    let Some((step, extra)) = one_step(succ, slot, extras + 1) else { return Ok(None) };
    steps.push(step);

    let mut patch = TypedModelPatch::default();
    let mut inputs = vec![];
    for inlet in node.inputs.iter() {
        inputs.push(patch.tap_model(model, *inlet)?);
    }
    if let Some(extra) = extra {
        inputs.push(patch.tap_model(model, extra)?);
    }
    let out =
        patch.wire_node(format!("{node_name}.chain"), WgpuElementWiseChain { steps }, &inputs)?;
    patch.shunt_outside(model, succ.id.into(), out[0])?;
    Ok(Some(patch))
}
