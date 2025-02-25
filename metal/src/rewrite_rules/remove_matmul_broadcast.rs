use crate::rewrite_rules::next_node;
use tract_core::internal::*;
use tract_core::ops::array::MultiBroadcastTo;
use tract_core::ops::einsum::BasicMatMul;
use tract_core::tract_data::itertools::Itertools;

pub fn remove_matmul_broadcast(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    _node_name: &str,
    _op: &MultiBroadcastTo,
) -> TractResult<Option<TypedModelPatch>> {
    let mut succ_node = match next_node(model, node) {
        Some(n) => n,
        None => return Ok(None),
    };

    let mut patch = TypedModelPatch::default();
    let inputs = patch.taps(model, &node.inputs)?;
    let mut prev_node_id = 0;
    while !succ_node.op_is::<BasicMatMul>() {
        if !succ_node.op_is::<AxisOp>() {
            return Ok(None);
        }
        prev_node_id = succ_node.id;
        succ_node = match next_node(model, succ_node) {
            Some(n) => n,
            None => return Ok(None),
        };
    }

    let move_out = patch.wire_node(&node.name, AxisOp::Rm(0), &inputs)?[0];

    let matmul_inputs = patch.taps(model, &succ_node.inputs)?;

    let matmul_inputs = succ_node
        .inputs
        .iter()
        .enumerate()
        .map(|(idx, o)| if o.node == prev_node_id { move_out } else { matmul_inputs[idx] })
        .collect_vec();

    let out = patch.wire_node(&succ_node.name, &succ_node.op, &matmul_inputs)?;

    patch.shunt_outside(model, succ_node.id.into(), out[0])?;
    Ok(Some(patch))
}
