use tract_nnef::tract_core::internal::*;
use tract_nnef::tract_core::ops::binary::TypedBinOp;
use tract_nnef::tract_core::ops::element_wise::ElementWiseOp;
use tract_nnef::tract_core::ops::math::Mul;
use tract_nnef::tract_core::ops::nn::Sigmoid;

use crate::ops::silu::BasicSilu;
use crate::rewrite_rules::next_node;
use crate::rule_ensure;


/// Search pattern => A = A * SIGMOID(A)
pub fn as_silu_rule(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    node_name: &str,
    op: &ElementWiseOp,
) -> TractResult<Option<TypedModelPatch>> {
    // Search pattern => A = A * SIGMOID(A);

    rule_ensure!(op.0.is::<Sigmoid>());

    let in_fact = model.node_input_facts(node.id)?[0];
    let dt = in_fact.datum_type;

    // Only F16 and F32 is supported.
    rule_ensure!(matches!(dt, DatumType::F32 | DatumType::F16));

    let mut patch = TypedModelPatch::default();
    let silu_input = patch.taps(model, &node.inputs)?;
    // Identify Mul
    let Some(mul_succ) = next_node(model, node) else { return Ok(None) };
    let Some(mul_succ_op) = mul_succ.op_as::<TypedBinOp>() else { return Ok(None) };
    rule_ensure!(mul_succ_op.0.is::<Mul>());
    rule_ensure!(mul_succ.inputs.contains(&node.inputs[0]));

    let out = patch.wire_node(format!("{node_name}.silu"), BasicSilu, &silu_input)?;

    patch.shunt_outside(model, mul_succ.id.into(), out[0])?;

    Ok(Some(patch))
}