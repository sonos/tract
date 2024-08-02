use crate::rewrite_rules::next_node;
use crate::rule_ensure;
use tract_core::internal::*;
use tract_core::ops::binary::BinMiniOp;
use tract_core::ops::binary::TypedBinOp;
use tract_core::ops::element_wise::ElementWiseOp;
use tract_core::ops::math::Mul;
use tract_core::ops::nn::Sigmoid;

#[derive(Clone, Debug, Hash)]
pub struct BasicSilu;

impl Op for BasicSilu {
    fn name(&self) -> Cow<str> {
        "BasicSilu".to_string().into()
    }
    op_as_typed_op!();
}

impl EvalOp for BasicSilu {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs);
        let dt = input.datum_type();
        let mut a = input.clone().into_tensor();
        Sigmoid {}.eval_in_place(&mut a, None)?;
        let a3 = Mul.eval(input.clone(), a.into_tvalue(), dt)?;
        Ok(tvec![a3.into()])
    }
}

impl TypedOp for BasicSilu {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let dt = inputs[0].datum_type;
        let fact = dt.fact(inputs[0].shape.clone());
        Ok(tvec!(fact))
    }

    as_op!();
}

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
