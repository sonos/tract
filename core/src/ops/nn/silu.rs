use crate::internal::*;
use crate::ops::binary::{BinMiniOp, TypedBinOp};
use crate::ops::element_wise::ElementWiseOp;
use crate::ops::math::Mul;
use crate::ops::nn::Sigmoid;

#[derive(Clone, Debug, Hash)]
pub struct Silu;

impl Op for Silu {
    fn name(&self) -> StaticName {
        "Silu".to_string().into()
    }
    op_as_typed_op!();
}

impl EvalOp for Silu {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs);
        let dt = input.datum_type();
        let mut a = input.clone().into_tensor();
        Sigmoid {}.eval_in_place(&mut a, None)?;
        let a3 = Mul.eval(input, a.into_tvalue(), dt)?;
        Ok(tvec![a3.into()])
    }
}

impl TypedOp for Silu {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let dt = inputs[0].datum_type;
        let fact = dt.fact(inputs[0].shape.clone());
        Ok(tvec!(fact))
    }

    as_op!();
}

/// Search pattern => A = A * SIGMOID(A)
pub fn detect_silu(model: &TypedModel, node: &TypedNode) -> TractResult<Option<TypedModelPatch>> {
    rule_if!(node.op_as::<ElementWiseOp>().is_some_and(|op| op.0.is::<Sigmoid>()));

    let in_fact = model.node_input_facts(node.id)?[0];
    let dt = in_fact.datum_type;

    // Only F16 and F32 is supported.
    rule_if!(matches!(dt, DatumType::F32 | DatumType::F16));

    // Identify Mul successor
    rule_if_some!(mul_succ = model.single_succ(node.id)?);
    rule_if_some!(mul_succ_op = mul_succ.op_as::<TypedBinOp>());
    rule_if!(mul_succ_op.0.is::<Mul>());
    rule_if!(mul_succ.inputs.contains(&node.inputs[0]));

    let mut patch = TypedModelPatch::default();
    let silu_input = patch.taps(model, &node.inputs)?;
    let out = patch.wire_node(format!("{}.silu", node.name), Silu, &silu_input)?;
    patch.shunt_outside(model, mul_succ.id.into(), out[0])?;
    Ok(Some(patch))
}
