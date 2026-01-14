use tract_nnef::internal::*;
use tract_nnef::tract_core::ops::binary::{BinMiniOp, TypedBinOp};
use tract_nnef::tract_core::ops::element_wise::ElementWiseOp;
use tract_nnef::tract_core::ops::math::Mul;
use tract_nnef::tract_core::ops::nn::Sigmoid;

use super::next_node;

pub fn register(registry: &mut Registry) {
    registry.register_dumper(ser_silu);
    registry.register_primitive(
        "tract_transformers_silu",
        &[TypeName::Scalar.tensor().named("input")],
        &[("output", TypeName::Scalar.tensor())],
        de_silu,
    );
}

fn de_silu(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let input = invocation.named_arg_as(builder, "input")?;
    builder.wire(Silu, &[input])
}

fn ser_silu(ast: &mut IntoAst, node: &TypedNode, _op: &Silu) -> TractResult<Option<Arc<RValue>>> {
    let input = ast.mapping[&node.inputs[0]].clone();
    Ok(Some(invocation("tract_transformers_silu", &[input], &[])))
}

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
pub fn silu_rule(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    node_name: &str,
    op: &ElementWiseOp,
) -> TractResult<Option<TypedModelPatch>> {
    // Search pattern => A = A * SIGMOID(A);

    rule_if!(op.0.is::<Sigmoid>());

    let in_fact = model.node_input_facts(node.id)?[0];
    let dt = in_fact.datum_type;

    // Only F16 and F32 is supported.
    rule_if!(matches!(dt, DatumType::F32 | DatumType::F16));

    let mut patch = TypedModelPatch::default();
    let silu_input = patch.taps(model, &node.inputs)?;
    // Identify Mul
    rule_if_some!(mul_succ = next_node(model, node));
    rule_if_some!(mul_succ_op = mul_succ.op_as::<TypedBinOp>());
    rule_if!(mul_succ_op.0.is::<Mul>());
    rule_if!(mul_succ.inputs.contains(&node.inputs[0]));

    let out = patch.wire_node(format!("{node_name}.silu"), Silu, &silu_input)?;

    patch.shunt_outside(model, mul_succ.id.into(), out[0])?;

    Ok(Some(patch))
}
