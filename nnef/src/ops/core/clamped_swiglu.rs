use crate::internal::*;
use tract_core::ops::nn::ClampedSwiGlu;

pub fn register(registry: &mut Registry) {
    registry.register_dumper(ser);
    registry.register_primitive(
        "tract_core_clamped_swiglu",
        &[
            TypeName::Scalar.tensor().named("gate"),
            TypeName::Scalar.tensor().named("up"),
            TypeName::Scalar.named("alpha"),
            TypeName::Scalar.named("limit"),
        ],
        &[("output", TypeName::Scalar.tensor())],
        de,
    );
}

fn de(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let gate = invocation.named_arg_as(builder, "gate")?;
    let up = invocation.named_arg_as(builder, "up")?;
    let alpha = invocation.named_arg_as(builder, "alpha")?;
    let limit = invocation.named_arg_as(builder, "limit")?;
    builder.wire(ClampedSwiGlu { alpha, limit }, &[gate, up])
}

fn ser(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &ClampedSwiGlu,
) -> TractResult<Option<Arc<RValue>>> {
    let facts =
        node.inputs.iter().map(|o| ast.model.outlet_fact(*o)).collect::<TractResult<Vec<_>>>()?;
    op.output_facts(&facts)?;
    let gate = ast.mapping[&node.inputs[0]].clone();
    let up = ast.mapping[&node.inputs[1]].clone();
    Ok(Some(invocation(
        "tract_core_clamped_swiglu",
        &[gate, up],
        &[("alpha", numeric(op.alpha)), ("limit", numeric(op.limit))],
    )))
}
