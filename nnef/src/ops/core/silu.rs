use crate::internal::*;
use tract_core::ops::nn::Silu;

pub fn register(registry: &mut Registry) {
    registry.register_dumper(ser_silu);
    registry.register_primitive(
        "tract_core_silu",
        &[TypeName::Scalar.tensor().named("input")],
        &[("output", TypeName::Scalar.tensor())],
        de_silu,
    );
    // Backward compatibility alias
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
    Ok(Some(invocation("tract_core_silu", &[input], &[])))
}
