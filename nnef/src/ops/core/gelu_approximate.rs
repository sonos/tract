use crate::internal::*;
use tract_core::ops::nn::GeluApproximate;

pub fn register(registry: &mut Registry) {
    registry.register_dumper(ser_gelu_approx);
    registry.register_primitive(
        "tract_core_gelu_approx",
        &[TypeName::Scalar.tensor().named("input"), TypeName::Logical.named("fast_impl")],
        &[("output", TypeName::Scalar.tensor())],
        de_gelu_approx,
    );
    // Backward compatibility alias
    registry.register_primitive(
        "tract_transformers_gelu_approx",
        &[TypeName::Scalar.tensor().named("input"), TypeName::Logical.named("fast_impl")],
        &[("output", TypeName::Scalar.tensor())],
        de_gelu_approx,
    );
}

fn de_gelu_approx(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<Value> {
    let input = invocation.named_arg_as(builder, "input")?;
    let fast_impl = invocation.named_arg_as(builder, "fast_impl")?;
    builder.wire(GeluApproximate { fast_impl }, &[input])
}

fn ser_gelu_approx(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &GeluApproximate,
) -> TractResult<Option<Arc<RValue>>> {
    let input = ast.mapping[&node.inputs[0]].clone();
    Ok(Some(invocation(
        "tract_core_gelu_approx",
        &[input],
        &[("fast_impl", logical(op.fast_impl))],
    )))
}
