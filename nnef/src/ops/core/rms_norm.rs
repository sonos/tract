use crate::internal::*;
use tract_core::ops::nn::RmsNorm;

pub fn register(registry: &mut Registry) {
    registry.register_dumper(ser_rms_norm);
    registry.register_primitive(
        "tract_core_rms_norm",
        &[
            TypeName::Scalar.tensor().named("input"),
            TypeName::Integer.named("axis"),
            TypeName::Scalar.named("eps").default(1e-6f32),
        ],
        &[("output", TypeName::Scalar.tensor())],
        de_rms_norm,
    );
    // Backward compatibility alias
    registry.register_primitive(
        "tract_transformers_rms_norm",
        &[
            TypeName::Scalar.tensor().named("input"),
            TypeName::Integer.named("axis"),
            TypeName::Scalar.named("eps").default(1e-6f32),
        ],
        &[("output", TypeName::Scalar.tensor())],
        de_rms_norm,
    );
}

fn de_rms_norm(builder: &mut ModelBuilder, invocation: &ResolvedInvocation) -> TractResult<Value> {
    let input = invocation.named_arg_as(builder, "input")?;
    let axis: usize = invocation.named_arg_as(builder, "axis")?;
    let eps = invocation.named_arg_as(builder, "eps")?;
    builder.wire(RmsNorm { axis, eps }, &[input])
}

fn ser_rms_norm(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &RmsNorm,
) -> TractResult<Option<Arc<RValue>>> {
    let input = ast.mapping[&node.inputs[0]].clone();
    Ok(Some(invocation(
        "tract_core_rms_norm",
        &[input],
        &[("axis", numeric(op.axis)), ("eps", numeric(op.eps.cast_to_scalar::<f32>()?))],
    )))
}
