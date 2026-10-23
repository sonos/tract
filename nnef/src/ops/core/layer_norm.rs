use crate::internal::*;
use tract_core::ops::nn::LayerNorm;

pub fn register(registry: &mut Registry) {
    registry.register_dumper(ser_layer_norm);
    registry.register_primitive(
        "tract_core_layer_norm",
        &[
            TypeName::Scalar.tensor().named("input"),
            TypeName::Integer.named("axis"),
            TypeName::Scalar.named("eps").default(1e-5f32),
        ],
        &[("output", TypeName::Scalar.tensor())],
        de_layer_norm,
    );
    registry.register_primitive(
        "tract_core_affine_layer_norm",
        &[
            TypeName::Scalar.tensor().named("input"),
            TypeName::Scalar.tensor().named("scale"),
            TypeName::Scalar.tensor().named("bias"),
            TypeName::Integer.named("axis"),
            TypeName::Scalar.named("eps").default(1e-5f32),
        ],
        &[("output", TypeName::Scalar.tensor())],
        de_affine_layer_norm,
    );
}

fn de_layer_norm(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<Value> {
    let input = invocation.named_arg_as(builder, "input")?;
    let axis: usize = invocation.named_arg_as(builder, "axis")?;
    let eps = invocation.named_arg_as(builder, "eps")?;
    builder.wire(LayerNorm { axis, eps }, &[input])
}

fn de_affine_layer_norm(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<Value> {
    let input = invocation.named_arg_as(builder, "input")?;
    let scale = invocation.named_arg_as(builder, "scale")?;
    let bias = invocation.named_arg_as(builder, "bias")?;
    let axis: usize = invocation.named_arg_as(builder, "axis")?;
    let eps = invocation.named_arg_as(builder, "eps")?;
    builder.wire(LayerNorm { axis, eps }, &[input, scale, bias])
}

fn ser_layer_norm(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &LayerNorm,
) -> TractResult<Option<Arc<RValue>>> {
    let inputs = node.inputs.iter().map(|i| ast.mapping[i].clone()).collect::<TVec<_>>();
    let name =
        if inputs.len() == 3 { "tract_core_affine_layer_norm" } else { "tract_core_layer_norm" };
    Ok(Some(invocation(
        name,
        &inputs,
        &[("axis", numeric(op.axis)), ("eps", numeric(op.eps.cast_to_scalar::<f32>()?))],
    )))
}
