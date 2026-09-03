use crate::internal::*;
use crate::ser::string;
use tract_core::ops::nn::{RmsNorm, ScaledRmsNorm};

pub fn register(registry: &mut Registry) {
    registry.register_dumper(ser_rms_norm);
    registry.register_dumper(ser_scaled_rms_norm);
    registry.register_primitive(
        "tract_core_scaled_rms_norm",
        &[
            TypeName::Scalar.tensor().named("input"),
            TypeName::Scalar.tensor().named("scale"),
            TypeName::Integer.named("axis"),
            TypeName::Scalar.named("eps").default(1e-6f32),
            TypeName::String.named("out_dt").default(""),
        ],
        &[("output", TypeName::Scalar.tensor())],
        de_scaled_rms_norm,
    );
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

fn de_scaled_rms_norm(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<Value> {
    let input = invocation.named_arg_as(builder, "input")?;
    let scale = invocation.named_arg_as(builder, "scale")?;
    let axis: usize = invocation.named_arg_as(builder, "axis")?;
    let eps = invocation.named_arg_as(builder, "eps")?;
    let out_dt: String = invocation.named_arg_as(builder, "out_dt")?;
    let out_dt = if out_dt.is_empty() { None } else { Some(out_dt.parse::<DatumType>()?) };
    builder.wire(ScaledRmsNorm { axis, eps, out_dt }, &[input, scale])
}

fn ser_scaled_rms_norm(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &ScaledRmsNorm,
) -> TractResult<Option<Arc<RValue>>> {
    let input = ast.mapping[&node.inputs[0]].clone();
    let scale = ast.mapping[&node.inputs[1]].clone();
    let out_dt = op.out_dt.map(|dt| format!("{dt:?}").to_lowercase()).unwrap_or_default();
    Ok(Some(invocation(
        "tract_core_scaled_rms_norm",
        &[input, scale],
        &[
            ("axis", numeric(op.axis)),
            ("eps", numeric(op.eps.cast_to_scalar::<f32>()?)),
            ("out_dt", string(out_dt)),
        ],
    )))
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
