use crate::internal::*;
use crate::ser::*;
use tract_core::ops::Downsample;

pub fn register(registry: &mut Registry) {
    registry.register_dumper(ser_downsample);
    registry.register_primitive(
        "tract_core_downsample",
        &[
            TypeName::Scalar.tensor().named("input"),
            TypeName::Integer.named("axis"),
            TypeName::Integer.named("stride"),
            TypeName::Integer.named("modulo").default(0),
        ],
        &[("output", TypeName::Scalar.tensor())],
        de_downsample,
    );
}

fn ser_downsample(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &Downsample,
) -> TractResult<Option<Arc<RValue>>> {
    let wire = ast.mapping[&node.inputs[0]].clone();
    Ok(Some(invocation(
        "tract_core_downsample",
        &[wire],
        &[
            ("axis", numeric(op.axis)),
            ("stride", numeric(op.stride)),
            ("modulo", numeric(op.modulo)),
        ],
    )))
}

fn de_downsample(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<Value> {
    let wire = invocation.named_arg_as(builder, "input")?;
    let axis = invocation.named_arg_as(builder, "axis")?;
    let stride = invocation.named_arg_as::<i64>(builder, "stride")? as isize;
    let modulo = invocation.named_arg_as(builder, "modulo")?;
    builder.wire(Downsample { axis, stride, modulo }, &[wire])
}
