use crate::internal::*;
use crate::ser::*;
use tract_core::ops::array::OneHot;

pub fn register(registry: &mut Registry) {
    registry.register_dumper(one_hot_dump);
    registry.register_primitive(
        "tract_core_one_hot",
        &one_hot_parameters(),
        &[("output", TypeName::Scalar.tensor())],
        one_hot_load,
    );
}

pub fn one_hot_parameters() -> Vec<Parameter> {
    vec![
        TypeName::Scalar.tensor().named("input"),
        TypeName::Integer.named("axis"),
        TypeName::Integer.named("dim"),
        TypeName::Scalar.named("value_off").default(0.0),
        TypeName::Scalar.named("value_on").default(1.0),
    ]
}

pub fn one_hot_dump(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &OneHot,
) -> TractResult<Option<Arc<RValue>>> {
    let input = ast.mapping[&node.inputs[0]].clone();
    Ok(Some(invocation(
        "tract_core_one_hot",
        &[input],
        &[
            ("axis", numeric(op.axis)),
            ("dim", numeric(op.dim)),
            ("value_off", numeric(op.off.cast_to_scalar::<f32>()?)),
            ("value_on", numeric(op.on.cast_to_scalar::<f32>()?)),
        ],
    )))
}

pub fn one_hot_load(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<Value> {
    let input = invocation.named_arg_as(builder, "input")?;
    let axis = invocation.named_arg_as(builder, "axis")?;
    let dim = invocation.named_arg_as(builder, "dim")?;
    let off = invocation.named_arg_as(builder, "value_off")?;
    let on = invocation.named_arg_as(builder, "value_on")?;
    let op = OneHot { axis, dim, on, off };
    builder.wire(op, &[input])
}
