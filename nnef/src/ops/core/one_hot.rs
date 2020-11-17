use crate::internal::*;
use crate::ser::*;
use tract_core::ops::array::OneHot;

pub fn register(registry: &mut Registry) {
    registry.register_dumper(TypeId::of::<OneHot>(), one_hot_dump);
    registry.register_primitive("tract_core_one_hot", &one_hot_parameters(), one_hot_load);
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

pub fn one_hot_dump(ast: &mut IntoAst, node: &TypedNode) -> TractResult<Option<Arc<RValue>>> {
    let one_hot = node.op_as::<OneHot>().unwrap();
    let input = ast.mapping[&node.inputs[0]].clone();
    Ok(Some(invocation(
        "tract_core_one_hot",
        &[input],
        &[
            ("axis", numeric(one_hot.axis)),
            ("dim", numeric(one_hot.dim)),
            ("value_off", numeric(one_hot.off.cast_to_scalar::<f32>()?)),
            ("value_on", numeric(one_hot.on.cast_to_scalar::<f32>()?)),
        ],
    )))
}

pub fn one_hot_load(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<TVec<OutletId>> {
    let input = invocation.named_arg_as(builder, "input")?;
    let axis = invocation.named_arg_as(builder, "axis")?;
    let dim = invocation.named_arg_as(builder, "dim")?;
    let off = invocation.named_arg_as(builder, "value_off")?;
    let on = invocation.named_arg_as(builder, "value_on")?;
    let op = OneHot { axis, dim, on, off };
    builder.wire(op, &[input])
}
