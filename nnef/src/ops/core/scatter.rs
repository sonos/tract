use crate::internal::*;
use tract_core::ops::array::ScatterElements;
use tract_core::ops::array::ScatterNd;

pub fn register(registry: &mut Registry) {
    use crate::internal::*;

    registry.register_dumper(ser_scatter_elements);
    registry.register_primitive(
        "tract_core_scatter_elements",
        &[
            TypeName::Scalar.tensor().named("input"),
            TypeName::Scalar.tensor().named("indices"),
            TypeName::Scalar.tensor().named("updates"),
            TypeName::Integer.named("axis"),
        ],
        &[("output", TypeName::Scalar.tensor())],
        de_scatter_elements,
    );

    registry.register_dumper(ser_scatter_nd);
    registry.register_primitive(
        "tract_core_scatter_nd",
        &[
            TypeName::Scalar.tensor().named("input"),
            TypeName::Scalar.tensor().named("indices"),
            TypeName::Scalar.tensor().named("updates"),
        ],
        &[("output", TypeName::Scalar.tensor())],
        de_scatter_nd,
    );
}

fn ser_scatter_nd(
    ast: &mut IntoAst,
    node: &TypedNode,
    _: &ScatterNd,
) -> TractResult<Option<Arc<RValue>>> {
    let wire = ast.mapping[&node.inputs[0]].clone();
    let indices = ast.mapping[&node.inputs[1]].clone();
    let updates = ast.mapping[&node.inputs[2]].clone();
    Ok(Some(invocation("tract_core_scatter_nd", &[wire, indices, updates], &[])))
}

fn de_scatter_nd(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<Value> {
    let wire = invocation.named_arg_as(builder, "input")?;
    let indices = invocation.named_arg_as(builder, "indices")?;
    let updates = invocation.named_arg_as(builder, "updates")?;
    builder.wire(ScatterNd, &[wire, indices, updates])
}

fn ser_scatter_elements(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &ScatterElements,
) -> TractResult<Option<Arc<RValue>>> {
    let wire = ast.mapping[&node.inputs[0]].clone();
    let indices = ast.mapping[&node.inputs[1]].clone();
    let updates = ast.mapping[&node.inputs[2]].clone();
    Ok(Some(invocation(
        "tract_core_scatter_elements",
        &[wire, indices, updates],
        &[("axis", numeric(op.axis))],
    )))
}

fn de_scatter_elements(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<Value> {
    let wire = invocation.named_arg_as(builder, "input")?;
    let indices = invocation.named_arg_as(builder, "indices")?;
    let updates = invocation.named_arg_as(builder, "updates")?;
    let axis = invocation.named_arg_as(builder, "axis")?;
    builder.wire(ScatterElements::new(axis), &[wire, indices, updates])
}
