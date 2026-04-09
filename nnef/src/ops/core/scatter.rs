use crate::internal::*;
use tract_core::ops::array::ScatterElements;
use tract_core::ops::array::ScatterNd;
use tract_core::ops::array::ScatterReduction;
use tract_core::ops::cast::wire_cast;

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
            TypeName::String.named("reduction").default("none"),
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
            TypeName::String.named("reduction").default("none"),
        ],
        &[("output", TypeName::Scalar.tensor())],
        de_scatter_nd,
    );
}

fn ser_scatter_nd(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &ScatterNd,
) -> TractResult<Option<Arc<RValue>>> {
    let wire = ast.mapping[&node.inputs[0]].clone();
    let indices = ast.mapping[&node.inputs[1]].clone();
    let updates = ast.mapping[&node.inputs[2]].clone();
    Ok(Some(invocation(
        "tract_core_scatter_nd",
        &[wire, indices, updates],
        &[("reduction", string(op.reduction.as_str()))],
    )))
}

fn de_scatter_nd(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<Value> {
    let wire: OutletId = invocation.named_arg_as(builder, "input")?;
    let indices: OutletId = invocation.named_arg_as(builder, "indices")?;
    let updates: OutletId = invocation.named_arg_as(builder, "updates")?;
    let reduction: String = invocation.named_arg_as(builder, "reduction")?;
    let super_type = DatumType::super_type_for([
        builder.model.outlet_fact(wire)?.datum_type,
        builder.model.outlet_fact(updates)?.datum_type,
    ])
    .context("No super type for ScatterNd data and updates")?;
    let prefix = builder.generate_node_name();
    let casted = wire_cast(&prefix, &mut builder.model, &[wire, updates], super_type)?;
    builder.wire(
        ScatterNd::new(ScatterReduction::parse(&reduction)?),
        &[casted[0], indices, casted[1]],
    )
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
        &[("axis", numeric(op.axis)), ("reduction", string(op.reduction.as_str()))],
    )))
}

fn de_scatter_elements(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<Value> {
    let wire: OutletId = invocation.named_arg_as(builder, "input")?;
    let indices: OutletId = invocation.named_arg_as(builder, "indices")?;
    let updates: OutletId = invocation.named_arg_as(builder, "updates")?;
    let axis = invocation.named_arg_as(builder, "axis")?;
    let reduction: String = invocation.named_arg_as(builder, "reduction")?;
    let super_type = DatumType::super_type_for([
        builder.model.outlet_fact(wire)?.datum_type,
        builder.model.outlet_fact(updates)?.datum_type,
    ])
    .context("No super type for ScatterElements data and updates")?;
    let prefix = builder.generate_node_name();
    let casted = wire_cast(&prefix, &mut builder.model, &[wire, updates], super_type)?;
    builder.wire(
        ScatterElements::new(axis, ScatterReduction::parse(&reduction)?),
        &[casted[0], indices, casted[1]],
    )
}
