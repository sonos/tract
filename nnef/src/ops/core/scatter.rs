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

#[cfg(test)]
mod tests {
    use super::*;
    use tract_core::ops::cast::Cast;

    fn build_scatter_nd_with_mismatch() -> TractResult<TypedModel> {
        let mut model = TypedModel::default();
        let data = model.add_source("data", i32::fact([2usize, 4]))?;
        let indices = model.add_source("indices", i64::fact([1usize, 1]))?;
        let updates = model.add_source("updates", i64::fact([1usize, 4]))?;
        let scatter = model.wire_node(
            "scatter",
            ScatterNd::new(ScatterReduction::None),
            &[data, indices, updates],
        )?;
        model.select_output_outlets(&scatter)?;
        Ok(model)
    }

    #[test]
    fn nnef_scatter_nd_round_trip_inserts_cast_on_type_mismatch() -> TractResult<()> {
        let model = build_scatter_nd_with_mismatch()?;
        // The original model has a type mismatch on inputs of ScatterNd; the NNEF
        // dump preserves the source types, and the deserializer must reconcile
        // them via wire_cast.
        let nnef = crate::nnef().with_tract_core();
        let mut buffer = vec![];
        nnef.write_to_tar(&model, &mut buffer)?;
        let reloaded = nnef.model_for_read(&mut &*buffer)?;

        let cast_count = reloaded.nodes().iter().filter(|n| n.op_is::<Cast>()).count();
        assert!(
            cast_count >= 1,
            "Expected at least one Cast node inserted by wire_cast, got {cast_count}"
        );

        // Locate the ScatterNd node and check both data and updates inputs share the same type.
        let scatter_node = reloaded
            .nodes()
            .iter()
            .find(|n| n.op_is::<ScatterNd>())
            .expect("ScatterNd node missing in reloaded model");
        let data_dt = reloaded.outlet_fact(scatter_node.inputs[0])?.datum_type;
        let updates_dt = reloaded.outlet_fact(scatter_node.inputs[2])?.datum_type;
        assert_eq!(
            data_dt, updates_dt,
            "ScatterNd data and updates types must match after wire_cast"
        );
        Ok(())
    }

    #[test]
    fn nnef_scatter_elements_round_trip_inserts_cast_on_type_mismatch() -> TractResult<()> {
        let mut model = TypedModel::default();
        let data = model.add_source("data", i32::fact([4usize]))?;
        let indices = model.add_source("indices", i64::fact([2usize]))?;
        let updates = model.add_source("updates", i64::fact([2usize]))?;
        let scatter = model.wire_node(
            "scatter",
            ScatterElements::new(0, ScatterReduction::None),
            &[data, indices, updates],
        )?;
        model.select_output_outlets(&scatter)?;

        let nnef = crate::nnef().with_tract_core();
        let mut buffer = vec![];
        nnef.write_to_tar(&model, &mut buffer)?;
        let reloaded = nnef.model_for_read(&mut &*buffer)?;

        let cast_count = reloaded.nodes().iter().filter(|n| n.op_is::<Cast>()).count();
        assert!(
            cast_count >= 1,
            "Expected at least one Cast node inserted by wire_cast, got {cast_count}"
        );

        let scatter_node = reloaded
            .nodes()
            .iter()
            .find(|n| n.op_is::<ScatterElements>())
            .expect("ScatterElements node missing in reloaded model");
        let data_dt = reloaded.outlet_fact(scatter_node.inputs[0])?.datum_type;
        let updates_dt = reloaded.outlet_fact(scatter_node.inputs[2])?.datum_type;
        assert_eq!(
            data_dt, updates_dt,
            "ScatterElements data and updates types must match after wire_cast"
        );
        Ok(())
    }
}
