use tract_core::internal::*;
use tract_core::ops::array::{ScatterElements, ScatterNd, ScatterReduction};
use tract_core::ops::cast::Cast;
use tract_nnef::internal::*;

fn round_trip(model: &TypedModel) -> TractResult<TypedModel> {
    let nnef = tract_nnef::nnef().with_tract_core();
    let mut buffer = vec![];
    nnef.write_to_tar(model, &mut buffer)?;
    nnef.model_for_read(&mut &*buffer)
}

#[test]
fn scatter_nd_round_trip_inserts_cast_on_type_mismatch() -> TractResult<()> {
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

    let reloaded = round_trip(&model)?;
    let cast_count = reloaded.nodes().iter().filter(|n| n.op_is::<Cast>()).count();
    assert!(
        cast_count >= 1,
        "Expected at least one Cast node inserted by wire_cast, got {cast_count}"
    );

    let scatter_node = reloaded
        .nodes()
        .iter()
        .find(|n| n.op_is::<ScatterNd>())
        .expect("ScatterNd node missing in reloaded model");
    let data_dt = reloaded.outlet_fact(scatter_node.inputs[0])?.datum_type;
    let updates_dt = reloaded.outlet_fact(scatter_node.inputs[2])?.datum_type;
    assert_eq!(data_dt, updates_dt, "ScatterNd data and updates types must match after wire_cast");
    Ok(())
}

#[test]
fn scatter_elements_round_trip_inserts_cast_on_type_mismatch() -> TractResult<()> {
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

    let reloaded = round_trip(&model)?;
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
