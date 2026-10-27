use tract_core::internal::*;
use tract_core::ops::array::{GatherNd, TypedConcat};
use tract_core::ops::binary::TypedBinOp;
use tract_core::ops::cast::Cast;
use tract_core::ops::change_axes::AxisOp;
use tract_core::ops::element_wise::ElementWiseOp;

/// Integers below this magnitude are exact in f32.
const F32_EXACT: i64 = 1 << 24;

/// Computes the indices of a `GatherNd` in f32 when they are rounded floats moved by integer
/// sums and clips, so they and the gather run on a device, which holds no TDim (nor, on wgpu,
/// 64-bit integers).
/// Below 2^24 every such value is exact in f32. Past it a coordinate is out of range of any
/// axis the rule accepts, and a clip back into range still lands on its bound.
pub fn float_gather_nd_indices(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    _name: &str,
    op: &GatherNd,
) -> TractResult<Option<TypedModelPatch>> {
    rule_if!(!model.outlet_fact(node.inputs[1])?.datum_type.is_float());
    let data = model.outlet_fact(node.inputs[0])?;
    rule_if!(data.shape.iter().all(|d| d.to_i64().is_ok_and(|d| d < F32_EXACT)));
    let mut patch = TypedModelPatch::default();
    rule_if_some!(indices = float_index(model, &mut patch, node.inputs[1], &mut HashMap::new())?);
    let data = patch.tap_model(model, node.inputs[0])?;
    let gathered = patch.wire_node(&node.name, op.clone(), &[data, indices])?[0];
    patch.shunt_outside(model, node.id.into(), gathered)?;
    Ok(Some(patch))
}

/// Wires the integer computation behind `outlet` again in f32, or `None` if a step of it is
/// not exact there.
fn float_index(
    model: &TypedModel,
    patch: &mut TypedModelPatch,
    outlet: OutletId,
    wired: &mut HashMap<OutletId, OutletId>,
) -> TractResult<Option<OutletId>> {
    if let Some(wire) = wired.get(&outlet) {
        return Ok(Some(*wire));
    }
    let node = model.node(outlet.node);
    let fact = model.outlet_fact(outlet)?;
    rule_if!(fact.datum_type.is_integer() || fact.datum_type.is_tdim());
    let name = model.unique_name(format!("{}.f32", node.name)).into_owned();
    let wire = if let Some(konst) = &fact.konst {
        let Ok(konst) = konst.cast_to::<i64>() else { return Ok(None) };
        let values = konst.try_as_plain_ram()?;
        rule_if!(values.as_slice::<i64>()?.iter().all(|v| v.abs() < F32_EXACT));
        patch.add_const(name, konst.cast_to::<f32>()?.into_owned())?
    } else if node.op_is::<Cast>() {
        let input = node.inputs[0];
        rule_if!(model.outlet_fact(input)?.datum_type == f32::datum_type());
        let rounded = model.node(input.node).op_as::<ElementWiseOp>().is_some_and(|op| {
            matches!(&*op.0.name(), "Floor" | "Ceil" | "Round" | "RoundHalfToEven")
        });
        rule_if!(rounded);
        patch.tap_model(model, input)?
    } else if let Some(op) = node.op_as::<TypedBinOp>() {
        rule_if!(matches!(op.0.name(), "Add" | "Sub" | "Max" | "Min"));
        rule_if_some!(inputs = float_indices(model, patch, &node.inputs, wired)?);
        patch.wire_node(name, TypedBinOp(op.0.clone(), None), &inputs)?[0]
    } else if node.op_is::<AxisOp>() || node.op_is::<TypedConcat>() {
        rule_if_some!(inputs = float_indices(model, patch, &node.inputs, wired)?);
        patch.wire_node(name, node.op.clone(), &inputs)?[0]
    } else {
        return Ok(None);
    };
    wired.insert(outlet, wire);
    Ok(Some(wire))
}

fn float_indices(
    model: &TypedModel,
    patch: &mut TypedModelPatch,
    outlets: &[OutletId],
    wired: &mut HashMap<OutletId, OutletId>,
) -> TractResult<Option<TVec<OutletId>>> {
    let mut wires = tvec![];
    for outlet in outlets {
        rule_if_some!(wire = float_index(model, patch, *outlet, wired)?);
        wires.push(wire);
    }
    Ok(Some(wires))
}
