use tract_core::internal::*;
use tract_core::ops::cast::Cast;
use tract_transformers::ops::rms_norm::RmsNorm;

/// Search pattern => A = CAST(RMS_NORM(CAST(A, F32)), F16)
pub fn remove_rms_norm_cast(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    node_name: &str,
    op: &RmsNorm,
) -> TractResult<Option<TypedModelPatch>> {
    // Identify Cast from F16 To F32
    let Some(cast_in_node) = model
        .single_prec(node.id)?
        .and_then(|n| n.op_as::<Cast>().and_then(|cast| (cast.to == DatumType::F32).then_some(n)))
        .filter(|n| {
            model.node_input_facts(n.id).map(|i| i[0].datum_type == DatumType::F16).unwrap_or(false)
        })
    else {
        return Ok(None);
    };

    // Identify Cast from F32 To F16
    let Some(cast_out_node) = model
        .single_succ(node.id)?
        .and_then(|n| n.op_as::<Cast>().and_then(|cast| (cast.to == DatumType::F16).then_some(n)))
        .filter(|n| {
            model.node_input_facts(n.id).map(|i| i[0].datum_type == DatumType::F32).unwrap_or(false)
        })
    else {
        return Ok(None);
    };

    let mut patch = TypedModelPatch::default();
    let rsm_input = patch.taps(model, &cast_in_node.inputs)?;
    let out = patch.wire_node(format!("{node_name}.without-cast"), op.clone(), &rsm_input)?;
    patch.shunt_outside(model, cast_out_node.id.into(), out[0])?;
    Ok(Some(patch))
}
