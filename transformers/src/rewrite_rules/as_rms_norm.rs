use tract_nnef::tract_core;
use crate::ops::rms_norm::BasicRmsNorm;
use crate::rewrite_rules::{collect_node_const_inputs, next_node, previous_node};
use crate::rule_ensure;
use tract_core::internal::*;
use tract_core::ops::binary::TypedBinOp;
use tract_core::ops::cast::Cast;
use tract_core::ops::element_wise::ElementWiseOp;
use tract_core::ops::math::{Add, Mul, Rsqrt};
use tract_core::ops::nn::{Reduce, Reducer};

/// Search pattern => A = A * RSQRT(MEAN_OF_SQUARES(A) + EPS)
pub fn as_rms_norm_rule(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    node_name: &str,
    op: &Reduce,
) -> TractResult<Option<TypedModelPatch>> {
    rule_ensure!(op.reducer == Reducer::MeanOfSquares);
    rule_ensure!(op.axes.len() == 1);
    let axis = op.axes[0];

    let in_fact = model.node_input_facts(node.id)?[0];
    let dt = in_fact.datum_type;

    // Only F16 and F32 is supported.
    rule_ensure!(matches!(dt, DatumType::F32 | DatumType::F16));

    // Identify Add operator
    let Some(add_succ) = next_node(model, node) else {
        return Ok(None);
    };
    let Some(add_succ_op) = add_succ.op_as::<TypedBinOp>() else {
        return Ok(None);
    };
    rule_ensure!(add_succ_op.0.is::<Add>());

    // Retrieve epsilon
    let add_consts = collect_node_const_inputs(model, add_succ);
    rule_ensure!(add_consts.len() == 1);
    let eps = add_consts[0].val().clone();
    rule_ensure!(eps.len() == 1);
    rule_ensure!(eps.datum_type() == dt);

    // Identify Rsqrt
    let Some(rsqrt_succ) = next_node(model, add_succ) else {
        return Ok(None);
    };
    let Some(rsqrt_succ_op) = rsqrt_succ.op_as::<ElementWiseOp>() else {
        return Ok(None);
    };
    rule_ensure!(rsqrt_succ_op.0.is::<Rsqrt>());

    // Identify Mul
    let Some(mul_succ) = next_node(model, rsqrt_succ) else {
        return Ok(None);
    };
    let Some(mul_succ_op) = mul_succ.op_as::<TypedBinOp>() else {
        return Ok(None);
    };
    rule_ensure!(mul_succ_op.0.is::<Mul>());
    rule_ensure!(mul_succ.inputs.contains(&node.inputs[0]));

    let mut patch = TypedModelPatch::default();
    let rsm_input = patch.taps(model, &node.inputs)?;
    let out =
        patch.wire_node(format!("{node_name}.rms_norm"), BasicRmsNorm { axis, eps }, &rsm_input)?;

    patch.shunt_outside(model, mul_succ.id.into(), out[0])?;
    Ok(Some(patch))
}

/// Search pattern => A = CAST(RMS_NORM(CAST(A, F32)), F16)
pub fn remove_rms_norm_cast(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    node_name: &str,
    op: &BasicRmsNorm,
) -> TractResult<Option<TypedModelPatch>> {
    // Identify Cast from F16 To F32
    let Some(cast_in_node) = previous_node(model, node)
        .and_then(|n| n.op_as::<Cast>().and_then(|cast| (cast.to == DatumType::F32).then_some(n)))
        .filter(|n| {
            model.node_input_facts(n.id).map(|i| i[0].datum_type == DatumType::F16).unwrap_or(false)
        })
    else {
        return Ok(None);
    };

    // Identify Cast from F32 To F16
    let Some(cast_out_node) = next_node(model, node)
        .and_then(|n| n.op_as::<Cast>().and_then(|cast| (cast.to == DatumType::F16).then_some(n)))
        .filter(|n| {
            model.node_input_facts(n.id).map(|i| i[0].datum_type == DatumType::F32).unwrap_or(false)
        })
    else {
        return Ok(None);
    };

    let eps = op.eps.cast_to_dt(DatumType::F16)?.into_owned().into();

    let mut patch = TypedModelPatch::default();
    let rsm_input = patch.taps(model, &cast_in_node.inputs)?;
    let out = patch.wire_node(
        format!("{node_name}.without-cast"),
        BasicRmsNorm { axis: op.axis, eps },
        &rsm_input,
    )?;
    patch.shunt_outside(model, cast_out_node.id.into(), out[0])?;
    Ok(Some(patch))
}
