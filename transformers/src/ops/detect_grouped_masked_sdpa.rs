use tract_core::ops::array::MultiBroadcastTo;
use tract_core::ops::cast::Cast;
use tract_core::ops::einsum::EinSum;
use tract_nnef::internal::*;

use super::scaled_masked_softmax::ScaledMaskedSoftmax;
use super::sdpa::Sdpa;

fn same_axes(actual: &AxesMapping, pattern: &str) -> TractResult<bool> {
    let expected: AxesMapping = pattern.parse()?;
    Ok(actual.iter_all_axes().count() == expected.iter_all_axes().count()
        && expected.iter_all_axes().all(|e| {
            actual.iter_all_axes().any(|a| a.inputs == e.inputs && a.outputs == e.outputs)
        }))
}

/// Opt-in conversion of grouped masked attention to SDPA's accumulation contract.
/// It accepts singleton batches and f32 scores with f16 probabilities, replacing
/// the explicit probability rounding with SDPA's fused accumulation.
pub fn detect_grouped_masked_sdpa_rule(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    name: &str,
    pv: &EinSum,
) -> TractResult<Option<TypedModelPatch>> {
    rule_if!(node.inputs.len() == 2 && pv.q_params.is_none());
    rule_if!(same_axes(&pv.axes, "hgsk,bhkd->hgsd")?);
    let fold = model.node(node.inputs[0].node);
    rule_if_some!(AxisOp::Reshape(0, from, to) = fold.op_as::<AxisOp>());
    rule_if!(from.len() == 1 && to.len() == 2);
    let probability_cast = model.node(fold.inputs[0].node);
    rule_if_some!(cast = probability_cast.op_as::<Cast>());
    rule_if!(cast.to == DatumType::F16);
    let softmax = model.node(probability_cast.inputs[0].node);
    rule_if_some!(sm = softmax.op_as::<ScaledMaskedSoftmax>());
    rule_if!(!sm.post_softmax_mask);
    let scores = model.node(softmax.inputs[0].node);
    rule_if_some!(qk = scores.op_as::<EinSum>());
    rule_if!(scores.inputs.len() == 2 && qk.q_params.is_none());
    rule_if!(same_axes(&qk.axes, "bhsd,bhkd->hsk")?);
    rule_if!(qk.operating_dt == DatumType::F32 && pv.operating_dt == DatumType::F16);
    let q_cast = model.node(scores.inputs[0].node);
    let k_cast = model.node(scores.inputs[1].node);
    rule_if!(q_cast.op_as::<Cast>().is_some_and(|c| c.to == DatumType::F32));
    rule_if!(k_cast.op_as::<Cast>().is_some_and(|c| c.to == DatumType::F32));
    let q = q_cast.inputs[0];
    let k_fold = model.node(k_cast.inputs[0].node);
    rule_if_some!(AxisOp::Reshape(1, k_from, k_to) = k_fold.op_as::<AxisOp>());
    rule_if!(k_from == to && k_to == from);
    let broadcast = model.node(k_fold.inputs[0].node);
    rule_if!(broadcast.op_is::<MultiBroadcastTo>());
    let add = model.node(broadcast.inputs[0].node);
    rule_if!(add.op_as::<AxisOp>() == Some(&AxisOp::Add(2)));
    let k = add.inputs[0];
    let v = node.inputs[1];
    let qf = model.outlet_fact(q)?;
    let kf = model.outlet_fact(k)?;
    let vf = model.outlet_fact(v)?;
    rule_if!(
        [qf, kf, vf]
            .iter()
            .all(|f| f.rank() == 4 && f.datum_type == DatumType::F16 && f.shape[0].is_one())
    );
    rule_if!(qf.shape[1] == from[0] && kf.shape[1] == to[0]);
    rule_if!(kf.shape == vf.shape && qf.shape[3] == kf.shape[3]);
    rule_if!(to[0].to_usize().is_ok_and(|h| h > 0) && to[1].to_usize().is_ok_and(|g| g > 0));
    rule_if!(from[0] == to[0].clone() * to[1].clone());
    let mask = softmax.inputs[1];
    let mf = model.outlet_fact(mask)?;
    rule_if!(mf.rank() == 3 && mf.datum_type == DatumType::F32);
    rule_if!(
        (mf.shape[0].is_one() || mf.shape[0] == qf.shape[1])
            && mf.shape[1] == qf.shape[2]
            && mf.shape[2] == kf.shape[2]
    );
    rule_if!(
        model.outlet_fact(k_fold.id.into())?.shape
            == tvec![1.to_dim(), qf.shape[1].clone(), kf.shape[2].clone(), kf.shape[3].clone()]
                .into()
    );
    let mut patch = TypedModelPatch::default();
    let q = patch.tap_model(model, q)?;
    let k = patch.tap_model(model, k)?;
    let v = patch.tap_model(model, v)?;
    let mask = patch.tap_model(model, mask)?;
    let mask = patch.wire_node(format!("{name}.mask_batch"), AxisOp::Add(0), &[mask])?[0];
    let sdpa = Sdpa {
        scale: Some(tensor0(sm.scale.cast_to_scalar::<f32>()?)),
        datum_type: DatumType::F16,
        acc_datum_type: DatumType::F32,
        is_causal: false,
    };
    let output = patch.wire_node(format!("{name}.sdpa"), sdpa, &[q, k, v, mask])?[0];
    let output = patch.wire_node(format!("{name}.unbatch"), AxisOp::Rm(0), &[output])?[0];
    let output = patch.wire_node(name, AxisOp::Reshape(0, from.clone(), to.clone()), &[output])?[0];
    patch.shunt_outside(model, node.id.into(), output)?;
    Ok(Some(patch))
}

/// Opt-in SDPA conversion for singleton-batch MHA with f16 score scaling.
/// Fused accumulation replaces explicit score and probability rounding.
pub fn detect_masked_mha_sdpa_rule(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    name: &str,
    pv: &EinSum,
) -> TractResult<Option<TypedModelPatch>> {
    use tract_core::ops::binary::TypedBinOp;
    use tract_core::ops::math::Add;
    use tract_core::ops::math::Mul;
    use tract_core::ops::nn::Softmax;
    use tract_core::ops::nn::SoftmaxKind;

    rule_if!(node.inputs.len() == 2 && pv.q_params.is_none());
    rule_if!(pv.operating_dt == DatumType::F16 && same_axes(&pv.axes, "hsk,bhkd->shd")?);
    let cast = model.node(node.inputs[0].node);
    rule_if!(cast.op_as::<Cast>().is_some_and(|c| c.to == DatumType::F16));
    let softmax = model.node(cast.inputs[0].node);
    rule_if_some!(sm = softmax.op_as::<Softmax>());
    rule_if!(
        sm.axes == tvec![2] && sm.kind == SoftmaxKind::Softmax && sm.quant_output_dt.is_none()
    );
    let add = model.node(softmax.inputs[0].node);
    rule_if!(add.op_as::<TypedBinOp>().is_some_and(|op| op.0.is::<Add>()));
    let cast = model.node(add.inputs[0].node);
    rule_if!(cast.op_as::<Cast>().is_some_and(|c| c.to == DatumType::F32));
    let scale_mul = model.node(cast.inputs[0].node);
    rule_if!(scale_mul.op_as::<TypedBinOp>().is_some_and(|op| op.0.is::<Mul>()));
    let consts = model.collect_const_inputs(scale_mul);
    rule_if!(consts.len() == 1 && consts[0].val().len() == 1);
    rule_if_some!(
        scores_outlet = scale_mul
            .inputs
            .iter()
            .find(|i| model.outlet_fact(**i).is_ok_and(|f| f.konst.is_none()))
    );
    let scores = model.node(scores_outlet.node);
    rule_if_some!(qk = scores.op_as::<EinSum>());
    rule_if!(scores.inputs.len() == 2 && qk.q_params.is_none());
    rule_if!(qk.operating_dt == DatumType::F16 && same_axes(&qk.axes, "bhsd,bhkd->hsk")?);
    let (q, k, v) = (scores.inputs[0], scores.inputs[1], node.inputs[1]);
    let (qf, kf, vf) = (model.outlet_fact(q)?, model.outlet_fact(k)?, model.outlet_fact(v)?);
    rule_if!(
        [qf, kf, vf]
            .iter()
            .all(|f| f.rank() == 4 && f.datum_type == DatumType::F16 && f.shape[0].is_one())
    );
    rule_if!(kf.shape == vf.shape && qf.shape[1] == kf.shape[1] && qf.shape[3] == kf.shape[3]);
    rule_if!(qf.shape[1].to_usize().is_ok_and(|h| h > 0));
    let mask = add.inputs[1];
    let mf = model.outlet_fact(mask)?;
    rule_if!(mf.rank() == 3 && mf.datum_type == DatumType::F32);
    rule_if!(
        (mf.shape[0].is_one() || mf.shape[0] == qf.shape[1])
            && mf.shape[1] == qf.shape[2]
            && mf.shape[2] == kf.shape[2]
    );
    let mut patch = TypedModelPatch::default();
    let q = patch.tap_model(model, q)?;
    let k = patch.tap_model(model, k)?;
    let v = patch.tap_model(model, v)?;
    let mask = patch.tap_model(model, mask)?;
    let mask = patch.wire_node(format!("{name}.mask_batch"), AxisOp::Add(0), &[mask])?[0];
    let sdpa = Sdpa {
        scale: Some(tensor0(consts[0].val().cast_to_scalar::<f32>()?)),
        datum_type: DatumType::F16,
        acc_datum_type: DatumType::F32,
        is_causal: false,
    };
    let output = patch.wire_node(format!("{name}.sdpa"), sdpa, &[q, k, v, mask])?[0];
    let output = patch.wire_node(format!("{name}.unbatch"), AxisOp::Rm(0), &[output])?[0];
    let output = patch.wire_node(name, AxisOp::Move(0, 1), &[output])?[0];
    patch.shunt_outside(model, node.id.into(), output)?;
    Ok(Some(patch))
}
