use tract_core::ops::array::MultiBroadcastTo;
use tract_core::ops::cast::Cast;
use tract_core::ops::einsum::EinSum;
use tract_nnef::internal::*;

use super::sdpa::Sdpa;

fn same_axes(actual: &AxesMapping, pattern: &str) -> TractResult<bool> {
    let expected: AxesMapping = pattern.parse()?;
    Ok(actual.iter_all_axes().count() == expected.iter_all_axes().count()
        && expected.iter_all_axes().all(|e| {
            actual.iter_all_axes().any(|a| a.inputs == e.inputs && a.outputs == e.outputs)
        }))
}

fn sink_qk_branch(
    model: &TypedModel,
    outlet: OutletId,
    heads: &TVec<TDim>,
    groups: &TVec<TDim>,
) -> TractResult<Option<(OutletId, OutletId, TDim, TDim)>> {
    use tract_core::ops::array::Slice;
    let fold = model.node(outlet.node);
    rule_if!(fold.op_as::<AxisOp>() == Some(&AxisOp::Reshape(0, groups.clone(), heads.clone())));
    let dot = model.node(fold.inputs[0].node);
    rule_if_some!(op = dot.op_as::<EinSum>());
    rule_if!(dot.inputs.len() == 2 && op.q_params.is_none() && op.operating_dt == DatumType::F16);
    rule_if!(same_axes(&op.axes, "bhgsd,bhgtd->hgst")?);
    let q_fold = model.node(dot.inputs[0].node);
    rule_if!(q_fold.op_as::<AxisOp>() == Some(&AxisOp::Reshape(1, heads.clone(), groups.clone())));
    let slice = model.node(dot.inputs[1].node);
    rule_if_some!(sl = slice.op_as::<Slice>());
    rule_if!(sl.axis == 4);
    let broadcast = model.node(slice.inputs[0].node);
    rule_if!(broadcast.op_is::<MultiBroadcastTo>());
    let add = model.node(broadcast.inputs[0].node);
    rule_if!(add.op_as::<AxisOp>() == Some(&AxisOp::Add(2)));
    let q = q_fold.inputs[0];
    let k = add.inputs[0];
    let qf = model.outlet_fact(q)?;
    let kf = model.outlet_fact(k)?;
    rule_if!(
        [qf, kf]
            .iter()
            .all(|f| f.rank() == 4 && f.shape[0].is_one() && f.datum_type == DatumType::F16)
    );
    rule_if!(qf.shape[1] == heads[0] && kf.shape[1] == groups[0]);
    rule_if!(qf.shape[3] == sl.end.clone() - &sl.start);
    rule_if!(
        model.outlet_fact(broadcast.id.into())?.shape.to_tvec()
            == tvec![
                1.to_dim(),
                groups[0].clone(),
                groups[1].clone(),
                kf.shape[2].clone(),
                kf.shape[3].clone()
            ]
    );
    Ok(Some((q, k, sl.start.clone(), sl.end.clone())))
}

/// Opt-in sink-aware GQA conversion for two contiguous QK head-dimension slices.
/// Preserves the additive mask and existing caches while replacing intermediate
/// f16 score/probability rounding with f32 SDPA accumulation.
pub fn detect_sink_sdpa_rule(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    name: &str,
    pv: &EinSum,
) -> TractResult<Option<TypedModelPatch>> {
    use tract_core::ops::array::Slice;
    use tract_core::ops::array::TypedConcat;
    use tract_core::ops::binary::TypedBinOp;
    use tract_core::ops::math::{Add, Mul, Sub};
    use tract_core::ops::nn::{Reduce, Reducer, Softmax, SoftmaxKind};

    rule_if!(node.inputs.len() == 2 && pv.q_params.is_none() && pv.operating_dt == DatumType::F16);
    rule_if!(same_axes(&pv.axes, "hgsk,bhkd->hgsd")?);
    let fold = model.node(node.inputs[0].node);
    rule_if_some!(AxisOp::Reshape(0, heads, groups) = fold.op_as::<AxisOp>());
    rule_if!(heads.len() == 1 && groups.len() == 2);
    rule_if!(
        groups.iter().all(|d| d.to_usize().is_ok_and(|n| n > 0))
            && heads[0] == groups[0].clone() * &groups[1]
    );
    let cast = model.node(fold.inputs[0].node);
    rule_if!(cast.op_as::<Cast>().is_some_and(|c| c.to == DatumType::F16));
    let drop_sink = model.node(cast.inputs[0].node);
    rule_if_some!(sl = drop_sink.op_as::<Slice>());
    rule_if!(sl.axis == 2 && sl.start == 0.to_dim());
    let softmax = model.node(drop_sink.inputs[0].node);
    rule_if_some!(sm = softmax.op_as::<Softmax>());
    rule_if!(
        sm.axes == tvec![2] && sm.kind == SoftmaxKind::Softmax && sm.quant_output_dt.is_none()
    );
    let sub = model.node(softmax.inputs[0].node);
    rule_if!(sub.op_as::<TypedBinOp>().is_some_and(|op| op.0.is::<Sub>()));
    let concat = model.node(sub.inputs[0].node);
    rule_if!(
        concat.op_as::<TypedConcat>().is_some_and(|op| op.axis == 2) && concat.inputs.len() == 2
    );
    let max = model.node(sub.inputs[1].node);
    rule_if!(
        max.op_as::<Reduce>().is_some_and(|op| op.axes == tvec![2] && op.reducer == Reducer::Max)
    );
    rule_if!(max.inputs[0] == sub.inputs[0]);
    let sinks_broadcast = model.node(concat.inputs[1].node);
    rule_if!(sinks_broadcast.op_is::<MultiBroadcastTo>());
    let sinks = sinks_broadcast.inputs[0];
    let sf = model.outlet_fact(sinks)?;
    rule_if!(
        sf.datum_type == DatumType::F32
            && sf.shape.to_tvec() == tvec![heads[0].clone(), 1.to_dim(), 1.to_dim()]
    );
    let masked = model.node(concat.inputs[0].node);
    rule_if!(masked.op_as::<TypedBinOp>().is_some_and(|op| op.0.is::<Add>()));
    let cast = model.node(masked.inputs[0].node);
    rule_if!(cast.op_as::<Cast>().is_some_and(|c| c.to == DatumType::F32));
    let scale_mul = model.node(cast.inputs[0].node);
    rule_if!(scale_mul.op_as::<TypedBinOp>().is_some_and(|op| op.0.is::<Mul>()));
    let consts = model.collect_const_inputs(scale_mul);
    rule_if!(consts.len() == 1 && consts[0].val().len() == 1);
    rule_if_some!(
        scores = scale_mul
            .inputs
            .iter()
            .find(|i| model.outlet_fact(**i).is_ok_and(|f| f.konst.is_none()))
    );
    let sum = model.node(scores.node);
    rule_if!(sum.op_as::<TypedBinOp>().is_some_and(|op| op.0.is::<Add>()));
    rule_if_some!(a = sink_qk_branch(model, sum.inputs[0], heads, groups)?);
    rule_if_some!(b = sink_qk_branch(model, sum.inputs[1], heads, groups)?);
    let (a, b) = if a.2 == 0.to_dim() { (a, b) } else { (b, a) };
    let kf = model.outlet_fact(a.1)?;
    rule_if!(a.1 == b.1 && a.2 == 0.to_dim() && a.3 == b.2 && b.3 == kf.shape[3]);
    let qf = model.outlet_fact(a.0)?;
    let qbf = model.outlet_fact(b.0)?;
    rule_if!(qf.shape[..3] == qbf.shape[..3]);
    let v = node.inputs[1];
    let vf = model.outlet_fact(v)?;
    rule_if!(vf.shape == kf.shape && vf.datum_type == DatumType::F16 && sl.end == kf.shape[2]);
    let mask = masked.inputs[1];
    let mf = model.outlet_fact(mask)?;
    rule_if!(mf.rank() == 3 && mf.datum_type == DatumType::F32);
    rule_if!(
        (mf.shape[0].is_one() || mf.shape[0] == heads[0])
            && mf.shape[1] == qf.shape[2]
            && mf.shape[2] == kf.shape[2]
    );
    rule_if!(
        model.outlet_fact(sinks_broadcast.id.into())?.shape.to_tvec()
            == tvec![heads[0].clone(), qf.shape[2].clone(), 1.to_dim()]
    );

    let mut patch = TypedModelPatch::default();
    let qa = patch.tap_model(model, a.0)?;
    let qb = patch.tap_model(model, b.0)?;
    let q = patch.wire_node(format!("{name}.q"), TypedConcat { axis: 3 }, &[qa, qb])?[0];
    let k = patch.tap_model(model, a.1)?;
    let v = patch.tap_model(model, v)?;
    let mask = patch.tap_model(model, mask)?;
    let mask = patch.wire_node(format!("{name}.mask_batch"), AxisOp::Add(0), &[mask])?[0];
    let sinks = patch.tap_model(model, sinks)?;
    let sinks = patch.wire_node(
        format!("{name}.sink_heads"),
        AxisOp::Reshape(0, sf.shape.to_tvec(), heads.clone()),
        &[sinks],
    )?[0];
    let sdpa = Sdpa {
        scale: Some(tensor0(consts[0].val().cast_to_scalar::<f32>()?)),
        datum_type: DatumType::F16,
        acc_datum_type: DatumType::F32,
        is_causal: false,
    };
    let output = patch.wire_node(format!("{name}.sink_sdpa"), sdpa, &[q, k, v, mask, sinks])?[0];
    let output = patch.wire_node(format!("{name}.unbatch"), AxisOp::Rm(0), &[output])?[0];
    let output =
        patch.wire_node(name, AxisOp::Reshape(0, heads.clone(), groups.clone()), &[output])?[0];
    patch.shunt_outside(model, node.id.into(), output)?;
    Ok(Some(patch))
}
