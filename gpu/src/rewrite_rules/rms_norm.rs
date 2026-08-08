use tract_core::internal::*;
use tract_core::model::TypedModelHelpers;
use tract_core::ops::binary::TypedBinOp;
use tract_core::ops::cast::Cast;
use tract_core::ops::math::Mul;
use tract_core::ops::nn::ScaledRmsNorm;
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
    rule_if_some!(
        cast_in_node = model
            .single_prec(node.id)?
            .and_then(|n| n
                .op_as::<Cast>()
                .and_then(|cast| (cast.to == DatumType::F32).then_some(n)))
            .filter(|n| {
                model
                    .node_input_facts(n.id)
                    .map(|i| i[0].datum_type == DatumType::F16)
                    .unwrap_or(false)
            })
    );

    // Identify Cast from F32 To F16
    rule_if_some!(
        cast_out_node = model
            .single_succ(node.id)?
            .and_then(|n| n
                .op_as::<Cast>()
                .and_then(|cast| (cast.to == DatumType::F16).then_some(n)))
            .filter(|n| {
                model
                    .node_input_facts(n.id)
                    .map(|i| i[0].datum_type == DatumType::F32)
                    .unwrap_or(false)
            })
    );

    let mut patch = TypedModelPatch::default();
    let rsm_input = patch.taps(model, &cast_in_node.inputs)?;
    let out = patch.wire_node(format!("{node_name}.without-cast"), op.clone(), &rsm_input)?;
    patch.shunt_outside(model, cast_out_node.id.into(), out[0])?;
    Ok(Some(patch))
}

/// Returns the (single) const input of a `Mul` bin op node when that const is
/// a per-axis vector: non-trivial only along `axis` of a rank-`rank` operand.
fn mul_axis_vector_const<'m>(
    model: &'m TypedModel,
    mul: &TypedNode,
    rank: usize,
    axis: usize,
    axis_dim: usize,
) -> Option<&'m tract_core::ops::konst::Const> {
    let mul_op = mul.op_as::<TypedBinOp>()?;
    if !mul_op.0.is::<Mul>() {
        return None;
    }
    let consts = model.collect_const_inputs(mul);
    if consts.len() != 1 {
        return None;
    }
    let w = consts[0].val();
    if !w.datum_type().is_float() || w.rank() > rank {
        return None;
    }
    let offset = rank - w.rank();
    for (ix, d) in w.shape().iter().enumerate() {
        if ix + offset == axis {
            if *d != axis_dim {
                return None;
            }
        } else if *d != 1 {
            return None;
        }
    }
    if w.len() != axis_dim {
        return None;
    }
    Some(consts[0])
}

/// Search pattern => RMS_NORM(A) * W (W a per-axis const vector, the classic
/// learned `gamma`). Rewrites to the fused `ScaledRmsNorm(A, W)` so GPU
/// backends run norm + weight multiply as one kernel. The scale is stored as
/// a rank-1 F32 const whatever its original dtype (the fused kernels multiply
/// in F32).
pub fn fuse_rms_norm_scale(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    node_name: &str,
    op: &RmsNorm,
) -> TractResult<Option<TypedModelPatch>> {
    let in_fact = model.node_input_facts(node.id)?[0];
    rule_if_some!(axis_dim = in_fact.shape[op.axis].to_usize().ok());
    rule_if_some!(mul = model.single_succ(node.id)?);
    rule_if_some!(w = mul_axis_vector_const(model, mul, in_fact.rank(), op.axis, axis_dim));

    let scale = w
        .val()
        .cast_to::<f32>()?
        .into_owned()
        .into_shape(&[axis_dim])?
        .into_arc_tensor();

    let mut patch = TypedModelPatch::default();
    let rsm_input = patch.taps(model, &node.inputs)?;
    let scale = patch.add_const(format!("{node_name}.scale"), scale)?;
    let out = patch.wire_node(
        format!("{node_name}.scaled"),
        ScaledRmsNorm { axis: op.axis, eps: op.eps.clone(), out_dt: None },
        &[rsm_input[0], scale],
    )?;
    patch.shunt_outside(model, mul.id.into(), out[0])?;
    Ok(Some(patch))
}

/// Search pattern => CAST(RMS_NORM(A), F16) * W_f16 (a downcast separating
/// the norm from its learned weight multiply). Moves the multiply above the
/// cast (promoting W to F32) so `fuse_rms_norm_scale` and the cast-removal
/// rules can then collapse the whole chain into one fused kernel.
pub fn swap_rms_norm_cast_mul(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    node_name: &str,
    op: &RmsNorm,
) -> TractResult<Option<TypedModelPatch>> {
    let in_fact = model.node_input_facts(node.id)?[0];
    rule_if!(in_fact.datum_type == DatumType::F32);
    rule_if_some!(axis_dim = in_fact.shape[op.axis].to_usize().ok());
    rule_if_some!(
        cast_out_node = model
            .single_succ(node.id)?
            .and_then(|n| n.op_as::<Cast>().and_then(|cast| (cast.to == DatumType::F16).then_some(n)))
    );
    rule_if_some!(mul = model.single_succ(cast_out_node.id)?);
    rule_if_some!(w = mul_axis_vector_const(model, mul, in_fact.rank(), op.axis, axis_dim));

    let w_shape = w.val().shape().to_vec();
    let w32 = w.val().cast_to::<f32>()?.into_owned().into_shape(&w_shape)?.into_arc_tensor();

    let mut patch = TypedModelPatch::default();
    let rsm_input = patch.taps(model, &node.inputs)?;
    let normed = patch.wire_node(format!("{node_name}.pre-mul"), op.clone(), &rsm_input)?;
    let w32 = patch.add_const(format!("{node_name}.scale-f32"), w32)?;
    let scaled = patch.wire_node(
        format!("{node_name}.mul-f32"),
        TypedBinOp(Box::new(Mul), None),
        &[normed[0], w32],
    )?;
    let out = patch.wire_node(
        format!("{node_name}.post-cast"),
        Cast { to: DatumType::F16 },
        &scaled,
    )?;
    patch.shunt_outside(model, mul.id.into(), out[0])?;
    Ok(Some(patch))
}

/// Folds a float-to-float cast feeding a `ScaledRmsNorm` into the op (the
/// kernel loads through an F32 accumulator whatever the input dtype). The
/// output dtype is pinned so downstream facts do not change.
pub fn fuse_scaled_rms_norm_in_cast(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    node_name: &str,
    op: &ScaledRmsNorm,
) -> TractResult<Option<TypedModelPatch>> {
    let prec = &model.nodes()[node.inputs[0].node];
    rule_if_some!(cast_in = prec.op_as::<Cast>().and_then(|c| c.to.is_float().then_some(prec)));
    let cast_src_dt = model.node_input_facts(cast_in.id)?[0].datum_type;
    rule_if!(matches!(cast_src_dt, DatumType::F16 | DatumType::F32));
    let out_dt = op.out_dt.unwrap_or(node.outputs[0].fact.datum_type);

    let mut patch = TypedModelPatch::default();
    let data_input = patch.taps(model, &cast_in.inputs)?;
    let scale_input = patch.tap_model(model, node.inputs[1])?;
    let out = patch.wire_node(
        format!("{node_name}.in-cast-folded"),
        ScaledRmsNorm { axis: op.axis, eps: op.eps.clone(), out_dt: Some(out_dt) },
        &[data_input[0], scale_input],
    )?;
    patch.shunt_outside(model, node.id.into(), out[0])?;
    Ok(Some(patch))
}

/// Folds a float-to-float cast consuming a `ScaledRmsNorm` into the op's
/// `out_dt` (the kernel casts once when writing its output).
pub fn fuse_scaled_rms_norm_out_cast(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    node_name: &str,
    op: &ScaledRmsNorm,
) -> TractResult<Option<TypedModelPatch>> {
    rule_if_some!(
        cast_out = model.single_succ(node.id)?.and_then(|n| {
            n.op_as::<Cast>().and_then(|c| {
                (c.to.is_float()
                    && matches!(c.to, DatumType::F16 | DatumType::F32)
                    && c.to != node.outputs[0].fact.datum_type)
                    .then_some(n)
            })
        })
    );
    let to = cast_out.op_as::<Cast>().unwrap().to;

    let mut patch = TypedModelPatch::default();
    let inputs = patch.taps(model, &node.inputs)?;
    let out = patch.wire_node(
        format!("{node_name}.out-cast-folded"),
        ScaledRmsNorm { axis: op.axis, eps: op.eps.clone(), out_dt: Some(to) },
        &inputs,
    )?;
    patch.shunt_outside(model, cast_out.id.into(), out[0])?;
    Ok(Some(patch))
}

/// Bypasses a float precision round-trip: `CAST(CAST(x, narrow), wide)` where
/// `x` is already `wide`. Skipping the intermediate rounding only gains
/// precision, and removes up to two kernel dispatches on GPU backends.
pub fn bypass_float_downcast_roundtrip(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    _node_name: &str,
    op: &Cast,
) -> TractResult<Option<TypedModelPatch>> {
    rule_if!(op.to.is_float());
    let prec = &model.nodes()[node.inputs[0].node];
    rule_if_some!(down = prec.op_as::<Cast>().and_then(|c| c.to.is_float().then_some(prec)));
    let down_op = down.op_as::<Cast>().unwrap();
    let src_outlet = down.inputs[0];
    let src_dt = model.outlet_fact(src_outlet)?.datum_type;
    // x (wide) -> narrow -> wide again
    rule_if!(src_dt == op.to);
    rule_if!(src_dt.size_of() > down_op.to.size_of());

    let mut patch = TypedModelPatch::default();
    let src = patch.tap_model(model, src_outlet)?;
    patch.shunt_outside(model, node.id.into(), src)?;
    Ok(Some(patch))
}
