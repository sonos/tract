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

/// Search pattern => RMS_NORM(A) whose learned weight multiply was split by
/// declutter's PushSliceUp over a downstream boundary (e.g. a partial-rotary
/// rope): every successor of the norm is a static `Slice` on the norm axis,
/// the slices tile [0, dim) exactly, and each slice feeds `* W_i` (a const
/// vector, the matching slice of the gamma), each optionally followed by a
/// single float cast, all branches ending on one dtype. Reassembles the full
/// gamma and rewrites to one fused `ScaledRmsNorm` re-sliced per branch: the
/// norm+scale runs as a single kernel instead of one norm plus a mul(+cast)
/// dispatch per slice. Bit-exact: the fused kernel computes
/// `cast((x * norm) * gamma)` in f32 with one rounding at the write, the
/// same op sequence as the split form.
pub fn fuse_rms_norm_split_scale(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    node_name: &str,
    op: &RmsNorm,
) -> TractResult<Option<TypedModelPatch>> {
    use tract_core::ops::array::Slice;

    if std::env::var_os("TRACT_GPU_DISABLE_RMS_NORM_SPLIT_SCALE").is_some() {
        return Ok(None);
    }
    let in_fact = model.node_input_facts(node.id)?[0];
    rule_if_some!(axis_dim = in_fact.shape[op.axis].to_usize().ok());
    rule_if!(node.outputs.len() == 1);
    let succs = node.outputs[0].successors.clone();
    rule_if!(succs.len() >= 2);

    // (start, end, gamma slice, outlet to shunt)
    let mut branches: Vec<(usize, usize, Arc<Tensor>, OutletId)> = Vec::new();
    let mut common_dt: Option<DatumType> = None;
    for inlet in &succs {
        let slice_node = &model.nodes()[inlet.node];
        rule_if_some!(slice = slice_node.op_as::<Slice>());
        rule_if!(slice.axis == op.axis);
        rule_if_some!(start = slice.start.to_usize().ok());
        rule_if_some!(end = slice.end.to_usize().ok());
        rule_if!(end > start);
        rule_if_some!(mul = model.single_succ(slice_node.id)?);
        rule_if_some!(
            w = mul_axis_vector_const(model, mul, in_fact.rank(), op.axis, end - start)
        );
        // Optional single float cast closing the branch.
        let (shunt, dt) = match model.single_succ(mul.id)? {
            Some(c)
                if c.op_as::<Cast>().is_some_and(|cast| cast.to.is_float())
                    && c.inputs.len() == 1 =>
            {
                (OutletId::new(c.id, 0), c.op_as::<Cast>().unwrap().to)
            }
            _ => (OutletId::new(mul.id, 0), mul.outputs[0].fact.datum_type),
        };
        rule_if!(common_dt.is_none_or(|d| d == dt));
        common_dt = Some(dt);
        let w32 = w.val().cast_to::<f32>()?.into_owned().into_shape(&[end - start])?;
        branches.push((start, end, w32.into_arc_tensor(), shunt));
    }
    // The slices must tile the norm axis exactly (no gap, no overlap).
    branches.sort_by_key(|b| b.0);
    rule_if!(branches.first().is_some_and(|b| b.0 == 0));
    rule_if!(branches.last().is_some_and(|b| b.1 == axis_dim));
    rule_if!(branches.windows(2).all(|w| w[0].1 == w[1].0));

    let mut gamma = vec![0f32; axis_dim];
    for (start, end, w, _) in &branches {
        let w = w.to_plain_array_view::<f32>()?;
        let w = w.as_slice().context("gamma slice must be contiguous")?;
        gamma[*start..*end].copy_from_slice(w);
    }

    let mut patch = TypedModelPatch::default();
    let rsm_input = patch.taps(model, &node.inputs)?;
    let gamma = patch.add_const(format!("{node_name}.split-scale"), tensor1(&gamma))?;
    let scaled = patch.wire_node(
        format!("{node_name}.split-scaled"),
        ScaledRmsNorm { axis: op.axis, eps: op.eps.clone(), out_dt: common_dt },
        &[rsm_input[0], gamma],
    )?;
    for (ix, (start, end, _, shunt)) in branches.iter().enumerate() {
        let sliced = patch.wire_node(
            format!("{node_name}.split-scaled.{ix}"),
            Slice::new(op.axis, *start, *end),
            &[scaled[0]],
        )?;
        patch.shunt_outside(model, *shunt, sliced[0])?;
    }
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
    // Only fold WIDENING casts (f16 -> f32): those are exact, so removing
    // the cast node cannot change semantics. Folding a narrowing cast would
    // delete a rounding step the source model may rely on (same invariant
    // as bypass_device_downcast_roundtrip's synthetic gate).
    let cast_to = cast_in.op_as::<Cast>().unwrap().to;
    rule_if!(cast_to.size_of() > cast_src_dt.size_of());
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

#[cfg(test)]
mod tests {
    use super::*;
    use tract_core::ops::array::Slice;

    /// Builds the PushSliceUp-split gamma pattern the rule targets:
    /// f16 input -> Cast(f32) -> RmsNorm -> per-range Slice -> Mul(gamma
    /// slice) -> Cast(f16), one output per range.
    fn split_scale_model(ranges: &[(usize, usize)]) -> TractResult<TypedModel> {
        let mut model = TypedModel::default();
        let input = model.add_source("input", f16::datum_type().fact([2usize, 8]))?;
        let cast =
            model.wire_node("cast-in", Cast { to: DatumType::F32 }, &[input])?;
        let norm = model.wire_node(
            "norm",
            RmsNorm { axis: 1, eps: Arc::new(tensor0(1e-4f32)) },
            &cast,
        )?;
        let mut outputs = tvec![];
        for (ix, (start, end)) in ranges.iter().enumerate() {
            let sliced = model.wire_node(
                format!("slice.{ix}"),
                Slice::new(1, *start, *end),
                &norm,
            )?;
            let gamma = model.add_const(
                format!("gamma.{ix}"),
                Tensor::from_shape(
                    &[1, end - start],
                    &(*start..*end).map(|i| 0.25 + i as f32 / 8.0).collect::<Vec<_>>(),
                )?,
            )?;
            let mul = model.wire_node(
                format!("mul.{ix}"),
                TypedBinOp(Box::new(Mul), None),
                &[sliced[0], gamma],
            )?;
            let out =
                model.wire_node(format!("cast-out.{ix}"), Cast { to: DatumType::F16 }, &mul)?;
            outputs.push(out[0]);
        }
        model.select_output_outlets(&outputs)?;
        Ok(model)
    }

    fn run_rule(model: &mut TypedModel) -> TractResult<()> {
        Rewriter::default()
            .with_rule_for("fuse_rms_norm_split_scale", fuse_rms_norm_split_scale)
            .rewrite(&(), model)
    }

    fn eval(model: &TypedModel) -> TractResult<TVec<TValue>> {
        let input = Tensor::from_shape(
            &[2, 8],
            &(0..16).map(|i| f16::from_f32(i as f32 / 3.0 - 2.0)).collect::<Vec<_>>(),
        )?;
        SimplePlan::new(model.clone())?.run(tvec![input.into_tvalue()])
    }

    #[test]
    fn split_scale_tiling_fuses() -> TractResult<()> {
        let mut model = split_scale_model(&[(0, 2), (2, 4), (4, 8)])?;
        let reference = eval(&model)?;
        run_rule(&mut model)?;
        ensure!(model.nodes().iter().any(|n| n.op_is::<ScaledRmsNorm>()));
        ensure!(!model.nodes().iter().any(|n| n.op_is::<RmsNorm>()));
        ensure!(!model.nodes().iter().any(|n| n.op_is::<TypedBinOp>()));
        let fused = eval(&model)?;
        for (r, f) in reference.iter().zip(fused.iter()) {
            r.close_enough(f, Approximation::Approximate)?;
        }
        Ok(())
    }

    #[test]
    fn split_scale_gap_does_not_fire() -> TractResult<()> {
        // 2..4 missing: slices do not tile the axis, the rule must not fire.
        let mut model = split_scale_model(&[(0, 2), (4, 8)])?;
        run_rule(&mut model)?;
        ensure!(model.nodes().iter().any(|n| n.op_is::<RmsNorm>()));
        ensure!(!model.nodes().iter().any(|n| n.op_is::<ScaledRmsNorm>()));
        Ok(())
    }
}
