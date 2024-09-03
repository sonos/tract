use tract_linalg::frame::block_quant::PackedBlockQuantFormat;
use tract_linalg::frame::PackedFormat;
use tract_linalg::mmm::{MMMInputFormat, MMMInputValue, MatMatMul};

use super::*;
use crate::ops::cast::cast;
use crate::ops::math::add;
use crate::ops::matmul::de_block_quant::BlockQuantValue;
use crate::ops::matmul::optimized::{
    AddMatMulGeometry, MapOutputAxisToInput, OptMatMul, ProtoFusedSpec,
};
use crate::ops::matmul::pack::MatMatMulPack;
use crate::ops::matmul::quant::{
    combine_scales, compensate_zero_points, requant, wire_ensure_q8_flavour,
};
use crate::ops::nn::{Reduce, Reducer};

#[allow(clippy::large_enum_variant)]
pub enum AxesOrPatch<'a> {
    Axes(&'a Axis, &'a Axis, &'a Axis),
    Patch(TypedModelPatch),
    NotAMatMul(&'a Axis),
}

pub(crate) fn codegen(
    op: &EinSum,
    model: &TypedModel,
    node: &TypedNode,
) -> TractResult<Option<TypedModelPatch>> {
    if (op.q_params.is_none() && node.inputs.len() != 2)
        || (op.q_params.is_some() && node.inputs.len() != 9)
    {
        return Ok(None);
    }
    let (m_axis, k_axis, n_axis) = match ensure_mkn_axes(op, model, node)? {
        AxesOrPatch::Axes(m, k, n) => (m, k, n),
        AxesOrPatch::Patch(p) => return Ok(Some(p)),
        AxesOrPatch::NotAMatMul(_) => return Ok(None),
    };
    if op.q_params.is_none() {
        optimized_mat_mul(op, model, node, (m_axis, k_axis, n_axis))
            .context("Translating to OptMatMul")
    } else {
        dequant(op, model, node, (m_axis, k_axis, n_axis)).context("Dequantize")
    }
}

pub(crate) fn ensure_mkn_axes<'a>(
    op: &'a EinSum,
    model: &TypedModel,
    node: &TypedNode,
) -> TractResult<AxesOrPatch<'a>> {
    let input_facts = model.node_input_facts(node.id)?;
    let input_shapes = op.actual_input_shapes_from_facts(&input_facts)?;
    let output_shape = super::eval::output_shape(&op.axes, &input_shapes);
    let candidate_k_axes: TVec<&Axis> = op
        .axes
        .iter_all_axes()
        // Filter possible candidates (should be one time in each inputs but not in output)
        .filter(|a| {
            a.inputs[0].len() == 1 && a.inputs[1].len() == 1 && a.outputs[0].len() == 0 &&
                input_shapes[0][a.inputs[0][0]] == input_shapes[1][a.inputs[1][0]]
        })
    .collect();

    let non_trivial_k_axis = candidate_k_axes
        .iter()
        .filter(|a| !input_shapes[0][a.inputs[0][0]].is_one())
        .collect::<TVec<_>>();

    let k_axis = if non_trivial_k_axis.len() > 1 {
        // TODO: handle case where multiple consecutive k in the same order in both input.
        bail!("Multiple k-axis candidate found");
    } else {
        non_trivial_k_axis.first().copied().or_else(|| candidate_k_axes.first()).copied()
    };
    let Some(k_axis) = k_axis else {
        return Ok(AxesOrPatch::Patch(inject_k_axis(op, model, node)?));
    };
    let m_axis = op
        .axes
        .iter_all_axes()
        .filter(|a| {
            a.inputs[0].len() == 1
                && (a.inputs[1].len() == 0 || input_shapes[1][a.inputs[1][0]].is_one())
                && a.outputs[0].len() == 1
        })
        .max_by_key(|a| output_shape[a.outputs[0][0]].as_i64().unwrap_or(i64::MAX));
    let Some(m_axis) = m_axis else {
        return Ok(AxesOrPatch::Patch(inject_m_or_n_axis(op, model, node, false, &[k_axis])?));
    };
    let n_axis = op
        .axes
        .iter_all_axes()
        .filter(|a| {
            (a.inputs[0].len() == 0 || input_shapes[0][a.inputs[0][0]].is_one())
                && a.inputs[1].len() == 1
                && a.outputs[0].len() == 1
        })
        .max_by_key(|a| output_shape[a.outputs[0][0]].as_i64().unwrap_or(i64::MAX));
    let Some(n_axis) = n_axis else {
        return Ok(AxesOrPatch::Patch(inject_m_or_n_axis(
            op,
            model,
            node,
            true,
            &[k_axis, m_axis],
        )?));
    };
    for axis in op.axes.iter_all_axes() {
        let one = TDim::one();
        let in_left =
            axis.inputs[0].first().map(|pos| &input_shapes[0][*pos]).unwrap_or(&one) != &one;
        let in_right =
            axis.inputs[1].first().map(|pos| &input_shapes[1][*pos]).unwrap_or(&one) != &one;
        let in_out = axis.outputs[0].first().map(|pos| &output_shape[*pos]).unwrap_or(&one) != &one;
        if (in_left ^ in_right) && !in_out {
            return Ok(AxesOrPatch::NotAMatMul(axis));
        }
    }
    Ok(AxesOrPatch::Axes(m_axis, k_axis, n_axis))
}

pub(super) fn inject_k_axis(
    op: &EinSum,
    model: &TypedModel,
    node: &TypedNode,
) -> TractResult<TypedModelPatch> {
    let mut new_axes = op.axes.clone();
    let name = &node.name;
    let mut patch = TypedModelPatch::new("inject k axis");
    let mut wire = patch.taps(model, &node.inputs)?;
    let repr = new_axes.available_label();
    new_axes = new_axes.with_extra_axis(repr, InOut::In(0), 0)?.with_extra_axis_occurency(
        repr,
        InOut::In(1),
        0,
    )?;
    wire[0] = patch.wire_node(format!("{name}.add_k.0"), AxisOp::Add(0), &[wire[0]])?[0];
    wire[1] = patch.wire_node(format!("{name}.add_k.1"), AxisOp::Add(0), &[wire[1]])?[0];
    wire = patch.wire_node(&node.name, EinSum { axes: new_axes, ..op.clone() }, &wire)?;
    patch.shunt_outside(model, node.id.into(), wire[0])?;
    Ok(patch)
}

pub(super) fn inject_m_or_n_axis(
    op: &EinSum,
    model: &TypedModel,
    node: &TypedNode,
    is_n: bool,
    exclude: &[&Axis],
) -> TractResult<TypedModelPatch> {
    let input_facts = model.node_input_facts(node.id)?;
    let input_shapes = op.actual_input_shapes_from_facts(&input_facts)?;
    let input_to_fix = is_n as usize;
    let label = if is_n { "n" } else { "m" };
    let quasi_m_or_n_axis = op.axes.iter_all_axes().filter(|a| !exclude.contains(a)).find(|a| {
        (a.inputs[1 - input_to_fix].len() == 0
            || input_shapes[1 - input_to_fix][a.inputs[1 - input_to_fix][0]].is_one())
            && (a.inputs[input_to_fix].len() == 1 || a.outputs[0].len() == 1)
    });
    let name = &node.name;
    let mut patch = TypedModelPatch::new("Injecting m or n axis");
    let mut wire = patch.taps(model, &node.inputs)?;
    if let Some(axis) = quasi_m_or_n_axis {
        if axis.inputs[input_to_fix].len() == 1 {
            let new_axes =
                op.axes.clone().with_extra_axis('$', InOut::Out(0), 0)?.linking(axis.repr, '$')?;
            wire = patch.wire_node(
                format!("{name}.einsum"),
                EinSum { axes: new_axes, ..op.clone() },
                &wire,
            )?;
            wire = patch.wire_node(&node.name, AxisOp::Rm(0), &wire)?;
        } else {
            let new_axes = op
                .axes
                .clone()
                .with_extra_axis('$', InOut::In(input_to_fix), 0)?
                .linking(axis.repr, '$')?;
            wire[input_to_fix] = patch.wire_node(
                format!("{name}.add_{label}"),
                AxisOp::Add(0),
                &[wire[input_to_fix]],
            )?[0];
            wire = patch.wire_node(&node.name, EinSum { axes: new_axes, ..op.clone() }, &wire)?;
        }
    } else {
        let repr = op.axes.available_label();
        let new_axes = op
            .axes
            .clone()
            .with_extra_axis(repr, InOut::In(input_to_fix), 0)?
            .with_extra_axis('$', InOut::Out(0), 0)?
            .linking(repr, '$')?;
        wire[input_to_fix] = patch.wire_node(
            format!("{name}.add_{label}"),
            AxisOp::Add(0),
            &[wire[input_to_fix]],
        )?[0];
        wire = patch.wire_node(
            format!("{name}.einsum"),
            EinSum { axes: new_axes, ..op.clone() },
            &wire,
        )?;
        wire = patch.wire_node(&node.name, AxisOp::Rm(0), &wire)?;
    }
    patch.shunt_outside(model, node.id.into(), wire[0])?;
    Ok(patch)
}

fn wire_axes_fix(
    patch: &mut TypedModelPatch,
    name: &str,
    var: &str,
    mapping: &AxesMapping,
    mut outlet: TVec<OutletId>,
) -> TractResult<TVec<OutletId>> {
    for (ix, axis_op) in mapping.translate_to_axis_ops()?.into_iter().enumerate() {
        outlet = patch.wire_node(format!("{name}.fix_{var}.{ix})"), axis_op, &outlet)?;
    }
    Ok(outlet)
}

fn dequant(
    op: &EinSum,
    model: &TypedModel,
    node: &TypedNode,
    (_, k_axis, _): (&Axis, &Axis, &Axis),
) -> TractResult<Option<TypedModelPatch>> {
    let name = &node.name;
    let mut patch = TypedModelPatch::new("Dequantizing einsum");

    let mut taps = patch.taps(model, &node.inputs)?;
    for ab in [0, 1] {
        let scale_input = 4 + ab * 2;
        if !patch.outlet_fact(taps[scale_input])?.shape.volume().is_one() {
            let q_axis_in_output = op.axes.axis((InOut::In(scale_input), 0))?.outputs[0][0];
            let output_rank = node.outputs[0].fact.rank();
            for i in 1..(output_rank - q_axis_in_output) {
                taps[scale_input] = patch.wire_node(
                    format!("{name}.scale_input{ab}_axis_fix_{i}"),
                    AxisOp::Add(i),
                    &[taps[scale_input]],
                )?[0];
            }
        }
    }

    let [mut a, mut b, bias, mut a0, a_scale, mut b0, b_scale, c0, c_scale] = *taps else {
        bail!("Expect exactly 9 inputs")
    };

    wire_ensure_q8_flavour(&mut patch, &node.name, &mut a, "a", &mut a0, i8::datum_type())?;
    wire_ensure_q8_flavour(&mut patch, &node.name, &mut b, "b", &mut b0, i8::datum_type())?;

    let mut output = patch.wire_node(
        &node.name,
        EinSum {
            q_params: None,
            axes: op.axes.extract_sub_mapping(&[0, 1], &[0])?,
            operating_dt: op.operating_dt,
        },
        &[a, b],
    )?;

    let a_i32 = patch.wire_node(format!("{name}.a_as_i32"), cast(i32::datum_type()), &[a])?[0];
    let b_i32 = patch.wire_node(format!("{name}.b_as_i32"), cast(i32::datum_type()), &[b])?[0];
    let sum_a = patch.wire_node(
        format!("{name}.sum_a"),
        Reduce::new(tvec!(k_axis.inputs[0][0]), Reducer::Sum),
        &[a_i32],
    )?;
    let sum_b = patch.wire_node(
        format!("{name}.sum_b"),
        Reduce::new(tvec!(k_axis.inputs[1][0]), Reducer::Sum),
        &[b_i32],
    )?;

    let sum_a =
        wire_axes_fix(&mut patch, name, "sum_a", &op.axes.extract_sub_mapping(&[0], &[0])?, sum_a)?;
    let sum_b =
        wire_axes_fix(&mut patch, name, "sum_b", &op.axes.extract_sub_mapping(&[1], &[0])?, sum_b)?;
    let bias = tvec!(bias);
    let bias =
        wire_axes_fix(&mut patch, name, "bias", &op.axes.extract_sub_mapping(&[2], &[0])?, bias)?;

    let abc_scale = combine_scales(&mut patch, name, a_scale, b_scale, c_scale)?;

    output = patch.wire_node(format!("{name}.add_bias"), add(), &[output[0], bias[0]])?;

    let k = model.outlet_fact(node.inputs[0])?.shape[k_axis.inputs[0][0]].clone();
    let output = compensate_zero_points(&mut patch, name, output[0], k, a0, b0, sum_a[0], sum_b[0])
        .context("Zero point compensation")?;
    let output = requant(&mut patch, name, output, op.q_params.unwrap(), abc_scale, c0)?;
    patch.shunt_outside(model, node.id.into(), output)?;
    Ok(Some(patch))
}

fn select_kernel_and_packing(
    model: &TypedModel,
    node: &TypedNode,
    m: &TDim,
    n: &TDim,
) -> TractResult<Option<(Box<dyn MatMatMul>, usize)>> {
    if let Some(bqv) = model
        .outlet_fact(node.inputs[0])?
        .konst
        .as_ref()
        .filter(|t| t.volume() == 1 && t.datum_type().is_opaque())
        .and_then(|t| t.as_slice::<Opaque>().unwrap()[0].downcast_ref::<BlockQuantValue>())
    {
        let mut options: Vec<(&Box<dyn MatMatMul>, usize)> = vec![];
        let b_dt = model.outlet_fact(node.inputs[1])?.datum_type;
        for imp in tract_linalg::ops().mmm_impls() {
            for (packing, (pack_a, pack_b)) in imp.packings().iter().enumerate() {
                if let (Some(input), Some(b)) = (
                    pack_a.downcast_ref::<PackedBlockQuantFormat>(),
                    pack_b.downcast_ref::<PackedFormat>(),
                ) {
                    if input.bq.same_as(&*bqv.fact.format) && b.dt == b_dt {
                        options.push((imp, packing));
                    }
                }
            }
        }
        if options.len() > 0 {
            let pair = if let (Some(m), Some(n)) = (m.as_i64(), n.as_i64()) {
                options
                    .iter()
                    .min_by_key(|a| {
                        ((m as usize).divceil(a.0.mr()) * (n as usize).divceil(a.0.nr()))
                            * (a.0.mr() * a.0.nr() + 100)
                    })
                    .unwrap()
            } else {
                options.iter().max_by_key(|a| a.0.mr() * a.0.nr()).unwrap()
            };
            return Ok(Some((pair.0.clone(), pair.1)));
        }
    }
    Ok(None)
}

fn wire_packing(
    model: &TypedModel,
    node: &TypedNode,
    input: usize,
    patch: &mut TypedModelPatch,
    packer: &dyn MMMInputFormat,
    k_axis: usize,
    mn_axis: usize,
) -> TractResult<OutletId> {
    let name = format!("{}.pack_{}", node.name, ['a', 'b'][input]);
    let a_fact = model.outlet_fact(node.inputs[0])?;
    if let Some(packed_format) = packer.downcast_ref::<PackedFormat>().cloned() {
        let wire = patch.tap_model(model, node.inputs[input])?;
        let pack_a = MatMatMulPack { packer: packed_format, k_axis, mn_axis };
        Ok(patch.wire_node(&name, pack_a, &[wire])?[0])
    } else if let (Some(bqf), Some(pbqf)) = (
        a_fact.opaque_fact.as_ref().and_then(|of| of.downcast_ref::<BlockQuantFact>()),
        packer.downcast_ref::<PackedBlockQuantFormat>(),
    ) {
        ensure!(k_axis == 1);
        ensure!(mn_axis == 0);
        ensure!(pbqf.bq.same_as(&*bqf.format));
        let Some(weights) = &a_fact.konst else {
            bail!("Block quant packing with non-const inputs")
        };
        ensure!(weights.datum_type() == Opaque::datum_type());
        let Some(weights) = weights.to_scalar::<Opaque>()?.downcast_ref::<BlockQuantValue>() else {
            bail!("Expected a BlockQuantValue, found {weights:?}")
        };
        let k = bqf.shape[k_axis].to_usize()?;
        let packed = pbqf.pack(&weights.value, k)?;
        let mmm_input: Box<dyn MMMInputValue> = Box::new(packed);
        patch.add_const(name, tensor0(Opaque::from(mmm_input)))
    } else {
        bail!("Unexpected packing format: {:?}", packer);
    }
}

fn optimized_mat_mul(
    op: &EinSum,
    model: &TypedModel,
    node: &TypedNode,
    (m_axis, k_axis, n_axis): (&Axis, &Axis, &Axis),
) -> TractResult<Option<TypedModelPatch>> {
    let input_facts = model.node_input_facts(node.id)?;
    let input_shapes = op.actual_input_shapes_from_facts(&input_facts)?;
    let a_m = m_axis.inputs[0][0];
    let a_k = k_axis.inputs[0][0];
    let b_n = n_axis.inputs[1][0];
    let b_k = k_axis.inputs[1][0];
    let c_m = m_axis.outputs[0][0];
    let c_n = n_axis.outputs[0][0];
    let m = &input_shapes[0][a_m];
    let k = &input_shapes[0][a_k];
    let n = &input_shapes[1][b_n];
    let must_transpose = match (m.as_i64(), n.as_i64()) {
        (Some(m), Some(n)) => m < n,
        (None, Some(n)) => n >= 8,
        _ => false,
    };
    if must_transpose {
        let expr = op
            .axes
            .iter_all_axes()
            .map(|axis| {
                let mut axis = axis.clone();
                axis.inputs.swap(0, 1);
                axis
            })
            .collect::<TVec<Axis>>();
        return TypedModelPatch::replace_single_op(
            model,
            node,
            &[node.inputs[1], node.inputs[0]],
            EinSum { axes: AxesMapping::new(node.inputs.len(), 1, expr)?, ..op.clone() },
        )
        .map(Some);
    }

    let (mmm, packing) = if let Some(pair) = select_kernel_and_packing(model, node, m, n)? {
        pair
    } else {
        let a_dt = input_facts[0].datum_type;
        let b_dt = input_facts[1].datum_type;
        let mmm = tract_linalg::ops()
            .mmm(op.operating_dt, m.to_usize().ok(), k.to_usize().ok(), n.to_usize().ok())
            .unwrap();
        let packing = mmm
            .packings()
            .iter()
            .position(|p| {
                p.0.can_prepare_types().contains(&a_dt.unquantized())
                    && p.1.can_prepare_types().contains(&b_dt.unquantized())
            })
            .with_context(|| format!("No packing for {mmm:?} with inputs {a_dt:?} and {b_dt:?}"))?;
        (mmm, packing)
    };

    let mut patch = TypedModelPatch::new("Einsum to OptMatMul");
    let packers = mmm.packings()[packing];

    let pa = wire_packing(model, node, 0, &mut patch, packers.0, a_k, a_m)
        .with_context(|| format!("Wiring packing {:?} for a: {:?}", packers.0, input_facts[0]))?;
    let pb = wire_packing(model, node, 1, &mut patch, packers.1, b_k, b_n)
        .with_context(|| format!("Wiring packing {:?} for b: {:?}", packers.1, input_facts[1]))?;

    let mut c_to_a_axis_mapping = tvec!();
    let mut c_to_b_axis_mapping = tvec!();
    for axis in op.axes.iter_all_axes().filter(|&axis| ![m_axis, k_axis, n_axis].contains(&axis)) {
        if let (&[c], &[a]) = (&*axis.outputs[0], &*axis.inputs[0]) {
            if input_shapes[0][a] != 1.to_dim() {
                let a = a - (a > a_m) as usize - (a > a_k) as usize;
                c_to_a_axis_mapping.push((c, a));
            }
        }
        if let (&[c], &[b]) = (&*axis.outputs[0], &*axis.inputs[1]) {
            if input_shapes[1][b] != 1.to_dim() {
                let b = b - (b > b_n) as usize - (b > b_k) as usize;
                c_to_b_axis_mapping.push((c, b));
            }
        }
    }

    let c_fact = op.output_facts(&input_facts)?.remove(0);
    let name = &node.name;
    let geo = AddMatMulGeometry {
        k: k.clone(),
        c_to_a_axis_mapping: MapOutputAxisToInput(c_to_a_axis_mapping),
        c_to_b_axis_mapping: MapOutputAxisToInput(c_to_b_axis_mapping),
    };
    let output = unsafe { mmm.c_view(c_m, c_n) };
    let opt = OptMatMul::new(
        mmm,
        c_fact,
        c_m,
        c_n,
        vec![ProtoFusedSpec::AddMatMul { geo, a: 0, b: 1, packing }, ProtoFusedSpec::Store(output)],
    )
    .context("Creating OptMatMul")?;
    let output = patch.wire_node(name, opt, &[pa, pb])?[0];
    patch.shunt_outside(model, node.id.into(), output)?;
    Ok(Some(patch))
}
