use super::*;
use crate::ops::cast::cast;
use crate::ops::math::add;
use crate::ops::matmul::lir_unary::{
    AddMatMulGeometry, LirMatMulUnary, MapOutputAxisToInput, ProtoFusedSpec,
};
use crate::ops::matmul::mir_quant::{
    combine_scales, compensate_zero_points, requant, wire_offset_u8_as_i8,
};
use crate::ops::matmul::pack::MatMatMulPack;
use crate::ops::nn::{Reduce, Reducer};

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
    if let Some(axes) = choose_mkn_axes(op, model, node).context("Choosing axes")? {
        if op.q_params.is_none() {
            lir_mat_mul_unary(op, model, node, axes).context("Translating to LirMatMul")
        } else {
            dequant_output(op, model, node, axes).context("Dequantizing output")
        }
    } else {
        Ok(None)
    }
}

fn choose_mkn_axes<'a>(
    op: &'a EinSum,
    model: &TypedModel,
    node: &TypedNode,
) -> TractResult<Option<(&'a Axis, &'a Axis, &'a Axis)>> {
    let input_facts = model.node_input_facts(node.id)?;
    let input_shapes: TVec<&[TDim]> = input_facts.iter().map(|f| &*f.shape).collect();
    let output_shape = super::eval::output_shape(&op.axes, &input_shapes);
    let k_axes: TVec<&Axis> = op
        .axes
        .iter_all_axes()
        .filter(|a| a.inputs[0].len() == 1 && a.inputs[1].len() == 1 && a.outputs[0].len() == 0)
        .filter(|a| {
            input_facts[0].shape[a.inputs[0][0]] == input_facts[1].shape[a.inputs[1][0]]
                && input_facts[0].shape[a.inputs[0][0]] != 1.to_dim()
        })
        .collect();
    let [k_axis] = &*k_axes else { return Ok(None) };
    let Some(m_axis) = op
        .axes
        .iter_all_axes()
        .filter(|a| a.inputs[0].len() == 1 && (a.inputs[1].len() == 0 || input_facts[1].shape[a.inputs[1][0]].is_one()) && a.outputs[0].len() == 1)
        .max_by_key(|a| &output_shape[a.outputs[0][0]])
    else {
        return Ok(None)
    };
    let Some(n_axis) = op
        .axes
        .iter_all_axes()
        .filter(|a| (a.inputs[0].len() == 0 || input_facts[0].shape[a.inputs[0][0]].is_one()) && a.inputs[1].len() == 1 && a.outputs[0].len() == 1)
        .max_by_key(|a| &output_shape[a.outputs[0][0]])
    else {
        return Ok(None)
    };
    Ok(Some((m_axis, k_axis, n_axis)))
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

fn dequant_output(
    op: &EinSum,
    model: &TypedModel,
    node: &TypedNode,
    (_, k_axis, _): (&Axis, &Axis, &Axis),
) -> TractResult<Option<TypedModelPatch>> {
    let name = &node.name;
    let mut patch = TypedModelPatch::new("Dequantizing einsum");
    let taps: Vec<OutletId> =
        node.inputs.iter().map(|i| patch.tap_model(model, *i)).collect::<TractResult<Vec<_>>>()?;
    let [a, b, bias, mut a0, a_scale, mut b0, b_scale, c0, c_scale] = *taps else {
        bail!("Expect exactly 9 inputs")
    };

    let a = wire_offset_u8_as_i8(&mut patch, &node.name, a, "a", &mut a0, "a0")?;
    let b = wire_offset_u8_as_i8(&mut patch, &node.name, b, "b", &mut b0, "b0")?;

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

fn lir_mat_mul_unary(
    op: &EinSum,
    model: &TypedModel,
    node: &TypedNode,
    (m_axis, k_axis, n_axis): (&Axis, &Axis, &Axis),
) -> TractResult<Option<TypedModelPatch>> {
    let input_facts = model.node_input_facts(node.id)?;
    let a_m = m_axis.inputs[0][0];
    let a_k = k_axis.inputs[0][0];
    let b_n = n_axis.inputs[1][0];
    let b_k = k_axis.inputs[1][0];
    let c_m = m_axis.outputs[0][0];
    let c_n = n_axis.outputs[0][0];
    let m = &input_facts[0].shape[a_m];
    let n = &input_facts[1].shape[b_n];
    let Ok(k) = input_facts[0].shape[a_k].to_usize() else {
        return Ok(None);
    };
    if m < n {
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
            EinSum { axes: AxesMapping::new(expr)?, ..op.clone() },
        )
        .map(Some);
    }
    let a_dt = input_facts[0].datum_type;
    let b_dt = input_facts[1].datum_type;
    let dt = op.operating_dt;
    let mmm = tract_linalg::ops()
        .mmm(a_dt, b_dt, dt, m.to_usize().ok(), Some(k), n.to_usize().ok())
        .unwrap();
    let name = &node.name;
    let mut patch = TypedModelPatch::new("Einsum to LirMatMulUnary");
    let a = patch.tap_model(model, node.inputs[0])?;
    let b = patch.tap_model(model, node.inputs[1])?;
    let pack_a = MatMatMulPack { packer: mmm.a_pack(), k_axis: a_k, mn_axis: a_m };
    let pack_b = MatMatMulPack { packer: mmm.b_pack(), k_axis: b_k, mn_axis: b_n };
    let pa = patch.wire_node(format!("{name}.pack_a"), pack_a, &[a])?[0];
    let pb = patch.wire_node(format!("{name}.pack_b"), pack_b, &[b])?[0];

    let mut c_to_a_axis_mapping = tvec!();
    let mut c_to_b_axis_mapping = tvec!();
    for axis in op.axes.iter_all_axes().filter(|&axis| ![m_axis, k_axis, n_axis].contains(&axis)) {
        if let (&[c], &[a]) = (&*axis.outputs[0], &*axis.inputs[0]) {
            if input_facts[0].shape[a] != 1.to_dim() {
                let a = a - (a > a_m) as usize - (a > a_k) as usize;
                c_to_a_axis_mapping.push((c, a));
            }
        }
        if let (&[c], &[b]) = (&*axis.outputs[0], &*axis.inputs[1]) {
            if input_facts[1].shape[b] != 1.to_dim() {
                let b = b - (b > b_n) as usize - (b > b_k) as usize;
                c_to_b_axis_mapping.push((c, b));
            }
        }
    }

    let c_fact = op.output_facts(&input_facts)?.remove(0);
    let name = &node.name;
    let geo = AddMatMulGeometry {
        k: k.to_dim(),
        a_storage: unsafe { mmm.a_packed(a_dt.size_of(), k) },
        b_storage: unsafe { mmm.b_packed(b_dt.size_of(), k) },
        c_to_a_axis_mapping: MapOutputAxisToInput(c_to_a_axis_mapping),
        c_to_b_axis_mapping: MapOutputAxisToInput(c_to_b_axis_mapping),
    };
    let output = unsafe { mmm.c_view(c_m, c_n) };
    let lir = LirMatMulUnary::new(
        mmm,
        c_fact,
        c_m,
        c_n,
        vec![
            ProtoFusedSpec::AddMatMul(geo, AttrOrInput::Input(0), AttrOrInput::Input(1)),
            ProtoFusedSpec::Store(output),
        ],
    )
    .context("Creating LirMatMulUnary")?;
    let output = patch.wire_node(name, lir, &[pa, pb])?[0];
    patch.shunt_outside(model, node.id.into(), output)?;
    Ok(Some(patch))
}
