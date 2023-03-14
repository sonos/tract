use super::*;
use crate::ops::matmul::lir_unary::{
    AddMatMulGeometry, LirMatMulUnary, MapOutputAxisToInput, ProtoFusedSpec,
};
use crate::ops::matmul::pack::MatMatMulPack;

pub(crate) fn codegen(
    op: &EinSum,
    model: &TypedModel,
    node: &TypedNode,
) -> TractResult<Option<TypedModelPatch>> {
    if node.inputs.len() != 2 {
        return Ok(None);
    }
    let input_facts = model.node_input_facts(node.id)?;
    let input_shapes: TVec<&[TDim]> = input_facts.iter().map(|f| &*f.shape).collect();
    let output_shape = super::eval::output_shape(&op.expr, &input_shapes);
    let Some(m_axis) = op
        .expr
        .iter_all_axes()
        .filter(|a| a.inputs[0].len() == 1 && (a.inputs[1].len() == 0 || input_facts[1].shape[a.inputs[1][0]].is_one()) && a.outputs[0].len() == 1)
        .max_by_key(|a| &output_shape[a.outputs[0][0]])
    else {
        return Ok(None)
    };
    let k_axes: TVec<&Axis> = op
        .expr
        .iter_all_axes()
        .filter(|a| a.inputs[0].len() == 1 && a.inputs[1].len() == 1 && a.outputs[0].len() == 0)
        .filter(|a| {
            input_facts[0].shape[a.inputs[0][0]] == input_facts[1].shape[a.inputs[1][0]]
                && input_facts[0].shape[a.inputs[0][0]] != 1.to_dim()
        })
        .collect();
    if k_axes.len() != 1 {
        return Ok(None);
    }
    let k_axis = k_axes[0];
    let Some(n_axis) = op
        .expr
        .iter_all_axes()
        .filter(|a| (a.inputs[0].len() == 0 || input_facts[0].shape[a.inputs[0][0]].is_one()) && a.inputs[1].len() == 1 && a.outputs[0].len() == 1)
                .max_by_key(|a| &output_shape[a.outputs[0][0]])
                else {
                    return Ok(None)
                };
    let a_m = m_axis.inputs[0][0];
    let a_k = k_axis.inputs[0][0];
    let b_n = n_axis.inputs[1][0];
    let b_k = k_axis.inputs[1][0];
    let c_m = m_axis.outputs[0][0];
    let c_n = n_axis.outputs[0][0];
    let m = &input_facts[0].shape[a_m];
    let k = input_facts[0].shape[a_k].to_usize()?;
    let n = &input_facts[1].shape[b_n];
    if m < n {
        let expr = op
            .expr
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
            EinSum { expr: AxesMapping::new(expr)?, ..op.clone() },
        )
        .map(Some);
    }
    let dt = op.operating_dt;
    let mmm =
        tract_linalg::ops().mmm(dt, dt, dt, m.to_usize().ok(), Some(k), n.to_usize().ok()).unwrap();
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
    for axis in op.expr.iter_all_axes().filter(|&axis| ![m_axis, k_axis, n_axis].contains(&axis)) {
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
    let geo = AddMatMulGeometry {
        k: k.to_dim(),
        a_storage: unsafe { mmm.a_packed(dt.size_of(), k) },
        b_storage: unsafe { mmm.b_packed(dt.size_of(), k) },
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
    )?;
    let output = patch.wire_node(name, lir, &[pa, pb])?[0];
    patch.shunt_outside(model, node.id.into(), output)?;
    Ok(Some(patch))
}
