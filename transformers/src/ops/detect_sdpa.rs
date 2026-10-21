use tract_nnef::internal::*;
use tract_nnef::tract_core::ops::binary::TypedBinOp;
use tract_nnef::tract_core::ops::change_axes::perm_to_ops;
use tract_nnef::tract_core::ops::einsum::EinSum;
use tract_nnef::tract_core::ops::math::Mul;
use tract_nnef::tract_core::ops::nn::{Softmax, SoftmaxKind};

use super::sdpa::Sdpa;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Role {
    Batch,
    Rows,
    Cols,
    Reduced,
}

/// An einsum axis in attention terms, with its positions in the first input, the second input
/// and the output (`None` when absent).
#[derive(Clone, Copy, Debug)]
struct AxisRole {
    role: Role,
    first: Option<usize>,
    second: Option<usize>,
    out: Option<usize>,
}

/// Role of every axis of a two-input einsum.
///
/// `Rows` lives in the first input and the output, `Cols` in the second input and the output,
/// `Reduced` in both inputs only, `Batch` everywhere. Any other axis (appearing twice in an
/// input, or summed out of a single input) makes the einsum unfit.
fn axis_roles(axes: &AxesMapping, first: usize) -> Option<TVec<AxisRole>> {
    let second = 1 - first;
    axes.iter_all_axes()
        .map(|axis| {
            let pos = |positions: &TVec<usize>| match positions.len() {
                0 => Some(None),
                1 => Some(Some(positions[0])),
                _ => None,
            };
            let (a, b, o) =
                (pos(&axis.inputs[first])?, pos(&axis.inputs[second])?, pos(&axis.outputs[0])?);
            let role = match (a.is_some(), b.is_some(), o.is_some()) {
                (true, true, true) => Role::Batch,
                (true, false, true) => Role::Rows,
                (false, true, true) => Role::Cols,
                (true, true, false) => Role::Reduced,
                _ => return None,
            };
            Some(AxisRole { role, first: a, second: b, out: o })
        })
        .collect()
}

/// Follow `outlet` up through axis ops to a `Mul` by a scalar constant. Returns the unscaled
/// outlet, the axis ops to replay on it (in wiring order) and the scalar.
fn peel_scalar_mul(
    model: &TypedModel,
    mut outlet: OutletId,
) -> TractResult<Option<(OutletId, Vec<AxisOp>, f32)>> {
    let mut axis_ops = vec![];
    loop {
        let node = model.node(outlet.node);
        if let Some(op) = node.op_as::<AxisOp>() {
            axis_ops.push(op.clone());
            outlet = node.inputs[0];
            continue;
        }
        rule_if_some!(bin = node.op_as::<TypedBinOp>());
        rule_if!(bin.0.is::<Mul>() && node.inputs.len() == 2);
        let consts = model.collect_const_inputs(node);
        rule_if!(consts.len() == 1 && consts[0].val().len() == 1);
        rule_if_some!(
            unscaled = node
                .inputs
                .iter()
                .find(|i| model.outlet_fact(**i).is_ok_and(|f| f.konst.is_none()))
        );
        rule_if!(model.outlet_fact(*unscaled)?.shape == node.outputs[0].fact.shape);
        let scale = consts[0].val().cast_to_scalar::<f32>()?;
        axis_ops.reverse();
        return Ok(Some((*unscaled, axis_ops, scale)));
    }
}

/// Fuse decomposed attention, `EinSum(Softmax(EinSum(q, k)), v)`, into an `Sdpa` node.
///
/// The score einsum must contract a single head-dim axis shared by Q and K, the softmax must
/// run over the key axis alone, and the value einsum must contract that key axis against V.
/// Every other axis is a batch axis, present with the same dim in Q, K, V and the output; at
/// most two of them are accepted. A scalar constant `Mul` on Q or K, behind any chain of axis
/// ops, is folded into the `Sdpa` scale. The fused op accumulates in the score einsum's
/// operating type, so the exploded fallback computes what the model did.
pub fn detect_sdpa_rule(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    node_name: &str,
    pv: &EinSum,
) -> TractResult<Option<TypedModelPatch>> {
    rule_if!(pv.q_params.is_none() && node.inputs.len() == 2);
    rule_if_some!(s_slot = (0..2).find(|&i| model.node(node.inputs[i].node).op_is::<Softmax>()));
    let v_slot = 1 - s_slot;
    let sm_node = model.node(node.inputs[s_slot].node);
    let softmax = sm_node.op_as::<Softmax>().unwrap();
    rule_if!(softmax.kind == SoftmaxKind::Softmax && softmax.quant_output_dt.is_none());
    rule_if!(softmax.axes.len() == 1 && sm_node.outputs[0].successors.len() == 1);
    let key_axis = softmax.axes[0];

    let qk_node = model.node(sm_node.inputs[0].node);
    rule_if_some!(qk = qk_node.op_as::<EinSum>());
    rule_if!(qk.q_params.is_none() && qk_node.inputs.len() == 2);
    rule_if!(qk_node.outputs[0].successors.len() == 1);
    let key_in_qk = qk.axes.axis((InOut::Out(0), key_axis))?;
    rule_if_some!(k_slot = (0..2).find(|&i| !key_in_qk.inputs[i].is_empty()));
    let q_slot = 1 - k_slot;

    let dt = node.outputs[0].fact.datum_type;
    rule_if!(matches!(dt, DatumType::F16 | DatumType::F32));
    let q_outlet = qk_node.inputs[q_slot];
    let k_outlet = qk_node.inputs[k_slot];
    let v_outlet = node.inputs[v_slot];
    let q_fact = model.outlet_fact(q_outlet)?;
    let k_fact = model.outlet_fact(k_outlet)?;
    let v_fact = model.outlet_fact(v_outlet)?;
    rule_if!([q_fact, k_fact, v_fact].iter().all(|f| f.datum_type == dt));

    rule_if_some!(qk_roles = axis_roles(&qk.axes, q_slot));
    rule_if_some!(pv_roles = axis_roles(&pv.axes, s_slot));
    let only = |roles: &[AxisRole], role| {
        let found: TVec<_> = roles.iter().filter(|r| r.role == role).copied().collect();
        if found.len() == 1 { Some(found[0]) } else { None }
    };
    rule_if_some!(qk_rows = only(&qk_roles, Role::Rows));
    rule_if_some!(qk_cols = only(&qk_roles, Role::Cols));
    rule_if_some!(qk_head = only(&qk_roles, Role::Reduced));
    rule_if_some!(pv_rows = only(&pv_roles, Role::Rows));
    rule_if_some!(pv_vdim = only(&pv_roles, Role::Cols));
    rule_if_some!(pv_keys = only(&pv_roles, Role::Reduced));
    rule_if!(qk_cols.out == Some(key_axis) && pv_keys.first == Some(key_axis));
    rule_if!(qk_rows.out == pv_rows.first);

    let mut qk_batch: TVec<_> =
        qk_roles.iter().filter(|r| r.role == Role::Batch).copied().collect();
    let pv_batch: TVec<_> = pv_roles.iter().filter(|r| r.role == Role::Batch).copied().collect();
    rule_if!(qk_batch.len() == pv_batch.len() && qk_batch.len() <= 2);
    qk_batch.sort_by_key(|r| r.out);
    let mut batch_dims = tvec![];
    for b in &qk_batch {
        rule_if_some!(pvb = pv_batch.iter().find(|p| p.first == b.out));
        let dims = [
            &q_fact.shape[b.first.unwrap()],
            &k_fact.shape[b.second.unwrap()],
            &v_fact.shape[pvb.second.unwrap()],
        ];
        rule_if!(dims[0] == dims[1] && dims[0] == dims[2]);
        batch_dims.push(*pvb);
    }

    let (q_src, q_replay, q_scale) = match peel_scalar_mul(model, q_outlet)? {
        Some(peeled) => peeled,
        None => (q_outlet, vec![], 1.0),
    };
    let (k_src, k_replay, k_scale) = match peel_scalar_mul(model, k_outlet)? {
        Some(peeled) => peeled,
        None => (k_outlet, vec![], 1.0),
    };

    let mut patch = TypedModelPatch::new(format!("Fuse attention into SDPA at {node_name}"));
    let canonical = |patch: &mut TypedModelPatch,
                     name: &str,
                     outlet: OutletId,
                     replay: &[AxisOp],
                     order: TVec<usize>|
     -> TractResult<OutletId> {
        let mut wire = patch.tap_model(model, outlet)?;
        for (ix, op) in replay.iter().chain(perm_to_ops(&order).iter()).enumerate() {
            wire = patch.wire_node(format!("{node_name}.{name}.{ix}"), op.clone(), &[wire])?[0];
        }
        if order.len() == 2 {
            wire = patch.wire_node(
                format!("{node_name}.{name}.add_batch"),
                AxisOp::Add(0),
                &[wire],
            )?[0];
        }
        Ok(wire)
    };
    let q_order = qk_batch
        .iter()
        .map(|b| b.first.unwrap())
        .chain([qk_rows.first.unwrap(), qk_head.first.unwrap()])
        .collect();
    let k_order = qk_batch
        .iter()
        .map(|b| b.second.unwrap())
        .chain([qk_cols.second.unwrap(), qk_head.second.unwrap()])
        .collect();
    let v_order = batch_dims
        .iter()
        .map(|b| b.second.unwrap())
        .chain([pv_keys.second.unwrap(), pv_vdim.second.unwrap()])
        .collect();
    let q = canonical(&mut patch, "q", q_src, &q_replay, q_order)?;
    let k = canonical(&mut patch, "k", k_src, &k_replay, k_order)?;
    let v = canonical(&mut patch, "v", v_outlet, &[], v_order)?;

    let sdpa = Sdpa {
        scale: Some(tensor0(q_scale * k_scale)),
        datum_type: dt,
        acc_datum_type: qk.operating_dt,
        is_causal: false,
    };
    let mut wire = patch.wire_node(format!("{node_name}.sdpa"), sdpa, &[q, k, v])?[0];
    if batch_dims.is_empty() {
        wire = patch.wire_node(format!("{node_name}.rm_batch"), AxisOp::Rm(0), &[wire])?[0];
    }
    let canonical_out: TVec<Option<usize>> =
        batch_dims.iter().map(|b| b.out).chain([pv_rows.out, pv_vdim.out]).collect();
    let out_order: TVec<usize> = (0..canonical_out.len())
        .map(|pos| canonical_out.iter().position(|p| *p == Some(pos)).unwrap())
        .collect();
    for (ix, op) in perm_to_ops(&out_order).into_iter().enumerate() {
        wire = patch.wire_node(format!("{node_name}.out.{ix}"), op, &[wire])?[0];
    }
    patch.shunt_outside(model, node.id.into(), wire)?;
    Ok(Some(patch))
}
