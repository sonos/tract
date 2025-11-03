use tract_core::internal::*;
use tract_core::ops::array::Range;
use tract_core::ops::binary::TypedBinOp;
use tract_core::ops::cast::Cast;
use tract_core::ops::logic::Comp;
use tract_transformers::ops::sdpa::Sdpa;
use tract_transformers::rule_ensure;

pub fn rewire_sdpa(model: &mut TypedModel) -> TractResult<()> {
    Rewriter::default().with_rule_for("flatten-sdpa", rewire_sdpa_op).rewrite(&(), model)
}

pub fn rewire_sdpa_op(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    _node_name: &str,
    op: &Sdpa,
) -> TractResult<Option<TypedModelPatch>> {
    op.patch_sdpa(model, node)
}

pub enum SdpaMaskMode {
    Neutral,
    Causal,
}

pub fn create_sdpa_mask_graph(
    model: &TypedModel,
    node: &TypedNode,
    node_name: &str,
    op: &Sdpa,
    dt: DatumType,
    mode: SdpaMaskMode,
) -> TractResult<Option<TypedModelPatch>> {
    // Find mask dimensions
    let in_facts = model.node_input_facts(node.id)?;
    let q_shape = &in_facts[0].shape;
    let k_shape = &in_facts[1].shape;
    let rank = q_shape.len();
    ensure!(k_shape.len() == rank);

    let s = &q_shape[rank - 2];
    let s_plus_p = &k_shape[rank - 2];

    let mut patch = TypedModelPatch::default();

    let s_plus_p_outlet = patch.add_const("S+P", tensor0(s_plus_p.clone()))?;
    let p_outlet = patch.add_const("P", tensor0(s_plus_p.clone() - s.clone()))?;

    let zero = patch.add_const("P", tensor0(TDim::Val(0)))?;
    let range_increment = patch.add_const("mask_s", tensor0(TDim::Val(1)))?;
    let s_range = patch.wire_node(
        "mask_s_range",
        Range::new(s.clone()),
        &[p_outlet, s_plus_p_outlet, range_increment],
    )?;
    let s_plus_p_range = patch.wire_node(
        "mask_s+p_range",
        Range::new(s_plus_p.clone()),
        &[zero, s_plus_p_outlet, range_increment],
    )?;
    let s_range_add_axis = patch.wire_node("mask_s_range.add_axis", AxisOp::Add(1), &s_range)?[0];
    let s_plus_p_range_add_axis =
        patch.wire_node("mask_s_plus_p_range.add_axis", AxisOp::Add(0), &s_plus_p_range)?[0];

    let greater =
        patch.wire_node("mask.greater", Comp::GT, &[s_plus_p_range_add_axis, s_range_add_axis])?[0];
    let cast_greater = patch.wire_node("mask.greater.cast", Cast::new(dt), &[greater])?[0];

    let multiplier = match mode {
        SdpaMaskMode::Causal => patch.add_const("P", dt.min_value())?,
        SdpaMaskMode::Neutral => patch.add_const("P", Tensor::zero_scalar_dt(dt)?)?,
    };
    let mult_reshape_op = AxisOp::Reshape(0, tvec![], tvec![TDim::Val(1); 2]);
    let reshaped_mult = patch.wire_node("mask.reshape", mult_reshape_op, &[multiplier])?[0];

    let mask = patch.wire_node(
        "mask",
        TypedBinOp(Box::new(tract_core::ops::math::Mul), None),
        &[cast_greater, reshaped_mult],
    )?[0];

    let mut inputs = node
        .inputs
        .iter()
        .map(|inp| patch.tap_model(model, *inp))
        .collect::<TractResult<Vec<_>>>()?;
    inputs.push(mask);

    let mut new_op = op.clone();
    new_op.is_causal = false;
    let new_sdpa = patch.wire_node(node_name, new_op, &inputs)?[0];
    patch.shunt_outside(model, node.id.into(), new_sdpa)?;

    Ok(Some(patch))
}

pub fn neutral_mask_for_full_attn(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    node_name: &str,
    op: &Sdpa,
) -> TractResult<Option<TypedModelPatch>> {
    rule_ensure!(!op.is_causal && node.inputs.len() == 3);
    create_sdpa_mask_graph(model, node, node_name, op, op.acc_datum_type, SdpaMaskMode::Neutral)
}

pub fn causal_mask_as_extern(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    node_name: &str,
    op: &Sdpa,
) -> TractResult<Option<TypedModelPatch>> {
    rule_ensure!(op.is_causal);
    create_sdpa_mask_graph(model, node, node_name, op, op.acc_datum_type, SdpaMaskMode::Causal)
}
