use tract_core::internal::*;
use tract_transformers::ops::sdpa::{Sdpa, SdpaMaskMode, wire_attention_mask};

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

pub fn create_sdpa_mask_graph(
    model: &TypedModel,
    node: &TypedNode,
    node_name: &str,
    op: &Sdpa,
    mode: SdpaMaskMode,
) -> TractResult<Option<TypedModelPatch>> {
    let in_facts = model.node_input_facts(node.id)?;
    let q_shape = &in_facts[0].shape;
    let k_shape = &in_facts[1].shape;
    let rank = q_shape.len();
    ensure!(k_shape.len() == rank);

    let q_len = &q_shape[rank - 2];
    let k_len = &k_shape[rank - 2];

    let mut patch = TypedModelPatch::default();
    let mut inputs = patch.taps(model, &node.inputs)?;

    let mask =
        wire_attention_mask(&mut patch, &node.name, op.acc_datum_type, mode, rank, q_len, k_len)?;
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
    rule_if!(!op.is_causal && node.inputs.len() == 3);
    create_sdpa_mask_graph(model, node, node_name, op, SdpaMaskMode::Neutral)
}

pub fn causal_mask_as_extern(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    node_name: &str,
    op: &Sdpa,
) -> TractResult<Option<TypedModelPatch>> {
    rule_if!(op.is_causal);
    create_sdpa_mask_graph(model, node, node_name, op, SdpaMaskMode::Causal)
}
