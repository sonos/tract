use tract_core::internal::*;
use tract_transformers::ops::sdpa::Sdpa;

pub fn rewire_sdpa(model: &mut TypedModel) -> TractResult<()> {
    Rewriter::default()
        .with_rule_for("flatten-sdpa", rewire_sdpa_op)
        .rewrite(&(), model)
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
