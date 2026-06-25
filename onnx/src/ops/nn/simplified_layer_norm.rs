use crate::model::ParsingContext;
use crate::ops::nn::layer_norm::LayerNorm;
use crate::pb::NodeProto;
use tract_hir::internal::*;

pub fn simplified_layer_norm(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let axis = node.get_attr_opt("axis")?.unwrap_or(-1);
    let epsilon = node.get_attr_opt("epsilon")?.unwrap_or(1e-5);
    let datum_type = node.get_attr_opt("stash_type")?.unwrap_or(DatumType::F32);
    Ok((expand(LayerNorm::new(axis, epsilon, datum_type)), vec![]))
}
