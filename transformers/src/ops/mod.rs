pub mod apply_rope;
pub mod dyn_kv_cache;
pub mod flash_sdpa;
pub mod scaled_masked_softmax;
pub mod sdpa;
pub mod streamed_sdpa;

// Re-export ops that moved to core
pub mod rms_norm {
    pub use tract_nnef::tract_core::ops::nn::RmsNorm;
    pub use tract_nnef::tract_core::ops::nn::rms_norm::*;
}
pub mod silu {
    pub use tract_nnef::tract_core::ops::nn::Silu;
    pub use tract_nnef::tract_core::ops::nn::silu::*;
}
pub mod gelu_approximate {
    pub use tract_nnef::tract_core::ops::nn::GeluApproximate;
    pub use tract_nnef::tract_core::ops::nn::gelu_approximate::*;
}

use tract_nnef::tract_core::internal::*;
use tract_nnef::tract_core::ops::konst::Const;

pub use apply_rope::{apply_rope_rule, rotate_half_rule};
pub use dyn_kv_cache::replace_kv_cache;
pub use scaled_masked_softmax::scaled_masked_softmax_rule;
pub use sdpa::fuse_kv_cache_broadcast_rule;

pub(crate) fn next_node<'a>(model: &'a TypedModel, node: &TypedNode) -> Option<&'a TypedNode> {
    if node.outputs.iter().map(|of| of.successors.len()).sum::<usize>() != 1 {
        return None;
    }
    let succ = node.outputs[0].successors[0];
    Some(&model.nodes()[succ.node])
}

pub(crate) fn previous_node<'a>(model: &'a TypedModel, node: &TypedNode) -> Option<&'a TypedNode> {
    if node.inputs.len() != 1 {
        return None;
    }
    Some(&model.nodes()[node.inputs[0].node])
}

pub(crate) fn previous_nodes<'a>(model: &'a TypedModel, node: &TypedNode) -> TVec<&'a TypedNode> {
    node.inputs.iter().map(|n| &model.nodes()[n.node]).collect()
}

pub(crate) fn collect_node_const_inputs<'a>(
    model: &'a TypedModel,
    node: &TypedNode,
) -> TVec<&'a Const> {
    node.inputs
        .iter()
        .filter_map(|i| {
            let prec = &model.nodes()[i.node];
            prec.op_as::<Const>()
        })
        .collect::<TVec<_>>()
}

pub(crate) fn single_prev_node_as<'a, O: TypedOp>(
    model: &'a TypedModel,
    node: &TypedNode,
) -> Option<(usize, &'a TypedNode)> {
    let prev_nodes = node
        .inputs
        .iter()
        .enumerate()
        .filter_map(|(in_idx, i)| {
            let prec = &model.nodes()[i.node];
            prec.op_is::<O>().then_some((in_idx, prec))
        })
        .collect::<TVec<_>>();

    if prev_nodes.len() != 1 { None } else { Some(prev_nodes[0]) }
}
