mod fuse_axis_op;
mod remove_matmul_broadcast;
mod rewire_metal_sync;
mod untranspose_matmul_output;

use tract_core::internal::*;

pub use fuse_axis_op::{fuse_axis_op, fuse_move_axis};
pub use remove_matmul_broadcast::remove_ggml_broadcast_pre_matmul;
pub use rewire_metal_sync::{rewire_metal_sync, rewire_metal_sync_after_const};
pub use untranspose_matmul_output::untranspose_matmul_output;

#[macro_export]
macro_rules! rule_ensure {
    ($cond:expr) => {
        if !$cond {
            return Ok(None);
        }
    };
}

fn next_node<'a>(model: &'a TypedModel, node: &TypedNode) -> Option<&'a TypedNode> {
    if node.outputs.iter().map(|of| of.successors.len()).sum::<usize>() != 1 {
        return None;
    }
    let succ = node.outputs[0].successors[0];
    Some(&model.nodes()[succ.node])
}

fn previous_node<'a>(model: &'a TypedModel, node: &TypedNode) -> Option<&'a TypedNode> {
    if node.inputs.len() != 1 {
        return None;
    }
    Some(&model.nodes()[node.inputs[0].node])
}

fn previous_nodes<'a>(model: &'a TypedModel, node: &TypedNode) -> TVec<&'a TypedNode> {
    node.inputs.iter().map(|n| &model.nodes()[n.node]).collect()
}
