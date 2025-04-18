mod fuse_axis_op;
mod remove_matmul_broadcast;
mod untranspose_matmul_output;

pub use tract_gpu::rewrite_rules::{next_node, previous_node, previous_nodes };
pub use fuse_axis_op::{fuse_axis_op, fuse_move_axis};
pub use remove_matmul_broadcast::remove_ggml_broadcast_pre_matmul;
pub use untranspose_matmul_output::untranspose_matmul_output;

#[macro_export]
macro_rules! rule_ensure {
    ($cond:expr) => {
        if !$cond {
            return Ok(None);
        }
    };
}