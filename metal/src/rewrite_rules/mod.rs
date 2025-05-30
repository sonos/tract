mod add_matmul_broadcast;
mod fuse_axis_op;
mod untranspose_matmul_output;
mod rms_norm;

pub use add_matmul_broadcast::add_broadcast_pre_matmul;
pub use fuse_axis_op::{fuse_axis_op, fuse_move_axis};
pub use untranspose_matmul_output::untranspose_matmul_output;
pub use rms_norm::remove_rms_norm_cast;

pub use tract_gpu::rewrite_rules::{next_node, previous_node, previous_nodes};
