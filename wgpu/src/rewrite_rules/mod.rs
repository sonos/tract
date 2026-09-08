mod fuse_axis_op;
mod fuse_elementwise;
mod fuse_epilogue;

pub use fuse_axis_op::{fuse_axis_op, fuse_move_axis};
pub use fuse_elementwise::{grow_elementwise_chain, start_binary_chain, start_elementwise_chain};
pub use fuse_epilogue::{fuse_conv_epilogue, fuse_gemm_epilogue};
