mod silu;
pub use silu::Silu;

const NN_OPS: &str = include_str!("unary.ptx");

