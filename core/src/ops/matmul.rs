pub mod lir;
pub mod mir;
pub mod mmm_wrapper;

pub use self::mir::{compute_shape, MatMul, MatMulUnary};
pub use mmm_wrapper::MMMWrapper;
