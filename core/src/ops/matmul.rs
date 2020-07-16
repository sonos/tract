pub mod lir;
pub mod mir;
pub mod mmm_wrapper;

pub use self::mir::{compute_shapes, MatMul, MatMulUnary};
pub use self::lir::FusableOps;
pub use mmm_wrapper::MMMWrapper;
