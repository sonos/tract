pub mod mmm_wrapper;
pub mod logic;
pub mod phy;

pub use mmm_wrapper::MMMWrapper;
pub use self::logic::{compute_shapes, MatMul, MatMulUnary};
