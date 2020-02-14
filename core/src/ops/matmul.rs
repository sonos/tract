pub mod logic;
pub mod mmm_wrapper;
pub mod phy;

pub use self::logic::{compute_shapes, MatMul, MatMulUnary};
pub use mmm_wrapper::MMMWrapper;
