mod conv;
mod pools;

pub use conv::Conv;
pub use pools::{HirMaxPool, HirSumPool};
pub use tract_core::ops::cnn::{PaddingSpec, PoolSpec};
