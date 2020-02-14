mod conv;
mod pools;

pub use conv::Conv;
pub use pools::{AvgPool, MaxPool};
pub use tract_core::ops::cnn::{ConvUnary, PaddingSpec, PoolSpec};
