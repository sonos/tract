mod conv;
mod pools;

pub use tract_core::ops::cnn::{ConvUnary, PaddingSpec, PoolSpec};
pub use conv::Conv;
pub use pools::{AvgPool, MaxPool};
