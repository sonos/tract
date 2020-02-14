mod conv;
mod pools;

pub use crate::ops::cnn::{ConvUnary, PaddingSpec, PoolSpec};
pub use conv::Conv;
pub use pools::{AvgPool, MaxPool};
