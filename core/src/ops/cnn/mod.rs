mod patches;
pub mod conv;
pub mod pools;
mod avgpool;
mod maxpool;
mod padding;

pub use self::pools::PoolSpec;
pub use self::avgpool::AvgPool;
pub use self::maxpool::MaxPool;
pub use self::padding::PaddingSpec;
pub use self::patches::{Patch, PatchSpec};
pub use self::conv::{Conv, ConvUnary, KernelFormat};
