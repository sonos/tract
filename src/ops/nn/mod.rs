mod avgpool;
mod conv;
mod global_pools;
mod layer_max;
mod maxpool;
mod patches;

pub use self::avgpool::AvgPool;
pub use self::conv::Conv;
pub use self::global_pools::{GlobalAvgPool, GlobalLpPool, GlobalMaxPool};
pub use self::layer_max::{LayerHardmax, LayerLogSoftmax, LayerSoftmax};
pub use self::maxpool::MaxPool;
pub use self::patches::PaddingSpec;

element_map!(Relu, [f32, i32], |x| if x < 0 as _ { 0 as _ } else { x });
element_map!(Sigmoid, [f32], |x| ((-x).exp() + 1.0).recip());
