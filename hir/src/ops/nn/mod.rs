mod global_pools;
mod layer_max;
mod reduce;

pub use global_pools::*;
pub use layer_max::*;
pub use reduce::{Reduce, Reducer};

pub use tract_core::ops::nn::{sigmoid, DataFormat};
