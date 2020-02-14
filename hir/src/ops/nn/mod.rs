mod arg_max_min;
mod global_pools;
mod layer_max;
mod reduce;

pub use arg_max_min::ArgMaxMin;
pub use global_pools::*;
pub use layer_max::*;
pub use reduce::{Reduce, Reducer};

pub use tract_core::ops::nn::{
    elu, hard_sigmoid, leaky_relu, parametric_softplus, scaled_tanh, selu, sigmoid, softplus,
    softsign, threshold_relu, DataFormat,
};
