#[macro_use]
pub mod macros;
pub mod framework;
pub mod model;

pub mod ops {
    pub mod array;
    pub mod binary;
    pub use crate::ops::cast;
    pub mod cnn;
    pub mod downsample;
    pub mod dummy;
    pub mod element_wise;
    pub mod identity;
    pub mod konst;
    pub mod logic;
    pub use crate::ops::math;
    pub mod matmul;
    pub mod nn;
    pub use crate::ops::quant;
    pub mod scan;
    pub mod source;
    pub mod unimpl;
}

pub mod prelude {
    pub use super::framework::Framework;
    pub use crate::infer::InferenceFact;
    pub use crate::infer::InferenceModel;
    pub use crate::prelude::*;
}

pub mod internal {
    pub use super::prelude::*;
    pub use crate::infer::*;
    pub use crate::internal::*;
}
