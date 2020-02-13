#[macro_use]
pub mod macros;

pub mod array;
pub mod binary;
pub mod dummy;
pub mod element_wise;
pub mod cnn;
pub mod downsample;
pub mod identity;
pub mod framework;
pub mod konst;
pub mod logic;
pub mod matmul;
pub mod model;
pub mod nn;
pub mod unimpl;
pub mod scan;
pub mod source;

pub mod prelude {
    pub use crate as tract_core;
    pub use crate::ndarray as tract_ndarray;
    pub use crate::prelude::*;
    pub use super::framework::Framework;
    pub use crate::infer::InferenceModel;
    pub use crate::infer::InferenceFact;
}

pub mod internal {
    pub use super::prelude::*;
    pub use crate::internal::*;
    pub use crate::infer::ShapeFactoid;
    pub use crate::infer::GenericFactoid;
    pub use crate::infer::InferenceOp;
}
