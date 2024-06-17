#![allow(clippy::missing_safety_doc)]

pub mod context;
pub mod func_constants;
pub mod kernels;
pub mod ops;
pub mod tensor;
pub mod transform;
pub mod utils;

pub use crate::context::{MetalContext, METAL_CONTEXT};
use crate::func_constants::{ConstantValues, Value};
pub use crate::kernels::{LibraryContent, LibraryName};
pub use crate::tensor::MetalTensor;
pub use crate::transform::MetalTransform;
use anyhow::Result;

pub trait IntoMetal<T> {
    fn into_metal(self) -> Result<T>;
}
