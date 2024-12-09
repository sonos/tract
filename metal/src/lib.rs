#![allow(clippy::missing_safety_doc)]
#![allow(clippy::missing_transmute_annotations)]

pub mod command_buffer;
pub mod context;
pub mod encoder;
pub mod fact;
pub mod func_constants;
pub mod kernels;
pub mod memory;
pub mod ops;
pub mod rewrite_rules;
pub mod session_handler;
pub mod tensor;
pub mod transform;
pub mod utils;

pub use crate::context::{MetalContext, METAL_CONTEXT};
use crate::func_constants::{ConstantValues, Value};
pub use crate::kernels::{matmul::MetalGemmImplKind, LibraryContent, LibraryName};
pub use crate::memory::MetalMemoryPool;
pub use crate::session_handler::MetalSessionHandler;
pub use crate::tensor::{MetalTensor, MetalTensorExt};
pub use crate::transform::MetalTransform;
use anyhow::Result;
pub use fact::{MetalFact, MetalOrigin};

pub trait IntoMetal<T> {
    fn into_metal(self) -> Result<T>;
}
