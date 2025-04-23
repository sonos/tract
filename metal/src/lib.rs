#![allow(clippy::missing_safety_doc)]
#![allow(clippy::missing_transmute_annotations)]

mod command_buffer;
mod context;
mod encoder;
mod func_constants;
mod kernels;
mod ops;
mod rewrite_rules;
mod tests;
mod transform;
mod utils;

use crate::context::MetalStream;
use crate::func_constants::{ConstantValues, Value};
pub use crate::kernels::matmul::MetalGemmImplKind;
use crate::kernels::LibraryName;

pub use crate::context::{MetalContext, METAL_STREAM};
pub use crate::transform::MetalTransform;
