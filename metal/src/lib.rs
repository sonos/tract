#![allow(clippy::missing_safety_doc)]
#![allow(clippy::missing_transmute_annotations)]

pub mod command_buffer;
pub mod context;
pub mod encoder;
pub mod func_constants;
pub mod kernels;
pub mod ops;
pub mod rewrite_rules;
pub mod transform;
pub mod utils;
mod tests;

pub use crate::context::{MetalStream, METAL_STREAM};
use crate::func_constants::{ConstantValues, Value};
pub use crate::kernels::{matmul::MetalGemmImplKind, LibraryContent, LibraryName};
pub use crate::transform::MetalTransform;
use anyhow::{anyhow, Result};

use objc::runtime::{objc_autoreleasePoolPop, objc_autoreleasePoolPush};
use std::os::raw::c_void;

use context::MetalContext;

pub(crate) fn get_metal_device() -> Result<Box<MetalContext>> {
    let gpu_device = tract_gpu::device::get_device()?;
    gpu_device.downcast::<MetalContext>().map_err(|_| anyhow!("GPU Device is not a Metal Device"))
}

// Copied code from objc crate to avoid closures
struct AutoReleaseHelper {
    context: *mut c_void,
}

impl AutoReleaseHelper {
    unsafe fn new() -> Self {
        AutoReleaseHelper { context: objc_autoreleasePoolPush() }
    }
}

impl Drop for AutoReleaseHelper {
    fn drop(&mut self) {
        unsafe { objc_autoreleasePoolPop(self.context) }
    }
}

pub(crate) fn autorelease_pool_init() -> AutoReleaseHelper {
    unsafe { AutoReleaseHelper::new() }
}
