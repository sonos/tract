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

use std::sync::Arc;
use anyhow::{anyhow, Result};
pub use crate::context::{MetalContext, METAL_CONTEXT};
use crate::func_constants::{ConstantValues, Value};
pub use crate::kernels::{matmul::MetalGemmImplKind, LibraryContent, LibraryName};
pub use crate::transform::MetalTransform;

use std::os::raw::c_void;
use objc::runtime::{objc_autoreleasePoolPush, objc_autoreleasePoolPop};

use context::MetalDevice;

pub(crate) fn get_metal_device() -> Result<Arc<MetalDevice>>
{
    let gpu_device = tract_gpu::context::get_device()?;

    Arc::downcast::<MetalDevice>(gpu_device).map_err(|_| anyhow!("GPU Device is not a Metal Device"))
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

pub(crate) fn autorelease_pool_init() -> AutoReleaseHelper
{
    unsafe { AutoReleaseHelper::new() }
}