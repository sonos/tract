#![allow(clippy::missing_safety_doc)]

use std::ffi::c_void;

use crate::MetalStream;
use crate::context::{MetalBuffer, StreamExt};
use metal::Buffer;
use objc::runtime::{objc_autoreleasePoolPop, objc_autoreleasePoolPush};
use tract_gpu::tensor::DeviceTensor;

// Copied code from objc crate to avoid closures
struct AutoReleaseHelper {
    context: *mut c_void,
}

impl AutoReleaseHelper {
    unsafe fn new() -> Self {
        AutoReleaseHelper { context: unsafe { objc_autoreleasePoolPush() } }
    }
}

impl Drop for AutoReleaseHelper {
    fn drop(&mut self) {
        unsafe { objc_autoreleasePoolPop(self.context) }
    }
}

pub fn with_borrowed_metal_stream<T, F: FnOnce(&MetalStream) -> TractResult<T>>(
    f: F,
) -> TractResult<T> {
    let _context = unsafe { AutoReleaseHelper::new() };
    crate::context::metal_context(); // ensures stream factory is registered
    tract_gpu::with_stream(|stream| {
        let stream = stream.metal()?;
        f(stream)
    })
}

pub fn get_metal_buffer(tensor: &DeviceTensor) -> &Buffer {
    if let Some(metal_buffer) = tensor.device_buffer().downcast_ref::<MetalBuffer>() {
        &metal_buffer.inner
    } else {
        panic!("Non-Metal Buffer accessed during Metal execution")
    }
}

use tract_core::internal::*;
