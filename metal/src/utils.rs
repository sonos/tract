#![allow(clippy::missing_safety_doc)]

use crate::context::MetalBuffer;
use metal::Buffer;
use tract_gpu::tensor::DeviceTensor;

pub fn get_metal_buffer(tensor: &DeviceTensor) -> &Buffer {
    if let Some(metal_buffer) = tensor.device_buffer().downcast_ref::<MetalBuffer>() {
        &metal_buffer.inner
    } else {
        panic!("Non-Metal Buffer accessed during Metal execution")
    }
}
#[cfg(test)]
pub use tests::with_borrowed_metal_stream;

#[cfg(test)]
mod tests {
    use std::ffi::c_void;

    use crate::MetalStream;
    use objc::runtime::{objc_autoreleasePoolPop, objc_autoreleasePoolPush};
    use tract_core::internal::*;

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
        crate::with_metal_stream(f)
    }
}
