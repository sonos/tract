#![allow(clippy::missing_safety_doc)]

use std::ffi::c_void;

use crate::context::MetalBuffer;
use crate::{MetalStream, METAL_STREAM};
use metal::Buffer;
use objc::runtime::{objc_autoreleasePoolPop, objc_autoreleasePoolPush};
use tract_gpu::tensor::DeviceTensor;

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

pub fn with_borrowed_metal_stream<T, F: FnOnce(&MetalStream) -> T>(f: F) -> T {
    let _context = unsafe { AutoReleaseHelper::new() };
    METAL_STREAM.with_borrow(f)
}

#[macro_export]
macro_rules! impl_eval_op_for_metal_op {
    ($op:ty) => {
        impl tract_core::internal::EvalOp for $op {
            fn is_stateless(&self) -> bool {
                false
            }

            #[allow(unused_variables)]
            fn state(
                &self,
                session: &mut tract_core::internal::SessionState,
                node_id: usize,
            ) -> TractResult<Option<Box<dyn OpState>>> {
                Ok(Some(Box::new($crate::ops::MetalOpState::new(node_id, self.clone()))))
            }
        }
    };
}

pub fn get_metal_buffer(tensor: &DeviceTensor) -> &Buffer {
    if let Some(metal_buffer) = tensor.device_buffer().downcast_ref::<MetalBuffer>() {
        &metal_buffer.inner
    } else {
        panic!("Non-Metal Buffer accessed during Metal execution")
    }
}
