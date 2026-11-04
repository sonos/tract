use metal::{ComputeCommandEncoderRef, MTLResourceUsage, MTLSize};
use std::cell::Cell;
use std::ops::Deref;
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

use crate::utils::get_metal_buffer;

pub trait EncoderExt {
    fn set_metal_tensor(&self, idx: u64, t: &DeviceTensor, usage: MTLResourceUsage);
    fn set_metal_tensor_with_offset(
        &self,
        idx: u64,
        t: &DeviceTensor,
        offset: u64,
        usage: MTLResourceUsage,
    );
    fn set_tensor(&self, idx: u64, t: &Tensor);
    fn set_slice<T: Copy>(&self, idx: u64, data: &[T]);
}

impl EncoderExt for ComputeCommandEncoderRef {
    fn set_metal_tensor(&self, idx: u64, t: &DeviceTensor, usage: MTLResourceUsage) {
        let buffer = get_metal_buffer(t);
        self.set_buffer(idx, Some(buffer), t.buffer_offset());
        self.use_resource(buffer, usage);
    }

    fn set_metal_tensor_with_offset(
        &self,
        idx: u64,
        t: &DeviceTensor,
        offset: u64,
        usage: MTLResourceUsage,
    ) {
        let buffer = get_metal_buffer(t);
        self.set_buffer(idx, Some(buffer), t.buffer_offset::<u64>() + offset);
        self.use_resource(buffer, usage);
    }

    fn set_tensor(&self, idx: u64, t: &Tensor) {
        self.set_bytes(idx, (t.datum_type().size_of() * t.len()) as _, unsafe {
            t.as_ptr_unchecked::<u8>()
        } as *const _);
    }

    fn set_slice<T: Copy>(&self, idx: u64, data: &[T]) {
        self.set_bytes(idx, std::mem::size_of_val(data) as _, data.as_ptr() as *const _)
    }
}

/// A compute encoder that counts kernel launches for command-buffer submission.
/// Buffer bindings and other encoder configuration do not count as dispatches.
pub struct TComputeCommandEncoder<'a> {
    inner: &'a ComputeCommandEncoderRef,
    dispatches: &'a Cell<usize>,
}

impl<'a> TComputeCommandEncoder<'a> {
    pub(crate) fn new(inner: &'a ComputeCommandEncoderRef, dispatches: &'a Cell<usize>) -> Self {
        Self { inner, dispatches }
    }

    /// Encode a threadgroup launch and count it toward this buffer's submission threshold.
    pub fn dispatch_thread_groups(&self, groups: MTLSize, threads_per_group: MTLSize) {
        self.inner.dispatch_thread_groups(groups, threads_per_group);
        self.dispatches.set(self.dispatches.get().saturating_add(1));
    }

    /// Encode a thread launch and count it toward this buffer's submission threshold.
    pub fn dispatch_threads(&self, threads: MTLSize, threads_per_group: MTLSize) {
        self.inner.dispatch_threads(threads, threads_per_group);
        self.dispatches.set(self.dispatches.get().saturating_add(1));
    }
}

impl Deref for TComputeCommandEncoder<'_> {
    type Target = ComputeCommandEncoderRef;

    fn deref(&self) -> &Self::Target {
        self.inner
    }
}
