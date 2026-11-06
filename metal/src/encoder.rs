use metal::{ComputeCommandEncoderRef, ComputePipelineStateRef, MTLResourceUsage, MTLSize};
use std::cell::{Cell, RefCell};
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

/// Bound pipeline and launch names shared by all handles of one command buffer.
#[derive(Debug, Default)]
pub(crate) struct CommandBufferTrace {
    pub pipeline: Option<String>,
    pub kernel_names: Vec<String>,
}

/// A compute encoder that retains bound tensor owners and counts kernel launches.
/// Tensor bindings retain their Rust owners on this command buffer; bindings and
/// other encoder configuration do not count as dispatches.
pub struct TComputeCommandEncoder<'a> {
    inner: &'a ComputeCommandEncoderRef,
    dispatches: &'a Cell<usize>,
    retained: &'a RefCell<Vec<DeviceTensor>>,
    trace: &'a RefCell<CommandBufferTrace>,
}

impl<'a> TComputeCommandEncoder<'a> {
    pub(crate) fn new(
        inner: &'a ComputeCommandEncoderRef,
        dispatches: &'a Cell<usize>,
        retained: &'a RefCell<Vec<DeviceTensor>>,
        trace: &'a RefCell<CommandBufferTrace>,
    ) -> Self {
        Self { inner, dispatches, retained, trace }
    }

    /// Bind a labeled pipeline; its label is recorded for each launch on this buffer.
    pub fn set_compute_pipeline_state(&self, pipeline: &ComputePipelineStateRef) {
        self.inner.set_compute_pipeline_state(pipeline);
        if cfg!(test) || log::log_enabled!(target: "tract_metal::context", log::Level::Trace) {
            self.trace.borrow_mut().pipeline = Some(pipeline.label().to_owned());
        } else {
            self.trace.borrow_mut().pipeline = None;
        }
    }

    fn record_launch(&self) {
        self.dispatches.set(self.dispatches.get().saturating_add(1));
        if cfg!(test) || log::log_enabled!(target: "tract_metal::context", log::Level::Trace) {
            let mut trace = self.trace.borrow_mut();
            if let Some(name) = trace.pipeline.clone() {
                trace.kernel_names.push(name);
            }
        }
    }

    /// Bind a tensor and retain its Rust owner on the consuming command buffer.
    pub fn set_metal_tensor(&self, idx: u64, tensor: &DeviceTensor, usage: MTLResourceUsage) {
        self.retained.borrow_mut().push(tensor.clone());
        self.inner.set_metal_tensor(idx, tensor, usage);
    }

    /// Bind a tensor at a byte offset and retain its Rust owner on this command buffer.
    pub fn set_metal_tensor_with_offset(
        &self,
        idx: u64,
        tensor: &DeviceTensor,
        offset: u64,
        usage: MTLResourceUsage,
    ) {
        self.retained.borrow_mut().push(tensor.clone());
        self.inner.set_metal_tensor_with_offset(idx, tensor, offset, usage);
    }

    /// Encode a threadgroup launch and count it toward this buffer's submission threshold.
    pub fn dispatch_thread_groups(&self, groups: MTLSize, threads_per_group: MTLSize) {
        self.inner.dispatch_thread_groups(groups, threads_per_group);
        self.record_launch();
    }

    /// Encode a thread launch and count it toward this buffer's submission threshold.
    pub fn dispatch_threads(&self, threads: MTLSize, threads_per_group: MTLSize) {
        self.inner.dispatch_threads(threads, threads_per_group);
        self.record_launch();
    }
}

impl Deref for TComputeCommandEncoder<'_> {
    type Target = ComputeCommandEncoderRef;

    fn deref(&self) -> &Self::Target {
        self.inner
    }
}
