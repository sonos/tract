use anyhow::Context;
use downcast_rs::{Downcast, impl_downcast};
use tract_core::prelude::TractResult;

pub mod device;
pub mod fact;
pub mod memory;
pub mod ops;
pub mod rewrite_rules;
pub mod session_handler;
pub mod sync;
pub mod tensor;
pub mod utils;

pub trait GpuStream: Downcast {}
impl_downcast!(GpuStream);

thread_local! {
    pub static GPU_STREAM: Option<Box<dyn GpuStream>> = None;
}

pub fn with_stream<R, F: FnOnce(&dyn GpuStream) -> TractResult<R>>(f: F) -> TractResult<R> {
    GPU_STREAM.with(|stream| stream.as_ref().context("GPU stream not setup").and_then(|s| f(&**s)))
}
