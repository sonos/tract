use downcast_rs::{Downcast, impl_downcast};
use std::cell::RefCell;
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

type StreamInitFn = fn() -> Box<dyn GpuStream>;

thread_local! {
    static GPU_STREAM: RefCell<Option<Box<dyn GpuStream>>> = const { RefCell::new(None) };
    static GPU_STREAM_INITIALIZING: std::cell::Cell<bool> = const { std::cell::Cell::new(false) };
}

static STREAM_INIT: std::sync::OnceLock<StreamInitFn> = std::sync::OnceLock::new();

/// Register a factory function that creates the GPU stream on first access per thread.
pub fn register_stream_factory(f: StreamInitFn) {
    STREAM_INIT.set(f).ok();
}

pub fn set_stream(stream: Box<dyn GpuStream>) {
    GPU_STREAM.with(|cell| {
        *cell.borrow_mut() = Some(stream);
    });
}

fn ensure_stream() {
    let needs_init = GPU_STREAM.with(|cell| cell.borrow().is_none());
    let initializing = GPU_STREAM_INITIALIZING.with(|cell| cell.get());
    if needs_init && !initializing {
        if let Some(init) = STREAM_INIT.get() {
            GPU_STREAM_INITIALIZING.with(|cell| cell.set(true));
            let stream = init();
            set_stream(stream);
            GPU_STREAM_INITIALIZING.with(|cell| cell.set(false));
        }
    }
}

pub fn with_stream<R>(f: impl FnOnce(&dyn GpuStream) -> TractResult<R>) -> TractResult<R> {
    ensure_stream();
    GPU_STREAM.with(|cell| {
        let borrow = cell.borrow();
        let stream = borrow
            .as_ref()
            .expect("GPU stream not set. Register a stream factory or call set_stream() first.");
        f(&**stream)
    })
}
