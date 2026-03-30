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

type WithStreamFn = fn(&mut dyn FnMut(&dyn GpuStream) -> TractResult<()>) -> TractResult<()>;

thread_local! {
    static GPU_WITH_STREAM: std::cell::Cell<Option<WithStreamFn>> = const { std::cell::Cell::new(None) };
}

pub fn register_stream(f: WithStreamFn) {
    GPU_WITH_STREAM.with(|cell| cell.set(Some(f)));
}

pub fn with_stream<R>(mut f: impl FnMut(&dyn GpuStream) -> TractResult<R>) -> TractResult<R> {
    let mut result: Option<R> = None;
    let mut err: Option<anyhow::Error> = None;
    GPU_WITH_STREAM.with(|cell| {
        let ws = cell.get().expect("GPU stream not registered");
        let _ = ws(&mut |stream| match f(stream) {
            Ok(r) => {
                result = Some(r);
                Ok(())
            }
            Err(e) => {
                err = Some(e);
                Ok(())
            }
        });
    });
    match (result, err) {
        (Some(r), _) => Ok(r),
        (_, Some(e)) => Err(e),
        _ => unreachable!(),
    }
}
