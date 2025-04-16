use core::fmt;
use std::cell::RefCell;

use tract_core::prelude::TractResult;

pub trait GpuContext {
    fn buffer_from_slice(&self, tensor: &[u8]) -> Box<dyn DeviceBuffer>;
    fn synchronize(&self) -> TractResult<()>;
}

pub trait DeviceBuffer: CloneBuff + Send + Sync {
    fn address(&self) -> usize;
}

impl fmt::Debug for Box<dyn DeviceBuffer> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DeviceBuffer{:?}", self)
    }
}

pub trait CloneBuff {
    fn clone_buff<'a>(&self) -> Box<dyn DeviceBuffer>;
}

impl<T> CloneBuff for T
where
    T: DeviceBuffer + Clone + 'static,
{
    fn clone_buff(&self) -> Box<dyn DeviceBuffer> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn DeviceBuffer> {
    fn clone(&self) -> Self {
        self.clone_buff()
    }
}

thread_local! {
    pub static GPU_CONTEXT: RefCell<Option<Box<dyn GpuContext>>> = RefCell::new(None);
}

pub fn set_context(new_context: impl GpuContext + 'static) -> TractResult<()> {
    GPU_CONTEXT.replace(Some(Box::new(new_context)));
    Ok(())
}

pub fn with_borrowed_gpu_context<F, R>(f: F) -> R 
    where F: FnOnce(&Box<dyn GpuContext>) -> R,
{
    GPU_CONTEXT.with_borrow(|ctxt| 
        if let Some(c) = ctxt {
            f(c)
        } else {
            panic!("No GPU context has been set")
        }
    )
}