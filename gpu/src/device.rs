use core::fmt;
use std::ffi::c_void;
use std::sync::RwLock;

use anyhow::{anyhow, bail};
use downcast_rs::{impl_downcast, Downcast};
use tract_core::prelude::TractResult;

pub trait DeviceContext: Downcast + ClonableDevice + Send + Sync {
    fn buffer_from_slice(&self, tensor: &[u8]) -> Box<dyn DeviceBuffer>;
    fn synchronize(&self) -> TractResult<()>;
}

impl_downcast!(DeviceContext);

pub trait ClonableDevice {
    fn clone_device<'a>(&self) -> Box<dyn DeviceContext>;
}

impl<T> ClonableDevice for T
where
    T: DeviceContext + Clone + 'static,
{
    fn clone_device(&self) -> Box<dyn DeviceContext> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn DeviceContext> {
    fn clone(&self) -> Self {
        self.clone_device()
    }
}

pub trait DeviceBuffer: ClonableBuffer + Downcast + Send + Sync {
    fn info(&self) -> String;
    fn ptr(&self) -> *const c_void;
}

impl_downcast!(DeviceBuffer);

impl fmt::Debug for dyn DeviceBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DeviceBuffer: {:?}", self.info())
    }
}

pub trait ClonableBuffer {
    fn clone_buff<'a>(&self) -> Box<dyn DeviceBuffer>;
}

impl<T> ClonableBuffer for T
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

pub static DEVICE_CONTEXT: RwLock<Option<Box<dyn DeviceContext>>> = RwLock::new(None);

pub fn set_context(curr_context: Box<dyn DeviceContext>) -> TractResult<()> {
    let mut context = DEVICE_CONTEXT.write().unwrap();
    if context.is_none() {
        *context = Some(curr_context);
        Ok(())
    } else {
        bail!("Context is already set")
    }
}

pub fn get_context() -> TractResult<Box<dyn DeviceContext>> {
    let guard = DEVICE_CONTEXT.read().map_err(|_| anyhow!("Cannot read GPU Context"))?;
    guard.as_ref().cloned().ok_or_else(|| anyhow!("GPU Context not initialized"))
}
