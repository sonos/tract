use core::fmt;
use std::ffi::c_void;
use std::sync::{Arc, Mutex};

use anyhow::{anyhow, bail};
use downcast_rs::{Downcast, impl_downcast};
use tract_core::dyn_clone;
use tract_core::prelude::{Tensor, TractResult};

use crate::tensor::OwnedDeviceTensor;

pub trait DeviceContext: Downcast + dyn_clone::DynClone + Send + Sync {
    fn tensor_to_device(&self, tensor: Tensor) -> TractResult<Box<dyn OwnedDeviceTensor>>;
    fn arc_tensor_to_device(&self, tensor: Arc<Tensor>) -> TractResult<Box<dyn OwnedDeviceTensor>>;
    fn synchronize(&self) -> TractResult<()>;
}

impl_downcast!(DeviceContext);
dyn_clone::clone_trait_object!(DeviceContext);

pub trait DeviceBuffer: Downcast + dyn_clone::DynClone + Send + Sync {
    fn info(&self) -> String;
    fn ptr(&self) -> *const c_void;
}

impl_downcast!(DeviceBuffer);
dyn_clone::clone_trait_object!(DeviceBuffer);

impl fmt::Debug for dyn DeviceBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DeviceBuffer: {:?}", self.info())
    }
}

pub static DEVICE_CONTEXT: Mutex<Option<Box<dyn DeviceContext>>> = Mutex::new(None);

pub fn set_context(curr_context: Box<dyn DeviceContext>) -> TractResult<()> {
    let mut context = DEVICE_CONTEXT.lock().unwrap();
    if context.is_none() {
        *context = Some(curr_context);
        Ok(())
    } else {
        bail!("Context is already set")
    }
}

pub fn get_context() -> TractResult<Box<dyn DeviceContext>> {
    let guard = DEVICE_CONTEXT.lock().map_err(|_| anyhow!("Cannot read GPU Context"))?;
    guard.as_ref().cloned().ok_or_else(|| anyhow!("GPU Context not initialized"))
}
