use std::ffi::c_void;
use std::sync::Mutex;

use anyhow::{anyhow, bail};
use downcast_rs::{Downcast, impl_downcast};
use tract_core::dyn_clone;
use tract_core::internal::OpaqueFact;
use tract_core::prelude::{DatumType, TractResult};
use tract_core::value::TValue;

use crate::tensor::OwnedDeviceTensor;

pub trait DeviceContext: Downcast + dyn_clone::DynClone + Send + Sync {
    fn tensor_to_device(&self, tensor: TValue) -> TractResult<Box<dyn OwnedDeviceTensor>>;
    fn uninitialized_device_tensor(
        &self,
        shape: &[usize],
        dt: DatumType,
    ) -> TractResult<Box<dyn OwnedDeviceTensor>>;
    fn uninitialized_device_opaque_tensor(
        &self,
        opaque_fact: Box<dyn OpaqueFact>,
    ) -> TractResult<Box<dyn OwnedDeviceTensor>>;
    fn synchronize(&self) -> TractResult<()>;
}

impl_downcast!(DeviceContext);
dyn_clone::clone_trait_object!(DeviceContext);

pub trait DeviceBuffer: Downcast + dyn_clone::DynClone + Send + Sync + std::fmt::Debug {
    fn ptr(&self) -> *const c_void;
}

impl_downcast!(DeviceBuffer);
dyn_clone::clone_trait_object!(DeviceBuffer);

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
