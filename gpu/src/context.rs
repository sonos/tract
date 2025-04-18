use core::fmt;
use std::sync::{Arc, RwLock};

use anyhow::{bail, Result};
use downcast_rs::{impl_downcast, Downcast};

pub trait GpuDevice: Send + Sync + Downcast + 'static {
    fn buffer_from_slice(&self, tensor: &[u8]) -> Box<dyn DeviceBuffer>;
    fn synchronize(&self) -> Result<()>;
}

impl_downcast!(GpuDevice);

pub trait DeviceBuffer: CloneBuff + Downcast + Send + Sync {
    fn info(&self) -> String;
    fn address(&self) -> usize;
}

impl_downcast!(DeviceBuffer);

impl fmt::Debug for dyn DeviceBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "DeviceBuffer: {:?}", self.info())
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

pub static GPU_DEVICE: RwLock<Option<Arc<dyn GpuDevice>>> = RwLock::new(None);

pub fn set_context(new_context: Arc<dyn GpuDevice>) -> Result<()> {
    let mut device = GPU_DEVICE.write().unwrap();
    *device = Some(new_context);
    Ok(())
}

pub fn get_device() -> Result<Arc<dyn GpuDevice>, anyhow::Error>
{   
    let guard = if let Some(guard) = GPU_DEVICE.read().ok() {
        guard
    }else {
        bail!("Cannot read GPU Device")
    };

    if let Some(gpu_ctxt) = guard
        .as_ref()
        .cloned() {
            Ok(gpu_ctxt)
        }
        else {
            bail!("GPU Device not initialized")
        }
}