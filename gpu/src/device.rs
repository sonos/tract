use core::fmt;
use std::sync::RwLock;

use anyhow::{anyhow, Result};
use downcast_rs::{impl_downcast, Downcast};

pub trait GpuDevice: Downcast + ClonableDevice + Send + Sync {
    fn buffer_from_slice(&self, tensor: &[u8]) -> Box<dyn DeviceBuffer>;
    fn synchronize(&self) -> Result<()>;
}

impl_downcast!(GpuDevice);

pub trait ClonableDevice {
    fn clone_device<'a>(&self) -> Box<dyn GpuDevice>;
}

impl<T> ClonableDevice for T
where
    T: GpuDevice + Clone + 'static,
{
    fn clone_device(&self) -> Box<dyn GpuDevice> {
        Box::new(self.clone())
    }
}

impl Clone for Box<dyn GpuDevice> {
    fn clone(&self) -> Self {
        self.clone_device()
    }
}

pub trait DeviceBuffer: ClonableBuffer + Downcast + Send + Sync {
    fn info(&self) -> String;
    fn address(&self) -> usize;
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

pub static GPU_DEVICE: RwLock<Option<Box<dyn GpuDevice>>> = RwLock::new(None);

pub fn set_device(curr_device: Box<dyn GpuDevice>) -> Result<()> {
    let mut device = GPU_DEVICE.write().unwrap();
    *device = Some(curr_device);
    Ok(())
}

pub fn get_device() -> Result<Box<dyn GpuDevice>> {
    let guard = GPU_DEVICE.read().map_err(|_| anyhow!("Cannot read GPU Device"))?;
    guard.as_ref().cloned().ok_or_else(|| anyhow!("GPU Device not initialized"))
}
