use std::ffi::c_void;
use std::ops::Range;
use std::sync::Mutex;

use anyhow::{anyhow, bail};
use downcast_rs::{Downcast, impl_downcast};
use tract_core::dyn_clone;
use tract_core::internal::*;
use tract_core::value::TValue;

use crate::tensor::{DeviceTensor, OwnedDeviceTensor};

pub trait DeviceContext: Downcast + dyn_clone::DynClone + Send + Sync {
    fn tensor_to_device(&self, tensor: TValue) -> TractResult<Box<dyn OwnedDeviceTensor>>;
    fn uninitialized_device_tensor(
        &self,
        shape: &[usize],
        dt: DatumType,
    ) -> TractResult<Box<dyn OwnedDeviceTensor>>;
    fn uninitialized_device_exotic_tensor(
        &self,
        exotic_fact: Box<dyn ExoticFact>,
    ) -> TractResult<Box<dyn OwnedDeviceTensor>>;
    fn synchronize(&self) -> TractResult<()>;
    fn copy_nd(
        &self,
        input: &DeviceTensor,
        input_offset: usize,
        input_strides: &[isize],
        output: &DeviceTensor,
        output_offset: usize,
        output_shape: &[usize],
        output_strides: &[isize],
    ) -> TractResult<()>;

    /// Copy a slice along `axis` from `src[src_range]` into `dst[dst_range]`.
    fn assign_slice(
        &self,
        dst: &DeviceTensor,
        dst_range: Range<usize>,
        src: &DeviceTensor,
        src_range: Range<usize>,
        axis: usize,
    ) -> TractResult<()> {
        let mut zone_shape: TVec<usize> = src.shape().into();
        zone_shape[axis] = src_range.len();
        if zone_shape.iter().product::<usize>() == 0 {
            return Ok(());
        }
        let src_offset =
            src_range.start * src.strides()[axis] as usize * src.datum_type().size_of();
        let dst_offset =
            dst_range.start * dst.strides()[axis] as usize * dst.datum_type().size_of();
        self.copy_nd(src, src_offset, src.strides(), dst, dst_offset, &zone_shape, dst.strides())
    }

    /// Copy from `src` into `dst` with given origins and strides.
    fn copy_with_origins(
        &self,
        zone_shape: &[usize],
        dst: &DeviceTensor,
        dst_origin: &[usize],
        dst_strides: &[isize],
        src: &DeviceTensor,
        src_origin: &[usize],
        src_strides: &[isize],
    ) -> TractResult<()> {
        if zone_shape.iter().product::<usize>() == 0 {
            return Ok(());
        }
        let dt_size = src.datum_type().size_of();
        let src_offset: usize =
            src_origin.iter().zip(src_strides).map(|(o, s)| o * *s as usize).sum::<usize>()
                * dt_size;
        let dst_offset: usize =
            dst_origin.iter().zip(dst_strides).map(|(o, s)| o * *s as usize).sum::<usize>()
                * dt_size;
        self.copy_nd(src, src_offset, src_strides, dst, dst_offset, zone_shape, dst_strides)
    }

    /// Flat memcpy of `byte_len` bytes.
    fn flat_copy(
        &self,
        src: &DeviceTensor,
        src_byte_offset: usize,
        dst: &DeviceTensor,
        dst_byte_offset: usize,
        byte_len: usize,
    ) -> TractResult<()> {
        if byte_len == 0 {
            return Ok(());
        }
        self.copy_nd(src, src_byte_offset, &[1], dst, dst_byte_offset, &[byte_len], &[1])
    }
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
