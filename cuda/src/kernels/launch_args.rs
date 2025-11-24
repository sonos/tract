use core::{mem::size_of, ptr};
use cudarc::driver::{CudaFunction, CudaView, DeviceRepr, LaunchArgs, LaunchConfig, PushKernelArg};
use num_traits::AsPrimitive;
use std::ops::Deref;
use tract_core::prelude::TractResult;

use crate::context::TractCudaStream;

/// A LaunchArgs that can take by-value params by stashing owned bytes
/// and handing `&'a T` refs to `inner.arg(...)`.
pub struct LaunchArgsOwned<'a> {
    inner: LaunchArgs<'a>,
    keepalive: Vec<Box<[u8]>>,
}

impl<'a> LaunchArgsOwned<'a> {
    pub fn new(stream: &'a TractCudaStream, func: &'a CudaFunction) -> Self {
        Self { inner: stream.launch_builder(func), keepalive: Vec::new() }
    }

    pub fn arg_val<T: DeviceRepr + Copy + 'a>(&mut self, v: T) {
        let mut buf = vec![0u8; size_of::<T>()].into_boxed_slice();
        unsafe {
            ptr::copy_nonoverlapping(&v as *const T as *const u8, buf.as_mut_ptr(), size_of::<T>());

            let r: &'a T = &*(buf.as_ptr() as *const T);
            self.inner.arg(r);
        }

        self.keepalive.push(buf);
    }

    pub fn set_slice<U>(&mut self, slice: &[impl AsPrimitive<U>])
    where
        U: DeviceRepr + Copy + 'static,
    {
        for s in slice.iter().copied() {
            self.arg_val::<U>(s.as_());
        }
    }

    pub fn set_el<U>(&mut self, x: impl AsPrimitive<U>)
    where
        U: DeviceRepr + Copy + 'static,
    {
        self.arg_val::<U>(x.as_());
    }

    pub fn set_view<T>(&mut self, x: &'a CudaView<'_, T>) {
        self.inner.arg(x);
    }

    pub fn launch(&mut self, cfg: LaunchConfig) -> TractResult<()> {
        unsafe {
            self.inner.launch(cfg)?;
        }
        Ok(())
    }
}
