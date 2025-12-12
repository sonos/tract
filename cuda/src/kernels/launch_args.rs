use core::{mem::size_of, ptr};
use cudarc::driver::{CudaFunction, CudaView, DeviceRepr, LaunchArgs, LaunchConfig, PushKernelArg};
use num_traits::AsPrimitive;
use std::ops::Deref;
use tract_core::prelude::TractResult;

use crate::context::TractCudaStream;

static VEC_CAPACITY: usize = 1024;

/// A LaunchArgs that can take by-value params by stashing owned bytes
/// and handing `&'a T` refs to `inner.arg(...)`.
pub struct TractLaunchArgs<'a> {
    inner: LaunchArgs<'a>,
    keepalive: Vec<u8>,
    keepalive_overflow: Vec<Box<[u8]>>,
}

impl<'a> TractLaunchArgs<'a> {
    pub fn new(stream: &'a TractCudaStream, func: &'a CudaFunction) -> Self {
        Self {
            inner: stream.launch_builder(func),
            keepalive: Vec::with_capacity(VEC_CAPACITY),
            keepalive_overflow: Vec::new(),
        }
    }

    fn arg_typed<T: DeviceRepr + Copy + 'a>(&mut self, v: T) {
        unsafe {
            let slice = std::slice::from_raw_parts((&v) as *const T as *const u8, size_of::<T>());
            if self.keepalive.len() + slice.len() < VEC_CAPACITY {
                let arg: *const T = self.keepalive.as_ptr().add(self.keepalive.len()) as *const T;
                self.keepalive.extend(slice);
                self.inner.arg(arg.as_ref().unwrap());
            } else {
                let mut buf = slice.to_vec().into_boxed_slice();

                let r: &'a T = &*(buf.as_ptr() as *const T);
                self.inner.arg(r);

                self.keepalive_overflow.push(buf);
            }
        }
    }

    pub fn push_slice<U>(&mut self, slice: &[impl AsPrimitive<U>])
    where
        U: DeviceRepr + Copy + 'static,
    {
        for s in slice.iter().copied() {
            self.arg_typed::<U>(s.as_());
        }
    }

    pub fn push_slice_i32(&mut self, slice: &[impl AsPrimitive<i32>]) {
        for s in slice.iter().copied() {
            self.arg_typed::<i32>(s.as_());
        }
    }

    pub fn push<U>(&mut self, x: impl AsPrimitive<U>)
    where
        U: DeviceRepr + Copy + 'static,
    {
        self.arg_typed::<U>(x.as_());
    }

    pub fn push_i32(&mut self, x: impl AsPrimitive<i32>) {
        self.arg_typed::<i32>(x.as_());
    }

    pub fn push_view<T>(&mut self, x: &'a CudaView<'_, T>) {
        self.inner.arg(x);
    }

    pub fn launch(&mut self, cfg: LaunchConfig) -> TractResult<()> {
        unsafe {
            self.inner.launch(cfg)?;
        }
        Ok(())
    }
}
