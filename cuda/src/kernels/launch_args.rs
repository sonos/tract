use cudarc::driver::{DeviceRepr, LaunchArgs, PushKernelArg};
use tract_core::internal::tract_smallvec::SmallVec;
use tract_gpu::tensor::DeviceTensor;

use crate::tensor::CudaBuffer;

pub trait LaunchArgsExt<'a> {
    fn set_slice<T: DeviceRepr>(&mut self, slice: &'a SmallVec<[T; 4]>);
}

impl<'a> LaunchArgsExt<'a> for LaunchArgs<'a> {
    fn set_slice<T: DeviceRepr>(&mut self, slice: &'a SmallVec<[T; 4]>) {
        for s in slice {
            self.arg(s);
        }
    }
}