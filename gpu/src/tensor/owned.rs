use downcast_rs::{Downcast, impl_downcast};
use dyn_clone::DynClone;
use std::fmt::Debug;
use tract_core::dyn_clone;
use tract_core::internal::*;

use crate::device::DeviceBuffer;

use super::DeviceTensor;

#[allow(clippy::len_without_is_empty)]
pub trait OwnedDeviceTensor: Downcast + DynClone + Send + Sync + Debug {
    fn datum_type(&self) -> DatumType;

    fn shape(&self) -> &[usize];

    fn strides(&self) -> &[isize];

    #[inline]
    fn len(&self) -> usize {
        self.shape().iter().product()
    }

    fn reshaped(&self, shape: TVec<usize>) -> TractResult<DeviceTensor>;
    fn restrided(&self, shape: TVec<isize>) -> TractResult<DeviceTensor>;

    fn as_arc_tensor(&self) -> Option<&Arc<Tensor>>;
    fn device_buffer(&self) -> &dyn DeviceBuffer;
    fn to_host(&self) -> TractResult<Arc<Tensor>>;
    fn view(&self) -> TensorView<'_>;
}

impl_downcast!(OwnedDeviceTensor);
dyn_hash::hash_trait_object!(OwnedDeviceTensor);
dyn_clone::clone_trait_object!(OwnedDeviceTensor);
