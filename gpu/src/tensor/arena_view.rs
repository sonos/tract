use num_traits::AsPrimitive;
use std::ffi::c_void;
use std::fmt::Display;
use tract_core::internal::*;

use crate::device::DeviceBuffer;
use crate::utils::check_strides_validity;

use super::OwnedDeviceTensor;

#[derive(Debug, Clone, Hash)]
pub struct DeviceArenaView {
    pub(crate) arena: Arc<OwnedDeviceTensor>,
    pub(crate) dt: DatumType,
    pub(crate) len: usize,
    pub(crate) shape: TVec<usize>,
    pub(crate) strides: TVec<isize>,
    pub(crate) offset_bytes: usize,
}

impl DeviceArenaView {
    #[inline]
    pub fn shape(&self) -> &[usize] {
        self.shape.as_slice()
    }

    /// Get the datum type of the tensor.
    #[inline]
    pub fn datum_type(&self) -> DatumType {
        self.dt
    }

    #[inline]
    pub fn strides(&self) -> &[isize] {
        self.strides.as_slice()
    }

    /// Get underlying inner device buffer.
    pub fn device_buffer(&self) -> &dyn DeviceBuffer {
        self.arena.device_buffer()
    }

    pub fn device_buffer_ptr(&self) -> *const c_void {
        self.arena.device_buffer().ptr()
    }

    /// Get underlying inner device buffer offset
    pub fn buffer_offset<I: Copy + 'static>(&self) -> I
    where
        usize: AsPrimitive<I>,
    {
        self.offset_bytes.as_()
    }

    /// Get the number of values in the tensor.
    #[inline]
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn as_bytes(&self) -> &[u8] {
        &self.arena.inner.as_arc_tensor().unwrap().as_bytes()
            [self.offset_bytes..self.offset_bytes + self.len() * self.dt.size_of()]
    }

    #[inline]
    pub fn view(&self) -> TensorView<'_> {
        unsafe {
            TensorView::from_bytes(
                self.arena.inner.as_arc_tensor().unwrap(),
                self.offset_bytes as _,
                self.shape.as_slice(),
                self.strides.as_slice(),
            )
        }
    }

    /// Reshaped tensor with given shape.
    pub fn reshaped(&self, shape: impl Into<TVec<usize>>) -> TractResult<Self> {
        let shape = shape.into();
        if self.len() != shape.iter().product::<usize>() {
            bail!("Invalid reshape {:?} to {:?}", self.shape(), shape);
        }
        if shape.as_slice() != self.shape() {
            Ok(Self {
                arena: Arc::clone(&self.arena),
                dt: self.dt,
                len: self.len,
                strides: Tensor::natural_strides(&shape),
                shape,
                offset_bytes: self.offset_bytes,
            })
        } else {
            Ok(self.clone())
        }
    }

    pub fn restrided(&self, strides: impl Into<TVec<isize>>) -> TractResult<Self> {
        let strides = strides.into();
        check_strides_validity(self.shape().into(), strides.clone())?;

        if strides.as_slice() != self.strides() {
            Ok(Self {
                arena: Arc::clone(&self.arena),
                dt: self.dt,
                len: self.len,
                strides,
                shape: self.shape.clone(),
                offset_bytes: self.offset_bytes,
            })
        } else {
            Ok(self.clone())
        }
    }
}

impl Display for DeviceArenaView {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let content =
            self.clone().into_tensor().dump(false).unwrap_or_else(|e| format!("Error : {e:?}"));
        write!(f, "DeviceArenaView: {{ {content} }}")
    }
}

impl IntoTensor for DeviceArenaView {
    fn into_tensor(self) -> Tensor {
        unsafe {
            Tensor::from_raw_dt(self.dt, &self.shape, self.as_bytes())
                .expect("Could not transform a DeviceArenaView to tensor")
        }
    }
}
