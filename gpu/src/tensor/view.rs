use num_traits::AsPrimitive;
use std::ffi::c_void;
use std::fmt::Display;
use tract_core::internal::*;
use tract_core::tract_linalg::block_quant::{BlockQuantFact, BlockQuantStorage};

use crate::device::{DeviceBuffer, get_context};
use crate::utils::check_strides_validity;

use super::OwnedDeviceTensor;

/// A window into a device buffer owned by someone else, described by shape,
/// strides and a byte offset.
///
/// The buffer is kept alive by the `Arc`, but its *contents* are not: the turn
/// arena hands out windows at offsets the memory schema deliberately reuses,
/// and a state-owned buffer is written by the op that owns it. So a view is
/// good for the turn that produced it and no longer -- it is read by the nodes
/// downstream of its producer, and copied out at the model boundary rather
/// than handed to the application.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub struct DeviceView {
    pub(crate) buffer: Arc<Box<dyn OwnedDeviceTensor>>,
    pub(crate) dt: DatumType,
    pub(crate) len: usize,
    pub(crate) shape: TVec<usize>,
    pub(crate) strides: TVec<isize>,
    pub(crate) offset_bytes: usize,
    pub(crate) exotic_fact: Option<Box<dyn ExoticFact>>,
}

impl DeviceView {
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
        self.buffer.device_buffer()
    }

    pub fn device_buffer_ptr(&self) -> *const c_void {
        self.buffer.device_buffer().ptr()
    }

    /// Get underlying inner device buffer offset
    pub fn buffer_offset<I: Copy + 'static>(&self) -> I
    where
        usize: AsPrimitive<I>,
    {
        self.offset_bytes.as_()
    }

    pub fn exotic_fact(&self) -> Option<&dyn ExoticFact> {
        self.exotic_fact.as_deref()
    }

    /// Get the number of values in the tensor.
    #[inline]
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.len
    }

    pub fn as_bytes(&self) -> Vec<u8> {
        let len = if let Some(of) = &self.exotic_fact {
            of.mem_size().as_i64().unwrap() as usize
        } else {
            self.len() * self.dt.size_of()
        };
        self.buffer.get_bytes_slice(self.offset_bytes, len)
    }

    /// Reshaped tensor with given shape.
    pub fn reshaped(&self, shape: impl Into<TVec<usize>>) -> TractResult<Self> {
        ensure!(self.exotic_fact.is_none(), "Can't reshape exotic tensor");
        let shape = shape.into();
        if self.len() != shape.iter().product::<usize>() {
            bail!("Invalid reshape {:?} to {:?}", self.shape(), shape);
        }
        if shape.as_slice() != self.shape() {
            Ok(Self {
                buffer: Arc::clone(&self.buffer),
                dt: self.dt,
                len: self.len,
                strides: Tensor::natural_strides(&shape),
                shape,
                offset_bytes: self.offset_bytes,
                exotic_fact: None,
            })
        } else {
            Ok(self.clone())
        }
    }

    pub fn restrided(&self, strides: impl Into<TVec<isize>>) -> TractResult<Self> {
        ensure!(self.exotic_fact.is_none(), "Can't restride exotic tensor");
        let strides = strides.into();
        check_strides_validity(self.shape().into(), strides.clone())?;

        if strides.as_slice() != self.strides() {
            Ok(Self {
                buffer: Arc::clone(&self.buffer),
                dt: self.dt,
                len: self.len,
                strides,
                shape: self.shape.clone(),
                offset_bytes: self.offset_bytes,
                exotic_fact: None,
            })
        } else {
            Ok(self.clone())
        }
    }

    pub fn to_host(&self) -> TractResult<Tensor> {
        get_context()?.synchronize()?;
        let content = self.as_bytes();
        unsafe {
            if let Some(bqf) =
                self.exotic_fact.as_ref().and_then(|of| of.downcast_ref::<BlockQuantFact>())
            {
                Ok(BlockQuantStorage::new(
                    bqf.format.clone(),
                    bqf.m(),
                    bqf.k(),
                    Arc::new(Blob::from_bytes(&content)?),
                )?
                .into_tensor_with_shape(self.dt, bqf.shape()))
            } else {
                Tensor::from_raw_dt(self.dt, &self.shape, &content)
            }
        }
    }
}

impl Display for DeviceView {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let content = self
            .clone()
            .to_host()
            .unwrap()
            .dump(false)
            .unwrap_or_else(|e| format!("Error : {e:?}"));
        write!(f, "DeviceView: {{ {content} }}")
    }
}
