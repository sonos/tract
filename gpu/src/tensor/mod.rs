#![allow(clippy::missing_safety_doc)]
#![allow(clippy::missing_transmute_annotations)]

mod arena_view;
mod lazy;
mod owned;

pub use arena_view::*;
pub use lazy::*;
pub use owned::*;

use num_traits::AsPrimitive;
use std::ffi::c_void;
use std::fmt::Display;
use tract_core::internal::*;
use tract_data::itertools::Itertools;

use crate::device::{DeviceBuffer, get_context};

/// Highest rank the backends' `copy_nd` kernels are compiled for.
const MAX_COPY_ND_RANK: usize = 6;

/// This struct represents a GPU tensor that can be either a owned tensor
/// or an arena view.
#[derive(Debug, Clone, Hash, PartialEq, Eq)]
pub enum DeviceTensor {
    Owned(Box<dyn OwnedDeviceTensor>),
    ArenaView(DeviceArenaView),
}

impl DeviceTensor {
    pub const SUPPORTED_DT: [DatumType; 11] = [
        DatumType::Bool,
        DatumType::F32,
        DatumType::F16,
        DatumType::I8,
        DatumType::U8,
        DatumType::I16,
        DatumType::U16,
        DatumType::I32,
        DatumType::U32,
        DatumType::I64,
        DatumType::U64,
    ];

    pub fn tname(dt: DatumType) -> TractResult<&'static str> {
        Ok(match dt {
            DatumType::F32 => "f32",
            DatumType::F16 => "f16",
            DatumType::U8 => "u8",
            DatumType::U16 => "u16",
            DatumType::U32 => "u32",
            DatumType::U64 => "u64",
            DatumType::I8 => "i8",
            DatumType::I16 => "i16",
            DatumType::I32 => "i32",
            DatumType::I64 => "i64",
            DatumType::Bool => "bool",
            _ => bail!("Unsupported dt {:?} for GPU Tensor", dt),
        })
    }

    /// Create an uninitialized DeviceTensor
    pub fn uninitialized_dt(dt: DatumType, shape: &[usize]) -> TractResult<DeviceTensor> {
        Ok(DeviceTensor::Owned(get_context()?.uninitialized_device_tensor(shape, dt)?))
    }

    pub fn uninitialized<T: Datum>(shape: &[usize]) -> TractResult<DeviceTensor> {
        Self::uninitialized_dt(T::datum_type(), shape)
    }

    pub fn uninitialized_exotic(exotic_fact: Box<dyn ExoticFact>) -> TractResult<DeviceTensor> {
        Ok(DeviceTensor::Owned(get_context()?.uninitialized_device_exotic_tensor(exotic_fact)?))
    }
    // Create a device tensor with a given shape and a slice of elements. The data is copied and aligned to size of T.
    pub fn from_shape<T: Copy + Datum>(shape: &[usize], data: &[T]) -> TractResult<DeviceTensor> {
        Tensor::from_shape(shape, data)?.into_device()
    }

    pub fn is_supported_dt(dt: DatumType) -> bool {
        Self::SUPPORTED_DT.contains(&dt)
    }

    /// Get the datum type of the tensor.
    #[inline]
    pub fn datum_type(&self) -> DatumType {
        match self {
            Self::Owned(owned) => owned.datum_type(),
            Self::ArenaView(view) => view.datum_type(),
        }
    }

    /// Get the number of dimensions (or axes) of the tensor.
    #[inline]
    pub fn rank(&self) -> usize {
        self.shape().len()
    }

    /// Get the shape of the tensor.
    #[inline]
    pub fn shape(&self) -> &[usize] {
        match self {
            Self::Owned(t) => t.shape(),
            Self::ArenaView(t) => t.shape(),
        }
    }

    /// Get the number of values in the tensor.
    #[inline]
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        match self {
            Self::Owned(t) => t.len(),
            Self::ArenaView(t) => t.len(),
        }
    }

    /// Get the strides of the tensor.
    #[inline]
    pub fn strides(&self) -> &[isize] {
        match self {
            Self::Owned(t) => t.strides(),
            Self::ArenaView(t) => t.strides(),
        }
    }

    /// Get underlying inner buffer.
    pub fn device_buffer(&self) -> &dyn DeviceBuffer {
        match self {
            Self::Owned(t) => t.device_buffer(),
            Self::ArenaView(t) => t.device_buffer(),
        }
    }

    /// Get underlying inner buffer offset
    pub fn buffer_offset<I: Copy + 'static>(&self) -> I
    where
        usize: AsPrimitive<I>,
    {
        match self {
            Self::Owned(_) => 0.as_(),
            Self::ArenaView(t) => t.buffer_offset(),
        }
    }

    pub fn device_buffer_ptr(&self) -> *const c_void {
        match self {
            Self::Owned(t) => t.device_buffer().ptr(),
            Self::ArenaView(t) => t.device_buffer().ptr(),
        }
    }

    /// Returns short description of the inner tensor.
    pub fn description(&self) -> String {
        format!("|{},{:?}|", self.shape().iter().join(","), self.datum_type(),)
    }

    /// Reshaped tensor with given shape.
    pub fn reshaped(&self, shape: TVec<usize>) -> TractResult<Self> {
        match self {
            Self::Owned(t) => Ok(t.reshaped(shape)?),
            Self::ArenaView(t) => Ok(Self::ArenaView(t.reshaped(shape)?)),
        }
    }

    pub fn restrided(&self, strides: TVec<isize>) -> TractResult<Self> {
        match self {
            Self::Owned(t) => Ok(t.restrided(strides)?),
            Self::ArenaView(t) => Ok(Self::ArenaView(t.restrided(strides)?)),
        }
    }

    /// Convert device tensor to a Tensor backed by device storage.
    ///
    /// The resulting tensor carries the real datum type and shape from the
    /// device tensor (e.g. F32 / \[2,3\]), rather than an exotic scalar wrapper.
    pub fn into_tensor(self) -> Tensor {
        let dt = self.datum_type();
        let shape: TVec<usize> = self.shape().into();
        Tensor::from_storage(dt, &shape, self)
    }

    /// Slice `[start, end)` along `axis` by a copy made on the device, or `None`
    /// where the backends' `copy_nd` kernels cannot serve the case and the
    /// caller has to fall back.
    ///
    /// `dt` and `shape` are the owning tensor's, and storage whose geometry has
    /// drifted from them is refused rather than sliced on stale dimensions.
    ///
    /// A copy rather than a view: a view would alias what it was cut from, which
    /// outlives neither an arena's turn nor a later write to the source. The
    /// result owns its buffer, and only the source of the copy is strided, which
    /// is what lets any axis be sliced and not just a dense one.
    ///
    /// The copy is ordered on the stream behind whatever produced the source and
    /// waits for nothing of its own; a caller that needs the bytes now is the
    /// one that synchronizes.
    pub(crate) fn slice_on_device(
        &self,
        dt: DatumType,
        shape: &[usize],
        axis: usize,
        start: usize,
        end: usize,
    ) -> TractResult<Option<DeviceTensor>> {
        ensure!(
            axis < shape.len() && start < end && end <= shape[axis],
            "Invalid slicing range {start}..{end} on axis {axis} of {shape:?}"
        );
        let servable = dt == self.datum_type()
            && shape == self.shape()
            && !self.is_exotic()
            && Self::is_supported_dt(dt)
            && (1..=MAX_COPY_ND_RANK).contains(&self.rank())
            && self.strides().iter().all(|s| *s > 0);
        if !servable {
            return Ok(None);
        }
        let mut sliced: TVec<usize> = shape.into();
        sliced[axis] = end - start;
        let output = DeviceTensor::uninitialized_dt(dt, &sliced)?;
        get_context()?.assign_slice(&output, 0..end - start, self, start..end, axis)?;
        Ok(Some(output))
    }

    /// Synchronize the GPU Tensor by completing all current
    /// commands on GPU and returns the inner tensor.
    pub fn to_host(&self) -> TractResult<Arc<Tensor>> {
        get_context()?.synchronize()?;

        Ok(match self {
            Self::Owned(o) => o.to_host()?,
            Self::ArenaView(v) => v.to_host()?.into(),
        })
    }
}

impl Display for DeviceTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Owned(o) => o.fmt(f),
            Self::ArenaView(v) => {
                let content =
                    v.to_host().unwrap().dump(false).unwrap_or_else(|e| format!("Error : {e:?}"));
                write!(f, "ArenaView: {{ {content} }}")
            }
        }
    }
}

pub trait IntoDevice<T> {
    fn into_device(self) -> TractResult<T>;
}

impl IntoDevice<DeviceTensor> for Tensor {
    fn into_device(self) -> TractResult<DeviceTensor> {
        Ok(DeviceTensor::Owned(get_context()?.tensor_to_device(self.into_tvalue())?))
    }
}

impl IntoDevice<DeviceTensor> for Arc<Tensor> {
    fn into_device(self) -> TractResult<DeviceTensor> {
        Ok(DeviceTensor::Owned(get_context()?.tensor_to_device(self.into_tvalue())?))
    }
}

impl TensorStorage for DeviceTensor {
    fn byte_len(&self) -> usize {
        self.len() * self.datum_type().size_of()
    }

    fn is_empty(&self) -> bool {
        self.byte_len() == 0
    }

    fn deep_clone(&self) -> Box<dyn TensorStorage> {
        Box::new(self.clone())
    }

    fn as_plain_ram(&self) -> Option<&PlainStorage> {
        None
    }

    fn as_plain_ram_mut(&mut self) -> Option<&mut PlainStorage> {
        None
    }

    fn into_plain_ram(self: Box<Self>) -> Option<PlainStorage> {
        None
    }

    fn dyn_hash(&self, _state: &mut dyn std::hash::Hasher) {
        // no meaningful hash for device memory
    }

    fn is_exotic(&self) -> bool {
        match self {
            Self::Owned(owned) => owned.exotic_fact().is_some(),
            Self::ArenaView(view) => view.exotic_fact().is_some(),
        }
    }

    fn in_ram(&self) -> bool {
        false
    }

    fn slice(
        &self,
        dt: DatumType,
        shape: &[usize],
        axis: usize,
        start: usize,
        end: usize,
    ) -> TractResult<Option<Tensor>> {
        Ok(self.slice_on_device(dt, shape, axis, start, end)?.map(|t| t.into_tensor()))
    }

    fn exotic_fact(&self, _shape: &[usize]) -> TractResult<Option<Box<dyn ExoticFact>>> {
        bail!(
            "DeviceTensor cannot reconstruct a DeviceFact: origin (FromHost/FromDevice) is not carried by storage"
        )
    }
}

impl From<DeviceArenaView> for DeviceTensor {
    fn from(view: DeviceArenaView) -> Self {
        Self::ArenaView(view)
    }
}

pub trait DeviceTensorExt {
    fn to_device_tensor(&self) -> TractResult<&DeviceTensor>;
    fn as_device_tensor(&self) -> Option<&DeviceTensor>;
}

impl DeviceTensorExt for Tensor {
    fn to_device_tensor(&self) -> TractResult<&DeviceTensor> {
        self.try_storage_as::<DeviceTensor>()
    }

    fn as_device_tensor(&self) -> Option<&DeviceTensor> {
        self.storage_as::<DeviceTensor>()
    }
}
