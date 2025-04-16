mod arena_view;
mod owned;

pub use arena_view::*;
pub use owned::*;

use anyhow::{Context, Result};
use num_traits::AsPrimitive;
use std::fmt::Display;
use tract_core::internal::*;
use tract_data::itertools::Itertools;

use crate::context::{with_borrowed_gpu_context, DeviceBuffer};

/// This struct represents a GPU tensor that can be either a owned tensor
/// or an arena view.
#[derive(Debug, Clone, Hash)]
pub enum GpuTensor {
    Owned(OwnedDeviceTensor),
    ArenaView(DeviceArenaView),
}

impl GpuTensor {
    pub const SUPPORTED_DT: [DatumType; 12] = [
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
        DatumType::Opaque,
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
            DatumType::Opaque => "opaque",
            _ => bail!("Unsupport dt {:?} for GPU Tensor", dt),
        })
    }

    /// Create an uninitialized GpuTensor
    pub unsafe fn uninitialized_dt(dt: DatumType, shape: &[usize]) -> Result<GpuTensor> {
        Tensor::uninitialized_dt(dt, shape)?.into_gpu()
    }

    pub unsafe fn uninitialized<T: Datum>(shape: &[usize]) -> Result<GpuTensor> {
        Self::uninitialized_dt(T::datum_type(), shape)
    }

    // Create a gpu tensor with a given shape and a slice of elements. The data is copied and aligned to size of T.
    pub fn from_shape<T: Copy + Datum>(shape: &[usize], data: &[T]) -> Result<GpuTensor> {
        Tensor::from_shape(shape, data)?.into_gpu()
    }

    pub fn is_supported_dt(dt: DatumType) -> bool {
        Self::SUPPORTED_DT.contains(&dt)
    }

    /// Get the datum type of the tensor.
    #[inline]
    pub fn datum_type(&self) -> DatumType {
        match self {
            Self::Owned(OwnedDeviceTensor { inner, .. }) => inner.datum_type(),
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
    pub fn device_buffer(&self) -> &Box<dyn DeviceBuffer> {
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
            Self::Owned(t) => t.buffer_offset(),
            Self::ArenaView(t) => t.buffer_offset(),
        }
    }

    pub fn device_buffer_address(&self) -> usize {
        match self {
            Self::Owned(t) => t.device_buffer_address(),
            Self::ArenaView(t) => t.device_buffer_address(),
        }
    }
    
    /// Get underlying inner tensor view.
    #[inline]
    pub fn view(&self) -> TensorView {
        match self {
            Self::Owned(t) => t.view(),
            Self::ArenaView(t) => t.view(),
        }
    }

    /// Returns short description of the inner tensor.
    pub fn description(&self) -> String {
        format!("|{},{:?}|", self.shape().iter().join(","), self.datum_type(),)
    }

    /// Reshaped tensor with given shape.
    pub fn reshaped(&self, shape: impl Into<TVec<usize>>) -> Result<Self> {
        match self {
            Self::Owned(t) => Ok(Self::Owned(t.reshaped(shape)?)),
            Self::ArenaView(t) => Ok(Self::ArenaView(t.reshaped(shape)?)),
        }
    }

    pub fn restrided(&self, strides: impl Into<TVec<isize>>) -> Result<Self> {
        match self {
            Self::Owned(t) => Ok(Self::Owned(t.restrided(strides)?)),
            Self::ArenaView(t) => Ok(Self::ArenaView(t.restrided(strides)?)),
        }
    }

    /// Convert GPU tensor to Opaque Tensor.
    pub fn into_opaque_tensor(self) -> Tensor {
        tensor0::<Opaque>(self.into())
    }

    /// Synchronize the GPU Tensor by completing all current
    /// commands on GPU and returns the inner tensor.
    pub fn to_cpu(&self) -> Result<Arc<Tensor>> {
        with_borrowed_gpu_context(|context| -> Result<Arc<Tensor>> {
                context.synchronize()?;
                Ok(match self {
                    Self::Owned(o) => o
                        .inner
                        .as_arc_tensor()
                        .cloned()
                        .unwrap_or_else(|| o.inner.clone().into_tensor().into_arc_tensor()),
                    Self::ArenaView(v) => v.clone().into_tensor().into(),
                })
            })
            .with_context(|| anyhow!("Error while synchronize gpu tensor to its cpu counterpart"))
    }
}

impl Display for GpuTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Owned(o) => match &o.inner {
                MValue::Const(t) | MValue::Var(t) => {
                    let content = t.dump(false).unwrap_or_else(|e| format!("Error : {e:?}"));
                    write!(f, "Owned: {{ {content} }}")
                }
                MValue::Reshaped { t, shape, .. } => {
                    let content = t.dump(false).unwrap_or_else(|e| format!("Error : {e:?}"));
                    write!(f, "Owned,Reshaped: {:?} - {{ {content} }}", shape)
                }
            },
            Self::ArenaView(v) => {
                let content = v
                    .clone()
                    .into_tensor()
                    .dump(false)
                    .unwrap_or_else(|e| format!("Error : {e:?}"));
                write!(f, "ArenaView: {{ {content} }}")
            }
        }
    }
}

pub trait IntoGpu<T> {
    fn into_gpu(self) -> Result<T>;
}

impl IntoGpu<GpuTensor> for Tensor {
    fn into_gpu(self) -> Result<GpuTensor> {
        Ok(GpuTensor::Owned(OwnedDeviceTensor::from_tensor(self)?))
    }
}

impl IntoGpu<GpuTensor> for Arc<Tensor> {
    fn into_gpu(self) -> Result<GpuTensor> {
        Ok(GpuTensor::Owned(OwnedDeviceTensor::from_tensor(self)?))
    }
}

impl From<GpuTensor> for Opaque {
    fn from(value: GpuTensor) -> Self {
        Opaque(Arc::new(value))
    }
}

impl From<DeviceArenaView> for GpuTensor {
    fn from(view: DeviceArenaView) -> Self {
        Self::ArenaView(view)
    }
}

impl OpaquePayload for GpuTensor {
    fn same_as(&self, other: &dyn OpaquePayload) -> bool {
        other
            .downcast_ref::<Self>()
            .is_some_and(|other| self.device_buffer_address() == other.device_buffer_address())
    }

    fn clarify_to_tensor(&self) -> TractResult<Option<Arc<Tensor>>> {
        Ok(Some(self.to_cpu()?))
    }
}

pub trait GpuTensorExt {
    fn to_gpu_tensor(&self) -> Result<&GpuTensor>;
    fn as_gpu_tensor(&self) -> Option<&GpuTensor>;
    fn to_gpu_tensor_mut(&mut self) -> Result<&mut GpuTensor>;
    fn as_gpu_tensor_mut(&mut self) -> Option<&mut GpuTensor>;
}

impl GpuTensorExt for Tensor {
    fn to_gpu_tensor_mut(&mut self) -> Result<&mut GpuTensor> {
        let opaque = self.to_scalar_mut::<Opaque>()?;
        opaque.downcast_mut::<GpuTensor>().ok_or_else(|| {
            anyhow::anyhow!("Could convert opaque tensor to mutable reference on a gpu tensor")
        })
    }

    fn as_gpu_tensor_mut(&mut self) -> Option<&mut GpuTensor> {
        let opaque = self.to_scalar_mut::<Opaque>().ok()?;
        opaque.downcast_mut::<GpuTensor>()
    }

    fn to_gpu_tensor(&self) -> Result<&GpuTensor> {
        let opaque = self.to_scalar::<Opaque>()?;
        opaque.downcast_ref::<GpuTensor>().ok_or_else(|| {
            anyhow::anyhow!("Could convert opaque tensor to reference on a gpu tensor")
        })
    }

    fn as_gpu_tensor(&self) -> Option<&GpuTensor> {
        let opaque = self.to_scalar::<Opaque>().ok()?;
        opaque.downcast_ref::<GpuTensor>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gpu_tensor() -> Result<()> {
        let a = GpuTensor::from_shape(&[1], &[0f32])?;
        assert_eq!(a.to_cpu()?.as_slice::<f32>()?, &[0.0]);
        Ok(())
    }
}
