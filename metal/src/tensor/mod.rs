mod arena_view;
mod owned;

pub use arena_view::*;
pub use owned::*;

use crate::IntoMetal;
use anyhow::{Context, Result};
use metal::Buffer;
use num_traits::AsPrimitive;
use std::fmt::Display;
use tract_core::internal::*;
use tract_data::itertools::Itertools;

/// This struct represents a metal tensor that can be either a owned tensor
/// or an arena view.
#[derive(Debug, Clone, Hash)]
pub enum MetalTensor {
    Owned(OwnedMetalTensor),
    ArenaView(MetalArenaView),
}

impl MetalTensor {
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
            _ => bail!("Unsupport dt {:?} for metal kernel function", dt),
        })
    }

    /// Create an uninitialized MetalTensor
    pub unsafe fn uninitialized_dt(dt: DatumType, shape: &[usize]) -> Result<MetalTensor> {
        Tensor::uninitialized_dt(dt, shape)?.into_metal()
    }

    pub unsafe fn uninitialized<T: Datum>(shape: &[usize]) -> Result<MetalTensor> {
        Self::uninitialized_dt(T::datum_type(), shape)
    }

    // Create a metal tensor with a given shape and a slice of elements. The data is copied and aligned to size of T.
    pub fn from_shape<T: Copy + Datum>(shape: &[usize], data: &[T]) -> Result<MetalTensor> {
        Tensor::from_shape(shape, data)?.into_metal()
    }

    pub fn is_supported_dt(dt: DatumType) -> bool {
        Self::SUPPORTED_DT.contains(&dt)
    }

    /// Get the datum type of the tensor.
    #[inline]
    pub fn datum_type(&self) -> DatumType {
        match self {
            Self::Owned(OwnedMetalTensor { inner, .. }) => inner.datum_type(),
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

    /// Get underlying inner metal buffer.
    pub fn metal(&self) -> &Buffer {
        match self {
            Self::Owned(t) => t.metal(),
            Self::ArenaView(t) => t.metal(),
        }
    }

    /// Get underlying inner metal buffer offset
    pub fn metal_offset<I: Copy + 'static>(&self) -> I
    where
        usize: AsPrimitive<I>,
    {
        match self {
            Self::Owned(t) => t.metal_offset(),
            Self::ArenaView(t) => t.metal_offset(),
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

    #[inline]
    pub fn retain_until_completion(&self) {
        crate::METAL_CONTEXT.with_borrow(|ctxt| ctxt.retain_tensor(self));
    }

    #[inline]
    pub fn retained_until_completion(&self) -> &Self {
        self.retain_until_completion();
        self
    }

    /// Convert Metal tensor to Opaque Tensor.
    pub fn into_opaque_tensor(self) -> Tensor {
        tensor0::<Opaque>(self.into())
    }

    /// Synchronize the Metal Tensor by completing all current
    /// commands on GPU and returns the inner tensor.
    pub fn to_cpu(&self) -> Result<Tensor> {
        crate::METAL_CONTEXT
            .with_borrow(|context| -> Result<Tensor> {
                context.wait_until_completed()?;
                Ok(match self {
                    Self::Owned(o) => o.inner.clone().into_tensor(),
                    Self::ArenaView(v) => v.clone().into_tensor(),
                })
            })
            .with_context(|| anyhow!("Error while synchronize metal tensor to its cpu counterpart"))
    }
}

impl Display for MetalTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Owned(o) => match &o.inner {
                MValue::Const(t) | MValue::Var(t) => {
                    let content = t.dump(false).unwrap_or_else(|e| format!("Error : {e:?}"));
                    write!(f, "Metal,Owned: {{ {content} }}")
                }
                MValue::Reshaped { t, shape, .. } => {
                    let content = t.dump(false).unwrap_or_else(|e| format!("Error : {e:?}"));
                    write!(f, "Metal,Owned,Reshaped: {:?} - {{ {content} }}", shape)
                }
            },
            Self::ArenaView(v) => {
                let content = v
                    .clone()
                    .into_tensor()
                    .dump(false)
                    .unwrap_or_else(|e| format!("Error : {e:?}"));
                write!(f, "Metal,ArenaView: {{ {content} }}")
            }
        }
    }
}

impl IntoMetal<MetalTensor> for Tensor {
    fn into_metal(self) -> Result<MetalTensor> {
        Ok(MetalTensor::Owned(OwnedMetalTensor::from_tensor(self)?))
    }
}

impl IntoMetal<MetalTensor> for Arc<Tensor> {
    fn into_metal(self) -> Result<MetalTensor> {
        Ok(MetalTensor::Owned(OwnedMetalTensor::from_tensor(self)?))
    }
}

impl From<MetalTensor> for Opaque {
    fn from(value: MetalTensor) -> Self {
        Opaque(Arc::new(value))
    }
}

impl From<MetalArenaView> for MetalTensor {
    fn from(view: MetalArenaView) -> Self {
        Self::ArenaView(view)
    }
}

impl OpaquePayload for MetalTensor {
    fn clarify_to_tensor(&self) -> Option<Result<Tensor>> {
        Some(self.to_cpu())
    }
}

pub trait MetalTensorExt {
    fn to_metal_tensor(&self) -> Result<&MetalTensor>;
    fn as_metal_tensor(&self) -> Option<&MetalTensor>;
    fn to_metal_tensor_mut(&mut self) -> Result<&mut MetalTensor>;
    fn as_metal_tensor_mut(&mut self) -> Option<&mut MetalTensor>;
}

impl MetalTensorExt for Tensor {
    fn to_metal_tensor_mut(&mut self) -> Result<&mut MetalTensor> {
        let opaque = self.to_scalar_mut::<Opaque>()?;
        opaque.downcast_mut::<MetalTensor>().ok_or_else(|| {
            anyhow::anyhow!("Could convert opaque tensor to mutable reference on a metal tensor")
        })
    }

    fn as_metal_tensor_mut(&mut self) -> Option<&mut MetalTensor> {
        let opaque = self.to_scalar_mut::<Opaque>().ok()?;
        opaque.downcast_mut::<MetalTensor>()
    }

    fn to_metal_tensor(&self) -> Result<&MetalTensor> {
        let opaque = self.to_scalar::<Opaque>()?;
        opaque.downcast_ref::<MetalTensor>().ok_or_else(|| {
            anyhow::anyhow!("Could convert opaque tensor to reference on a metal tensor")
        })
    }

    fn as_metal_tensor(&self) -> Option<&MetalTensor> {
        let opaque = self.to_scalar::<Opaque>().ok()?;
        opaque.downcast_ref::<MetalTensor>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metal_tensor() -> Result<()> {
        let a = MetalTensor::from_shape(&[1], &[0f32])?;
        assert_eq!(a.to_cpu()?.as_slice::<f32>()?, &[0.0]);
        Ok(())
    }
}
