use crate::IntoMetal;
use anyhow::Result;
use metal::{Buffer, MTLResourceOptions, NSUInteger};
use std::fmt::Display;
use tract_core::internal::*;
use tract_data::itertools::Itertools;

/// This struct represents a metal tensor that can be accessed from the
/// GPU and the CPU. Metal's MTLResourceStorageModeShared is used.

#[derive(Debug)]
pub struct MetalTensor {
    inner: Arc<Tensor>,
    metal: Buffer,
}

impl Hash for MetalTensor {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.inner.hash(state);
    }
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

    // Create a metal tensor with a given shape and a slice of elements. The data is copied and aligned to size of T.
    pub fn from_shape<T: Copy + Datum>(shape: &[usize], data: &[T]) -> Result<MetalTensor> {
        Tensor::from_shape(shape, data)?.into_metal()
    }

    /// Create an uninitialized MetalTensor
    pub unsafe fn uninitialized_dt(dt: DatumType, shape: &[usize]) -> Result<MetalTensor> {
        Tensor::uninitialized_dt(dt, shape)?.into_metal()
    }

    /// Create a MetalTensor filled of zeros
    pub fn zero_dt(datum_type: DatumType, shape: &[usize]) -> Result<MetalTensor> {
        Tensor::zero_dt(datum_type, shape)?.into_metal()
    }

    pub fn is_supported_dt(dt: DatumType) -> bool {
        Self::SUPPORTED_DT.contains(&dt)
    }

    /// Create a metal tensor from a cpu tensor.
    pub fn from_tensor(tensor: Arc<Tensor>) -> Result<Self> {
        crate::METAL_CONTEXT.with_borrow(|ctxt| {
            ensure!(
                Self::is_supported_dt(tensor.datum_type()),
                "Tensor of {:?} is not copied. No Metal buffer can be allocated for it.",
                tensor.datum_type(),
            );
            let size = (tensor.datum_type().size_of() * tensor.len()) as NSUInteger;
            let buffer = ctxt.device().new_buffer_with_bytes_no_copy(
                tensor.as_bytes().as_ptr() as *const core::ffi::c_void,
                size,
                MTLResourceOptions::StorageModeShared,
                None,
            );
            Ok(Self { inner: tensor, metal: buffer })
        })
    }

    /// Get the datum type of the tensor.
    #[inline]
    pub fn datum_type(&self) -> DatumType {
        self.inner.datum_type()
    }

    /// Get the number of dimensions (or axes) of the tensor.
    #[inline]
    pub fn rank(&self) -> usize {
        self.inner.rank()
    }

    /// Get the shape of the tensor.
    #[inline]
    pub fn shape(&self) -> &[usize] {
        self.inner.shape()
    }

    /// Get the number of values in the tensor.
    #[inline]
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Get the strides of the tensor.
    #[inline]
    pub fn strides(&self) -> &[isize] {
        self.inner.strides()
    }

    /// Get underlying inner metal buffer.
    pub fn metal(&self) -> &Buffer {
        &self.metal
    }

    /// Get underlying inner tensor.
    pub fn tensor(&self) -> &Tensor {
        &self.inner
    }

    /// Get mutable underlying inner metal buffer.
    pub fn metal_mut(&mut self) -> &mut Buffer {
        &mut self.metal
    }

    /// Returns short description of the inner tensor.
    pub fn description(&self) -> String {
        format!("|{},{:?}|", self.shape().iter().join(","), self.datum_type(),)
    }

    pub fn from_opaque_tensor(t: &Tensor) -> Result<&MetalTensor> {
        let opaque = t.to_scalar::<Opaque>()?;
        opaque.downcast_ref::<MetalTensor>().ok_or_else(|| {
            anyhow::anyhow!("Could convert opaque tensor to reference on a metal tensor")
        })
    }

    pub fn assert_sane_floats(&self) -> Result<()> {
        if let Ok(floats) = self.inner.as_slice::<f32>() {
            if let Some(pos) = floats.iter().position(|f| !f.is_finite()) {
                bail!("Found {} in at position {:?}", floats[pos], pos);
            }
        } else if let Ok(floats) = self.inner.as_slice::<f16>() {
            if let Some(pos) = floats.iter().position(|f| !f.is_finite()) {
                bail!("Found {} in at position {:?}", floats[pos], pos);
            }
        }
        Ok(())
    }

    /// Convert Metal tensor to Opaque Tensor.
    pub fn into_opaque_tensor(self) -> Tensor {
        tensor0::<Opaque>(self.into())
    }

    pub fn copy(&self) -> Result<MetalTensor> {
        self.inner.as_ref().clone().into_metal()
    }

    pub fn to_cpu(self) -> Tensor {
        log::warn!("MetalTensor to Tensor");
        self.inner.into_tensor()
    }
}

impl Display for MetalTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let content = self.inner.dump(false).unwrap_or_else(|e| format!("Error : {e:?}"));
        write!(f, "Metal {{ {content} }}")
    }
}

impl IntoMetal<MetalTensor> for Tensor {
    fn into_metal(self) -> Result<MetalTensor> {
        MetalTensor::from_tensor(Arc::new(self))
    }
}

impl IntoMetal<MetalTensor> for Arc<Tensor> {
    fn into_metal(self) -> Result<MetalTensor> {
        MetalTensor::from_tensor(self)
    }
}

impl From<MetalTensor> for Opaque {
    fn from(value: MetalTensor) -> Self {
        Opaque(Arc::new(value))
    }
}

impl OpaquePayload for MetalTensor {}

pub trait MetalTensorExt {
    fn to_opaque_metal_tensor(&self) -> Result<&MetalTensor>;
    fn as_opaque_metal_tensor(&self) -> Option<&MetalTensor>;
}

impl MetalTensorExt for Tensor {
    fn to_opaque_metal_tensor(&self) -> Result<&MetalTensor> {
        let opaque = self.to_scalar::<Opaque>()?;
        opaque.downcast_ref::<MetalTensor>().ok_or_else(|| {
            anyhow::anyhow!("Could convert opaque tensor to reference on a metal tensor")
        })
    }

    fn as_opaque_metal_tensor(&self) -> Option<&MetalTensor> {
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
        assert_eq!(a.to_cpu().as_slice::<f32>()?, &[0.0]);
        Ok(())
    }
}
