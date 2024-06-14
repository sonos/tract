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
    inner: Tensor,
    metal: Buffer,
}

impl Hash for MetalTensor {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.inner.hash(state);
    }
}

impl MetalTensor {
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

    /// Create a metal tensor from a cpu tensor.
    pub fn from_tensor(tensor: Tensor) -> Result<Self> {
        crate::METAL_CONTEXT.with_borrow(|ctxt| {
            ensure!(
                tensor.datum_type().is_copy(),
                "Tensor is not copied. No Metal buffer can be allocated for it."
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
        format!("{},{:?}", self.shape().iter().join(","), self.datum_type(),)
    }

    /// Convert Metal tensor to Opaque Tensor.
    pub fn into_opaque_tensor(self) -> Tensor {
        tensor0::<Opaque>(self.into())
    }
}

impl Display for MetalTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let content = self.description();
        write!(f, "{content}")
    }
}

impl IntoTensor for MetalTensor {
    fn into_tensor(self) -> Tensor {
        self.inner
    }
}

impl IntoMetal<MetalTensor> for Tensor {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_metal_tensor() -> Result<()> {
        let a = MetalTensor::from_shape(&[1], &[0f32])?;
        assert_eq!(a.into_tensor().as_slice::<f32>()?, &[0.0]);
        Ok(())
    }
}
