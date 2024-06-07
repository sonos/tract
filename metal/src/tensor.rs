use crate::{IntoMetal, MetalContext};
use anyhow::Result;
use metal::{Buffer, MTLResourceOptions, NSUInteger};
use tract_core::internal::*;

impl IntoMetal<MetalTensor> for Tensor {
    fn into_metal(self) -> Result<MetalTensor> {
        crate::METAL_CONTEXT.with_borrow(|ctxt| MetalTensor::from_tensor(ctxt, self))
    }
}

/// This struct represents a metal tensor that can be accessed from the
/// GPU and the CPU. Metal's MTLResourceStorageModeShared is used.
#[derive(Debug)]
pub struct MetalTensor {
    pub inner: Tensor,
    metal: Buffer,
}

impl MetalTensor {
    // Create a metal tensor with a given shape and a slice of elements. The data is copied and aligned to size of T.
    pub fn from_shape<T: Copy + Datum>(shape: &[usize], data: &[T]) -> Result<MetalTensor> {
        crate::METAL_CONTEXT.with_borrow(|ctxt| {
            let tensor = Tensor::from_shape(shape, data)?;
            Self::from_tensor(ctxt, tensor)
        })
    }

    /// Create an uninitialized MetalTensor
    pub unsafe fn uninitialized_dt(dt: DatumType, shape: &[usize]) -> Result<MetalTensor> {
        crate::METAL_CONTEXT.with_borrow(|ctxt| {
            let tensor = Tensor::uninitialized_dt(dt, shape)?;
            Self::from_tensor(ctxt, tensor)
        })
    }

    /// Create a MetalTensor filled of zeros
    pub fn zero_dt(
        context: &MetalContext,
        datum_type: DatumType,
        shape: &[usize],
    ) -> Result<MetalTensor> {
        let t = Tensor::zero_dt(datum_type, shape)?;
        Self::from_tensor(context, t)
    }

    /// Create a metal tensor from a cpu tensor.
    pub fn from_tensor(context: &MetalContext, tensor: Tensor) -> Result<Self> {
        ensure!(
            tensor.datum_type().is_copy(),
            "Tensor is not copied. No Metal buffer can be allocated for it."
        );
        let size = (tensor.datum_type().size_of() * tensor.len()) as NSUInteger;
        let buffer = context.device().new_buffer_with_bytes_no_copy(
            tensor.as_bytes().as_ptr() as *const core::ffi::c_void,
            size,
            MTLResourceOptions::StorageModeShared,
            None,
        );
        Ok(Self { inner: tensor, metal: buffer })
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
    pub fn len(&self) -> usize {
        self.inner.len()
    }

    /// Get the strides of the tensor.
    #[inline]
    pub fn strides(&self) -> &[isize] {
        &self.inner.strides()
    }

    /// Get underlying inner metal buffer.
    pub fn metal<'a>(&'a self) -> &'a Buffer {
        &self.metal
    }

    /// Get mutable underlying inner metal buffer.
    pub fn metal_mut<'a>(&'a mut self) -> &'a mut Buffer {
        &mut self.metal
    }
}

impl IntoTensor for MetalTensor {
    fn into_tensor(self) -> Tensor {
        self.inner
    }
}

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
