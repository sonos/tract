use crate::IntoMetal;
use anyhow::Result;
use metal::{Buffer, MTLResourceOptions, NSUInteger};
use std::fmt::Display;
use tract_core::internal::*;
use tract_data::itertools::Itertools;

#[derive(Debug, Clone, Hash)]
pub enum MValue {
    Const(Arc<Tensor>),
    Var(Arc<Tensor>),
    Reshaped { t: Arc<Tensor>, shape: TVec<usize>, strides: TVec<isize> },
}

impl MValue {
    #[inline]
    pub fn view(&self) -> TensorView<'_> {
        match self {
            Self::Const(t) => t.view(),
            Self::Var(t) => t.view(),
            Self::Reshaped { t, shape, strides } => unsafe {
                TensorView::from_bytes(t, 0, shape.as_slice(), strides.as_slice())
            },
        }
    }
}

impl IntoTensor for MValue {
    fn into_tensor(self) -> Tensor {
        match self {
            Self::Var(t) => Arc::try_unwrap(t).unwrap_or_else(|t| (*t).clone()),
            Self::Const(t) => Arc::try_unwrap(t).unwrap_or_else(|t| (*t).clone()),
            Self::Reshaped { t, shape, strides: _ } => {
                let mut t = Arc::try_unwrap(t).unwrap_or_else(|t| (*t).clone());
                t.set_shape(&shape).expect("Could not apply shape to reshaped metal tensor");
                t
            }
        }
    }
}

impl From<Tensor> for MValue {
    fn from(v: Tensor) -> Self {
        Self::Var(Arc::new(v))
    }
}

impl From<Arc<Tensor>> for MValue {
    fn from(v: Arc<Tensor>) -> Self {
        Self::Const(v)
    }
}

/// This struct represents a metal tensor that can be accessed from the
/// GPU and the CPU. Metal's MTLResourceStorageModeShared is used.
#[derive(Debug, Clone)]
pub struct MetalTensor {
    dt: DatumType,
    inner: MValue,
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

    // Create a metal tensor with a given shape and a slice of elements. The data is copied and aligned to size of T.
    pub fn from_shape<T: Copy + Datum>(shape: &[usize], data: &[T]) -> Result<MetalTensor> {
        Tensor::from_shape(shape, data)?.into_metal()
    }

    /// Create an uninitialized MetalTensor
    pub unsafe fn uninitialized_dt(dt: DatumType, shape: &[usize]) -> Result<MetalTensor> {
        Tensor::uninitialized_dt(dt, shape)?.into_metal()
    }

    /// Create an uninitialized MetalTensor
    pub unsafe fn zero_dt(dt: DatumType, shape: &[usize]) -> Result<MetalTensor> {
        Tensor::zero_dt(dt, shape)?.into_metal()
    }

    pub unsafe fn uninitialized<T: Datum>(shape: &[usize]) -> Result<MetalTensor> {
        Self::uninitialized_dt(T::datum_type(), shape)
    }

    pub fn is_supported_dt(dt: DatumType) -> bool {
        Self::SUPPORTED_DT.contains(&dt)
    }

    pub fn value(&self) -> &MValue {
        &self.inner
    }

    /// Create a metal tensor from a cpu tensor.
    pub fn from_tensor<T: Into<MValue>>(tensor: T) -> Result<Self> {
        crate::METAL_CONTEXT.with_borrow(|ctxt| {
            let m_value = tensor.into();
            let tensor_view = m_value.view();
            ensure!(
                Self::is_supported_dt(tensor_view.datum_type()),
                "Tensor of {:?} is not copied. No Metal buffer can be allocated for it.",
                tensor_view.datum_type(),
            );
            let size = (tensor_view.datum_type().size_of() * tensor_view.len()) as NSUInteger;
            let buffer = ctxt.device().new_buffer_with_bytes_no_copy(
                tensor_view.tensor.as_bytes().as_ptr() as *const core::ffi::c_void,
                size,
                MTLResourceOptions::StorageModeShared,
                None,
            );
            Ok(Self { dt: tensor_view.datum_type(), inner: m_value, metal: buffer })
        })
    }

    /// Get the datum type of the tensor.
    #[inline]
    pub fn datum_type(&self) -> DatumType {
        self.dt
    }

    /// Get the number of dimensions (or axes) of the tensor.
    #[inline]
    pub fn rank(&self) -> usize {
        self.shape().len()
    }

    /// Get the shape of the tensor.
    #[inline]
    pub fn shape(&self) -> &[usize] {
        match &self.inner {
            MValue::Const(t) | MValue::Var(t) => t.shape(),
            MValue::Reshaped { shape, .. } => shape,
        }
    }

    /// Get the number of values in the tensor.
    #[inline]
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.inner.view().len()
    }

    /// Get the strides of the tensor.
    #[inline]
    pub fn strides(&self) -> &[isize] {
        match &self.inner {
            MValue::Const(t) | MValue::Var(t) => t.strides(),
            MValue::Reshaped { strides, .. } => strides,
        }
    }

    /// Get underlying inner metal buffer.
    pub fn metal(&self) -> &Buffer {
        &self.metal
    }

    /// Get underlying inner tensor view.
    #[inline]
    pub fn view(&self) -> TensorView {
        self.inner.view()
    }

    /// Returns short description of the inner tensor.
    pub fn description(&self) -> String {
        format!("|{},{:?}|", self.shape().iter().join(","), self.datum_type(),)
    }

    /// Reshaped tensor with given shape.
    pub fn reshaped(&self, shape: impl Into<TVec<usize>>) -> Result<Self> {
        let shape = shape.into();
        if self.len() != shape.iter().product::<usize>() {
            bail!("Invalid reshape {:?} to {:?}", self.shape(), shape);
        }
        if shape.as_slice() != self.shape() {
            match &self.inner {
                MValue::Const(t) | MValue::Var(t) | MValue::Reshaped { t, .. } => Ok(Self {
                    dt: self.dt,
                    inner: MValue::Reshaped {
                        t: Arc::clone(t),
                        strides: Tensor::natural_strides(&shape),
                        shape,
                    },
                    metal: self.metal.clone(),
                }),
            }
        } else {
            Ok(Self { dt: self.dt, inner: self.inner.clone(), metal: self.metal.clone() })
        }
    }

    /// Reshaped tensor with given shape and strides, no consistency check.
    pub unsafe fn reshaped_with_geometry_unchecked(
        &self,
        shape: impl Into<TVec<usize>>,
        strides: impl Into<TVec<isize>>,
    ) -> Self {
        match &self.inner {
            MValue::Const(t) | MValue::Var(t) | MValue::Reshaped { t, .. } => Self {
                dt: self.dt,
                inner: MValue::Reshaped {
                    t: Arc::clone(t),
                    strides: strides.into(),
                    shape: shape.into(),
                },
                metal: self.metal.clone(),
            },
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

    pub fn assert_sane_floats(&self) -> Result<()> {
        if let Ok(floats) = self.inner.view().as_slice::<f32>() {
            if let Some(pos) = floats.iter().position(|f| !f.is_finite()) {
                bail!("Found {} in at position {:?}", floats[pos], pos);
            }
        } else if let Ok(floats) = self.inner.view().as_slice::<f16>() {
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

    pub fn to_cpu(&self) -> Tensor {
        self.inner.clone().into_tensor()
    }
}

impl Display for MetalTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.inner {
            MValue::Const(t) | MValue::Var(t) => {
                let content = t.dump(false).unwrap_or_else(|e| format!("Error : {e:?}"));
                write!(f, "Metal {{ {content} }}")
            }
            MValue::Reshaped { t, shape, strides: _ } => {
                let content = t.dump(false).unwrap_or_else(|e| format!("Error : {e:?}"));
                write!(f, "Metal reshaped: {:?} - {{ {content} }}", shape)
            }
        }
    }
}

impl IntoMetal<MetalTensor> for Tensor {
    fn into_metal(self) -> Result<MetalTensor> {
        MetalTensor::from_tensor(self)
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
        assert_eq!(a.to_cpu().as_slice::<f32>()?, &[0.0]);
        Ok(())
    }
}
