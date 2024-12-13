use crate::MetalTensor;
use anyhow::Result;
use metal::Buffer;
use num_traits::AsPrimitive;
use std::fmt::Display;
use tract_core::internal::*;

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

    /// Get the datum type of the tensor.
    #[inline]
    pub fn datum_type(&self) -> DatumType {
        match self {
            Self::Const(t) => t.datum_type(),
            Self::Var(t) => t.datum_type(),
            Self::Reshaped { t, .. } => t.datum_type(),
        }
    }

    #[inline]
    pub fn shape(&self) -> &[usize] {
        match self {
            MValue::Const(t) | MValue::Var(t) => t.shape(),
            MValue::Reshaped { shape, .. } => shape,
        }
    }

    /// Get the number of values.
    #[inline]
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.shape().iter().product()
    }

    /// Reshaped tensor with given shape.
    pub fn reshaped(&self, shape: impl Into<TVec<usize>>) -> Result<Self> {
        let shape = shape.into();
        if self.len() != shape.iter().product::<usize>() {
            bail!("Invalid reshape {:?} to {:?}", self.shape(), shape);
        }
        if shape.as_slice() != self.shape() {
            match &self {
                MValue::Const(t) | MValue::Var(t) | MValue::Reshaped { t, .. } => {
                    Ok(Self::Reshaped {
                        t: Arc::clone(t),
                        strides: Tensor::natural_strides(&shape),
                        shape,
                    })
                }
            }
        } else {
            Ok(self.clone())
        }
    }

    /// Reshaped tensor with given shape and strides, no consistency check.
    pub unsafe fn reshaped_with_geometry_unchecked(
        &self,
        shape: impl Into<TVec<usize>>,
        strides: impl Into<TVec<isize>>,
    ) -> Self {
        match self {
            MValue::Const(t) | MValue::Var(t) | MValue::Reshaped { t, .. } => {
                MValue::Reshaped { t: Arc::clone(t), strides: strides.into(), shape: shape.into() }
            }
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

/// This struct represents a owned metal tensor that can be accessed from the
/// GPU and the CPU. Metal's MTLResourceStorageModeShared is used.
#[derive(Debug, Clone)]
pub struct OwnedMetalTensor {
    pub inner: MValue,
    pub metal: Buffer,
}

impl Hash for OwnedMetalTensor {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.inner.hash(state)
    }
}

impl OwnedMetalTensor {
    /// Create a owned metal tensor from a cpu tensor.
    pub fn from_tensor<T: Into<MValue>>(tensor: T) -> Result<Self> {
        crate::METAL_CONTEXT.with_borrow(|ctxt| {
            let m_value = tensor.into();
            let tensor_view = m_value.view();
            ensure!(
                MetalTensor::is_supported_dt(tensor_view.datum_type()),
                "Tensor of {:?} is not copied. No Metal buffer can be allocated for it.",
                tensor_view.datum_type(),
            );
            let buffer = ctxt.buffer_from_slice(tensor_view.tensor.as_bytes());
            Ok(OwnedMetalTensor { inner: m_value, metal: buffer })
        })
    }

    #[inline]
    pub fn shape(&self) -> &[usize] {
        self.inner.shape()
    }

    /// Get the number of values in the tensor.
    #[inline]
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.shape().iter().product()
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
    #[inline]
    pub fn metal(&self) -> &Buffer {
        &self.metal
    }

    /// Get underlying inner metal buffer offset
    #[inline]
    pub fn metal_offset<I: Copy + 'static>(&self) -> I
    where
        usize: AsPrimitive<I>,
    {
        0usize.as_()
    }

    /// Reshaped tensor with given shape.
    #[inline]
    pub fn reshaped(&self, shape: impl Into<TVec<usize>>) -> Result<Self> {
        Ok(Self { inner: self.inner.reshaped(shape)?, metal: self.metal.clone() })
    }

    /// Reshaped tensor with given shape and strides, no consistency check.
    #[inline]
    pub unsafe fn reshaped_with_geometry_unchecked(
        &self,
        shape: impl Into<TVec<usize>>,
        strides: impl Into<TVec<isize>>,
    ) -> Self {
        Self {
            inner: self.inner.reshaped_with_geometry_unchecked(shape, strides),
            metal: self.metal.clone(),
        }
    }

    #[inline]
    pub fn view(&self) -> TensorView<'_> {
        self.inner.view()
    }
}

impl Display for OwnedMetalTensor {
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
