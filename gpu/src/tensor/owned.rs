use crate::device::{get_context, DeviceBuffer};
use crate::tensor::DeviceTensor;
use crate::utils::{as_q40_tensor, check_strides_validity};
use num_traits::AsPrimitive;
use std::ffi::c_void;
use std::fmt::Display;
use tract_core::internal::*;

#[derive(Debug, Clone, Hash)]
pub enum DValue {
    Natural(Arc<Tensor>),
    Reshaped { t: Arc<Tensor>, shape: TVec<usize>, strides: TVec<isize> },
}

impl DValue {
    #[inline]
    pub fn view(&self) -> TensorView<'_> {
        match self {
            Self::Natural(t) => t.view(),
            Self::Reshaped { t, shape, strides } => unsafe {
                TensorView::from_bytes(t, 0, shape.as_slice(), strides.as_slice())
            },
        }
    }

    /// Get the datum type of the tensor.
    #[inline]
    pub fn datum_type(&self) -> DatumType {
        match self {
            Self::Natural(t) => t.datum_type(),
            Self::Reshaped { t, .. } => t.datum_type(),
        }
    }

    #[inline]
    pub fn shape(&self) -> &[usize] {
        match self {
            DValue::Natural(t) => t.shape(),
            DValue::Reshaped { shape, .. } => shape,
        }
    }

    /// Get the number of values.
    #[inline]
    #[allow(clippy::len_without_is_empty)]
    pub fn len(&self) -> usize {
        self.shape().iter().product()
    }

    /// Reshaped tensor with given shape.
    pub fn reshaped(&self, shape: impl Into<TVec<usize>>) -> TractResult<Self> {
        let shape = shape.into();
        if self.len() != shape.iter().product::<usize>() {
            bail!("Invalid reshape {:?} to {:?}", self.shape(), shape);
        }
        if shape.as_slice() != self.shape() {
            match &self {
                DValue::Natural(t) | DValue::Reshaped { t, .. } => Ok(Self::Reshaped {
                    t: Arc::clone(t),
                    strides: Tensor::natural_strides(&shape),
                    shape,
                }),
            }
        } else {
            Ok(self.clone())
        }
    }

    pub fn restrided(&self, strides: impl Into<TVec<isize>>) -> TractResult<Self> {
        let strides = strides.into();
        check_strides_validity(self.shape().into(), strides.clone())?;

        match &self {
            DValue::Natural(t) => {
                Ok(Self::Reshaped { t: Arc::clone(t), strides, shape: self.shape().into() })
            }
            DValue::Reshaped { t, strides: old_strides, .. } => {
                if &strides != old_strides {
                    Ok(Self::Reshaped { t: Arc::clone(t), strides, shape: self.shape().into() })
                } else {
                    Ok(self.clone())
                }
            }
        }
    }

    pub fn as_arc_tensor(&self) -> Option<&Arc<Tensor>> {
        match self {
            DValue::Natural(t) => Some(t),
            DValue::Reshaped { .. } => None,
        }
    }

    /// Reshaped tensor with given shape and strides, no consistency check.
    pub unsafe fn reshaped_with_geometry_unchecked(
        &self,
        shape: impl Into<TVec<usize>>,
        strides: impl Into<TVec<isize>>,
    ) -> Self {
        match self {
            DValue::Natural(t) | DValue::Reshaped { t, .. } => {
                DValue::Reshaped { t: Arc::clone(t), strides: strides.into(), shape: shape.into() }
            }
        }
    }
}

impl IntoTensor for DValue {
    fn into_tensor(self) -> Tensor {
        match self {
            Self::Natural(t) => Arc::try_unwrap(t).unwrap_or_else(|t| (*t).clone()),
            Self::Reshaped { t, shape, strides: _ } => {
                let mut t = Arc::try_unwrap(t).unwrap_or_else(|t| (*t).clone());
                t.set_shape(&shape).expect("Could not apply shape to reshaped GPU tensor");
                t
            }
        }
    }
}

impl From<Tensor> for DValue {
    fn from(v: Tensor) -> Self {
        Self::Natural(Arc::new(v))
    }
}

impl From<Arc<Tensor>> for DValue {
    fn from(v: Arc<Tensor>) -> Self {
        Self::Natural(v)
    }
}

/// This struct represents a owned tensor that can be accessed from the
/// GPU and the CPU.
#[derive(Debug, Clone)]
pub struct OwnedDeviceTensor {
    pub inner: DValue,
    pub device_buffer: Box<dyn DeviceBuffer>,
}

impl Hash for OwnedDeviceTensor {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.inner.hash(state)
    }
}

impl OwnedDeviceTensor {
    /// Create a owned gpu tensor from a cpu tensor.
    pub fn from_tensor<T: Into<DValue>>(tensor: T) -> TractResult<Self> {
        let m_value: DValue = tensor.into();
        let tensor_view = m_value.view();
        ensure!(
            DeviceTensor::is_supported_dt(tensor_view.datum_type()),
            "Tensor of {:?} is not copied. No device buffer can be allocated for it.",
            tensor_view.datum_type(),
        );

        let data_bytes = as_q40_tensor(tensor_view.tensor)
            .map(|bqv| bqv.value.as_bytes())
            .unwrap_or(tensor_view.tensor.as_bytes());

        let device_buffer = get_context()?.buffer_from_slice(data_bytes);

        Ok(OwnedDeviceTensor { inner: m_value, device_buffer })
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
            DValue::Natural(t) => t.strides(),
            DValue::Reshaped { strides, .. } => strides,
        }
    }

    /// Get underlying inner device buffer.
    #[inline]
    pub fn device_buffer(&self) -> &dyn DeviceBuffer {
        &(*self.device_buffer)
    }

    pub fn device_buffer_ptr(&self) -> *const c_void {
        self.device_buffer.ptr()
    }

    /// Get underlying inner buffer offset
    #[inline]
    pub fn buffer_offset<I: Copy + 'static>(&self) -> I
    where
        usize: AsPrimitive<I>,
    {
        // No offset for non-arena tensor
        0usize.as_()
    }

    /// Reshaped tensor with given shape.
    #[inline]
    pub fn reshaped(&self, shape: impl Into<TVec<usize>>) -> TractResult<Self> {
        Ok(Self { inner: self.inner.reshaped(shape)?, device_buffer: self.device_buffer.clone() })
    }

    /// Change tensor stride.
    #[inline]
    pub fn restrided(&self, strides: impl Into<TVec<isize>>) -> TractResult<Self> {
        Ok(Self {
            inner: self.inner.restrided(strides)?,
            device_buffer: self.device_buffer.clone(),
        })
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
            device_buffer: self.device_buffer.clone(),
        }
    }

    #[inline]
    pub fn view(&self) -> TensorView<'_> {
        self.inner.view()
    }
}

impl Display for OwnedDeviceTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.inner {
            DValue::Natural(t) => {
                let content = t.dump(false).unwrap_or_else(|e| format!("Error : {e:?}"));
                write!(f, "GPU {{ {content} }}")
            }
            DValue::Reshaped { t, shape, strides: _ } => {
                let content = t.dump(false).unwrap_or_else(|e| format!("Error : {e:?}"));
                write!(f, "GPU reshaped: {:?} - {{ {content} }}", shape)
            }
        }
    }
}
