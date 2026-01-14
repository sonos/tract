use std::fmt::Display;
use tract_core::internal::*;
use tract_gpu::device::DeviceBuffer;
use tract_gpu::tensor::{DeviceTensor, OwnedDeviceTensor};
use tract_gpu::utils::check_strides_validity;

use crate::context::MetalBuffer;

#[derive(Debug, Clone, Hash)]
pub enum MValue {
    Natural(Arc<Tensor>),
    Reshaped { t: Arc<Tensor>, shape: TVec<usize>, strides: TVec<isize> },
}

impl MValue {
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
            MValue::Natural(t) => t.shape(),
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
    pub fn reshaped(&self, shape: impl Into<TVec<usize>>) -> TractResult<Self> {
        let shape = shape.into();
        if self.len() != shape.iter().product::<usize>() {
            bail!("Invalid reshape {:?} to {:?}", self.shape(), shape);
        }
        if shape.as_slice() != self.shape() {
            match &self {
                MValue::Natural(t) | MValue::Reshaped { t, .. } => Ok(Self::Reshaped {
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
            MValue::Natural(t) => {
                Ok(Self::Reshaped { t: Arc::clone(t), strides, shape: self.shape().into() })
            }
            MValue::Reshaped { t, strides: old_strides, .. } => {
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
            MValue::Natural(t) => Some(t),
            MValue::Reshaped { .. } => None,
        }
    }
}

impl IntoTensor for MValue {
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

impl From<Tensor> for MValue {
    fn from(v: Tensor) -> Self {
        Self::Natural(Arc::new(v))
    }
}

impl From<Arc<Tensor>> for MValue {
    fn from(v: Arc<Tensor>) -> Self {
        Self::Natural(v)
    }
}

/// This struct represents a owned tensor that can be accessed from the
/// GPU and the CPU.
#[derive(Clone)]
pub struct MetalTensor {
    pub inner: MValue,
    pub device_buffer: MetalBuffer,
    pub opaque_fact: Option<Box<dyn OpaqueFact>>,
}

impl std::fmt::Debug for MetalTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "MetalTensor: {:?}", self.inner)
    }
}

impl Hash for MetalTensor {
    #[inline]
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        self.inner.hash(state)
    }
}

impl OwnedDeviceTensor for MetalTensor {
    fn datum_type(&self) -> DatumType {
        self.inner.datum_type()
    }

    #[inline]
    fn shape(&self) -> &[usize] {
        self.inner.shape()
    }

    /// Get the number of values in the tensor.
    #[inline]
    #[allow(clippy::len_without_is_empty)]
    fn len(&self) -> usize {
        self.shape().iter().product()
    }

    /// Get the strides of the tensor.
    #[inline]
    fn strides(&self) -> &[isize] {
        match &self.inner {
            MValue::Natural(t) => t.strides(),
            MValue::Reshaped { strides, .. } => strides,
        }
    }

    /// Get underlying inner device buffer.
    #[inline]
    fn device_buffer(&self) -> &dyn DeviceBuffer {
        &self.device_buffer
    }

    /// Reshaped tensor with given shape.
    #[inline]
    fn reshaped(&self, shape: TVec<usize>) -> TractResult<DeviceTensor> {
        Ok(DeviceTensor::Owned(Box::new(Self {
            inner: self.inner.reshaped(shape)?,
            device_buffer: self.device_buffer.clone(),
            opaque_fact: self.opaque_fact.clone(),
        })))
    }

    /// Change tensor stride.
    #[inline]
    fn restrided(&self, strides: TVec<isize>) -> TractResult<DeviceTensor> {
        Ok(DeviceTensor::Owned(Box::new(Self {
            inner: self.inner.restrided(strides)?,
            device_buffer: self.device_buffer.clone(),
            opaque_fact: self.opaque_fact.clone(),
        })))
    }

    fn to_host(&self) -> TractResult<Arc<Tensor>> {
        Ok(self
            .inner
            .as_arc_tensor()
            .cloned()
            .unwrap_or_else(|| self.inner.clone().into_tensor().into_arc_tensor()))
    }

    fn opaque_fact(&self) -> Option<&dyn OpaqueFact> {
        self.opaque_fact.as_deref()
    }

    fn get_bytes_slice(&self, offset: usize, len: usize) -> Vec<u8> {
        self.inner.as_arc_tensor().unwrap().as_bytes()[offset..offset + len].to_vec()
    }
}

impl Display for MetalTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match &self.inner {
            MValue::Natural(t) => {
                let content = t.dump(false).unwrap_or_else(|e| format!("Error : {e:?}"));
                write!(f, "GPU {{ {content} }}")
            }
            MValue::Reshaped { t, shape, strides: _ } => {
                let content = t.dump(false).unwrap_or_else(|e| format!("Error : {e:?}"));
                write!(f, "GPU reshaped: {:?} - {{ {content} }}", shape)
            }
        }
    }
}
