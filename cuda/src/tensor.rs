use std::ops::{Deref, DerefMut};

use cudarc::driver::{CudaSlice, DevicePtr};
use tract_core::internal::tract_smallvec::ToSmallVec;
use tract_core::internal::*;
use tract_core::prelude::{DatumType, TVec};
use tract_core::tract_linalg::block_quant::{BlockQuantFact, BlockQuantStorage, Q8_1};
use tract_gpu::device::DeviceBuffer;
use tract_gpu::tensor::{DeviceTensor, OwnedDeviceTensor};
use tract_gpu::utils::{as_q40_tensor, check_strides_validity};

use crate::ops::GgmlQuantQ81Fact;

#[derive(Debug, Clone)]
pub struct CudaBuffer {
    pub inner: CudaSlice<u8>,
}

impl DeviceBuffer for CudaBuffer {
    fn ptr(&self) -> *const std::ffi::c_void {
        crate::with_cuda_stream(|stream| Ok(self.inner.device_ptr(stream).0 as _)).unwrap()
    }
}
impl Deref for CudaBuffer {
    type Target = CudaSlice<u8>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for CudaBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl PartialEq for CudaBuffer {
    fn eq(&self, other: &Self) -> bool {
        self.ptr() == other.ptr() && self.inner.len() == other.inner.len()
    }
}
impl Eq for CudaBuffer {}

#[derive(Clone, PartialEq, Eq)]
pub struct CudaTensor {
    buffer: Arc<CudaBuffer>,
    datum_type: DatumType,
    shape: TVec<usize>,
    strides: TVec<isize>,
    exotic_fact: Option<Box<dyn ExoticFact>>,
}

impl CudaTensor {
    pub fn from_tensor(tensor: &Tensor) -> TractResult<Self> {
        if let Some(bqs) = as_q40_tensor(tensor) {
            let bqf = BlockQuantFact::new(
                tract_core::dyn_clone::clone_box(bqs.format()),
                tensor.shape().into(),
            );
            let data = bqs.value().as_bytes();
            crate::with_cuda_stream(|stream| {
                let device_data = stream
                    .clone_htod(data)
                    .with_context(|| format!("Data address: {:?}", data.as_ptr()))?;
                let buffer = Arc::new(CudaBuffer { inner: device_data });
                Ok(CudaTensor {
                    buffer,
                    datum_type: tensor.datum_type(),
                    shape: tensor.shape().into(),
                    strides: tensor.strides().into(),
                    exotic_fact: Some(Box::new(bqf)),
                })
            })
        } else {
            let data = tensor.as_bytes();
            crate::with_cuda_stream(|stream| {
                let device_data = stream
                    .clone_htod(data)
                    .with_context(|| format!("Data address: {:?}", data.as_ptr()))?;
                let buffer = Arc::new(CudaBuffer { inner: device_data });
                Ok(CudaTensor {
                    buffer,
                    datum_type: tensor.datum_type(),
                    shape: tensor.shape().into(),
                    strides: tensor.strides().into(),
                    exotic_fact: None,
                })
            })
        }
    }

    pub fn uninitialized_dt(shape: &[usize], dt: DatumType) -> TractResult<Self> {
        crate::with_cuda_stream(|stream| unsafe {
            let device_data = stream.alloc(shape.iter().product::<usize>() * dt.size_of()).unwrap();
            let buffer = Arc::new(CudaBuffer { inner: device_data });
            Ok(CudaTensor {
                buffer,
                datum_type: dt,
                shape: shape.to_smallvec(),
                strides: natural_strides(shape),
                exotic_fact: None,
            })
        })
    }

    pub fn uninitialized_exotic(exotic_fact: Box<dyn ExoticFact>) -> TractResult<Self> {
        if let Some(bqf) = exotic_fact.downcast_ref::<BlockQuantFact>() {
            let shape = bqf.shape();
            let format = bqf.format.clone();
            let len = shape.iter().product::<usize>();
            ensure!(len % format.block_len() == 0);
            crate::with_cuda_stream(|stream| unsafe {
                let device_data = stream.alloc(len * format.block_bytes() / format.block_len())?;
                let buffer = Arc::new(CudaBuffer { inner: device_data });
                Ok(CudaTensor {
                    buffer,
                    datum_type: f32::datum_type(),
                    shape: tvec!(),
                    strides: tvec!(),
                    exotic_fact: Some(Box::new(bqf.clone())),
                })
            })
        } else if let Some(ggml_q81_fact) = exotic_fact.downcast_ref::<GgmlQuantQ81Fact>() {
            let mem_size = ggml_q81_fact.mem_size().as_i64().unwrap() as usize;

            crate::with_cuda_stream(|stream| unsafe {
                let device_data = stream.alloc(mem_size)?;
                let buffer = Arc::new(CudaBuffer { inner: device_data });
                Ok(CudaTensor {
                    buffer,
                    datum_type: f32::datum_type(),
                    shape: tvec!(),
                    strides: tvec!(),
                    exotic_fact: Some(Box::new(ggml_q81_fact.clone())),
                })
            })
        } else {
            bail!("Unsupported exotic type")
        }
    }
}

impl std::fmt::Debug for CudaTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaTensor")
            .field("datum_type", &self.datum_type)
            .field("shape", &self.shape)
            .field("block_quant_fact", &self.exotic_fact)
            .finish()
    }
}

impl OwnedDeviceTensor for CudaTensor {
    fn datum_type(&self) -> DatumType {
        self.datum_type
    }

    fn shape(&self) -> &[usize] {
        &self.shape
    }

    fn strides(&self) -> &[isize] {
        &self.strides
    }

    fn reshaped(&self, shape: TVec<usize>) -> TractResult<DeviceTensor> {
        if self.len() != shape.iter().product::<usize>() {
            bail!("Invalid reshape {:?} to {:?}", self.shape(), shape);
        }
        if shape.as_slice() != self.shape() {
            Ok(DeviceTensor::Owned(Box::new(CudaTensor {
                strides: Tensor::natural_strides(&shape),
                shape,
                ..self.clone()
            })))
        } else {
            Ok(DeviceTensor::Owned(Box::new(self.clone())))
        }
    }

    fn restrided(&self, strides: TVec<isize>) -> TractResult<DeviceTensor> {
        check_strides_validity(self.shape().into(), strides.clone())?;
        if strides.as_slice() != self.strides() {
            Ok(DeviceTensor::Owned(Box::new(CudaTensor { strides, ..self.clone() })))
        } else {
            Ok(DeviceTensor::Owned(Box::new(self.clone())))
        }
    }

    fn device_buffer(&self) -> &dyn tract_gpu::device::DeviceBuffer {
        self.buffer.as_ref()
    }

    fn to_host(&self) -> TractResult<Arc<Tensor>> {
        crate::with_cuda_stream(|stream| {
            let t: Tensor = if let Some(of) = &self.exotic_fact {
                let mut blob =
                    unsafe { Blob::new_for_size_and_align(self.buffer.len(), vector_size()) };
                stream.memcpy_dtoh(&self.buffer.inner, blob.as_bytes_mut())?;
                let bqf = if let Some(bqf) = of.downcast_ref::<BlockQuantFact>() {
                    (*bqf).clone()
                } else if let Some(ggml_q81) = of.downcast_ref::<GgmlQuantQ81Fact>() {
                    let out_shape = ggml_q81.concrete_out_shape()?;
                    BlockQuantFact::new(Box::new(Q8_1), out_shape.into())
                } else {
                    bail!("Unknown exotic fact")
                };
                let total_m = bqf.m();
                let k = bqf.k();
                BlockQuantStorage::new(bqf.format.clone(), total_m, k, Arc::new(blob))?
                    .into_tensor_with_shape(self.datum_type, &self.shape)
            } else {
                let mut tensor = unsafe { Tensor::uninitialized_dt(self.datum_type, &self.shape)? };
                stream.memcpy_dtoh(&self.buffer.inner, tensor.as_bytes_mut())?;
                tensor
            };

            Ok(Arc::new(t))
        })
    }

    fn exotic_fact(&self) -> Option<&dyn ExoticFact> {
        self.exotic_fact.as_deref()
    }

    fn get_bytes_slice(&self, offset: usize, len: usize) -> Vec<u8> {
        crate::with_cuda_stream(|stream| {
            Ok(stream.clone_dtoh(&self.buffer.slice(offset..offset + len)).unwrap())
        })
        .unwrap()
    }
}
