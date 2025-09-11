use std::ops::Deref;

use cudarc::driver::{CudaSlice, DevicePtr};
use tract_core::internal::tract_smallvec::ToSmallVec;
use tract_core::internal::*;
use tract_core::prelude::{DatumType, TVec};
use tract_core::tract_linalg::block_quant::{BlockQuantFact, BlockQuantValue};
use tract_gpu::device::DeviceBuffer;
use tract_gpu::tensor::{DeviceTensor, OwnedDeviceTensor};
use tract_gpu::utils::{as_q40_tensor, check_strides_validity};

use crate::context::CUDA_STREAM;

#[derive(Debug, Clone)]
pub struct CudaBuffer {
    pub inner: CudaSlice<u8>,
}

impl DeviceBuffer for CudaBuffer {
    fn ptr(&self) -> *const std::ffi::c_void {
        CUDA_STREAM.with(|stream| self.inner.device_ptr(stream).0 as _)
    }
}
impl Deref for CudaBuffer {
    type Target = CudaSlice<u8>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

#[derive(Clone)]
pub struct CudaTensor {
    buffer: Arc<CudaBuffer>,
    datum_type: DatumType,
    shape: TVec<usize>,
    strides: TVec<isize>,
    opaque_fact: Option<Box<dyn OpaqueFact>>,
}

impl CudaTensor {
    pub fn from_tensor(tensor: &Tensor) -> TractResult<Self> {
        let (data, bqf) = as_q40_tensor(tensor)
            .map(|bqv| (bqv.value.as_bytes(), Some(bqv.fact.clone().into())))
            .unwrap_or((tensor.as_bytes(), None));
        CUDA_STREAM.with(|stream| {
            let device_data = stream
                .memcpy_stod(data)
                .with_context(|| format!("Data address: {:?}", data.as_ptr()))?;
            let buffer = Arc::new(CudaBuffer { inner: device_data });
            Ok(CudaTensor {
                buffer,
                datum_type: tensor.datum_type(),
                shape: tensor.shape().into(),
                strides: tensor.strides().into(),
                opaque_fact: bqf,
            })
        })
    }

    pub fn uninitialized_dt(shape: &[usize], dt: DatumType) -> TractResult<Self> {
        CUDA_STREAM.with(|stream| unsafe {
            let device_data = stream.alloc(shape.iter().product::<usize>() * dt.size_of()).unwrap();
            let buffer = Arc::new(CudaBuffer { inner: device_data });
            Ok(CudaTensor {
                buffer,
                datum_type: dt,
                shape: shape.to_smallvec(),
                strides: natural_strides(shape),
                opaque_fact: None,
            })
        })
    }

    pub fn uninitialized_opaque(opaque_fact: &dyn OpaqueFact) -> TractResult<Self> {
        if let Some(bqf) = opaque_fact.downcast_ref::<BlockQuantFact>() {
            let shape = bqf.shape();
            let format = bqf.format.clone();
            let len = shape.iter().product::<usize>();
            ensure!(len % format.block_len() == 0);
            CUDA_STREAM.with(|stream| unsafe {
                let device_data = stream.alloc(len * format.block_bytes() / format.block_len())?;
                let buffer = Arc::new(CudaBuffer { inner: device_data });
                let bqf = BlockQuantFact::new(format, shape.to_smallvec());
                Ok(CudaTensor {
                    buffer,
                    datum_type: DatumType::Opaque,
                    shape: tvec!(),
                    strides: natural_strides(shape),
                    opaque_fact: Some(Box::new(bqf)),
                })
            })
        } else {
            bail!("Only BlockQuant Tensor allocation supported for now");
        }
    }

    pub fn opaque_fact(&self) -> Option<&dyn OpaqueFact> {
        self.opaque_fact.as_deref()
    }
}

impl std::fmt::Debug for CudaTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CudaTensor")
            .field("datum_type", &self.datum_type)
            .field("shape", &self.shape)
            .field("block_quant_fact", &self.opaque_fact)
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

    fn as_arc_tensor(&self) -> Option<&Arc<Tensor>> {
        log::warn!("As arc tensor called on Cuda Tensor!");
        None
    }

    fn device_buffer(&self) -> &dyn tract_gpu::device::DeviceBuffer {
        self.buffer.as_ref()
    }

    fn to_host(&self) -> TractResult<Arc<Tensor>> {
        CUDA_STREAM.with(|stream| {
            let t: Tensor = if let Some(bqf) = &self.opaque_fact {
                ensure!(self.shape.iter().product::<usize>() == 1, "Only support Scalar Opaque");
                let mut blob =
                    unsafe { Blob::new_for_size_and_align(self.buffer.len(), vector_size()) };
                stream.memcpy_dtoh(&self.buffer.inner, blob.as_bytes_mut())?;
                let bqv = BlockQuantValue {
                    fact: *bqf.clone().downcast::<BlockQuantFact>().unwrap(),
                    value: Arc::new(blob),
                };
                Opaque(Arc::new(bqv)).into()
            } else {
                let mut tensor = unsafe { Tensor::uninitialized_dt(self.datum_type, &self.shape)? };
                stream.memcpy_dtoh(&self.buffer.inner, tensor.as_bytes_mut())?;
                tensor
            };

            Ok(Arc::new(t.into_shape(&self.shape)?))
        })
    }

    fn view(&self) -> TensorView<'_> {
        todo!()
    }
}
