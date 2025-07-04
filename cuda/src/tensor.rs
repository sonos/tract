use std::ops::Deref;

use cudarc::driver::{CudaSlice, DevicePtr};
use tract_core::internal::*;
use tract_core::prelude::{DatumType, TVec};
use tract_gpu::device::DeviceBuffer;
use tract_gpu::tensor::{DeviceTensor, OwnedDeviceTensor};
use tract_gpu::utils::check_strides_validity;

use crate::context::CUDA_STREAM;

#[derive(Debug, Clone)]
pub struct CudaBuffer {
    pub inner: CudaSlice<u8>,
}

impl DeviceBuffer for CudaBuffer {
    fn info(&self) -> String {
        format!("Buffer: {:?}", self.inner)
    }

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

#[derive(Debug, Clone)]
pub struct CudaTensor {
    buffer: CudaBuffer,
    datum_type: DatumType,
    shape: TVec<usize>,
    strides: TVec<isize>,
}

impl CudaTensor {
    pub fn from_bytes(data: &[u8], dt: DatumType, shape: &[usize], strides: &[isize]) -> Self {
        CUDA_STREAM.with(|stream| {
            let device_data = stream.memcpy_stod(data).unwrap();
            let buffer = CudaBuffer { inner: device_data };
            CudaTensor { buffer, datum_type: dt, shape: shape.into(), strides: strides.into() }
        })
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
        println!("As arc tensor called on Cuda Tensor");
        None
    }

    fn device_buffer(&self) -> &dyn tract_gpu::device::DeviceBuffer {
        &self.buffer
    }

    fn to_host(&self) -> TractResult<Arc<Tensor>> {
        CUDA_STREAM.with(|stream| {
            let res = stream.memcpy_dtov(&self.buffer.inner.try_clone()?)?;
            let t: Tensor =
                unsafe { Tensor::from_raw_dt(self.datum_type, &self.shape, &res).unwrap() };
            Ok(Arc::new(t))
        })
    }

    fn view(&self) -> TensorView<'_> {
        todo!()
    }
}
