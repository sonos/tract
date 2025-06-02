use cust::util::SliceExt;
use tract_core::internal::*;
use cust::memory::{CopyDestination, DeviceBuffer};
use tract_core::prelude::{DatumType, TVec};
use tract_gpu::tensor::{DeviceTensor, OwnedDeviceTensor};
use tract_gpu::utils::check_strides_validity;

#[derive(Debug, Clone)]
pub struct CudaBuffer {
    pub inner: Arc<DeviceBuffer<u8>>
}

impl tract_gpu::device::DeviceBuffer for CudaBuffer {
    fn info(&self) -> String {
        format!("Buffer: {:?}", self.inner)
    }

    fn ptr(&self) -> *const std::ffi::c_void {
        self.inner.as_device_ptr().as_ptr() as _
    }
}

#[derive(Debug, Clone)]
pub struct CudaTensor {
    buffer: CudaBuffer,
    datum_type: DatumType,
    shape: TVec<usize>,
    strides: TVec<isize>
}

impl CudaTensor {
    pub fn from_bytes(data: &[u8], dt: DatumType, shape: &[usize], strides: &[isize]) -> Self {
        let buffer = CudaBuffer{ inner: Arc::new(data.as_dbuf().unwrap()) };
        CudaTensor { buffer, datum_type: dt, shape: shape.into(), strides: strides.into() }
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
            Ok(DeviceTensor::Owned(Box::new(
                CudaTensor {
                    strides: Tensor::natural_strides(&shape),
                    shape,
                    ..self.clone()
                }
            )))
        } else {
            Ok(DeviceTensor::Owned(Box::new(self.clone())))
        }
    }

    fn restrided(&self, strides: TVec<isize>) -> TractResult<DeviceTensor> {
        check_strides_validity(self.shape().into(), strides.clone())?;
        if strides.as_slice() != self.strides() {
            Ok(DeviceTensor::Owned(Box::new(
                CudaTensor {
                    strides: strides,
                    ..self.clone()
                }
            )))
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

    fn to_host(&self) -> Arc<Tensor> {
        let tensor = unsafe {
            let mut tensor = Tensor::uninitialized_dt(self.datum_type, &self.shape).unwrap();
            assert!(dbg!(tensor.as_bytes_mut().len()) == dbg!(self.buffer.inner.len()));

            self.buffer.inner.copy_to(&mut tensor.as_bytes_mut()).unwrap();
            tensor
        };
        Arc::new(tensor)
    }

    fn view(&self) -> TensorView {
        todo!()
    }
}