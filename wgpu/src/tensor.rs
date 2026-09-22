use std::fmt::Display;
use std::sync::Arc;

use tract_core::internal::*;
use tract_gpu::device::DeviceBuffer;
use tract_gpu::tensor::{DeviceTensor, OwnedDeviceTensor};
use tract_gpu::utils::check_strides_validity;

use crate::context::WgpuBuffer;
#[cfg(not(all(target_arch = "wasm32", feature = "jspi")))]
use crate::context::with_wgpu_queue;

#[derive(Clone)]
pub struct WgpuTensor {
    pub buffer: Arc<WgpuBuffer>,
    pub datum_type: DatumType,
    pub shape: TVec<usize>,
    pub strides: TVec<isize>,
    pub exotic_fact: Option<Box<dyn ExoticFact>>,
}

impl std::fmt::Debug for WgpuTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WgpuTensor")
            .field("datum_type", &self.datum_type)
            .field("shape", &self.shape)
            .finish()
    }
}

impl PartialEq for WgpuTensor {
    fn eq(&self, other: &Self) -> bool {
        std::ptr::eq(self.buffer.as_ref(), other.buffer.as_ref())
            && self.datum_type == other.datum_type
            && self.shape == other.shape
            && self.strides == other.strides
    }
}
impl Eq for WgpuTensor {}

impl OwnedDeviceTensor for WgpuTensor {
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
            Ok(DeviceTensor::Owned(Box::new(WgpuTensor {
                strides: Tensor::natural_strides(&shape),
                shape,
                buffer: Arc::clone(&self.buffer),
                datum_type: self.datum_type,
                exotic_fact: self.exotic_fact.clone(),
            })))
        } else {
            Ok(DeviceTensor::Owned(Box::new(self.clone())))
        }
    }

    fn restrided(&self, strides: TVec<isize>) -> TractResult<DeviceTensor> {
        check_strides_validity(self.shape().into(), strides.clone())?;
        if strides.as_slice() != self.strides() {
            Ok(DeviceTensor::Owned(Box::new(WgpuTensor {
                strides,
                buffer: Arc::clone(&self.buffer),
                datum_type: self.datum_type,
                shape: self.shape.clone(),
                exotic_fact: self.exotic_fact.clone(),
            })))
        } else {
            Ok(DeviceTensor::Owned(Box::new(self.clone())))
        }
    }

    fn device_buffer(&self) -> &dyn DeviceBuffer {
        self.buffer.as_ref()
    }

    fn to_host(&self) -> TractResult<Arc<Tensor>> {
        let (offset, len) = if let Some(of) = &self.exotic_fact {
            (0u64, of.mem_size().as_i64().context("Symbolic exotic tensor size")? as u64)
        } else {
            (0u64, (self.len() * self.datum_type.size_of()) as u64)
        };
        let bytes = download_owned_bytes(&self.buffer.inner, offset, len)?;
        let t = if let Some(of) = &self.exotic_fact {
            let bqf = of
                .downcast_ref::<tract_core::tract_linalg::block_quant::BlockQuantFact>()
                .context("Unknown exotic fact")?;
            let blob = tract_data::internal::Blob::from_bytes(&bytes)?;
            tract_core::tract_linalg::block_quant::BlockQuantStorage::new(
                bqf.format.clone(),
                bqf.m(),
                bqf.k(),
                Arc::new(blob),
            )?
            .into_tensor_with_shape(self.datum_type, &self.shape)
        } else {
            unsafe { Tensor::from_raw_dt(self.datum_type, &self.shape, &bytes)? }
        };
        Ok(Arc::new(t))
    }

    fn exotic_fact(&self) -> Option<&dyn ExoticFact> {
        self.exotic_fact.as_deref()
    }

    /// Only reached through tract-gpu's arena view, which needs a
    /// `DeviceSessionHandler` this backend never installs. The signature cannot
    /// report a failure, so were it reached, a device-side fetch that failed
    /// would abort instead: on web it cannot succeed at all without `jspi`.
    fn get_bytes_slice(&self, offset: usize, len: usize) -> Vec<u8> {
        download_owned_bytes(&self.buffer.inner, offset as u64, len as u64).unwrap()
    }
}

fn download_owned_bytes(buffer: &wgpu::Buffer, offset: u64, len: u64) -> TractResult<Vec<u8>> {
    #[cfg(all(target_arch = "wasm32", feature = "jspi"))]
    {
        crate::jspi::download_jspi(buffer, offset, len)
    }
    #[cfg(not(all(target_arch = "wasm32", feature = "jspi")))]
    {
        with_wgpu_queue(|q| q.download(buffer, offset, len))
    }
}

impl Display for WgpuTensor {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "WgpuTensor({:?} {:?})", self.datum_type, self.shape)
    }
}
