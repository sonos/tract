use std::ops::{Deref, DerefMut, RangeBounds};

use cudarc::driver::{CudaSlice, DevicePtr, PushKernelArg};
use tract_core::internal::tract_smallvec::ToSmallVec;
use tract_core::internal::*;
use tract_core::prelude::{DatumType, TVec};
use tract_core::tract_data::itertools::izip;
use tract_core::tract_linalg::block_quant::{BlockQuantFact, BlockQuantValue, Q8_1};
use tract_gpu::device::DeviceBuffer;
use tract_gpu::tensor::{DeviceTensor, OwnedDeviceTensor};
use tract_gpu::utils::{as_q40_tensor, check_strides_validity};

use crate::context::{cuda_context, TractCudaStream, CUDA_STREAM};
use crate::kernels::launch_args::LaunchArgsExt;
use crate::kernels::utils::cuda_launch_cfg_for_cpy;
use crate::kernels::{get_sliced_cuda_view, BroadcastKind, LibraryName};
use crate::ops::GgmlQuantQ81Fact;

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

impl DerefMut for CudaBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
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

    pub fn uninitialized_opaque(opaque_fact: Box<dyn OpaqueFact>) -> TractResult<Self> {
        if let Some(bqf) = opaque_fact.downcast_ref::<BlockQuantFact>() {
            let shape = bqf.shape();
            let format = bqf.format.clone();
            let len = shape.iter().product::<usize>();
            ensure!(len % format.block_len() == 0);
            CUDA_STREAM.with(|stream| unsafe {
                let device_data = stream.alloc(len * format.block_bytes() / format.block_len())?;
                let buffer = Arc::new(CudaBuffer { inner: device_data });
                Ok(CudaTensor {
                    buffer,
                    datum_type: DatumType::Opaque,
                    shape: tvec!(),
                    strides: tvec!(),
                    opaque_fact: Some(Box::new(bqf.clone())),
                })
            })
        } else if let Some(ggml_q81_fact) = opaque_fact.downcast_ref::<GgmlQuantQ81Fact>() {
            let mem_size = ggml_q81_fact.mem_size().as_i64().unwrap() as usize;

            CUDA_STREAM.with(|stream| unsafe {
                let device_data = stream.alloc(mem_size)?;
                let buffer = Arc::new(CudaBuffer { inner: device_data });
                Ok(CudaTensor {
                    buffer,
                    datum_type: DatumType::Opaque,
                    shape: tvec!(),
                    strides: tvec!(),
                    opaque_fact: Some(Box::new(ggml_q81_fact.clone())),
                })
            })
        } else {
            bail!("Unsupported opaque type")
        }
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

    fn device_buffer(&self) -> &dyn tract_gpu::device::DeviceBuffer {
        self.buffer.as_ref()
    }

    fn to_host(&self) -> TractResult<Arc<Tensor>> {
        CUDA_STREAM.with(|stream| {
            let t: Tensor = if let Some(of) = &self.opaque_fact {
                ensure!(self.shape.iter().product::<usize>() == 1, "Only support Scalar Opaque");
                let mut blob =
                    unsafe { Blob::new_for_size_and_align(self.buffer.len(), vector_size()) };
                stream.memcpy_dtoh(&self.buffer.inner, blob.as_bytes_mut())?;
                let bqf = if let Some(bqf) = of.downcast_ref::<BlockQuantFact>() {
                    (*bqf).clone()
                } else if let Some(ggml_q81) = of.downcast_ref::<GgmlQuantQ81Fact>() {
                    let out_shape = ggml_q81.concrete_out_shape()?;
                    BlockQuantFact::new(Box::new(Q8_1), out_shape.into())
                } else {
                    bail!("Unknown Opaque Fact")
                };
                let bqv = BlockQuantValue { fact: bqf, value: Arc::new(blob) };
                Opaque(Arc::new(bqv)).into()
            } else {
                let mut tensor = unsafe { Tensor::uninitialized_dt(self.datum_type, &self.shape)? };
                stream.memcpy_dtoh(&self.buffer.inner, tensor.as_bytes_mut())?;
                tensor
            };

            Ok(Arc::new(t.into_shape(&self.shape)?))
        })
    }

    fn opaque_fact(&self) -> Option<&dyn OpaqueFact> {
        self.opaque_fact.as_deref()
    }

    fn get_bytes_slice(&self, offset: usize, len: usize) -> Vec<u8> {
        CUDA_STREAM
            .with(|stream| stream.memcpy_dtov(&self.buffer.slice(offset..offset + len)).unwrap())
    }
}

pub fn device_tensor_assign_slice(
    stream: &TractCudaStream,
    dst: &DeviceTensor,
    dst_range: impl RangeBounds<usize>,
    src: &DeviceTensor,
    src_range: impl RangeBounds<usize>,
    axis: usize,
) -> TractResult<()> {
    ensure!(src.datum_type() == dst.datum_type());
    ensure!(src.datum_type().is_copy() && src.datum_type().is_number());
    ensure!(src.rank() == dst.rank() && axis < src.rank());
    let src_range = clip_range_bounds(src.shape()[axis], src_range);
    let dst_range = clip_range_bounds(dst.shape()[axis], dst_range);
    if src_range.is_empty() {
        return Ok(());
    }
    ensure!(dst_range.len() == src_range.len());
    ensure!(
        tract_itertools::izip!(dst.shape(), src.shape(), 0..).all(|(d, s, a)| a == axis || s == d)
    );

    let mut shape = src.shape().to_vec();
    shape[axis] = src_range.len();

    let mut dst_origin = tvec!(0usize; shape.len());
    dst_origin[axis] = dst_range.start;
    let src_origin = tvec!(0usize; shape.len());

    device_tensor_launch_copy(
        stream,
        &shape,
        dst,
        &dst_origin,
        dst.strides(),
        src,
        &src_origin,
        src.strides(),
    )
}

#[allow(clippy::too_many_arguments)]
pub fn device_tensor_launch_copy(
    stream: &TractCudaStream,
    zone_shape: &[usize],
    dst: &DeviceTensor,
    dst_origin: &[usize],
    dst_strides: &[isize],
    src: &DeviceTensor,
    src_origin: &[usize],
    src_strides: &[isize],
) -> TractResult<()> {
    ensure!(src.datum_type() == dst.datum_type());
    ensure!(src.datum_type().is_copy() && src.datum_type().is_number());
    ensure!(zone_shape.len() == dst.rank());
    ensure!(zone_shape.len() == dst_origin.len());
    ensure!(zone_shape.len() == dst_strides.len());
    ensure!(zone_shape.len() == src_origin.len());
    ensure!(zone_shape.len() == src_strides.len());
    let broadcast_kind = BroadcastKind::from_rank(dst.rank()).with_context(|| {
        format!(
            "Unsupported broadcast for assign slice: (in: {:?}, out: {:?})",
            src.shape(),
            dst.shape()
        )
    })?;
    ensure!(src.len() > 0);

    let tname = DeviceTensor::tname(src.datum_type())?;
    let broadcast_name = broadcast_kind.name();
    let kernel_name = format!("copy_{broadcast_name}_{tname}");
    let func = cuda_context().load_pipeline(LibraryName::Array, kernel_name)?;

    let dst_offset = izip!(dst_origin, dst_strides).map(|(a, b)| a * *b as usize).sum::<usize>()
        * dst.datum_type().size_of();
    let dst_len = dst.len() * dst.datum_type().size_of();
    let dst_view = get_sliced_cuda_view(dst, dst_offset, dst_len - dst_offset)?;

    let src_offset = izip!(src_origin, src_strides).map(|(a, b)| a * *b as usize).sum::<usize>()
        * src.datum_type().size_of();
    let src_len = src.len() * src.datum_type().size_of();
    let src_view = get_sliced_cuda_view(src, src_offset, src_len - src_offset)?;

    let mut launch_args = stream.launch_builder(&func);
    launch_args.arg(&src_view);
    launch_args.arg(&dst_view);
    launch_args.set_slice(src_strides);
    launch_args.set_slice(zone_shape);
    launch_args.set_slice(dst_strides);

    let cfg = cuda_launch_cfg_for_cpy(zone_shape);
    unsafe { launch_args.launch(cfg)? };
    Ok(())
}
