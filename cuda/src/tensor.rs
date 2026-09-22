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

#[cfg(test)]
mod tests {
    use std::ops::Range;

    use super::*;
    use tract_gpu::device::get_context;
    use tract_gpu::memory::{DeviceMemoryPool, DeviceResolvedMemSchema};
    use tract_gpu::ops::dyn_kv_cache::{GpuDynKVCache, GpuDynKVCacheState};
    use tract_gpu::tensor::{DeviceTensorExt, IntoDevice, LazyHostStorage};

    fn iota(shape: &[usize]) -> TractResult<Tensor> {
        let len = shape.iter().product::<usize>();
        Tensor::from_shape(shape, &(0..len).map(|i| i as f32).collect::<Vec<_>>())
    }

    /// Slice on device and check the result against the same slice done on host.
    fn check_slice(
        input: &Tensor,
        device: &DeviceTensor,
        axis: usize,
        range: Range<usize>,
    ) -> TractResult<()> {
        let sliced = device.clone().into_tensor().slice(axis, range.start, range.end)?;
        let device = sliced.to_device_tensor()?;
        assert!(matches!(device, DeviceTensor::Owned(_)));
        device
            .to_host()?
            .close_enough(&input.slice(axis, range.start, range.end)?, Approximation::Exact)
    }

    #[test]
    fn slice_owned_device_tensor_stays_on_device() -> TractResult<()> {
        crate::with_cuda_stream(|_| {
            let input = iota(&[4, 6])?;
            let device = input.clone().into_device()?;
            assert!(matches!(device, DeviceTensor::Owned(_)));
            check_slice(&input, &device, 0, 1..3)
        })
    }

    /// The inner axis: the copy is strided on the source, which is exactly what
    /// a view could not have represented.
    #[test]
    fn slice_owned_device_tensor_on_inner_axis() -> TractResult<()> {
        crate::with_cuda_stream(|_| {
            let input = iota(&[4, 6])?;
            let device = input.clone().into_device()?;
            check_slice(&input, &device, 1, 1..4)
        })
    }

    /// What the application actually holds: an output that crossed the boundary
    /// still on device. The slice must stay there, and stay lazy, or the next
    /// run pays a transfer both ways.
    #[test]
    fn slice_of_a_lazy_host_tensor_stays_on_device() -> TractResult<()> {
        crate::with_cuda_stream(|_| {
            let input = iota(&[4, 6])?;
            let lazy = LazyHostStorage::new(input.clone().into_device()?)?.into_tensor();
            let sliced = lazy.slice(0, 1, 3)?;
            let storage = sliced
                .storage_as::<LazyHostStorage>()
                .context("slice of a lazy host tensor came back on host")?;
            assert!(!storage.is_materialized());
            assert!(storage.device().is_some());
            sliced.close_enough(&input.slice(0, 1, 3)?, Approximation::Exact)
        })
    }

    /// Once the bytes are back, the host copy is the cheap path and the device
    /// is not asked again.
    #[test]
    fn slice_of_a_materialized_lazy_tensor_stays_on_host() -> TractResult<()> {
        crate::with_cuda_stream(|_| {
            let input = iota(&[4, 6])?;
            let lazy = LazyHostStorage::new(input.clone().into_device()?)?.into_tensor();
            lazy.try_as_plain_ram()?;
            let sliced = lazy.slice(0, 1, 3)?;
            assert!(sliced.storage_as::<LazyHostStorage>().is_none());
            sliced.close_enough(&input.slice(0, 1, 3)?, Approximation::Exact)
        })
    }

    /// An arena view sits at a non-zero byte offset in a buffer it does not own:
    /// the slice must read from the view, and own its own buffer afterwards.
    #[test]
    fn slice_arena_view_owns_its_result() -> TractResult<()> {
        crate::with_cuda_stream(|_| {
            let input = iota(&[4, 6])?;
            let pool = DeviceMemoryPool::from_schema(DeviceResolvedMemSchema {
                offsets_by_node: vec![Some(tvec!(tvec!(256)))],
                memory_size: 4096,
            })?;
            let view = pool.tensor_for_node(0, f32::datum_type(), &[4, 6])?;
            assert!(matches!(view, DeviceTensor::View(_)));
            let ctx = get_context()?;
            ctx.flat_copy(&input.clone().into_device()?, 0, &view, 0, 4 * 6 * 4)?;
            ctx.synchronize()?;
            check_slice(&input, &view, 0, 2..4)
        })
    }

    fn kv_op() -> GpuDynKVCache {
        GpuDynKVCache {
            name: "kv".to_string(),
            axis: 2,
            past_sequence_fact: f32::fact([1, 2, 0, 3]),
            input_sequence_fact: f32::fact([1, 2, 1, 3]),
            window_output: false,
        }
    }

    fn kv_state() -> GpuDynKVCacheState {
        GpuDynKVCacheState::new("kv".to_string(), 2, f32::fact([1, 2, 0, 3]))
    }

    /// Token `t` of a `[1, 2, S, 3]` cache, every value `t`, so the cache at
    /// length `n` is `[[0..n], [0..n]]` whatever order it was built in.
    fn kv_token(t: usize) -> TractResult<Tensor> {
        Tensor::from_shape(&[1, 2, 1, 3], &[t as f32; 6])
    }

    fn kv_expected(tokens: &[usize]) -> TractResult<Tensor> {
        let n = tokens.len();
        let mut data = vec![0f32; 2 * n * 3];
        for h in 0..2 {
            for (s, t) in tokens.iter().enumerate() {
                for d in 0..3 {
                    data[h * n * 3 + s * 3 + d] = *t as f32;
                }
            }
        }
        Tensor::from_shape(&[1, 2, n, 3], &data)
    }

    fn kv_push(
        state: &mut GpuDynKVCacheState,
        op: &GpuDynKVCache,
        t: usize,
    ) -> TractResult<Arc<Tensor>> {
        let token = kv_token(t)?.into_device()?.into_tensor().into_tvalue();
        let out = state.eval(&EvalContext::out_of_plan(), op, tvec!(token))?;
        out[0].to_device_tensor()?.to_host()
    }

    /// One token per turn, the decode shape. The cache reads as everything
    /// pushed, and re-seats its buffer a logarithmic number of times rather than
    /// once a turn -- which is the whole point of holding spare capacity.
    #[test]
    fn kv_cache_grows_geometrically() -> TractResult<()> {
        crate::with_cuda_stream(|_| {
            let (op, mut state) = (kv_op(), kv_state());
            for t in 0..64 {
                let out = kv_push(&mut state, &op, t)?;
                out.close_enough(
                    &kv_expected(&(0..=t).collect::<Vec<_>>())?,
                    Approximation::Exact,
                )?;
            }
            assert!(
                state.reallocs() <= 8,
                "64 single-token appends re-seated the buffer {} times",
                state.reallocs()
            );
            assert!(state.capacity() >= 64);
            Ok(())
        })
    }

    /// Rolling back to a shared prefix is a shorter read of the same buffer, and
    /// the decode goes on appending over the tail it dropped.
    #[test]
    fn kv_cache_truncates_and_grows_again() -> TractResult<()> {
        crate::with_cuda_stream(|_| {
            let (op, mut state) = (kv_op(), kv_state());
            for t in 0..10 {
                kv_push(&mut state, &op, t)?;
            }
            let (capacity, reallocs) = (state.capacity(), state.reallocs());
            state.truncate(4)?;
            assert_eq!(state.capacity(), capacity, "truncation moved the buffer");
            assert_eq!(state.reallocs(), reallocs);

            let out = kv_push(&mut state, &op, 100)?;
            out.close_enough(&kv_expected(&[0, 1, 2, 3, 100])?, Approximation::Exact)
        })
    }

    /// A checkpoint carries the live prefix and nothing of the spare tail.
    #[test]
    fn kv_cache_checkpoint_carries_the_live_prefix() -> TractResult<()> {
        crate::with_cuda_stream(|_| {
            let (op, mut state) = (kv_op(), kv_state());
            for t in 0..5 {
                kv_push(&mut state, &op, t)?;
            }
            assert!(state.capacity() > 5, "this test wants a buffer with a spare tail");
            let mut saved = vec![];
            state.save_to(&mut saved)?;
            saved[0].close_enough(&kv_expected(&[0, 1, 2, 3, 4])?, Approximation::Exact)
        })
    }

    /// A window with spare capacity is not its own bytes in order: reading it
    /// back has to walk it, not take a flat run from the offset.
    #[test]
    fn a_strided_window_reads_back_whole() -> TractResult<()> {
        crate::with_cuda_stream(|_| {
            let input = iota(&[2, 3, 8, 4])?;
            let device = input.clone().into_device()?;
            let window = device.prefix_window(2, 5)?;
            assert!(matches!(window, DeviceTensor::View(_)));
            assert_eq!(window.shape(), &[2, 3, 5, 4]);
            window.to_host()?.close_enough(&input.slice(2, 0, 5)?, Approximation::Exact)
        })
    }
}
