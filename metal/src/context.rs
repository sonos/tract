use crate::command_buffer::TCommandBuffer;
use crate::func_constants::ConstantValues;
use crate::kernels::{LibraryContent, LibraryName};
use crate::tensor::{MValue, MetalTensor};

use metal::NSUInteger;
use tract_core::tract_linalg::block_quant::{BlockQuantFact, BlockQuantStorage};
use tract_gpu::device::{DeviceBuffer, DeviceContext};
use tract_gpu::tensor::{DeviceTensor, OwnedDeviceTensor};
use tract_gpu::utils::as_q40_tensor;

use std::alloc::Layout;
use std::cell::RefCell;
use std::ffi::c_void;
use std::ops::{Deref, DerefMut};
use std::path::Path;
use std::sync::Mutex;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock, RwLock};

use anyhow::{Context, anyhow};
use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, Function,
    FunctionConstantValues, Library, MTLResourceOptions,
};
use std::collections::HashMap;
use std::collections::VecDeque;
use tract_core::internal::*;

thread_local! {
    static METAL_STREAM: RefCell<Option<MetalStream>> = const { RefCell::new(None) };
}

const MAX_POOLED_PER_KEY: usize = 16;
const MAX_POOLED_BYTES: usize = 512 * 1024 * 1024;
const MAX_IN_FLIGHT_COMMAND_BUFFERS: usize = 8;

pub fn with_metal_stream<R>(f: impl FnOnce(&MetalStream) -> TractResult<R>) -> TractResult<R> {
    metal_context(); // ensures context is initialized
    METAL_STREAM.with(|cell| {
        let needs_init = cell.borrow().is_none();
        if needs_init {
            let stream = MetalStream::new();
            *cell.borrow_mut() = Some(stream);
        }
        let borrow = cell.borrow();
        f(borrow.as_ref().unwrap())
    })
}

pub fn metal_context() -> MetalContext {
    static INSTANCE: OnceLock<MetalContext> = OnceLock::new();
    INSTANCE
        .get_or_init(|| {
            let ctxt = MetalContext::new().expect("Could not create Metal context");
            tract_gpu::device::set_context(Box::new(ctxt.clone()))
                .expect("Could not set Metal context");
            ctxt
        })
        .clone()
}

#[derive(Debug, Clone)]
pub struct MetalContext {
    device: Device,
    cache_libraries: Arc<RwLock<HashMap<LibraryName, Library>>>,
    #[allow(clippy::type_complexity)]
    cache_pipelines:
        Arc<RwLock<HashMap<(LibraryName, String, Option<ConstantValues>), ComputePipelineState>>>,
    /// Recycled (host allocation, MTLBuffer) pairs keyed by exact
    /// (dtype, shape). Creating and destroying Metal buffers goes through an
    /// IOGPU kernel trap each way (~17% of decode CPU time before pooling);
    /// transformer decode reallocates the same transient shapes every token,
    /// so an exact-shape pool absorbs nearly all of it.
    #[allow(clippy::type_complexity)]
    buffer_pool: Arc<Mutex<HashMap<(DatumType, TVec<usize>), Vec<(Arc<Tensor>, Buffer, u64)>>>>,
    pooled_bytes: Arc<std::sync::atomic::AtomicUsize>,
    /// Monotonic insertion stamp driving oldest-first pool eviction.
    pool_stamp: Arc<std::sync::atomic::AtomicU64>,
}

impl MetalContext {
    fn pool_take(&self, dt: DatumType, shape: &[usize]) -> Option<(Arc<Tensor>, Buffer)> {
        let mut pool = self.buffer_pool.lock().ok()?;
        let entry = pool.get_mut(&(dt, TVec::from_slice(shape)));
        let hit = entry?.pop()?;
        self.pooled_bytes
            .fetch_sub(hit.0.len() * dt.size_of(), std::sync::atomic::Ordering::Relaxed);
        Some(hit.0.clone()).map(|host| (host, hit.1))
    }

    fn pool_put(&self, host: Arc<Tensor>, buffer: Buffer) {
        let dt = host.datum_type();
        if !DeviceTensor::is_supported_dt(dt) {
            return;
        }
        let bytes = host.len() * dt.size_of();
        if bytes > MAX_POOLED_BYTES {
            return;
        }
        let Ok(mut pool) = self.buffer_pool.lock() else { return };
        // Evict oldest entries (globally, by insertion stamp) until the new
        // buffer fits the budget: recent shapes stay hot, stale shapes from
        // an earlier context length get released for real.
        while self.pooled_bytes.load(std::sync::atomic::Ordering::Relaxed) + bytes
            > MAX_POOLED_BYTES
        {
            let oldest_key = pool
                .iter()
                .filter(|(_, v)| !v.is_empty())
                .min_by_key(|(_, v)| v.first().map(|e| e.2).unwrap_or(u64::MAX))
                .map(|(k, _)| k.clone());
            let Some(key) = oldest_key else { break };
            let Some(entry) = pool.get_mut(&key) else { break };
            let (evicted_host, _, _) = entry.remove(0);
            let evicted_bytes = evicted_host.len() * evicted_host.datum_type().size_of();
            self.pooled_bytes.fetch_sub(evicted_bytes, std::sync::atomic::Ordering::Relaxed);
            if entry.is_empty() {
                pool.remove(&key);
            }
        }
        let entry = pool.entry((dt, TVec::from_slice(host.shape()))).or_default();
        if entry.len() >= MAX_POOLED_PER_KEY {
            return;
        }
        let stamp = self.pool_stamp.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.pooled_bytes.fetch_add(bytes, std::sync::atomic::Ordering::Relaxed);
        entry.push((host, buffer, stamp));
    }

    pub fn new() -> TractResult<Self> {
        let device = Device::system_default()
            .with_context(|| "Could not find system default Metal device")?;

        let ctxt = Self {
            device,
            cache_libraries: Arc::new(RwLock::new(HashMap::new())),
            cache_pipelines: Arc::new(RwLock::new(HashMap::new())),
            buffer_pool: Arc::new(Mutex::new(HashMap::new())),
            pooled_bytes: Arc::new(std::sync::atomic::AtomicUsize::new(0)),
            pool_stamp: Arc::new(std::sync::atomic::AtomicU64::new(0)),
        };
        ctxt.preload_pipelines()?;
        Ok(ctxt)
    }

    pub fn preload_pipelines(&self) -> TractResult<()> {
        for ew_func in crate::kernels::element_wise::all_functions() {
            let _ = self.load_pipeline(LibraryName::ElementWiseOps, &ew_func);
        }
        for bin_func in crate::kernels::bin_ops::all_functions() {
            let _ = self.load_pipeline(LibraryName::BinOps, &bin_func);
        }
        for func in crate::kernels::array::all_functions() {
            let _ = self.load_pipeline(LibraryName::ArrayOps, &func);
        }
        for func in crate::kernels::nn::all_functions() {
            let _ = self.load_pipeline(LibraryName::NNOps, &func);
        }
        Ok(())
    }

    pub fn load_library(&self, name: LibraryName) -> TractResult<Library> {
        {
            let cache_libraries = self.cache_libraries.read().map_err(|e| anyhow!("{:?}", e))?;
            if let Some(library) = cache_libraries.get(&name) {
                return Ok(library.clone());
            }
        }
        let mut cache_libraries = self.cache_libraries.write().map_err(|e| anyhow!("{:?}", e))?;
        let library = match name.content() {
            LibraryContent::Data(lib_data) => self
                .device
                .new_library_with_data(lib_data)
                .map_err(|e| anyhow!("{}", e))
                .with_context(|| {
                    format!("Error while loading Metal library from data: {:?}", name)
                })?,
            LibraryContent::Source(lib_source) => self
                .device
                .new_library_with_source(lib_source, &CompileOptions::new())
                .map_err(|e| anyhow!("{}", e))
                .with_context(|| {
                    format!("Error while loading Metal library from source: {:?}", name)
                })?,
        };
        cache_libraries.insert(name, library.clone());
        Ok(library)
    }

    pub fn load_function(
        &self,
        library_name: LibraryName,
        func_name: &str,
        constants: Option<FunctionConstantValues>,
    ) -> TractResult<Function> {
        let func = self
            .load_library(library_name)?
            .get_function(func_name, constants)
            .map_err(|e| anyhow!("{}", e))
            .with_context(|| {
                format!(
                    "Error while loading function {func_name} from library: {:?} with constants",
                    library_name
                )
            })?;
        Ok(func)
    }

    pub(crate) fn load_pipeline_with_constants(
        &self,
        library_name: LibraryName,
        func_name: &str,
        constants: Option<ConstantValues>,
    ) -> TractResult<ComputePipelineState> {
        let key = (library_name, func_name.to_string(), constants);
        {
            let cache_pipelines = self.cache_pipelines.read().map_err(|e| anyhow!("{:?}", e))?;
            if let Some(pipeline) = cache_pipelines.get(&key) {
                return Ok(pipeline.clone());
            }
        }
        let mut cache_pipelines = self.cache_pipelines.write().map_err(|e| anyhow!("{:?}", e))?;

        let (library_name, func_name, constants) = key;
        let func = self.load_function(
            library_name,
            &func_name,
            constants.as_ref().map(|c| c.function_constant_values()),
        )?;
        let pipeline = self.device
            .new_compute_pipeline_state_with_function(&func)
            .map_err(|e| anyhow!("{}", e))
            .with_context(|| format!("Error while creating compute pipeline for function {func_name} from source: {:?}", library_name))?;
        cache_pipelines.insert((library_name, func_name.to_string(), constants), pipeline.clone());
        Ok(pipeline)
    }

    pub fn load_pipeline(
        &self,
        library_name: LibraryName,
        func_name: &str,
    ) -> TractResult<ComputePipelineState> {
        self.load_pipeline_with_constants(library_name, func_name, None)
    }
}

impl DeviceContext for MetalContext {
    fn synchronize(&self) -> TractResult<()> {
        with_metal_stream(|stream| stream.wait_until_completed())
    }

    fn tensor_to_device(&self, tensor: TValue) -> TractResult<Box<dyn OwnedDeviceTensor>> {
        let view = tensor.view();
        ensure!(
            DeviceTensor::is_supported_dt(view.datum_type()),
            "Tensor of {:?} is not copied. No device buffer can be allocated for it.",
            view.datum_type(),
        );
        let bqs = as_q40_tensor(view.tensor);

        let (data_bytes, bqf) = if let Some(bqs) = bqs {
            (
                bqs.value().as_bytes(),
                Some(Box::new(BlockQuantFact::new(
                    tract_core::dyn_clone::clone_box(bqs.format()),
                    tensor.view().tensor.shape().into(),
                )) as Box<dyn ExoticFact>),
            )
        } else {
            (view.tensor.as_bytes(), None)
        };

        // Handle empty data
        static ZERO: [u8; 1] = [0];
        let data = if data_bytes.is_empty() { &ZERO } else { data_bytes };

        let size = core::mem::size_of_val(data) as NSUInteger;
        let buffer = self.device.new_buffer_with_bytes_no_copy(
            data.as_ptr() as *const core::ffi::c_void,
            size,
            MTLResourceOptions::StorageModeShared,
            None,
        );
        let host = tensor.into_arc_tensor();
        let device_buffer = MetalBuffer {
            inner: buffer.clone(),
            pool: if bqf.is_none() {
                Some(Arc::new(BufferPoolGuard { host: host.clone(), buffer }))
            } else {
                None
            },
        };

        Ok(Box::new(MetalTensor { inner: MValue::Natural(host), device_buffer, exotic_fact: bqf }))
    }

    fn uninitialized_device_tensor(
        &self,
        shape: &[usize],
        dt: DatumType,
    ) -> TractResult<Box<dyn OwnedDeviceTensor>> {
        if let Some((host, buffer)) = self.pool_take(dt, shape) {
            let device_buffer = MetalBuffer {
                inner: buffer.clone(),
                pool: Some(Arc::new(BufferPoolGuard { host: host.clone(), buffer })),
            };
            return Ok(Box::new(MetalTensor {
                inner: MValue::Natural(host),
                device_buffer,
                exotic_fact: None,
            }));
        }
        let tensor = unsafe {
            Tensor::uninitialized_dt(dt, shape).with_context(|| {
                format!("Error while allocating a {dt:?} tensor of shape {shape:?}")
            })?
        };
        self.tensor_to_device(tensor.into())
    }

    fn uninitialized_device_exotic_tensor(
        &self,
        exotic_fact: Box<dyn ExoticFact>,
    ) -> TractResult<Box<dyn OwnedDeviceTensor>> {
        if let Some(bqf) = exotic_fact.downcast_ref::<BlockQuantFact>() {
            let blocks = bqf.shape().iter().product::<usize>() / bqf.format.block_len();
            let blob = unsafe {
                Blob::for_layout(
                    Layout::from_size_align(blocks * bqf.format.block_bytes(), vector_size())
                        .unwrap(),
                )
            };
            let tensor =
                BlockQuantStorage::new(bqf.format.clone(), bqf.m(), bqf.k(), Arc::new(blob))?
                    .into_tensor_with_shape(f32::datum_type(), bqf.shape());
            self.tensor_to_device(tensor.into())
        } else {
            bail!("Only BlockQuant Tensor allocation supported for now")
        }
    }

    fn copy_nd(
        &self,
        input: &DeviceTensor,
        input_offset: usize,
        input_strides: &[isize],
        output: &DeviceTensor,
        output_offset: usize,
        output_shape: &[usize],
        output_strides: &[isize],
    ) -> TractResult<()> {
        crate::kernels::array::metal_copy_nd_dispatch(
            input,
            input_offset,
            input_strides,
            output,
            output_offset,
            output_shape,
            output_strides,
        )
    }
}

/// One committed-but-unawaited command buffer with everything that must
/// stay alive (retained tensors) or stay out of the recycling pool
/// (deferred pool pairs) until it completes.
#[derive(Debug)]
struct InFlightBuffer {
    buffer: TCommandBuffer,
    retained: Vec<DeviceTensor>,
    /// (host allocation, MTLBuffer) pairs whose last host-side owner dropped
    /// while GPU work was pending: recycling them into the pool is deferred
    /// to this buffer's completion, because a buffer committed earlier (or
    /// this one) may still read or write them. Recycling at host-drop time
    /// raced exactly there: the pool handed the pair to a new tensor while
    /// an in-flight command buffer still referenced it (byte-level run-to-run
    /// nondeterminism at low in-flight depth, qwen3.5-35B, 2026-08-12).
    recycle: Vec<(Arc<Tensor>, Buffer)>,
}

#[derive(Debug)]
pub struct MetalStream {
    context: MetalContext,
    command_queue: CommandQueue,
    command_buffer: RefCell<Option<TCommandBuffer>>,
    /// Buffers committed by `commit_current` and not yet awaited, oldest
    /// first, each with the tensors that must stay alive until it completes.
    /// The queue is FIFO, so waiting on the newest implies all have completed.
    committed_command_buffers: RefCell<VecDeque<InFlightBuffer>>,
    /// Pool pairs dropped since the last commit (see
    /// [`InFlightBuffer::recycle`]): they move onto the next committed
    /// buffer, or recycle directly at the next blocking wait.
    recycle_stash: RefCell<Vec<(Arc<Tensor>, Buffer)>>,
    command_buffer_id: AtomicUsize,
    retained_tensors: RefCell<Vec<DeviceTensor>>,
}

impl Default for MetalStream {
    fn default() -> Self {
        Self::new()
    }
}

impl MetalStream {
    pub fn new() -> Self {
        let context = metal_context();
        let command_queue = context.device.new_command_queue();
        Self {
            context,
            command_queue,
            command_buffer: RefCell::new(None),
            committed_command_buffers: RefCell::new(VecDeque::new()),
            recycle_stash: RefCell::new(Vec::new()),
            command_buffer_id: AtomicUsize::new(0),
            retained_tensors: RefCell::new(vec![]),
        }
    }

    pub fn load_library(&self, name: LibraryName) -> TractResult<Library> {
        self.context.load_library(name)
    }

    pub fn load_pipeline(
        &self,
        library_name: LibraryName,
        func_name: &str,
    ) -> TractResult<ComputePipelineState> {
        self.context.load_pipeline(library_name, func_name)
    }

    pub(crate) fn load_pipeline_with_constants(
        &self,
        library_name: LibraryName,
        func_name: &str,
        constants: Option<ConstantValues>,
    ) -> TractResult<ComputePipelineState> {
        self.context.load_pipeline_with_constants(library_name, func_name, constants)
    }

    pub fn retain_tensor(&self, tensor: &DeviceTensor) {
        self.retained_tensors.borrow_mut().push(tensor.clone());
    }

    pub fn command_buffer(&self) -> TCommandBuffer {
        self.command_buffer
            .borrow_mut()
            .get_or_insert_with(|| {
                TCommandBuffer::new(self.command_queue.new_command_buffer().to_owned())
            })
            .to_owned()
    }

    /// Commit the current command buffer without blocking the CPU on its
    /// completion. The next `command_buffer()` call opens a fresh one; the
    /// queue guarantees the committed buffer executes before it. Tensors
    /// retained so far move into the in-flight entry and are released once
    /// that buffer completes.
    pub fn commit_current(&self) -> TractResult<()> {
        let Some(command_buffer) = self.command_buffer.borrow_mut().take() else {
            return Ok(());
        };
        match command_buffer.status() {
            metal::MTLCommandBufferStatus::Committed
            | metal::MTLCommandBufferStatus::Scheduled
            | metal::MTLCommandBufferStatus::Completed => {
                anyhow::bail!("Current Metal command buffer is already committed.")
            }
            _ => {}
        }
        command_buffer.encoder().end_encoding();
        command_buffer.commit();
        let retained = std::mem::take(&mut *self.retained_tensors.borrow_mut());
        // Pool pairs dropped while this buffer was open may still be
        // referenced by it (or by an earlier one, which the queue's FIFO
        // order covers): recycle them only once this buffer has completed.
        let recycle = std::mem::take(&mut *self.recycle_stash.borrow_mut());
        let mut committed = self.committed_command_buffers.borrow_mut();
        committed.push_back(InFlightBuffer { buffer: command_buffer, retained, recycle });
        // Retire every already-completed buffer at the front (no wait): their
        // recycle pairs go back to the pool at true completion time instead
        // of pop time, keeping the pool hot within a step at deep in-flight
        // settings. Then enforce the in-flight cap with blocking waits.
        loop {
            let done = committed
                .front()
                .is_some_and(|e| e.buffer.status() == metal::MTLCommandBufferStatus::Completed);
            if !done && committed.len() <= MAX_IN_FLIGHT_COMMAND_BUFFERS {
                break;
            }
            let oldest = committed.pop_front().unwrap();
            oldest.buffer.wait_until_completed();
            for (host, buffer) in oldest.recycle {
                self.context.pool_put(host, buffer);
            }
            // Dropping the retained tensors can fire BufferPoolGuard drops,
            // which re-enter through the recycle stash (the committed deque
            // is mutably borrowed here, so the guard defers): those pairs
            // ride to the next commit or the next blocking wait.
            drop(oldest.retained);
        }
        Ok(())
    }

    /// Recycle everything that was deferred to command-buffer completion.
    /// Only call with the device fully quiesced (all buffers waited): pairs
    /// go straight into the pool. Loops because dropping retained tensors
    /// can push new pairs into the stash.
    fn flush_recycle_stash(&self) {
        loop {
            let pairs = std::mem::take(&mut *self.recycle_stash.borrow_mut());
            if pairs.is_empty() {
                return;
            }
            for (host, buffer) in pairs {
                self.context.pool_put(host, buffer);
            }
        }
    }

    pub fn wait_until_completed(&self) -> TractResult<()> {
        let Some(command_buffer) = self.command_buffer.borrow().to_owned() else {
            // No open buffer, but commit_current buffers may still be in
            // flight: the host must not read results before they land. FIFO:
            // waiting on the newest is enough.
            let drained: Vec<_> = self.committed_command_buffers.borrow_mut().drain(..).collect();
            if let Some(newest) = drained.last() {
                newest.buffer.wait_until_completed();
            }
            for entry in drained {
                for (host, buffer) in entry.recycle {
                    self.context.pool_put(host, buffer);
                }
                drop(entry.retained);
            }
            self.retained_tensors.borrow_mut().clear();
            self.flush_recycle_stash();
            return Ok(());
        };

        command_buffer.encoder().end_encoding();

        match command_buffer.status() {
            metal::MTLCommandBufferStatus::Committed
            | metal::MTLCommandBufferStatus::Scheduled
            | metal::MTLCommandBufferStatus::Completed => {
                anyhow::bail!("Current Metal command buffer is already committed.")
            }
            _ => {}
        }
        let command_buffer_id = self.command_buffer_id.load(Ordering::Relaxed);
        command_buffer.commit();
        log::trace!("Command buffer {:?} commit", command_buffer_id);
        command_buffer.wait_until_completed();
        log::trace!("Command buffer {:?} has completed (Blocking call)", command_buffer_id);

        // The queue is FIFO: the buffer above completing implies every buffer
        // committed earlier by commit_current has completed too.
        let drained: Vec<_> = self.committed_command_buffers.borrow_mut().drain(..).collect();
        for entry in drained {
            for (host, buffer) in entry.recycle {
                self.context.pool_put(host, buffer);
            }
            drop(entry.retained);
        }

        // Clear local retained values used by the command buffer
        self.retained_tensors.borrow_mut().clear();
        self.flush_recycle_stash();

        *self.command_buffer.borrow_mut() = None;
        Ok(())
    }

    pub fn capture_trace<P, F>(&self, path: P, compute: F) -> TractResult<()>
    where
        P: AsRef<Path>,
        F: FnOnce(&Self) -> TractResult<()>,
    {
        self.wait_until_completed()?;

        anyhow::ensure!(path.as_ref().is_absolute());

        let capture = metal::CaptureManager::shared();
        let descriptor = metal::CaptureDescriptor::new();
        descriptor.set_destination(metal::MTLCaptureDestination::GpuTraceDocument);
        descriptor.set_capture_device(&self.context.device);
        descriptor.set_output_url(path);

        capture.start_capture(&descriptor).map_err(|e| anyhow!("Error Metal Capture: {:?}", e))?;

        (compute)(self)?;

        self.wait_until_completed()?;
        capture.stop_capture();
        Ok(())
    }
}

impl Drop for MetalStream {
    fn drop(&mut self) {
        let drained: Vec<_> = self.committed_command_buffers.borrow_mut().drain(..).collect();
        if let Some(newest) = drained.last() {
            newest.buffer.wait_until_completed();
        }
        drop(drained);
        if let Some(command_buffer) = self.command_buffer.borrow_mut().take() {
            match command_buffer.status() {
                metal::MTLCommandBufferStatus::Committed
                | metal::MTLCommandBufferStatus::Scheduled
                | metal::MTLCommandBufferStatus::Completed => {
                    panic!("Current Metal command buffer is already committed.")
                }
                _ => {}
            }

            command_buffer.encoder().end_encoding();
            command_buffer.commit();
            command_buffer.wait_until_completed();
        }
        // Everything is waited: deferred pairs are simply released (no
        // recycling into the process pool from a dying stream).
        self.recycle_stash.borrow_mut().clear();
    }
}

/// Returns its (host allocation, MTLBuffer) pair to the context pool when
/// the last owner drops it, provided nothing else still references the host
/// tensor (`to_host` on unified memory hands out the same allocation, so an
/// escaped `Arc<Tensor>` blocks recycling and the pair is simply released).
#[derive(Debug)]
pub(crate) struct BufferPoolGuard {
    pub(crate) host: Arc<Tensor>,
    pub(crate) buffer: Buffer,
}

impl Drop for BufferPoolGuard {
    fn drop(&mut self) {
        if Arc::strong_count(&self.host) != 1 {
            return;
        }
        let mut pair = Some((self.host.clone(), self.buffer.clone()));
        // Defer recycling to command-buffer completion when this thread's
        // stream has (or may have, when a RefCell is busy because this drop
        // runs inside commit_current/wait) GPU work in flight that could
        // still reference the pair: recycling at host-drop time handed the
        // buffer to a new tensor while an in-flight command buffer still
        // read or wrote it. A thread without a stream never dispatched
        // anything referencing the pair, so it recycles immediately (the
        // historical behavior).
        let _ = METAL_STREAM.try_with(|cell| {
            let Ok(stream_ref) = cell.try_borrow() else { return };
            let Some(stream) = stream_ref.as_ref() else { return };
            let busy = stream.command_buffer.try_borrow().map_or(true, |cb| cb.is_some())
                || stream
                    .committed_command_buffers
                    .try_borrow()
                    .map_or(true, |committed| !committed.is_empty());
            if busy {
                stream.recycle_stash.borrow_mut().push(pair.take().unwrap());
            }
        });
        if let Some((host, buffer)) = pair {
            metal_context().pool_put(host, buffer);
        }
    }
}

#[derive(Debug, Clone)]
pub struct MetalBuffer {
    pub inner: Buffer,
    /// Shared across clones of the owning tensor; the last drop recycles.
    pub(crate) pool: Option<Arc<BufferPoolGuard>>,
}

impl PartialEq for MetalBuffer {
    fn eq(&self, other: &Self) -> bool {
        self.inner.length() == other.inner.length() && self.inner.length() == other.inner.length()
    }
}
impl Eq for MetalBuffer {}

impl Deref for MetalBuffer {
    type Target = Buffer;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl DerefMut for MetalBuffer {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}
impl DeviceBuffer for MetalBuffer {
    fn ptr(&self) -> *const c_void {
        self.inner.gpu_address() as *const c_void
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Regression test for the transient buffer-pool recycling race: a
    /// pooled tensor dropped while GPU work is pending must NOT be recyclable
    /// until that work completes. Before the deferral fix the pair entered
    /// the pool at host-drop time and could be handed to a new tensor while
    /// an in-flight command buffer still referenced it (byte-level
    /// run-to-run nondeterminism on qwen3.5-35B at in-flight depth 2,
    /// 2026-08-12). The race itself needs GPU/CPU timing to bite, so this
    /// tests the lifetime mechanics deterministically instead.
    #[test]
    fn pool_recycling_defers_to_command_buffer_completion() -> TractResult<()> {
        crate::utils::with_borrowed_metal_stream(|stream| {
            // A shape no other test uses, so pool state is ours alone.
            let shape = [4099usize];
            let dt = DatumType::F32;
            let context = metal_context();
            // Drain anything a previous run of this test left behind.
            while context.pool_take(dt, &shape).is_some() {}

            // Idle stream: a drop recycles immediately (historical behavior).
            let t = DeviceTensor::uninitialized_dt(dt, &shape)?;
            drop(t);
            ensure!(
                context.pool_take(dt, &shape).is_some(),
                "idle-stream drop must recycle immediately"
            );

            // Busy stream: with a command buffer open, the drop must defer.
            let t = DeviceTensor::uninitialized_dt(dt, &shape)?;
            let _cb = stream.command_buffer();
            drop(t);
            ensure!(
                context.pool_take(dt, &shape).is_none(),
                "drop under an open command buffer must not recycle yet"
            );

            // A committed buffer can complete before the CPU returns from
            // commit_current. Recycling is safe in that case; otherwise the
            // pair remains deferred until the blocking wait below.
            stream.commit_current()?;
            let pending = stream.committed_command_buffers.borrow().back().is_some_and(|entry| {
                entry.buffer.status() != metal::MTLCommandBufferStatus::Completed
            });
            if pending {
                ensure!(
                    context.pool_take(dt, &shape).is_none(),
                    "drop under a pending committed buffer must not recycle yet"
                );
            }

            // Fully waited: the deferred pair lands in the pool.
            stream.wait_until_completed()?;
            ensure!(
                context.pool_take(dt, &shape).is_some(),
                "the pair must recycle once the in-flight work completed"
            );
            Ok(())
        })
    }
}
