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
use std::sync::Mutex;
use std::path::Path;
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
    const MAX_POOLED_PER_KEY: usize = 16;

    /// Hard cap on recycled bytes. The budget must hold the session memory
    /// arena (recycled once per decode step, tens to hundreds of MB at long
    /// context) with room to spare for the small fixed-shape transients;
    /// entries beyond it are evicted oldest-first, so stale shapes from a
    /// grown context cannot pin wired memory forever (unbounded pinning is
    /// what used to slow the weight-streaming kernels when large buffers
    /// were pooled without eviction).
    fn max_pooled_bytes() -> usize {
        static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        *N.get_or_init(|| {
            std::env::var("TRACT_METAL_POOL_MAX_MB")
                .ok()
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(512)
                * 1024
                * 1024
        })
    }

    fn pool_take(&self, dt: DatumType, shape: &[usize]) -> Option<(Arc<Tensor>, Buffer)> {
        if std::env::var_os("TRACT_METAL_DISABLE_BUFFER_POOL").is_some() {
            return None;
        }
        let mut pool = self.buffer_pool.lock().ok()?;
        let entry = pool.get_mut(&(dt, TVec::from_slice(shape)))?;
        let hit = entry.pop()?;
        self.pooled_bytes.fetch_sub(
            hit.0.len() * dt.size_of(),
            std::sync::atomic::Ordering::Relaxed,
        );
        Some(hit.0.clone()).map(|host| (host, hit.1))
    }

    fn pool_put(&self, host: Arc<Tensor>, buffer: Buffer) {
        if std::env::var_os("TRACT_METAL_DISABLE_BUFFER_POOL").is_some() {
            return;
        }
        let dt = host.datum_type();
        if !DeviceTensor::is_supported_dt(dt) {
            return;
        }
        let bytes = host.len() * dt.size_of();
        let budget = Self::max_pooled_bytes();
        if bytes > budget {
            return;
        }
        let Ok(mut pool) = self.buffer_pool.lock() else { return };
        // Evict oldest entries (globally, by insertion stamp) until the new
        // buffer fits the budget: recent shapes stay hot, stale shapes from
        // an earlier context length get released for real.
        while self.pooled_bytes.load(std::sync::atomic::Ordering::Relaxed) + bytes > budget {
            let oldest_key = pool
                .iter()
                .filter(|(_, v)| !v.is_empty())
                .min_by_key(|(_, v)| v.first().map(|e| e.2).unwrap_or(u64::MAX))
                .map(|(k, _)| k.clone());
            let Some(key) = oldest_key else { break };
            let Some(entry) = pool.get_mut(&key) else { break };
            let (evicted_host, _, _) = entry.remove(0);
            self.pooled_bytes.fetch_sub(
                evicted_host.len() * evicted_host.datum_type().size_of(),
                std::sync::atomic::Ordering::Relaxed,
            );
            if entry.is_empty() {
                pool.remove(&key);
            }
        }
        let entry = pool.entry((dt, TVec::from_slice(host.shape()))).or_default();
        if entry.len() >= Self::MAX_POOLED_PER_KEY {
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

        Ok(Box::new(MetalTensor {
            inner: MValue::Natural(host),
            device_buffer,
            exotic_fact: bqf,
        }))
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

#[derive(Debug)]
pub struct MetalStream {
    context: MetalContext,
    command_queue: CommandQueue,
    command_buffer: RefCell<Option<TCommandBuffer>>,
    /// Buffers committed by `commit_current` and not yet awaited, oldest
    /// first, each with the tensors that must stay alive until it completes.
    /// The queue is FIFO, so waiting on the newest implies all have completed.
    committed_command_buffers: RefCell<VecDeque<(TCommandBuffer, Vec<DeviceTensor>, Vec<String>)>>,
    /// Kernel names dispatched into the current (open) command buffer, only
    /// populated under TRACT_METAL_PROFILE_KERNELS.
    pending_kernel_names: RefCell<Vec<String>>,
    command_buffer_id: AtomicUsize,
    retained_tensors: RefCell<Vec<DeviceTensor>>,
    /// `command_buffer()` acquisitions since the last cadence commit; only
    /// maintained when TRACT_METAL_COMMIT_EVERY_N_DISPATCHES is set.
    dispatches_since_commit: std::cell::Cell<usize>,
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
            pending_kernel_names: RefCell::new(Vec::new()),
            command_buffer_id: AtomicUsize::new(0),
            retained_tensors: RefCell::new(vec![]),
            dispatches_since_commit: std::cell::Cell::new(0),
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
        if std::env::var_os("TRACT_METAL_PROFILE_KERNELS").is_some() {
            // One command buffer per dispatch: per-buffer GPU clocks become
            // per-kernel GPU times, logged with the name recorded here.
            self.commit_current()?;
            self.pending_kernel_names.borrow_mut().push(func_name.to_string());
        }
        self.context.load_pipeline(library_name, func_name)
    }

    pub(crate) fn load_pipeline_with_constants(
        &self,
        library_name: LibraryName,
        func_name: &str,
        constants: Option<ConstantValues>,
    ) -> TractResult<ComputePipelineState> {
        if std::env::var_os("TRACT_METAL_PROFILE_KERNELS").is_some() {
            // Same per-dispatch attribution as `load_pipeline`.
            self.commit_current()?;
            self.pending_kernel_names.borrow_mut().push(func_name.to_string());
        }
        self.context.load_pipeline_with_constants(library_name, func_name, constants)
    }

    pub fn retain_tensor(&self, tensor: &DeviceTensor) {
        self.retained_tensors.borrow_mut().push(tensor.clone());
    }

    /// Commit cadence: split the token/forward into command buffers every N
    /// `command_buffer()` acquisitions (roughly every N kernel dispatches), so
    /// the GPU starts executing early layers while the CPU still encodes late
    /// ones. 0 disables the cadence (single buffer per forward, plus whatever
    /// boundaries ops request themselves). Callers with fn-local scratch
    /// tensors must re-retain them after encoding (see
    /// `dispatch_route_topk_f32`): a cadence commit moves the retained list
    /// onto the buffer being closed.
    fn commit_every_n_dispatches() -> usize {
        static N: std::sync::OnceLock<usize> = std::sync::OnceLock::new();
        *N.get_or_init(|| {
            std::env::var("TRACT_METAL_COMMIT_EVERY_N_DISPATCHES")
                .ok()
                .and_then(|v| v.parse().ok())
                .unwrap_or(0)
        })
    }

    pub fn command_buffer(&self) -> TCommandBuffer {
        let cadence = Self::commit_every_n_dispatches();
        if cadence > 0 {
            let n = self.dispatches_since_commit.get() + 1;
            if n > cadence && self.command_buffer.borrow().is_some() {
                // Ignore failure modes commit_current already guards against.
                let _ = self.commit_current();
                self.dispatches_since_commit.set(1);
            } else {
                self.dispatches_since_commit.set(n);
            }
        }
        self.command_buffer
            .borrow_mut()
            .get_or_insert_with(|| {
                TCommandBuffer::new(self.command_queue.new_command_buffer().to_owned())
            })
            .to_owned()
    }

    fn log_gpu_time(buffer: &TCommandBuffer, tag: &str) {
        Self::log_gpu_time_named(buffer, tag, &[]);
    }

    fn log_gpu_time_named(buffer: &TCommandBuffer, tag: &str, names: &[String]) {
        if std::env::var_os("TRACT_METAL_LOG_GPU_TIME").is_some()
            || std::env::var_os("TRACT_METAL_PROFILE_KERNELS").is_some()
        {
            // metal-rs does not wrap GPUStartTime/GPUEndTime; go through objc.
            use objc::{msg_send, sel, sel_impl};
            let raw: &metal::CommandBufferRef = buffer;
            let start: f64 = unsafe { msg_send![raw, GPUStartTime] };
            let end: f64 = unsafe { msg_send![raw, GPUEndTime] };
            let label = if names.is_empty() { String::new() } else { format!(" [{}]", names.join("+")) };
            eprintln!("gpu-time {tag}{label}: {:.3} ms", (end - start) * 1e3);
        }
    }

    /// How many committed-but-unawaited buffers `commit_current` keeps in
    /// flight. Depth 2 overlaps CPU encoding with GPU execution; the wait on
    /// the oldest buffer is the backpressure that bounds transient memory
    /// (without it, a long-context forward retains every layer's transients
    /// at once and thrashes).
    const MAX_COMMITTED_IN_FLIGHT: usize = 2;

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
        let names = std::mem::take(&mut *self.pending_kernel_names.borrow_mut());
        let mut committed = self.committed_command_buffers.borrow_mut();
        committed.push_back((command_buffer, retained, names));
        while committed.len() > Self::MAX_COMMITTED_IN_FLIGHT {
            let (oldest, tensors, names) = committed.pop_front().unwrap();
            oldest.wait_until_completed();
            Self::log_gpu_time_named(&oldest, "segment", &names);
            drop(tensors);
        }
        Ok(())
    }

    pub fn wait_until_completed(&self) -> TractResult<()> {
        let Some(command_buffer) = self.command_buffer.borrow().to_owned() else {
            // No open buffer, but commit_current buffers may still be in
            // flight: the host must not read results before they land. FIFO:
            // waiting on the newest is enough.
            let drained: Vec<_> = self.committed_command_buffers.borrow_mut().drain(..).collect();
            if let Some((newest, _, _)) = drained.last() {
                newest.wait_until_completed();
            }
            for (buffer, _, names) in &drained {
                Self::log_gpu_time_named(buffer, "segment-tail", names);
            }
            drop(drained);
            self.retained_tensors.borrow_mut().clear();
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
        Self::log_gpu_time(&command_buffer, "final");
        log::trace!("Command buffer {:?} has completed (Blocking call)", command_buffer_id);

        // The queue is FIFO: the buffer above completing implies every buffer
        // committed earlier by commit_current has completed too.
        self.committed_command_buffers.borrow_mut().clear();

        // Clear local retained values used by the command buffer
        self.retained_tensors.borrow_mut().clear();

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
        if let Some((newest, _, _)) = drained.last() {
            newest.wait_until_completed();
        }
        drop(drained);
        let Some(command_buffer) = self.command_buffer.borrow_mut().to_owned() else { return };

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
        if Arc::strong_count(&self.host) == 1 {
            metal_context().pool_put(self.host.clone(), self.buffer.clone());
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
