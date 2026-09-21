use std::cell::{Cell, RefCell};
use std::collections::HashMap;
use std::ffi::c_void;
use std::mem::ManuallyDrop;
use std::num::NonZeroU64;
use std::sync::{Arc, Mutex, OnceLock, RwLock};

#[cfg(not(target_arch = "wasm32"))]
use anyhow::Context;
use anyhow::{anyhow, bail};
use tract_core::internal::*;
use tract_core::tract_linalg::block_quant::BlockQuantFact;
use tract_gpu::device::{DeviceBuffer, DeviceContext};
use tract_gpu::tensor::{DeviceTensor, OwnedDeviceTensor};
use tract_gpu::utils::as_q40_tensor;
use wgpu::util::DeviceExt;

#[cfg(all(target_arch = "wasm32", target_feature = "atomics"))]
compile_error!(
    "tract-wgpu cannot be built with wasm atomics. It is mutually exclusive with the \
     wasm-bindgen-rayon tier (+atomics,+bulk-memory,+mutable-globals,+simd128). \
     Build wasm32-unknown-unknown without +atomics; see linalg/MULTITHREAD_BENCHMARKS.md."
);

use crate::kernels::shaders::{
    EntryPoint, GRID_LIMIT, LayoutKind, ModuleKey, PipelineKey, ShaderDtype,
};
use crate::tensor::WgpuTensor;

pub const UNIFORM_ALIGN: u64 = 256;
/// Slots in the uniform ring. One dispatch takes one slot, and running out
/// forces a submit-and-wait in the middle of a frame, so the ring holds more
/// than a graph's worth of dispatches.
const UNIFORM_SLOTS: u64 = 256;
const UNIFORM_BYTES: u64 = UNIFORM_ALIGN * UNIFORM_SLOTS;
const WORKGROUP: u32 = 64;

/// Bind groups kept between frames before the cache is dropped wholesale.
const BIND_GROUP_CACHE_CAP: usize = 8192;

thread_local! {
    static WGPU_QUEUE: RefCell<Option<WgpuQueue>> = const { RefCell::new(None) };
    static LAST_PIPE_ID: Cell<u32> = const { Cell::new(0) };
    static LAST_BG_ID: Cell<u32> = const { Cell::new(0) };
}

pub fn with_wgpu_queue<R>(f: impl FnOnce(&WgpuQueue) -> TractResult<R>) -> TractResult<R> {
    wgpu_context();
    WGPU_QUEUE.with(|cell| {
        if cell.borrow().is_none() {
            *cell.borrow_mut() = Some(WgpuQueue::new()?);
        }
        let borrow = cell.borrow();
        f(borrow.as_ref().unwrap())
    })
}

/// Cache behaviour, for the ignored probes: bind groups and pooled buffers
/// found versus built.
pub static CACHE_STATS: [std::sync::atomic::AtomicUsize; 5] = [
    std::sync::atomic::AtomicUsize::new(0),
    std::sync::atomic::AtomicUsize::new(0),
    std::sync::atomic::AtomicUsize::new(0),
    std::sync::atomic::AtomicUsize::new(0),
    std::sync::atomic::AtomicUsize::new(0),
];

fn bump(i: usize) {
    CACHE_STATS[i].fetch_add(1, std::sync::atomic::Ordering::Relaxed);
}

fn context_slot() -> &'static OnceLock<WgpuContext> {
    static INSTANCE: OnceLock<WgpuContext> = OnceLock::new();
    &INSTANCE
}

fn install_context(ctxt: WgpuContext) -> TractResult<WgpuContext> {
    let _ = tract_gpu::device::set_context(Box::new(ctxt.clone()));
    let _ = context_slot().set(ctxt.clone());
    Ok(context_slot().get().cloned().unwrap_or(ctxt))
}

/// Native: creates the adapter synchronously. Wasm: panics unless
/// [`wgpu_context_async`] has already completed — `pollster` cannot block
/// the browser event loop.
pub fn wgpu_context() -> WgpuContext {
    try_wgpu_context().expect("Could not create wgpu context")
}

/// As [`wgpu_context`], reporting a host with no usable adapter instead of
/// panicking, which is what `Runtime::check` needs to answer.
pub fn try_wgpu_context() -> TractResult<WgpuContext> {
    if let Some(ctxt) = context_slot().get() {
        return Ok(ctxt.clone());
    }
    #[cfg(target_arch = "wasm32")]
    bail!(
        "tract-wgpu on wasm32: await wgpu_context_async() once at startup \
         (requestAdapter/requestDevice are async; PollType::Wait is a no-op on web)"
    );
    #[cfg(not(target_arch = "wasm32"))]
    install_context(WgpuContext::new()?)
}

/// One-shot async device init. Safe to call from wasm `wasm_bindgen` async
/// startup; subsequent [`wgpu_context`] calls are free.
pub async fn wgpu_context_async() -> TractResult<WgpuContext> {
    if let Some(ctxt) = context_slot().get() {
        return Ok(ctxt.clone());
    }
    let ctxt = WgpuContext::new_async().await?;
    install_context(ctxt)
}

#[derive(Clone)]
pub struct WgpuContext {
    inner: Arc<WgpuContextInner>,
}

struct WgpuContextInner {
    device: wgpu::Device,
    queue: wgpu::Queue,
    shader_f16: bool,
    modules: RwLock<HashMap<ModuleKey, wgpu::ShaderModule>>,
    pipelines: RwLock<HashMap<PipelineKey, (Arc<wgpu::ComputePipeline>, u32)>>,
    /// Keyed by [`ProgramKey`], the pair of hashes of whatever names a
    /// generated program: a frame looks these up once per dispatch, and
    /// formatting a string for each of them was a measurable share of the
    /// host time under wasm.
    chain_pipelines: RwLock<HashMap<ProgramKey, (Arc<wgpu::ComputePipeline>, u32)>>,
    layouts: RwLock<HashMap<LayoutKind, (wgpu::BindGroupLayout, wgpu::PipelineLayout)>>,
    bind_groups: Mutex<HashMap<BindGroupKey, (Arc<wgpu::BindGroup>, u32)>>,
    /// Staging buffers for downloads, each with whether a download still
    /// holds it: a caller that starts the next frame before awaiting the last
    /// frame's map needs a second one.
    staging: Mutex<Vec<(wgpu::Buffer, Arc<std::sync::atomic::AtomicBool>)>>,
    next_gpu_id: std::sync::atomic::AtomicU32,
}

/// The buffers one pool position has been handed, oldest first.
type Slot = Vec<Arc<WgpuBuffer>>;

/// Frees a staging buffer when the download holding it ends, however it ends.
struct Release(Arc<std::sync::atomic::AtomicBool>);

impl Drop for Release {
    fn drop(&mut self) {
        self.0.store(false, std::sync::atomic::Ordering::Release);
    }
}

/// What a generated program is keyed on: two independent 64-bit hashes of
/// the values that name it (kernel, dtype, epilogue steps, flags), so the
/// lookup costs a hash rather than a formatted string, and a collision would
/// need both to agree.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ProgramKey(u64, u64);

impl ProgramKey {
    pub fn of(key: &impl std::hash::Hash) -> Self {
        use std::hash::Hasher;
        let mut sip = std::collections::hash_map::DefaultHasher::new();
        key.hash(&mut sip);
        let mut fnv = Fnv1a(0xcbf2_9ce4_8422_2325);
        key.hash(&mut fnv);
        ProgramKey(sip.finish(), fnv.finish())
    }
}

struct Fnv1a(u64);

impl std::hash::Hasher for Fnv1a {
    fn finish(&self) -> u64 {
        self.0
    }

    fn write(&mut self, bytes: &[u8]) {
        for b in bytes {
            self.0 ^= *b as u64;
            self.0 = self.0.wrapping_mul(0x0100_0000_01b3);
        }
    }
}

/// Identity of a bind group. Buffer ids are stable for pooled slots, so a
/// graph that allocates in the same order reuses last frame's groups. The
/// uniform ring is the same buffer for every dispatch on this queue, so it
/// is not part of the key.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
struct BindGroupKey {
    layout: LayoutKind,
    n: u8,
    buffers: [u64; 8],
}

impl std::fmt::Debug for WgpuContext {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WgpuContext").field("shader_f16", &self.inner.shader_f16).finish()
    }
}

impl WgpuContext {
    async fn create_device() -> TractResult<(wgpu::Device, wgpu::Queue, bool)> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor::new_without_display_handle());
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
                ..Default::default()
            })
            .await
            .map_err(|e| anyhow!("No wgpu adapter: {e}"))?;

        let info = adapter.get_info();
        log::info!(
            "tract-wgpu: adapter {:?} ({:?}) backend={:?}",
            info.name,
            info.device_type,
            info.backend
        );

        let shader_f16 = adapter.features().contains(wgpu::Features::SHADER_F16);
        let mut features = wgpu::Features::empty();
        if shader_f16 {
            features |= wgpu::Features::SHADER_F16;
        }
        if adapter.features().contains(wgpu::Features::TIMESTAMP_QUERY) {
            features |= wgpu::Features::TIMESTAMP_QUERY;
        }

        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("tract-wgpu"),
                required_features: features,
                required_limits: adapter.limits(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
            })
            .await
            .map_err(|e| anyhow!("request_device failed: {e}"))?;
        Ok((device, queue, shader_f16))
    }

    fn from_device(
        device: wgpu::Device,
        queue: wgpu::Queue,
        shader_f16: bool,
    ) -> TractResult<Self> {
        let ctxt = Self {
            inner: Arc::new(WgpuContextInner {
                device,
                queue,
                shader_f16,
                modules: RwLock::new(HashMap::new()),
                pipelines: RwLock::new(HashMap::new()),
                chain_pipelines: RwLock::new(HashMap::new()),
                layouts: RwLock::new(HashMap::new()),
                bind_groups: Mutex::new(HashMap::new()),
                staging: Mutex::new(Vec::new()),
                next_gpu_id: std::sync::atomic::AtomicU32::new(1),
            }),
        };
        #[cfg(target_arch = "wasm32")]
        crate::wasm_record::tract_install_record_hooks();
        ctxt.preload_pipelines()?;
        Ok(ctxt)
    }

    pub fn new() -> TractResult<Self> {
        #[cfg(target_arch = "wasm32")]
        {
            bail!(
                "WgpuContext::new() cannot block on wasm32. Await WgpuContext::new_async() \
                 (or wgpu_context_async()) once at startup."
            )
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let (device, queue, shader_f16) = pollster::block_on(Self::create_device())?;
            Self::from_device(device, queue, shader_f16)
        }
    }

    pub async fn new_async() -> TractResult<Self> {
        let (device, queue, shader_f16) = Self::create_device().await?;
        Self::from_device(device, queue, shader_f16)
    }

    pub fn device(&self) -> &wgpu::Device {
        &self.inner.device
    }

    pub fn queue(&self) -> &wgpu::Queue {
        &self.inner.queue
    }

    pub fn shader_f16(&self) -> bool {
        self.inner.shader_f16
    }

    pub fn timestamps(&self) -> bool {
        self.inner.device.features().contains(wgpu::Features::TIMESTAMP_QUERY)
    }

    fn preload_pipelines(&self) -> TractResult<()> {
        for key in crate::kernels::all_pipeline_keys(self.shader_f16()) {
            let _ = self.pipeline(key);
        }
        Ok(())
    }

    pub fn layout(
        &self,
        kind: LayoutKind,
    ) -> TractResult<(wgpu::BindGroupLayout, wgpu::PipelineLayout)> {
        {
            let cache = self.inner.layouts.read().map_err(|e| anyhow!("{e}"))?;
            if let Some(v) = cache.get(&kind) {
                return Ok(v.clone());
            }
        }
        let mut cache = self.inner.layouts.write().map_err(|e| anyhow!("{e}"))?;
        if let Some(v) = cache.get(&kind) {
            return Ok(v.clone());
        }
        let entries = kind.bind_group_layout_entries();
        let bgl = self.inner.device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some(kind.label()),
            entries: &entries,
        });
        let pl = self.inner.device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some(kind.label()),
            bind_group_layouts: &[Some(&bgl)],
            immediate_size: 0,
        });
        cache.insert(kind, (bgl.clone(), pl.clone()));
        Ok((bgl, pl))
    }

    fn module(&self, key: ModuleKey) -> TractResult<wgpu::ShaderModule> {
        {
            let cache = self.inner.modules.read().map_err(|e| anyhow!("{e}"))?;
            if let Some(m) = cache.get(&key) {
                return Ok(m.clone());
            }
        }
        let mut cache = self.inner.modules.write().map_err(|e| anyhow!("{e}"))?;
        if let Some(m) = cache.get(&key) {
            return Ok(m.clone());
        }
        if key.dtype == ShaderDtype::F16 && !self.shader_f16() {
            bail!("f16 requested but adapter lacks shader-f16");
        }
        let src = key.wgsl();
        let module = self.inner.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(&key.label()),
            source: wgpu::ShaderSource::Wgsl(src.into()),
        });
        cache.insert(key, module.clone());
        Ok(module)
    }

    fn next_gpu_id(&self) -> u32 {
        self.inner.next_gpu_id.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    }

    /// Handles come out of the caches behind an `Arc`: on the web backend a
    /// `wgpu` handle clone is two calls into JavaScript, and a frame clones
    /// each dispatch's pipeline and bind group several times over.
    pub fn pipeline(&self, key: PipelineKey) -> TractResult<Arc<wgpu::ComputePipeline>> {
        {
            let cache = self.inner.pipelines.read().map_err(|e| anyhow!("{e}"))?;
            if let Some((p, id)) = cache.get(&key) {
                LAST_PIPE_ID.with(|c| c.set(*id));
                return Ok(p.clone());
            }
        }
        let mut cache = self.inner.pipelines.write().map_err(|e| anyhow!("{e}"))?;
        if let Some((p, id)) = cache.get(&key) {
            LAST_PIPE_ID.with(|c| c.set(*id));
            return Ok(p.clone());
        }
        let module = self.module(key.module)?;
        let (_bgl, pl) = self.layout(key.module.layout())?;
        let entry = key.entry.name();
        let id = self.next_gpu_id();
        let label = format!("tp{id}");
        let pipeline =
            self.inner.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&label),
                layout: Some(&pl),
                module: &module,
                entry_point: Some(&entry),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        LAST_PIPE_ID.with(|c| c.set(id));
        let pipeline = Arc::new(pipeline);
        cache.insert(key, (pipeline.clone(), id));
        Ok(pipeline)
    }

    /// Pipeline for a generated program, keyed by the program's own name;
    /// `source` is only called on a miss.
    pub fn chain_pipeline(
        &self,
        key: impl std::hash::Hash,
        layout: LayoutKind,
        entry: EntryPoint,
        source: impl FnOnce() -> String,
    ) -> TractResult<Arc<wgpu::ComputePipeline>> {
        let key = ProgramKey::of(&key);
        {
            let cache = self.inner.chain_pipelines.read().map_err(|e| anyhow!("{e}"))?;
            if let Some((p, id)) = cache.get(&key) {
                LAST_PIPE_ID.with(|c| c.set(*id));
                return Ok(p.clone());
            }
        }
        let mut cache = self.inner.chain_pipelines.write().map_err(|e| anyhow!("{e}"))?;
        if let Some((p, id)) = cache.get(&key) {
            LAST_PIPE_ID.with(|c| c.set(*id));
            return Ok(p.clone());
        }
        let module = self.inner.device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some(layout.label()),
            source: wgpu::ShaderSource::Wgsl(source().into()),
        });
        let (_bgl, pl) = self.layout(layout)?;
        let id = self.next_gpu_id();
        let label = format!("tp{id}");
        let pipeline =
            self.inner.device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some(&label),
                layout: Some(&pl),
                module: &module,
                entry_point: Some(&entry.name()),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: None,
            });
        LAST_PIPE_ID.with(|c| c.set(id));
        let pipeline = Arc::new(pipeline);
        cache.insert(key, (pipeline.clone(), id));
        Ok(pipeline)
    }

    pub fn bind_group(
        &self,
        kind: LayoutKind,
        buffers: &[&WgpuBuffer],
        uniform: &wgpu::Buffer,
    ) -> TractResult<Arc<wgpu::BindGroup>> {
        ensure!(kind != LayoutKind::Ingest, "Ingest layout binds a texture; use bind_group_ingest");
        ensure!(buffers.len() <= 8, "bind group has {} buffers, max 8", buffers.len());
        let mut ids = [0u64; 8];
        for (i, b) in buffers.iter().enumerate() {
            ids[i] = b.id;
        }
        let key = BindGroupKey { layout: kind, n: buffers.len() as u8, buffers: ids };
        {
            let cache = self.inner.bind_groups.lock().map_err(|e| anyhow!("{e}"))?;
            if let Some((bg, id)) = cache.get(&key) {
                bump(0);
                LAST_BG_ID.with(|c| c.set(*id));
                return Ok(bg.clone());
            }
        }
        bump(1);
        let (bgl, _) = self.layout(kind)?;
        let mut entries: Vec<wgpu::BindGroupEntry<'_>> = Vec::with_capacity(buffers.len() + 1);
        for (i, buf) in buffers.iter().enumerate() {
            entries.push(wgpu::BindGroupEntry {
                binding: i as u32,
                resource: buf.inner.as_entire_binding(),
            });
        }
        entries.push(uniform_entry(buffers.len() as u32, uniform));
        let id = self.next_gpu_id();
        let label = format!("tb{id}");
        let bg = self.inner.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&label),
            layout: &bgl,
            entries: &entries,
        });
        LAST_BG_ID.with(|c| c.set(id));
        let mut cache = self.inner.bind_groups.lock().map_err(|e| anyhow!("{e}"))?;
        let bg = Arc::new(bg);
        cache.insert(key, (bg.clone(), id));
        Ok(bg)
    }

    /// One storage buffer read, one storage texture written.
    pub fn bind_group_export(
        &self,
        input: &wgpu::Buffer,
        view: &wgpu::TextureView,
        uniform: &wgpu::Buffer,
    ) -> TractResult<Arc<wgpu::BindGroup>> {
        let (bgl, _) = self.layout(LayoutKind::Export)?;
        let entries = [
            wgpu::BindGroupEntry { binding: 0, resource: input.as_entire_binding() },
            wgpu::BindGroupEntry { binding: 1, resource: wgpu::BindingResource::TextureView(view) },
            uniform_entry(2, uniform),
        ];
        let id = self.next_gpu_id();
        let label = format!("tb{id}");
        LAST_BG_ID.with(|c| c.set(id));
        Ok(Arc::new(self.inner.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&label),
            layout: &bgl,
            entries: &entries,
        })))
    }

    pub fn bind_group_ingest(
        &self,
        view: &wgpu::TextureView,
        output: &wgpu::Buffer,
        uniform: &wgpu::Buffer,
    ) -> TractResult<Arc<wgpu::BindGroup>> {
        let (bgl, _) = self.layout(LayoutKind::Ingest)?;
        let entries = [
            wgpu::BindGroupEntry { binding: 0, resource: wgpu::BindingResource::TextureView(view) },
            wgpu::BindGroupEntry { binding: 1, resource: output.as_entire_binding() },
            uniform_entry(2, uniform),
        ];
        let id = self.next_gpu_id();
        let label = format!("tb{id}");
        LAST_BG_ID.with(|c| c.set(id));
        Ok(Arc::new(self.inner.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some(&label),
            layout: &bgl,
            entries: &entries,
        })))
    }

    pub fn create_storage_buffer(&self, bytes: &[u8]) -> wgpu::Buffer {
        let padded = pad4(bytes);
        self.inner.device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("tract-wgpu-storage"),
            contents: &padded,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
        })
    }

    /// Starts a new allocation sequence. Called when the queue flushes, which
    /// is where a graph's intermediates are released.
    /// Bind groups a frame no longer needs. The cache is only trimmed between
    /// frames: sized to hold a graph's working set several times over, it
    /// bounds a pathological model without ever evicting mid-frame what the
    /// frame in flight is still binding.
    pub(crate) fn trim_bind_groups(&self) {
        if let Ok(mut cache) = self.inner.bind_groups.lock()
            && cache.len() > BIND_GROUP_CACHE_CAP
        {
            cache.clear();
        }
    }

    fn next_buffer_id(&self) -> u64 {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    }

    pub fn alloc_storage(&self, size: u64) -> Arc<WgpuBuffer> {
        with_wgpu_queue(|q| Ok(q.alloc_storage(size)))
            .expect("wgpu queue unavailable for allocation")
    }

    pub fn wrap_storage(&self, inner: wgpu::Buffer) -> WgpuBuffer {
        WgpuBuffer { inner, id: self.next_buffer_id(), pooled: false }
    }

    fn wrap_pooled_storage(&self, inner: wgpu::Buffer) -> WgpuBuffer {
        WgpuBuffer { inner, id: self.next_buffer_id(), pooled: true }
    }

    pub fn create_empty_storage(&self, size: u64) -> wgpu::Buffer {
        let size = size.max(4).next_multiple_of(4);
        self.inner.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tract-wgpu-storage"),
            size,
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_SRC
                | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        })
    }

    /// A free staging buffer of at least `size` bytes, marked busy until the
    /// download that took it releases it. Buffers too small for a request are
    /// replaced; a request while every buffer is busy gets a new one.
    fn staging_at_least(&self, size: u64) -> (wgpu::Buffer, Arc<std::sync::atomic::AtomicBool>) {
        use std::sync::atomic::Ordering;
        let mut slots = self.inner.staging.lock().unwrap();
        slots.retain(|(b, busy)| busy.load(Ordering::Acquire) || b.size() >= size);
        if let Some((b, busy)) = slots.iter().find(|(_, busy)| !busy.load(Ordering::Acquire)) {
            busy.store(true, Ordering::Release);
            return (b.clone(), busy.clone());
        }
        let buffer = self.inner.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tract-wgpu-staging"),
            size: size.max(4),
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let busy = Arc::new(std::sync::atomic::AtomicBool::new(true));
        slots.push((buffer.clone(), busy.clone()));
        (buffer, busy)
    }

    /// Awaitable GPU→CPU copy. On web this is the single yield at the end of
    /// `run()`; do not call it mid-graph. The copy rides the pending work's
    /// own submission, so a frame costs one submit, not two.
    pub async fn download_async(
        &self,
        buffer: &wgpu::Buffer,
        offset: u64,
        len: u64,
    ) -> TractResult<Vec<u8>> {
        if len == 0 {
            with_wgpu_queue(|q| q.flush())?;
            return Ok(vec![]);
        }
        let copy_len = len.next_multiple_of(4);
        let (staging, busy) = self.staging_at_least(copy_len);
        let release = Release(busy);
        with_wgpu_queue(|q| q.flush_after_copy(buffer, offset, &staging, copy_len))?;

        #[cfg(target_arch = "wasm32")]
        {
            let slice = staging.slice(..copy_len);
            let promise = js_sys::Promise::new(&mut |resolve, reject| {
                slice.map_async(wgpu::MapMode::Read, move |r| match r {
                    Ok(()) => {
                        let _ = resolve.call0(&wasm_bindgen::JsValue::UNDEFINED);
                    }
                    Err(e) => {
                        let _ = reject.call1(
                            &wasm_bindgen::JsValue::UNDEFINED,
                            &wasm_bindgen::JsValue::from_str(&format!("{e}")),
                        );
                    }
                });
            });
            wasm_bindgen_futures::JsFuture::from(promise)
                .await
                .map_err(|e| anyhow!("mapAsync: {e:?}"))?;
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            let slice = staging.slice(..copy_len);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            self.poll_wait()?;
            rx.recv().context("staging map channel")??;
        }
        let data = staging.slice(..copy_len).get_mapped_range()?;
        let out = data[..len as usize].to_vec();
        drop(data);
        staging.unmap();
        drop(release);
        Ok(out)
    }

    fn poll_wait(&self) -> TractResult<()> {
        // PollType::Wait is a no-op on web — the browser pumps GPU work.
        // Never rely on it to complete mapAsync there.
        #[cfg(target_arch = "wasm32")]
        {
            let _ = self.inner.device.poll(wgpu::PollType::Poll);
            Ok(())
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.inner
                .device
                .poll(wgpu::PollType::wait_indefinitely())
                .map(|_| ())
                .map_err(|e| anyhow!("wgpu poll: {e}"))
        }
    }
}

/// The uniform slot every layout ends with: one 256-byte window, addressed by
/// the dynamic offset a dispatch passes.
fn uniform_entry(binding: u32, uniform: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
            buffer: uniform,
            offset: 0,
            size: Some(NonZeroU64::new(UNIFORM_ALIGN).unwrap()),
        }),
    }
}

fn pad4(bytes: &[u8]) -> Vec<u8> {
    let mut v = bytes.to_vec();
    while !v.len().is_multiple_of(4) {
        v.push(0);
    }
    if v.is_empty() {
        v.resize(4, 0);
    }
    v
}

impl DeviceContext for WgpuContext {
    fn synchronize(&self) -> TractResult<()> {
        with_wgpu_queue(|q| q.flush())
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
        let buffer = Arc::new(self.wrap_storage(self.create_storage_buffer(data_bytes)));
        Ok(Box::new(WgpuTensor {
            buffer,
            datum_type: view.datum_type(),
            shape: view.shape().into(),
            strides: view.strides().into(),
            exotic_fact: bqf,
        }))
    }

    fn uninitialized_device_tensor(
        &self,
        shape: &[usize],
        dt: DatumType,
    ) -> TractResult<Box<dyn OwnedDeviceTensor>> {
        let bytes = shape.iter().product::<usize>() * dt.size_of();
        let buffer = self.alloc_storage(bytes as u64);
        Ok(Box::new(WgpuTensor {
            buffer,
            datum_type: dt,
            shape: shape.into(),
            strides: Tensor::natural_strides(shape),
            exotic_fact: None,
        }))
    }

    fn uninitialized_device_exotic_tensor(
        &self,
        exotic_fact: Box<dyn ExoticFact>,
    ) -> TractResult<Box<dyn OwnedDeviceTensor>> {
        if let Some(bqf) = exotic_fact.downcast_ref::<BlockQuantFact>() {
            let blocks = bqf.shape().iter().product::<usize>() / bqf.format.block_len();
            let bytes = blocks * bqf.format.block_bytes();
            let buffer = self.alloc_storage(bytes as u64);
            Ok(Box::new(WgpuTensor {
                buffer,
                datum_type: f32::datum_type(),
                shape: tvec!(),
                strides: tvec!(),
                exotic_fact: Some(exotic_fact),
            }))
        } else {
            bail!("Only BlockQuant tensor allocation supported for now")
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
        crate::kernels::copy::wgpu_copy_nd_dispatch(
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

struct Dispatch {
    label: &'static str,
    pipeline: Arc<wgpu::ComputePipeline>,
    bind_group: Arc<wgpu::BindGroup>,
    #[allow(dead_code)]
    pipe_id: u32,
    #[allow(dead_code)]
    bg_id: u32,
    dynamic_offset: u32,
    groups: [u32; 3],
}

/// GPU time attributed to one kernel family over a run.
#[derive(Debug, Clone, Default)]
pub struct KernelTime {
    pub calls: usize,
    pub nanos: f64,
}

pub struct WgpuQueue {
    context: WgpuContext,
    encoder: RefCell<Option<wgpu::CommandEncoder>>,
    pending: RefCell<Vec<Dispatch>>,
    uniform_staging: RefCell<Vec<u8>>,
    /// A graph allocates the same sizes in the same order every frame, so a
    /// buffer is keyed by that position rather than by size alone: the Nth
    /// allocation of a given size always gets the same buffer, and the bind
    /// group built for it last frame still matches. Both maps belong to the
    /// thread that owns the frame: a buffer must not be handed to another
    /// thread while a dispatch this one recorded still reads it.
    /// Every buffer a position has ever been handed, oldest first: with a
    /// frame still in flight the position's last buffer is alive, and the one
    /// before it is what keeps the bind groups keyed on it from being rebuilt.
    buffer_slots: RefCell<HashMap<(u64, u32), Slot>>,
    /// The blocked matmul's weights, transposed once. Keyed by everything the
    /// packing depends on, and only ever by an unpooled buffer, whose id no
    /// later tensor can take.
    repacked: RefCell<HashMap<RepackKey, DeviceTensor>>,
    slot_cursor: RefCell<HashMap<u64, u32>>,
    profile: Cell<bool>,
    profiled: RefCell<Vec<(&'static str, u32)>>,
    queries: RefCell<Option<Queries>>,
    uniform: ManuallyDrop<wgpu::Buffer>,
    uniform_cursor: Cell<u64>,
    retained: RefCell<Vec<DeviceTensor>>,
    retained_textures: RefCell<Vec<wgpu::Texture>>,
    staging: RefCell<Option<wgpu::Buffer>>,
}

/// One timestamp pair per dispatch, plus the buffers to read them back.
struct Queries {
    set: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    read: wgpu::Buffer,
    capacity: u32,
}

/// Timestamps are written per compute pass, so a profiled run puts every
/// dispatch in a pass of its own: the attribution is real GPU time, but the
/// total is not what an unprofiled run costs.
const MAX_QUERIES: u32 = 4096;

impl WgpuQueue {
    /// GPU time per kernel family over `f`. Returns nothing when the adapter has
    /// no timestamp queries.
    pub fn profile<R>(
        &self,
        f: impl FnOnce() -> TractResult<R>,
    ) -> TractResult<(R, Vec<(&'static str, KernelTime)>)> {
        if !self.context.timestamps() {
            return Ok((f()?, vec![]));
        }
        self.flush()?;
        self.ensure_queries()?;
        self.profiled.borrow_mut().clear();
        self.profile.set(true);
        let out = f();
        let _ = self.flush();
        self.profile.set(false);
        Ok((out?, self.read_profile()?))
    }

    fn ensure_queries(&self) -> TractResult<()> {
        let mut slot = self.queries.borrow_mut();
        if slot.is_some() {
            return Ok(());
        }
        let device = self.context.device();
        let set = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("tract-wgpu-timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: MAX_QUERIES,
        });
        let size = MAX_QUERIES as u64 * 8;
        let resolve = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tract-wgpu-timestamps-resolve"),
            size,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let read = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("tract-wgpu-timestamps-read"),
            size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        *slot = Some(Queries { set, resolve, read, capacity: MAX_QUERIES });
        Ok(())
    }

    fn read_profile(&self) -> TractResult<Vec<(&'static str, KernelTime)>> {
        let profiled = std::mem::take(&mut *self.profiled.borrow_mut());
        if profiled.is_empty() {
            return Ok(vec![]);
        }
        let slot = self.queries.borrow();
        let queries = slot.as_ref().unwrap();
        let used = profiled.len() as u32 * 2;
        {
            let mut encoder =
                self.context.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
                    label: Some("tract-wgpu-resolve"),
                });
            encoder.resolve_query_set(&queries.set, 0..used, &queries.resolve, 0);
            encoder.copy_buffer_to_buffer(&queries.resolve, 0, &queries.read, 0, used as u64 * 8);
            self.context.queue().submit(Some(encoder.finish()));
        }
        let slice = queries.read.slice(..used as u64 * 8);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |r| {
            let _ = tx.send(r);
        });
        self.context.poll_wait()?;
        rx.recv().context("timestamp map channel")??;
        let period = self.context.queue().get_timestamp_period() as f64;
        let ticks: Vec<u64> = {
            let data = slice.get_mapped_range()?;
            data.chunks_exact(8).map(|c| u64::from_le_bytes(c.try_into().unwrap())).collect()
        };
        queries.read.unmap();

        let mut by_label: HashMap<&'static str, KernelTime> = HashMap::new();
        for (label, base) in profiled {
            let (start, end) = (ticks[base as usize], ticks[base as usize + 1]);
            let e = by_label.entry(label).or_default();
            e.calls += 1;
            e.nanos += end.saturating_sub(start) as f64 * period;
        }
        let mut out: Vec<_> = by_label.into_iter().collect();
        out.sort_by(|a, b| b.1.nanos.total_cmp(&a.1.nanos));
        Ok(out)
    }
}

impl WgpuQueue {
    fn new() -> TractResult<Self> {
        let context = wgpu_context();
        let uniform = context.device().create_buffer(&wgpu::BufferDescriptor {
            label: Some("tract-wgpu-uniform"),
            size: UNIFORM_BYTES,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Ok(Self {
            context,
            encoder: RefCell::new(None),
            pending: RefCell::new(vec![]),
            uniform_staging: RefCell::new(vec![]),
            buffer_slots: RefCell::new(HashMap::new()),
            repacked: RefCell::new(HashMap::new()),
            slot_cursor: RefCell::new(HashMap::new()),
            profile: Cell::new(false),
            profiled: RefCell::new(vec![]),
            queries: RefCell::new(None),
            uniform: ManuallyDrop::new(uniform),
            uniform_cursor: Cell::new(0),
            retained: RefCell::new(vec![]),
            retained_textures: RefCell::new(vec![]),
            staging: RefCell::new(None),
        })
    }

    pub fn context(&self) -> &WgpuContext {
        &self.context
    }

    pub fn uniform(&self) -> &wgpu::Buffer {
        &self.uniform
    }

    /// wgpu-core buffer drop takes a TLS snatch lock. Doing that from this
    /// thread-local's destructor panics (`cannot access a Thread Local Storage
    /// value during or after destruction`). Leak GPU objects; the device lives
    /// in a process-level `OnceLock` for the rest of the run.
    fn leak_gpu_objects(&mut self) {
        std::mem::forget(self.encoder.replace(None));
        std::mem::forget(self.staging.replace(None));
        std::mem::forget(std::mem::take(&mut *self.retained.borrow_mut()));
        std::mem::forget(std::mem::take(&mut *self.retained_textures.borrow_mut()));
        std::mem::forget(std::mem::take(&mut *self.buffer_slots.borrow_mut()));
        std::mem::forget(std::mem::take(&mut *self.repacked.borrow_mut()));
        unsafe {
            std::mem::forget(ManuallyDrop::take(&mut self.uniform));
        }
    }

    /// A storage buffer of `size` bytes for this position in the frame's
    /// allocation sequence. Reuses a buffer that position held before, the
    /// oldest one nothing holds any more; a position whose buffers are all
    /// alive — the previous frame not yet submitted — gets one more, and from
    /// then on alternates between them, so every buffer id the position ever
    /// binds is a stable one.
    pub fn alloc_storage(&self, size: u64) -> Arc<WgpuBuffer> {
        let size = size.max(4).next_multiple_of(4);
        let seq = {
            let mut cursor = self.slot_cursor.borrow_mut();
            let n = cursor.entry(size).or_insert(0);
            let seq = *n;
            *n += 1;
            seq
        };
        let mut slots = self.buffer_slots.borrow_mut();
        let held = slots.entry((size, seq)).or_default();
        if let Some(free) = held.iter().find(|b| Arc::strong_count(b) == 1) {
            bump(2);
            return free.clone();
        }
        let fresh =
            Arc::new(self.context.wrap_pooled_storage(self.context.create_empty_storage(size)));
        bump(3);
        held.push(fresh.clone());
        fresh
    }

    pub fn repacked(&self, key: &RepackKey) -> Option<DeviceTensor> {
        self.repacked.borrow().get(key).cloned()
    }

    pub fn store_repacked(&self, key: RepackKey, packed: DeviceTensor) {
        self.repacked.borrow_mut().insert(key, packed);
    }

    pub fn retain_tensor(&self, t: &DeviceTensor) {
        self.retained.borrow_mut().push(t.clone());
    }

    pub fn retain_texture(&self, t: wgpu::Texture) {
        self.retained_textures.borrow_mut().push(t);
    }

    fn encoder(&self) -> std::cell::RefMut<'_, wgpu::CommandEncoder> {
        let mut slot = self.encoder.borrow_mut();
        if slot.is_none() {
            *slot = Some(self.context.device().create_command_encoder(
                &wgpu::CommandEncoderDescriptor { label: Some("tract-wgpu") },
            ));
        }
        std::cell::RefMut::map(slot, |o| o.as_mut().unwrap())
    }

    /// Stages one dispatch's uniform block. The ring is uploaded once per
    /// flush: a write per dispatch costs an allocation and a queue call each,
    /// and a graph issues them by the hundred.
    pub fn alloc_uniform(&self, bytes: &[u8]) -> TractResult<u32> {
        ensure!(
            bytes.len() as u64 <= UNIFORM_ALIGN,
            "uniform payload {} > {UNIFORM_ALIGN}",
            bytes.len()
        );
        let mut cursor = self.uniform_cursor.get();
        if cursor + UNIFORM_ALIGN > UNIFORM_BYTES {
            self.flush()?;
            cursor = 0;
        }
        let mut staging = self.uniform_staging.borrow_mut();
        let at = cursor as usize;
        if staging.len() < at + UNIFORM_ALIGN as usize {
            staging.resize(at + UNIFORM_ALIGN as usize, 0);
        }
        staging[at..at + bytes.len()].copy_from_slice(bytes);
        staging[at + bytes.len()..at + UNIFORM_ALIGN as usize].fill(0);
        self.uniform_cursor.set(cursor + UNIFORM_ALIGN);
        Ok(cursor as u32)
    }

    /// Uploads the slots staged since the last flush, before anything reads
    /// them.
    fn upload_uniforms(&self) {
        let used = self.uniform_cursor.get();
        if used == 0 {
            return;
        }
        let staging = self.uniform_staging.borrow();
        self.context.queue().write_buffer(&self.uniform, 0, &staging[..used as usize]);
    }

    pub fn dispatch(
        &self,
        label: &'static str,
        pipeline: &Arc<wgpu::ComputePipeline>,
        bind_group: &Arc<wgpu::BindGroup>,
        dynamic_offset: u32,
        n_elements: u64,
    ) -> TractResult<()> {
        if n_elements == 0 {
            return Ok(());
        }
        let groups = n_elements.div_ceil(WORKGROUP as u64);
        let x = groups.min(GRID_LIMIT as u64) as u32;
        let y = groups.div_ceil(GRID_LIMIT as u64) as u32;
        self.dispatch_grid(label, pipeline, bind_group, dynamic_offset, [x, y, 1])
    }

    /// A kernel whose workgroups tile a 2-D or 3-D iteration space itself.
    pub fn dispatch_grid(
        &self,
        label: &'static str,
        pipeline: &Arc<wgpu::ComputePipeline>,
        bind_group: &Arc<wgpu::BindGroup>,
        dynamic_offset: u32,
        groups: [u32; 3],
    ) -> TractResult<()> {
        if groups.contains(&0) {
            return Ok(());
        }
        self.pending.borrow_mut().push(Dispatch {
            label,
            pipeline: pipeline.clone(),
            bind_group: bind_group.clone(),
            pipe_id: LAST_PIPE_ID.with(|c| c.get()),
            bg_id: LAST_BG_ID.with(|c| c.get()),
            dynamic_offset,
            groups,
        });
        Ok(())
    }

    /// Records everything queued since the last drain as a single compute pass.
    /// WebGPU orders dispatches inside a pass and synchronizes their storage
    /// accesses, so one pass per graph costs far less than one pass per kernel.
    /// Anything that encodes a copy must drain first to keep its place in line.
    fn drain_pending(&self) {
        let pending = std::mem::take(&mut *self.pending.borrow_mut());
        if pending.is_empty() {
            return;
        }
        if self.profile.get() {
            self.drain_profiled(&pending);
            return;
        }
        #[cfg(target_arch = "wasm32")]
        {
            self.drain_pending_wasm(&pending);
            return;
        }
        let mut encoder = self.encoder();
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("tract-wgpu"),
            timestamp_writes: None,
        });
        for d in &pending {
            pass.set_pipeline(&d.pipeline);
            pass.set_bind_group(0, &*d.bind_group, &[d.dynamic_offset]);
            pass.dispatch_workgroups(d.groups[0], d.groups[1], d.groups[2]);
        }
    }

    #[cfg(target_arch = "wasm32")]
    fn drain_pending_wasm(&self, pending: &[Dispatch]) {
        let mut bytes = vec![0u8; pending.len() * 24];
        for (i, d) in pending.iter().enumerate() {
            let o = i * 24;
            bytes[o..o + 4].copy_from_slice(&d.pipe_id.to_le_bytes());
            bytes[o + 4..o + 8].copy_from_slice(&d.bg_id.to_le_bytes());
            bytes[o + 8..o + 12].copy_from_slice(&d.dynamic_offset.to_le_bytes());
            bytes[o + 12..o + 16].copy_from_slice(&d.groups[0].to_le_bytes());
            bytes[o + 16..o + 20].copy_from_slice(&d.groups[1].to_le_bytes());
            bytes[o + 20..o + 24].copy_from_slice(&d.groups[2].to_le_bytes());
        }
        let mut encoder = self.encoder();
        {
            let _pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("tract-wgpu"),
                timestamp_writes: None,
            });
            crate::wasm_record::tract_record_pass(&bytes);
        }
    }

    /// Records `copy_buffer_to_buffer` behind the pending work and flushes the
    /// lot in one submission.
    pub fn flush_after_copy(
        &self,
        src: &wgpu::Buffer,
        offset: u64,
        dst: &wgpu::Buffer,
        copy_len: u64,
    ) -> TractResult<()> {
        self.upload_uniforms();
        self.drain_pending();
        self.encoder().copy_buffer_to_buffer(src, offset, dst, 0, copy_len);
        self.flush()
    }

    fn drain_profiled(&self, pending: &[Dispatch]) {
        let slot = self.queries.borrow();
        let Some(queries) = slot.as_ref() else { return };
        let mut profiled = self.profiled.borrow_mut();
        let mut encoder = self.encoder();
        for d in pending {
            let base = profiled.len() as u32 * 2;
            if base + 2 > queries.capacity {
                break;
            }
            profiled.push((d.label, base));
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some(d.label),
                timestamp_writes: Some(wgpu::ComputePassTimestampWrites {
                    query_set: &queries.set,
                    beginning_of_pass_write_index: Some(base),
                    end_of_pass_write_index: Some(base + 1),
                }),
            });
            pass.set_pipeline(&d.pipeline);
            pass.set_bind_group(0, &*d.bind_group, &[d.dynamic_offset]);
            pass.dispatch_workgroups(d.groups[0], d.groups[1], d.groups[2]);
        }
    }

    pub fn flush(&self) -> TractResult<()> {
        self.upload_uniforms();
        self.drain_pending();
        if let Some(encoder) = self.encoder.replace(None) {
            self.context.queue().submit(Some(encoder.finish()));
        }
        self.context.poll_wait()?;
        self.retained.borrow_mut().clear();
        self.retained_textures.borrow_mut().clear();
        self.uniform_cursor.set(0);
        self.context.trim_bind_groups();
        self.slot_cursor.borrow_mut().clear();
        Ok(())
    }

    fn copy_to_staging(
        &self,
        buffer: &wgpu::Buffer,
        offset: u64,
        copy_len: u64,
    ) -> TractResult<()> {
        {
            let mut staging = self.staging.borrow_mut();
            let need_new = staging.as_ref().map(|b| b.size() < copy_len).unwrap_or(true);
            if need_new {
                *staging = Some(self.context.device().create_buffer(&wgpu::BufferDescriptor {
                    label: Some("tract-wgpu-staging"),
                    size: copy_len.max(4),
                    usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                }));
            }
        }
        let staging = self.staging.borrow();
        let staging = staging.as_ref().unwrap();
        let mut encoder =
            self.context.device().create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("tract-wgpu-download"),
            });
        encoder.copy_buffer_to_buffer(buffer, offset, staging, 0, copy_len);
        self.context.queue().submit(Some(encoder.finish()));
        Ok(())
    }

    fn read_mapped_staging(&self, copy_len: u64, len: u64) -> TractResult<Vec<u8>> {
        let staging = self.staging.borrow();
        let staging = staging.as_ref().unwrap();
        let slice = staging.slice(..copy_len);
        let data = slice.get_mapped_range()?;
        let out = data[..len as usize].to_vec();
        drop(data);
        staging.unmap();
        Ok(out)
    }

    /// Blocking readback. On wasm without JSPI this errors: `PollType::Wait`
    /// is a no-op. With `--features jspi`, [`crate::tensor::WgpuTensor::to_host`]
    /// suspends via JSPI instead of calling this. Use [`Self::download_async`]
    /// from an `async` entry.
    pub fn download(&self, buffer: &wgpu::Buffer, offset: u64, len: u64) -> TractResult<Vec<u8>> {
        #[cfg(target_arch = "wasm32")]
        {
            let _ = (buffer, offset, len);
            bail!(
                "blocking GPU readback is not possible on web (PollType::Wait is a no-op). \
                 After run(), await download_async / to_host_async once."
            )
        }
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.flush()?;
            if len == 0 {
                return Ok(vec![]);
            }
            let copy_len = len.next_multiple_of(4);
            self.copy_to_staging(buffer, offset, copy_len)?;
            let staging = self.staging.borrow();
            let staging = staging.as_ref().unwrap();
            let slice = staging.slice(..copy_len);
            let (tx, rx) = std::sync::mpsc::channel();
            slice.map_async(wgpu::MapMode::Read, move |r| {
                let _ = tx.send(r);
            });
            let _ = staging;
            self.context.poll_wait()?;
            rx.recv().context("staging map channel")??;
            self.read_mapped_staging(copy_len, len)
        }
    }

    /// One await at the end of a run. Native path still blocks inside; web
    /// yields to the browser so `mapAsync` can complete.
    pub async fn download_async(
        &self,
        buffer: &wgpu::Buffer,
        offset: u64,
        len: u64,
    ) -> TractResult<Vec<u8>> {
        #[cfg(not(target_arch = "wasm32"))]
        {
            self.download(buffer, offset, len)
        }
        #[cfg(target_arch = "wasm32")]
        {
            self.flush()?;
            if len == 0 {
                return Ok(vec![]);
            }
            let copy_len = len.next_multiple_of(4);
            self.copy_to_staging(buffer, offset, copy_len)?;
            let staging = self.staging.borrow().as_ref().unwrap().clone();
            let slice = staging.slice(..copy_len);
            let promise = js_sys::Promise::new(&mut |resolve, reject| {
                slice.map_async(wgpu::MapMode::Read, move |r| match r {
                    Ok(()) => {
                        let _ = resolve.call0(&wasm_bindgen::JsValue::UNDEFINED);
                    }
                    Err(e) => {
                        let _ = reject.call1(
                            &wasm_bindgen::JsValue::UNDEFINED,
                            &wasm_bindgen::JsValue::from_str(&format!("{e}")),
                        );
                    }
                });
            });
            wasm_bindgen_futures::JsFuture::from(promise)
                .await
                .map_err(|e| anyhow!("mapAsync: {e:?}"))?;
            self.read_mapped_staging(copy_len, len)
        }
    }
}

impl Drop for WgpuQueue {
    fn drop(&mut self) {
        // Do not flush or drop wgpu buffers here: this type lives in TLS and
        // wgpu-core buffer drop takes a TLS snatch lock.
        self.leak_gpu_objects();
    }
}

/// Identifies one blocked-matmul weight packing: the buffer it was read from,
/// where in that buffer, and the shape it was written as.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct RepackKey {
    pub buffer: u64,
    pub offset: usize,
    pub k: usize,
    pub n: usize,
}

#[derive(Clone)]
/// The identity a bind group is keyed on. It follows the buffer through the
/// pool, so a graph that reuses its buffers reuses its bind groups; keying on
/// the `wgpu::Buffer` itself would keep every buffer alive in the cache and
/// leave the pool nothing to hand out.
pub struct WgpuBuffer {
    pub inner: wgpu::Buffer,
    pub id: u64,
    /// Whether this buffer came from the frame pool, which hands it to the next
    /// tensor that fits once nothing holds it. Its `id` is stable only while it
    /// lives, so a cache must not key on a pooled buffer.
    pub pooled: bool,
}

impl std::fmt::Debug for WgpuBuffer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("WgpuBuffer").field("size", &self.inner.size()).finish()
    }
}

impl DeviceBuffer for WgpuBuffer {
    fn ptr(&self) -> *const c_void {
        // Opaque identity of the wgpu buffer handle. WebGPU has no device
        // address; launch functions downcast `WgpuBuffer` instead of using this.
        std::ptr::from_ref(&self.inner) as *const c_void
    }
}
