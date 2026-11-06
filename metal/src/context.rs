use crate::command_buffer::TCommandBuffer;
use crate::func_constants::ConstantValues;
use crate::kernels::{LibraryContent, LibraryName};
use crate::profile::{MetalProfile, MetalProfileHandle};
use crate::tensor::{MValue, MetalTensor};

use metal::NSUInteger;
use tract_core::tract_linalg::block_quant::{BlockQuantFact, BlockQuantStorage};
use tract_gpu::device::{DeviceBuffer, DeviceContext};
use tract_gpu::tensor::{DeviceTensor, OwnedDeviceTensor};

use std::alloc::Layout;
use std::cell::RefCell;
use std::ffi::c_void;
use std::ops::{Deref, DerefMut};
use std::path::Path;
use std::rc::Rc;
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
}

impl MetalContext {
    pub fn new() -> TractResult<Self> {
        let device = Device::system_default()
            .with_context(|| "Could not find system default Metal device")?;

        let ctxt = Self {
            device,
            cache_libraries: Arc::new(RwLock::new(HashMap::new())),
            cache_pipelines: Arc::new(RwLock::new(HashMap::new())),
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
        let start = std::time::Instant::now();
        let func = self.load_function(
            library_name,
            &func_name,
            constants.as_ref().map(|c| c.function_constant_values()),
        )?;
        let descriptor = metal::ComputePipelineDescriptor::new();
        descriptor.set_compute_function(Some(&func));
        descriptor.set_label(&func_name);
        let pipeline = self.device
            .new_compute_pipeline_state(&descriptor)
            .map_err(|e| anyhow!("{}", e))
            .with_context(|| format!("Error while creating compute pipeline for function {func_name} from source: {:?}", library_name))?;
        log::trace!(
            "pipeline created: {:?}::{func_name} in {:.2} ms",
            library_name,
            start.elapsed().as_secs_f64() * 1e3
        );
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
        let tensor = tensor.into_arc_tensor();
        let view = tensor.view();
        ensure!(
            DeviceTensor::is_supported_dt(view.datum_type()),
            "Tensor of {:?} is not copied. No device buffer can be allocated for it.",
            view.datum_type(),
        );
        let bqs = view
            .tensor
            .storage_as::<BlockQuantStorage>()
            .filter(|bqs| crate::kernels::matmul::GgmlQuant::of(bqs.format()).is_some());

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
        // Metal owns the captured storage until its last reference to the no-copy buffer drops.
        let storage = tensor.clone();
        let deallocator = block::ConcreteBlock::new(move |_: *const c_void, _: NSUInteger| {
            drop(storage.clone());
        })
        .copy();
        let device_buffer = MetalBuffer {
            inner: self.device.new_buffer_with_bytes_no_copy(
                data.as_ptr() as *const core::ffi::c_void,
                size,
                MTLResourceOptions::StorageModeShared,
                Some(&deallocator),
            ),
        };

        Ok(Box::new(MetalTensor {
            inner: MValue::Natural(tensor),
            device_buffer,
            exotic_fact: bqf,
        }))
    }

    fn uninitialized_device_tensor(
        &self,
        shape: &[usize],
        dt: DatumType,
    ) -> TractResult<Box<dyn OwnedDeviceTensor>> {
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

/// A committed command buffer and the tensors retained until it completes.
#[derive(Debug)]
struct InFlightBuffer {
    buffer: TCommandBuffer,
    retained: Vec<DeviceTensor>,
}

/// State accumulated for the next command buffer submission.
#[derive(Debug, Default)]
struct OpenCommandBuffer {
    buffer: Option<TCommandBuffer>,
    retained: Vec<DeviceTensor>,
}

#[derive(Debug)]
pub struct MetalStream {
    context: MetalContext,
    command_queue: CommandQueue,
    open: RefCell<OpenCommandBuffer>,
    committed_command_buffers: RefCell<VecDeque<InFlightBuffer>>,
    command_error: RefCell<Option<String>>,
    dispatch_tensors: RefCell<Vec<DeviceTensor>>,
    commit_every_n_dispatches: usize,
    max_in_flight: usize,
    profile: RefCell<Option<Rc<MetalProfileHandle>>>,
}

/// Retains raw-buffer dispatch owners on every command buffer acquired in this scope.
/// Scopes must be dropped in reverse creation order.
pub(crate) struct DispatchTensorScope<'a> {
    tensors: &'a RefCell<Vec<DeviceTensor>>,
    previous_len: usize,
}

impl Drop for DispatchTensorScope<'_> {
    fn drop(&mut self) {
        self.tensors.borrow_mut().truncate(self.previous_len);
    }
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
            open: RefCell::new(OpenCommandBuffer::default()),
            committed_command_buffers: RefCell::new(VecDeque::new()),
            command_error: RefCell::new(None),
            dispatch_tensors: RefCell::new(vec![]),
            commit_every_n_dispatches: crate::tuning::TRACT_METAL_COMMIT_EVERY_N_DISPATCHES.get(),
            max_in_flight: crate::tuning::TRACT_METAL_MAX_IN_FLIGHT.get().max(1),
            profile: RefCell::new(None),
        }
    }

    /// Time every dispatch from here on, each against the node that asked for
    /// it. A profiled turn runs a pass per dispatch, so it is slower than the
    /// turn it measures.
    pub fn enable_profiling(&self) -> TractResult<()> {
        let profile = MetalProfile::new(&self.context.device)?;
        *self.profile.borrow_mut() =
            Some(Rc::new(MetalProfileHandle { profile: RefCell::new(profile), node_id: 0.into() }));
        Ok(())
    }

    pub fn is_profiling(&self) -> bool {
        self.profile.borrow().is_some()
    }

    /// Which node the dispatches that follow belong to.
    pub fn set_current_node(&self, node_id: usize) {
        if let Some(handle) = self.profile.borrow().as_ref() {
            handle.node_id.set(node_id);
        }
    }

    /// What each dispatch spent on the device since the last drain. Call once
    /// the work has completed.
    pub fn drain_profile(&self) -> TractResult<Vec<(usize, std::time::Duration)>> {
        let handle = self.profile.borrow().clone();
        let Some(handle) = handle else { return Ok(vec![]) };
        handle.profile.borrow_mut().drain(&self.context.device)
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

    /// Keep raw-buffer kernel owners on every buffer acquired until the scope ends.
    pub(crate) fn retain_for_dispatch(&self, tensors: &[&DeviceTensor]) -> DispatchTensorScope<'_> {
        let mut retained = self.dispatch_tensors.borrow_mut();
        let previous_len = retained.len();
        retained.extend(tensors.iter().map(|t| (*t).clone()));
        DispatchTensorScope { tensors: &self.dispatch_tensors, previous_len }
    }

    pub fn retain_tensor(&self, tensor: &DeviceTensor) {
        self.open.borrow_mut().retained.push(tensor.clone());
    }

    /// Return the current buffer, submitting it first if its launch count reached the limit.
    /// Submission happens at acquisition boundaries; an existing handle can encode more launches.
    pub fn command_buffer(&self) -> TCommandBuffer {
        let should_commit =
            self.commit_every_n_dispatches > 0
                && self.open.borrow().buffer.as_ref().is_some_and(|buffer| {
                    buffer.dispatch_count() >= self.commit_every_n_dispatches
                });
        if should_commit && let Err(error) = self.commit_current() {
            self.command_error.borrow_mut().get_or_insert_with(|| error.to_string());
        }
        let mut open = self.open.borrow_mut();
        let profile = self.profile.borrow().clone();
        let buffer = open
            .buffer
            .get_or_insert_with(|| {
                // Keep Rust's owned reference without leaving temporary
                // ownership in the caller's (possibly absent) autorelease pool.
                objc::rc::autoreleasepool(|| {
                    TCommandBuffer::new(self.command_queue.new_command_buffer().to_owned(), profile)
                })
            })
            .to_owned();
        for tensor in self.dispatch_tensors.borrow().iter() {
            buffer.retain_tensor(tensor);
        }
        buffer
    }

    fn log_completed_kernels(tag: &str, names: &[String]) {
        log::trace!("completed Metal command buffer {tag}: {names:?}");
    }

    fn record_command_error(&self, buffer: &TCommandBuffer) {
        if buffer.status() == metal::MTLCommandBufferStatus::Error {
            self.command_error.borrow_mut().get_or_insert_with(|| {
                format!("Metal command buffer failed on the GPU: {}", command_buffer_error(buffer))
            });
        }
    }

    fn check_command_error(&self) -> TractResult<()> {
        if let Some(error) = self.command_error.borrow().as_ref() {
            bail!("{error}");
        }
        Ok(())
    }

    /// Commit the current command buffer, waiting for older work if needed to
    /// make room within the in-flight limit. The next `command_buffer()` call
    /// opens a fresh one, ordered after this buffer by the queue. Tensors
    /// retained so far move into the in-flight entry and are released once
    /// that buffer completes. GPU failures remain visible until `wait_until_completed`
    /// drains the stream and reports them.
    pub fn commit_current(&self) -> TractResult<()> {
        let Some(command_buffer) = self.open.borrow().buffer.clone() else {
            return self.check_command_error();
        };
        match command_buffer.status() {
            metal::MTLCommandBufferStatus::Committed
            | metal::MTLCommandBufferStatus::Scheduled
            | metal::MTLCommandBufferStatus::Completed => {
                anyhow::bail!("Current Metal command buffer is already committed.")
            }
            _ => {}
        }
        command_buffer.end_encoding();
        let mut committed = self.committed_command_buffers.borrow_mut();
        loop {
            let done = committed.front().is_some_and(|e| {
                matches!(
                    e.buffer.status(),
                    metal::MTLCommandBufferStatus::Completed | metal::MTLCommandBufferStatus::Error
                )
            });
            if !done && committed.len() < self.max_in_flight {
                break;
            }
            let oldest = committed.pop_front().unwrap();
            oldest.buffer.wait_until_completed();
            self.record_command_error(&oldest.buffer);
            Self::log_completed_kernels("segment", &oldest.buffer.trace.borrow().kernel_names);

            drop(oldest.retained);
        }
        command_buffer.commit();
        let open = std::mem::take(&mut *self.open.borrow_mut());
        committed.push_back(InFlightBuffer { buffer: command_buffer, retained: open.retained });
        self.check_command_error()
    }

    /// Wait for all submitted work, then report and clear the recorded GPU failure.
    /// A subsequent run starts without the previous run's error.
    pub fn wait_until_completed(&self) -> TractResult<()> {
        let submitted = self.commit_current();
        let drained = std::mem::take(&mut *self.committed_command_buffers.borrow_mut());
        if let Some(newest) = drained.back() {
            newest.buffer.wait_until_completed();
        }
        for entry in drained {
            self.record_command_error(&entry.buffer);
            Self::log_completed_kernels("segment-tail", &entry.buffer.trace.borrow().kernel_names);
        }
        *self.open.borrow_mut() = OpenCommandBuffer::default();
        let command_error = self.command_error.borrow_mut().take();
        submitted?;
        if let Some(error) = command_error {
            bail!("{error}");
        }
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
        let _ = self.wait_until_completed();
    }
}

#[derive(Debug, Clone)]
pub struct MetalBuffer {
    pub inner: Buffer,
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
/// The description of the `NSError` a failed command buffer carries, which the
/// `metal` crate does not expose.
// The objc msg_send!/sel! macros expand to a cargo-clippy cfg check that older
// toolchains report at the call site; the allow covers it.
#[allow(unexpected_cfgs)]
fn command_buffer_error(command_buffer: &metal::CommandBufferRef) -> String {
    use metal::foreign_types::ForeignTypeRef;
    use objc::runtime::Object;
    use objc::{msg_send, sel, sel_impl};

    unsafe {
        let command_buffer = command_buffer.as_ptr() as *mut Object;
        let error: *mut Object = msg_send![command_buffer, error];
        if error.is_null() {
            return "no error reported".to_string();
        }
        let description: *mut Object = msg_send![error, description];
        let chars: *const std::os::raw::c_char = msg_send![description, UTF8String];
        if chars.is_null() {
            return "an error without a description".to_string();
        }
        std::ffi::CStr::from_ptr(chars).to_string_lossy().into_owned()
    }
}

#[cfg(test)]
mod submission_tests {
    use super::*;

    #[test]
    fn submission_traces_consuming_buffer_launches() -> TractResult<()> {
        use tract_gpu::tensor::IntoDevice;

        for profile in [false, true] {
            let mut stream = MetalStream::new();
            stream.commit_every_n_dispatches = 1;
            if profile {
                if !stream
                    .context
                    .device
                    .supports_counter_sampling(metal::MTLCounterSamplingPoint::AtStageBoundary)
                {
                    continue;
                }
                stream.enable_profiling()?;
            }
            let input = tensor1(&[1u32, 2, 3, 4]).into_device()?;
            let output = DeviceTensor::uninitialized_dt(DatumType::U32, &[4])?;
            let name_a = "array_ops::copy_unicast_u32";
            let name_b = "array_ops::copy_unicast_u16";
            let a = stream.load_pipeline(LibraryName::ArrayOps, name_a)?;
            let first = stream.command_buffer();
            first.encode(|encoder| encoder.set_compute_pipeline_state(&a));
            assert!(first.trace.borrow().kernel_names.is_empty());
            first.encode(|encoder| {
                encoder.set_compute_pipeline_state(&a);
                encoder.set_metal_tensor(0, &input, metal::MTLResourceUsage::Read);
                encoder.set_metal_tensor(1, &output, metal::MTLResourceUsage::Write);
                encoder
                    .dispatch_threads(metal::MTLSize::new(4, 1, 1), metal::MTLSize::new(1, 1, 1));
                encoder.dispatch_thread_groups(
                    metal::MTLSize::new(4, 1, 1),
                    metal::MTLSize::new(1, 1, 1),
                );
            });
            let b = stream.load_pipeline_with_constants(
                LibraryName::ArrayOps,
                name_b,
                Some(ConstantValues::new(vec![])),
            )?;
            stream.load_pipeline(LibraryName::ArrayOps, "array_ops::copy_unicast_u8")?;
            let second = stream.command_buffer();
            assert_eq!(first.trace.borrow().kernel_names, [name_a, name_a]);
            assert!(second.trace.borrow().kernel_names.is_empty());
            second.encode(|encoder| {
                encoder.set_compute_pipeline_state(&b);
                encoder.set_metal_tensor(0, &input, metal::MTLResourceUsage::Read);
                encoder.set_metal_tensor(1, &output, metal::MTLResourceUsage::Write);
                encoder
                    .dispatch_threads(metal::MTLSize::new(8, 1, 1), metal::MTLSize::new(1, 1, 1));
            });
            stream.wait_until_completed()?;
            assert_eq!(first.trace.borrow().kernel_names, [name_a, name_a]);
            assert_eq!(second.trace.borrow().kernel_names, [name_b]);
            output.to_host()?.close_enough(&tensor1(&[1u32, 2, 3, 4]), Approximation::Exact)?;
        }
        Ok(())
    }

    #[test]
    fn submission_waits_before_exceeding_in_flight_limit() -> TractResult<()> {
        use std::sync::mpsc;
        use std::time::Duration;

        for limit in [1, 2, 8] {
            let (ready_tx, ready_rx) = mpsc::channel();
            let (done_tx, done_rx) = mpsc::channel();
            let worker = std::thread::spawn(move || -> TractResult<()> {
                let mut stream = MetalStream::new();
                stream.max_in_flight = limit;
                let gate = stream.context.device.new_shared_event();
                let first = stream.command_buffer();
                first.encode_wait_for_event(&gate, 1);
                stream.commit_current()?;
                for _ in 1..limit {
                    stream.command_buffer();
                    stream.commit_current()?;
                }
                let next = stream.command_buffer();
                ready_tx.send((gate, std::ops::Deref::deref(&next).clone()))?;
                let result = stream.commit_current();
                done_tx.send(())?;
                result?;
                stream.wait_until_completed()?;
                Ok(())
            });
            let (gate, next) = ready_rx.recv_timeout(Duration::from_secs(10))?;
            let early_return = done_rx.recv_timeout(Duration::from_millis(250));
            let status_before_release = next.status();
            gate.set_signaled_value(1);
            let result = worker.join().expect("submission worker panicked");
            result?;
            assert!(matches!(early_return, Err(mpsc::RecvTimeoutError::Timeout)));
            assert_eq!(status_before_release, metal::MTLCommandBufferStatus::NotEnqueued);
            assert_eq!(next.status(), metal::MTLCommandBufferStatus::Completed);
        }
        Ok(())
    }

    #[test]
    fn submission_retains_consuming_tensor_views() -> TractResult<()> {
        use crate::kernels::array::Memcpy;
        use crate::utils::get_metal_buffer;
        use tract_gpu::tensor::IntoDevice;

        for profile in [false, true] {
            for raw_buffers in [false, true] {
                let mut stream = MetalStream::new();
                stream.commit_every_n_dispatches = 1;
                stream.max_in_flight = 1;
                if profile {
                    if !stream
                        .context
                        .device
                        .supports_counter_sampling(metal::MTLCounterSamplingPoint::AtStageBoundary)
                    {
                        continue;
                    }
                    stream.enable_profiling()?;
                }
                let input = tensor1(&[1u32, 2, 3, 4]).into_device()?.prefix_window(0, 4)?;
                let output =
                    DeviceTensor::uninitialized_dt(DatumType::U32, &[4])?.prefix_window(0, 4)?;
                let pipeline =
                    stream.load_pipeline(LibraryName::ArrayOps, "array_ops::copy_unicast_u32")?;
                let retained = raw_buffers.then(|| stream.retain_for_dispatch(&[&input, &output]));
                for _ in 0..3 {
                    if raw_buffers {
                        stream.command_buffer().encode(|encoder| {
                            encoder.set_compute_pipeline_state(&pipeline);
                            encoder.set_buffer(0, Some(get_metal_buffer(&input)), 0);
                            encoder.set_buffer(1, Some(get_metal_buffer(&output)), 0);
                            encoder.dispatch_threads(
                                metal::MTLSize::new(4, 1, 1),
                                metal::MTLSize::new(1, 1, 1),
                            );
                        });
                    } else {
                        Memcpy.dispatch_eval(&stream, &input, 0, &output)?;
                    }
                    let open = stream.open.borrow();
                    let buffer = open.buffer.as_ref().unwrap();
                    assert!(buffer.retains_buffer_of(&input));
                    assert!(buffer.retains_buffer_of(&output));
                }
                drop(retained);
                assert!(stream.dispatch_tensors.borrow().is_empty());
                drop(input);
                stream.wait_until_completed()?;
                output.to_host()?.close_enough(&tensor1(&[1u32, 2, 3, 4]), Approximation::Exact)?;
            }
        }
        Ok(())
    }

    #[test]
    fn submission_counts_kernel_launches() -> TractResult<()> {
        use tract_gpu::tensor::IntoDevice;

        for profile in [false, true] {
            let mut stream = MetalStream::new();
            stream.commit_every_n_dispatches = 2;
            stream.max_in_flight = 1;
            if profile {
                if !stream
                    .context
                    .device
                    .supports_counter_sampling(metal::MTLCounterSamplingPoint::AtStageBoundary)
                {
                    continue;
                }
                stream.enable_profiling()?;
            }
            let input = tensor1(&[1u32, 2, 3, 4]).into_device()?;
            let output = DeviceTensor::uninitialized_dt(DatumType::U32, &[4])?;
            let pipeline =
                stream.load_pipeline(LibraryName::ArrayOps, "array_ops::copy_unicast_u32")?;
            let first = stream.command_buffer();
            for _ in 0..4 {
                stream.command_buffer().encode(|encoder| {
                    encoder.set_compute_pipeline_state(&pipeline);
                });
            }
            assert_eq!(first.dispatch_count(), 0);
            assert_eq!(first.status(), metal::MTLCommandBufferStatus::NotEnqueued);
            let alias = first.clone();
            first.encode(|encoder| {
                encoder.set_metal_tensor(0, &input, metal::MTLResourceUsage::Read);
                encoder.set_metal_tensor(1, &output, metal::MTLResourceUsage::Write);
                encoder.set_compute_pipeline_state(&pipeline);
                encoder.dispatch_thread_groups(
                    metal::MTLSize::new(4, 1, 1),
                    metal::MTLSize::new(1, 1, 1),
                );
            });
            assert_eq!(alias.dispatch_count(), 1);
            let same = stream.command_buffer();
            assert_eq!(same.dispatch_count(), 1);
            assert_eq!(first.status(), metal::MTLCommandBufferStatus::NotEnqueued);
            same.encode(|encoder| {
                encoder.set_compute_pipeline_state(&pipeline);
                encoder.set_metal_tensor(0, &input, metal::MTLResourceUsage::Read);
                encoder.set_metal_tensor(1, &output, metal::MTLResourceUsage::Write);
                encoder
                    .dispatch_threads(metal::MTLSize::new(4, 1, 1), metal::MTLSize::new(1, 1, 1));
                encoder.dispatch_thread_groups(
                    metal::MTLSize::new(4, 1, 1),
                    metal::MTLSize::new(1, 1, 1),
                );
            });
            assert_eq!(alias.dispatch_count(), 3);
            assert_eq!(stream.command_buffer().dispatch_count(), 0);
            assert!(matches!(
                first.status(),
                metal::MTLCommandBufferStatus::Committed
                    | metal::MTLCommandBufferStatus::Scheduled
                    | metal::MTLCommandBufferStatus::Completed
            ));
            stream.wait_until_completed()?;
            output.to_host()?.close_enough(&tensor1(&[1u32, 2, 3, 4]), Approximation::Exact)?;
            assert_eq!(stream.command_buffer().dispatch_count(), 0);
            stream.wait_until_completed()?;
        }
        Ok(())
    }

    #[test]
    fn submission_clears_open_buffer_metadata() -> TractResult<()> {
        let stream = MetalStream::new();
        for wait in [false, true, false, true] {
            stream.load_pipeline(LibraryName::Ggml, "kernel_mul_mv_f32_f32")?;
            stream.command_buffer();
            {
                let open = stream.open.borrow();
                assert_eq!(open.buffer.as_ref().unwrap().dispatch_count(), 0);
                assert!(open.buffer.as_ref().unwrap().trace.borrow().kernel_names.is_empty());
            }
            if wait {
                stream.wait_until_completed()?;
            } else {
                stream.commit_current()?;
            }
            let open = stream.open.borrow();
            assert!(open.buffer.is_none());
            assert!(open.retained.is_empty());
        }
        stream.wait_until_completed()?;
        Ok(())
    }
}
