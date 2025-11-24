use crate::command_buffer::TCommandBuffer;
use crate::func_constants::ConstantValues;
use crate::kernels::{LibraryContent, LibraryName};
use crate::tensor::{MValue, MetalTensor};

use metal::NSUInteger;
use tract_core::tract_linalg::block_quant::{BlockQuantFact, BlockQuantValue};
use tract_gpu::device::{DeviceBuffer, DeviceContext};
use tract_gpu::tensor::{DeviceTensor, OwnedDeviceTensor};
use tract_gpu::utils::as_q40_tensor;

use std::alloc::Layout;
use std::cell::RefCell;
use std::ffi::c_void;
use std::ops::{Deref, DerefMut};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock, RwLock};

use anyhow::{Context, anyhow};
use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, Function,
    FunctionConstantValues, Library, MTLResourceOptions,
};
use std::collections::HashMap;
use tract_core::internal::*;

thread_local! {
    pub static METAL_STREAM: RefCell<MetalStream> = RefCell::new(MetalStream::new());
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
        for ew_func in crate::kernels::ElementWiseOps::all_functions() {
            let _ = self.load_pipeline(LibraryName::ElementWiseOps, &ew_func);
        }
        for bin_func in crate::kernels::BinOps::all_functions() {
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
        METAL_STREAM.with_borrow(|stream| stream.wait_until_completed())
    }

    fn tensor_to_device(&self, tensor: TValue) -> TractResult<Box<dyn OwnedDeviceTensor>> {
        let view = tensor.view();
        ensure!(
            DeviceTensor::is_supported_dt(view.datum_type()),
            "Tensor of {:?} is not copied. No device buffer can be allocated for it.",
            view.datum_type(),
        );
        let bqv = as_q40_tensor(view.tensor);

        let (data_bytes, bqf) = bqv
            .map(|bqv| (bqv.value.as_bytes(), Some(bqv.fact.clone().into())))
            .unwrap_or((view.tensor.as_bytes(), None));

        // Handle empty data
        static ZERO: [u8; 1] = [0];
        let data = if data_bytes.is_empty() { &ZERO } else { data_bytes };

        let size = core::mem::size_of_val(data) as NSUInteger;
        let device_buffer = MetalBuffer {
            inner: self.device.new_buffer_with_bytes_no_copy(
                data.as_ptr() as *const core::ffi::c_void,
                size,
                MTLResourceOptions::StorageModeShared,
                None,
            ),
        };

        Ok(Box::new(MetalTensor {
            inner: MValue::Natural(tensor.into_arc_tensor()),
            device_buffer,
            opaque_fact: bqf,
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

    fn uninitialized_device_opaque_tensor(
        &self,
        opaque_fact: Box<dyn OpaqueFact>,
    ) -> TractResult<Box<dyn OwnedDeviceTensor>> {
        if let Some(bqf) = opaque_fact.downcast_ref::<BlockQuantFact>() {
            let blocks = bqf.shape().iter().product::<usize>() / bqf.format.block_len();
            let blob = unsafe {
                Blob::for_layout(
                    Layout::from_size_align(blocks * bqf.format.block_bytes(), vector_size())
                        .unwrap(),
                )
            };
            let value = BlockQuantValue { fact: bqf.clone(), value: Arc::new(blob) };
            let tensor = tensor0(Opaque(Arc::new(value)));
            self.tensor_to_device(tensor.into())
        } else {
            bail!("Only BlockQuant Tensor allocation supported for now")
        }
    }
}

#[derive(Debug)]
pub struct MetalStream {
    context: MetalContext,
    command_queue: CommandQueue,
    command_buffer: RefCell<Option<TCommandBuffer>>,
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

    pub fn wait_until_completed(&self) -> TractResult<()> {
        let Some(command_buffer) = self.command_buffer.borrow().to_owned() else { return Ok(()) };

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

#[derive(Debug, Clone)]
pub struct MetalBuffer {
    pub inner: Buffer,
}

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
