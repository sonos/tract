use crate::func_constants::ConstantValues;
pub use crate::kernels::{LibraryContent, LibraryName};
pub use crate::tensor::MetalTensor;
use metal::Buffer;
use metal::MTLResourceOptions;
use metal::NSUInteger;
use std::cell::RefCell;
use std::path::Path;
use std::sync::Arc;
use std::sync::{OnceLock, RwLock};

use anyhow::{anyhow, Context, Result};
use metal::{
    CommandBuffer, CommandQueue, CompileOptions, ComputePipelineState, Device, Function,
    FunctionConstantValues, Library,
};
use std::collections::HashMap;

thread_local! {
    pub static METAL_CONTEXT: RefCell<MetalContext> = RefCell::new(MetalContext::new());
}

fn shared_metal_context() -> SharedMetalContext {
    static INSTANCE: OnceLock<SharedMetalContext> = OnceLock::new();
    INSTANCE
        .get_or_init(|| SharedMetalContext::new().expect("Could not create shared metal context"))
        .clone()
}

#[derive(Debug, Clone)]
pub struct SharedMetalContext {
    device: Device,
    cache_libraries: Arc<RwLock<HashMap<LibraryName, Library>>>,
    #[allow(clippy::type_complexity)]
    cache_pipelines:
        Arc<RwLock<HashMap<(LibraryName, String, Option<ConstantValues>), ComputePipelineState>>>,
}

impl SharedMetalContext {
    pub fn new() -> Result<Self> {
        let device = Device::system_default()
            .with_context(|| "Could not find system default Metal device")?;
        Ok(Self {
            device,
            cache_libraries: Arc::new(RwLock::new(HashMap::new())),
            cache_pipelines: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    pub fn flush_pipeline_cache(&self) -> Result<()> {
        self.cache_pipelines.write().map_err(|e| anyhow!("{:?}", e))?.clear();
        Ok(())
    }

    pub fn load_library(&self, name: LibraryName) -> Result<Library> {
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
                    anyhow!("Error while loading Metal library from data: {:?}", name)
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
    ) -> Result<Function> {
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
    ) -> Result<ComputePipelineState> {
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
    ) -> Result<ComputePipelineState> {
        self.load_pipeline_with_constants(library_name, func_name, None)
    }
}

#[derive(Debug)]
pub struct MetalContext {
    shared: SharedMetalContext,
    command_queue: CommandQueue,
    command_buffer: RwLock<CommandBuffer>,
    command_buffer_idx: RwLock<usize>,
    command_buffer_capacity: usize,
}

impl Default for MetalContext {
    fn default() -> Self {
        Self::new()
    }
}

impl MetalContext {
    pub fn new() -> Self {
        let shared = shared_metal_context();
        let command_queue = shared.device.new_command_queue();
        let command_buffer = command_queue.new_command_buffer().to_owned();
        command_buffer.enqueue();
        Self {
            shared,
            command_queue,
            command_buffer: RwLock::new(command_buffer),
            command_buffer_idx: RwLock::new(0),
            command_buffer_capacity: 10,
        }
    }

    pub fn device(&self) -> &Device {
        &self.shared.device
    }

    pub fn shared_context(&self) -> &SharedMetalContext {
        &self.shared
    }

    pub fn buffer_from_slice<T>(&self, data: &[T]) -> Buffer {
        let size = core::mem::size_of_val(data) as NSUInteger;
        self.device().new_buffer_with_bytes_no_copy(
            data.as_ptr() as *const core::ffi::c_void,
            size,
            MTLResourceOptions::StorageModeShared,
            None,
        )
    }

    pub fn buffer_from_slice_mut<T>(&self, data: &mut [T]) -> Buffer {
        let size = core::mem::size_of_val(data) as NSUInteger;
        self.device().new_buffer_with_bytes_no_copy(
            data.as_ptr() as *const core::ffi::c_void,
            size,
            MTLResourceOptions::StorageModeShared,
            None,
        )
    }

    pub fn command_buffer(&self) -> Result<CommandBuffer> {
        let mut self_command_buffer =
            self.command_buffer.try_write().map_err(|e| anyhow!("{:?}", e))?;
        let mut command_buffer = self_command_buffer.to_owned();

        let mut command_buffer_idx =
            self.command_buffer_idx.try_write().map_err(|e| anyhow!("{:?}", e))?;

        if *command_buffer_idx > self.command_buffer_capacity {
            *command_buffer_idx = 0;
            command_buffer.commit();
            command_buffer = self.command_queue.new_command_buffer().to_owned();
            *self_command_buffer = command_buffer.clone();
            Ok(command_buffer.to_owned())
        } else {
            *command_buffer_idx += 1;
            Ok(command_buffer)
        }
    }

    pub fn wait_until_completed(&self) -> Result<()> {
        let mut command_buffer = self.command_buffer.try_write().map_err(|e| anyhow!("{:?}", e))?;

        match command_buffer.status() {
            metal::MTLCommandBufferStatus::Committed
            | metal::MTLCommandBufferStatus::Scheduled
            | metal::MTLCommandBufferStatus::Completed => {
                anyhow::bail!("Current Metal command buffer is already committed.")
            }
            _ => {}
        }
        command_buffer.commit();
        command_buffer.wait_until_completed();
        *command_buffer = self.command_queue.new_command_buffer().to_owned();
        Ok(())
    }

    pub fn capture_trace<P, F>(&self, path: P, compute: F) -> Result<()>
    where
        P: AsRef<Path>,
        F: Fn(&Self) -> Result<()>,
    {
        self.wait_until_completed()?;

        let capture = metal::CaptureManager::shared();
        let descriptor = metal::CaptureDescriptor::new();
        descriptor.set_destination(metal::MTLCaptureDestination::GpuTraceDocument);
        descriptor.set_capture_device(self.device());
        descriptor.set_output_url(path);

        capture.start_capture(&descriptor).map_err(|e| anyhow!("{:?}", e))?;

        (compute)(self)?;

        self.wait_until_completed()?;
        capture.stop_capture();
        Ok(())
    }
}

impl Drop for MetalContext {
    fn drop(&mut self) {
        let command_buffer =
            self.command_buffer.try_write().map_err(|e| anyhow!("{:?}", e)).unwrap();

        match command_buffer.status() {
            metal::MTLCommandBufferStatus::Committed
            | metal::MTLCommandBufferStatus::Scheduled
            | metal::MTLCommandBufferStatus::Completed => {
                panic!("Current Metal command buffer is already committed.")
            }
            _ => {}
        }
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }
}
