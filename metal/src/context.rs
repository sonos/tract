use crate::command_buffer::{MetalProfiler, TCommandBuffer};
use crate::func_constants::ConstantValues;
use crate::get_metal_device;
use crate::kernels::{LibraryContent, LibraryName};

use tract_gpu::tensor::DeviceTensor;
use metal::NSUInteger;
use tract_gpu::context::{DeviceBuffer, GpuDevice};
use std::cell::RefCell;
use std::ops::{Deref, DerefMut};
use std::path::Path;
use std::rc::Rc;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, OnceLock, RwLock};
use std::time::Duration;

use anyhow::{anyhow, Context, Result};
use metal::{
    Buffer, CommandQueue, CompileOptions, ComputePipelineState, Device, Function,
    FunctionConstantValues, Library, MTLResourceOptions,
};
use std::collections::HashMap;
use tract_core::internal::*;

thread_local! {
    pub static METAL_CONTEXT: RefCell<MetalContext> = RefCell::new(MetalContext::new());
}

fn metal_device() -> MetalDevice {
    static INSTANCE: OnceLock<MetalDevice> = OnceLock::new();
    INSTANCE
        .get_or_init(|| MetalDevice::new().expect("Could not create shared metal context"))
        .clone()
}

#[derive(Debug, Clone)]
pub struct MetalDevice {
    device: Device,
    cache_libraries: Arc<RwLock<HashMap<LibraryName, Library>>>,
    #[allow(clippy::type_complexity)]
    cache_pipelines:
        Arc<RwLock<HashMap<(LibraryName, String, Option<ConstantValues>), ComputePipelineState>>>,
}

impl MetalDevice {
    pub fn new() -> Result<Self> {
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

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn preload_pipelines(&self) -> Result<()> {
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
    
    pub fn register() -> Result<()> {
        tract_gpu::context::set_context(Arc::new(metal_device()))
    }
}

#[derive(Debug)]
pub struct MetalContext {
    command_queue: CommandQueue,
    command_buffer: RefCell<Option<TCommandBuffer>>,
    command_buffer_id: AtomicUsize,
    retained_tensors: RefCell<Vec<DeviceTensor>>,
    profiler: RefCell<Option<Rc<RefCell<MetalProfiler>>>>,
}

impl Default for MetalContext {
    fn default() -> Self {
        Self::new()
    }
}

impl MetalContext {
    pub fn new() -> Self {
        let device = metal_device();
        let command_queue = device.device.new_command_queue();
        Self {
            command_queue,
            command_buffer: RefCell::new(None),
            command_buffer_id: AtomicUsize::new(0),
            retained_tensors: RefCell::new(vec![]),
            profiler: RefCell::new(None),
        }
    }

    pub fn load_library(&self, name: LibraryName) -> Result<Library> {
        let metal_device = get_metal_device()?;
        metal_device.load_library(name)
    }

    pub fn load_pipeline(
        &self,
        library_name: LibraryName,
        func_name: &str,
    ) -> Result<ComputePipelineState> {
        let metal_device = get_metal_device()?;
        metal_device.load_pipeline(library_name, func_name)
    }

    pub(crate) fn load_pipeline_with_constants(
        &self,
        library_name: LibraryName,
        func_name: &str,
        constants: Option<ConstantValues>,
    ) -> Result<ComputePipelineState> {
        let metal_device = get_metal_device()?;
        metal_device.load_pipeline_with_constants(library_name, func_name, constants)
    }

    pub fn retain_tensor(&self, tensor: &DeviceTensor) {
        self.retained_tensors.borrow_mut().push(tensor.clone());
    }

    pub fn command_buffer(&self) -> TCommandBuffer {
        let command_buffer = self
            .command_buffer
            .borrow_mut()
            .get_or_insert_with(|| {                
                let command_buffer = TCommandBuffer::new(
                    self.command_queue.new_command_buffer().to_owned(),
                    self.profiler.borrow().clone(),
                );
                command_buffer
            })
            .to_owned();
        command_buffer
    }

    pub fn wait_until_completed(&self) -> Result<()> {
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

    pub fn capture_trace<P, F>(&self, path: P, compute: F) -> Result<()>
    where
        P: AsRef<Path>,
        F: FnOnce(&Self) -> Result<()>,
    {
        self.wait_until_completed()?;

        anyhow::ensure!(path.as_ref().is_absolute());

        let capture = metal::CaptureManager::shared();
        let descriptor = metal::CaptureDescriptor::new();
        descriptor.set_destination(metal::MTLCaptureDestination::GpuTraceDocument);
        descriptor.set_capture_device(&metal_device().device);
        descriptor.set_output_url(path);

        capture.start_capture(&descriptor).map_err(|e| anyhow!("Error Metal Capture: {:?}", e))?;

        (compute)(self)?;

        self.wait_until_completed()?;
        capture.stop_capture();
        Ok(())
    }

    pub fn profiler(&self) -> Option<Rc<RefCell<MetalProfiler>>> {
        self.profiler.borrow().clone()
    }

    pub fn profile<EvalCallback>(
        &self,
        num_nodes: usize,
        eval: EvalCallback,
    ) -> TractResult<(TVec<TValue>, Duration, Vec<u64>)>
    where
        EvalCallback: FnOnce() -> TractResult<(TVec<TValue>, Duration)>,
    {
        self.wait_until_completed()?;

        let device: &Device = &metal_device().device;
        assert!(
            device.supports_counter_sampling(metal::MTLCounterSamplingPoint::AtStageBoundary)
        );

        let profiler = Rc::new(RefCell::new(MetalProfiler::new(device.to_owned(), num_nodes)));

        self.profiler.replace(Some(profiler.clone()));

        let (output, eval_duration) = eval()?;
        let profile_buffers = profiler.borrow_mut().get_profile_data();

        self.profiler.replace(None);
        self.wait_until_completed()?;

        Ok((output, eval_duration, profile_buffers))
    }
}

impl Drop for MetalContext {
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
        command_buffer.commit();
        command_buffer.wait_until_completed();
    }
}

#[derive(Clone)]
pub struct MetalBuffer {
    pub inner: Buffer
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
    fn info(&self) -> String {
        format!("{:?}", self.inner)
    }

    fn address(&self) -> usize {
        self.inner.gpu_address() as usize
    }
}

impl GpuDevice for MetalDevice {
    fn buffer_from_slice(&self, data: &[u8]) -> Box<dyn tract_gpu::context::DeviceBuffer> {
        static ZERO: [u8; 1] = [0];
        // Handle empty data
        let data = if data.len() == 0 { &ZERO } else { data };

        let size = core::mem::size_of_val(data) as NSUInteger;
        Box::new(MetalBuffer { inner: self.device.new_buffer_with_bytes_no_copy(
            data.as_ptr() as *const core::ffi::c_void,
            size,
            MTLResourceOptions::StorageModeShared,
            None,
        )}
    )}

    fn synchronize(&self) -> TractResult<()> {
        METAL_CONTEXT.with_borrow(|ctxt| {
            ctxt.wait_until_completed()
        })
    }
}


#[cfg(test)]
mod tests {
    use super::*;
    use crate::{autorelease_pool_init, MetalTransform};
    use tract_core::ops::einsum::prefix_matmul::PrefixMatMul;
    use tract_core::ops::math::{add, mul};
    use tract_core::ops::nn::{Softmax, SoftmaxExp};
    use tract_core::transform::ModelTransform;
    use tract_gpu::memory::DeviceMemSchema;
    use tract_gpu::tensor::IntoGpu;

    #[test]
    fn test_alloc_zero() -> TractResult<()>{
        MetalDevice::register()?;
        let _ = autorelease_pool_init();
        Tensor::from_shape::<f32>(&[0], &[])?.into_gpu()?;
        Ok(())
    }

    fn wire_sdpa_layer(
        model: &mut TypedModel,
        name: impl ToString,
        q: OutletId,
        k: OutletId,
        v: OutletId,
    ) -> TractResult<TVec<OutletId>> {
        let name = name.to_string();

        // Reshape Q
        let q_shape = model.outlet_fact(q)?.shape.to_tvec();
        let embed_dim: TDim = q_shape[1].clone();
        let head_dim: TDim = q_shape[3].clone();
        let batch: TDim = q_shape[0].clone();
        let seq_len: TDim = q_shape[2].clone();
        ensure!(batch.to_i64()? == 1, "Input 'q' shape is {:?} (expect batch = 1)", q_shape);
        ensure!(q_shape.len() == 4, "Input 'q' shape is {:?} (expect 4D)", q_shape);
        let q_reshaped = model.wire_node(
            format!("q_reshape_{}", name),
            AxisOp::Reshape(
                0,
                q_shape.clone(),
                tvec![embed_dim.clone(), batch.clone(), seq_len.clone(), head_dim.clone(),],
            ),
            &[q],
        )?[0];

        // Reshape K
        let k_shape = model.outlet_fact(k)?.shape.to_tvec();
        ensure!(k_shape.len() == 4, "Input 'k' shape is {:?} (expect 4D)", k_shape);
        let seq_plus_prompt_len: TDim = k_shape[2].clone();

        let k_reshaped = model.wire_node(
            format!("k_reshape_{}", name),
            AxisOp::Reshape(
                0,
                k_shape.clone(),
                tvec![
                    embed_dim.clone(),
                    batch.clone(),
                    seq_plus_prompt_len.clone(),
                    head_dim.clone(),
                ],
            ),
            &[k],
        )?[0];

        // Compute Q * K^T
        let qk = model.wire_node(
            format!("qk_{}", name),
            PrefixMatMul {
                transpose_a: false,
                transpose_b: true,
                transpose_c: false,
                quantize_output: None,
            },
            &[q_reshaped, k_reshaped],
        )?[0];

        let qk_squeezed = model.wire_node(
            format!("qk_squeezed_{}", name),
            AxisOp::Reshape(
                0,
                tvec![
                    embed_dim.clone(),
                    batch.clone(),
                    seq_len.clone(),
                    seq_plus_prompt_len.clone(),
                ],
                tvec![embed_dim.clone(), seq_len.clone(), seq_plus_prompt_len.clone(),],
            ),
            &[qk],
        )?[0];

        // Scale factor for attention
        let scale = model.add_const(
            format!("scale_{}", name),
            tensor3(&[[[1.0f32 / (head_dim.to_i64()? as f32).sqrt()]]]),
        )?;
        let qk_scaled =
            model.wire_node(format!("qk_scaled_{}", name), mul(), &[qk_squeezed, scale])?[0];

        // Mask QK
        let mask = model.add_const("mask", tensor3(&[[[1.0f32]]]))?;
        let qk_scaled_masked =
            model.wire_node(format!("qk_scaled_masked_{}", name), add(), &[qk_scaled, mask])?[0];

        // Apply softmax
        let attention = model.wire_node(
            format!("attention_weights_{}", name),
            Softmax::new(tvec![2], None, SoftmaxExp::Libc),
            &[qk_scaled_masked],
        )?[0];

        // Reshape V
        let v_reshaped = model.wire_node(
            format!("v_reshape_{}", name),
            AxisOp::Reshape(
                0,
                k_shape,
                tvec![embed_dim.clone(), seq_plus_prompt_len.clone(), head_dim.clone(),],
            ),
            &[v],
        )?[0];

        // Multiply with V
        let output = model.wire_node(
            format!("attention_output_{}", name),
            PrefixMatMul {
                transpose_a: false,
                transpose_b: false,
                transpose_c: false,
                quantize_output: None,
            },
            &[attention, v_reshaped],
        )?[0];

        // Reshape output
        let output_reshaped = model.wire_node(
            format!("output_reshape_{}", name),
            AxisOp::Reshape(
                0,
                tvec![embed_dim.clone(), seq_len.clone(), head_dim.clone(),],
                q_shape,
            ),
            &[output],
        )?;
        Ok(output_reshaped)
    }

    #[test]
    fn test_build_schema_from_model() -> TractResult<()> {
        // Given
        const EMBED_DIM: i64 = 32;
        const HEAD_DIM: i64 = 64;
        const SEQUENCE_LENGTH: i64 = 1;
        const PAST_SEQUENCE_LENGTH: i64 = 8;
        const EXPECTED_PEAK_SIZE: i64 = 9344;
        const EXPECTED_USAGE: f32 = 0.89;

        // Build a model with Scaled Dot-Product Attention (SDPA) layers
        let mut model = TypedModel::default();

        // Input shapes for Q, K, V
        let s = TDim::Sym(model.sym("S"));
        let p = TDim::Sym(model.sym("P"));
        let q_fact = f32::fact(tvec![1.into(), EMBED_DIM.into(), s.clone(), HEAD_DIM.into()]);
        let k_fact = f32::fact(tvec![1.into(), EMBED_DIM.into(), s + p, HEAD_DIM.into()]);
        let v_fact = k_fact.clone();

        // Create inputs for Q, K, V
        let q = model.add_source("q", q_fact)?;
        let k = model.add_source("k", k_fact)?;
        let v = model.add_source("v", v_fact)?;

        let outputs = wire_sdpa_layer(&mut model, "0", q, k, v)?;
        let outputs = wire_sdpa_layer(&mut model, "1", outputs[0], k, v)?;

        model.set_output_outlets(&outputs)?;

        MetalDevice::register()?;
        // Transform model for Metal execution
        let model = MetalTransform::default().transform_into(model)?;

        // Get execution order
        let order = model.eval_order()?;

        // Hint symbol values
        let mut symbol_values = SymbolValues::default();
        symbol_values.set(&model.symbols.get("S").context("Missing symbol S")?, SEQUENCE_LENGTH);
        symbol_values
            .set(&model.symbols.get("P").context("Missing symbol P")?, PAST_SEQUENCE_LENGTH);

        // Build memory schema
        let schema = DeviceMemSchema::build(&model, &order, &symbol_values)?;

        // Verify number of nodes
        assert!(schema.model_num_nodes > 1, "Schema should contain at least 2 nodes");

        // Verify number of partitions
        assert!(schema.by_partition.len() > 1, "Schema should contain at least 2 partitions");

        // Verify steps
        assert_eq!(schema.by_steps.len(), order.len());
        for step in 0..schema.by_steps.len() {
            for partition in schema.by_partition.iter() {
                let partition_size = partition.eval_size_to_i64(&symbol_values)?;

                // No empty partition
                assert!(!partition.nodes.is_empty());

                if let Some(this) = partition.find_node_alive_at_step(step) {
                    // Node memory requirement should be <= the partition size
                    let node_size = this.mem_size.eval_to_i64(&symbol_values)?;
                    assert!(node_size <= partition_size);
                    assert!(node_size > 0);

                    // All nodes should have a valid lifetime
                    assert!(this.lifetime.start < this.lifetime.end);

                    // No other node in the partition should be alive at this step
                    for other in partition.nodes.iter().filter(|it| it.node != this.node) {
                        assert!(
                            !other.lifetime.is_alive_at_step(step)
                                && other.lifetime.is_disjoint(&this.lifetime),
                            "Lifetime conflict @ step {}\n{:?}\n{:?}",
                            step,
                            this,
                            other
                        );
                    }

                    // This node should not be alive in another partition at the same step
                    for p in schema.by_partition.iter().filter(|it| it != &partition) {
                        if let Some(other) = p.find_node_alive_at_step(step) {
                            assert!(other.node != this.node);
                        }
                    }
                }
            }
        }

        // Verify schema usage
        let usage = schema.eval_usage(&symbol_values)?;
        assert!(usage >= EXPECTED_USAGE, "Usage {}, expected >= {}", usage, EXPECTED_USAGE);

        // Verify peak memory size
        let peak_memory_size = schema.eval_peak_memory_size(&symbol_values)?;
        assert_eq!(peak_memory_size, EXPECTED_PEAK_SIZE, "Peak memory size mismatch");

        Ok(())
    }
}
