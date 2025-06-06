use cust::prelude::*;

use tract_gpu::device::DeviceContext;
use tract_gpu::tensor::OwnedDeviceTensor;

use std::cell::RefCell;

use std::ops::Deref;
use std::sync::{OnceLock, RwLock};

use tract_core::internal::*;

use crate::kernels::LibraryName;
use crate::tensor::CudaTensor;

thread_local! {
    pub static CUDA_STREAM: RefCell<CudaStream> = RefCell::new(CudaStream::new());
}

pub fn cuda_context() -> CudaContext {
    static INSTANCE: OnceLock<CudaContext> = OnceLock::new();
    INSTANCE
        .get_or_init(|| {
            let ctxt = CudaContext::new().expect("Could not create CUDA context");
            tract_gpu::device::set_context(Box::new(ctxt.clone()))
                .expect("Could not set CUDA context");
            ctxt
        })
        .clone()
}

#[derive(Debug, Clone)]
pub struct CudaContext {
    inner: Context,
    cached_modules: Arc<RwLock<HashMap<LibraryName, Arc<Module>>>>,
    cached_pipelines: Arc<RwLock<HashMap<(LibraryName, String), Arc<Function<'static>>>>>,
}

impl Deref for CudaContext {
    type Target = Context;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl CudaContext {
    pub fn new() -> TractResult<Self> {
    let context =
        cust::quick_init().with_context(|| "Could not find system default CUDA device")?;

    Ok(Self { inner: context,
            cached_modules: Arc::new(RwLock::new(HashMap::new())),
            cached_pipelines: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    pub fn load_library(&self, name: &LibraryName) -> TractResult<Arc<Module>> {
        {
            let cache = self.cached_modules.read().map_err(|e| anyhow!("{:?}", e))?;
            if let Some(module) = cache.get(name) {
                return Ok(module.clone());
            }
        }

        let module = Arc::new(Module::from_ptx(name.content(), &[])?);

        let mut cache = self.cached_modules.write().map_err(|e| anyhow!("{:?}", e))?;
        cache.insert(name.clone(), module.clone());

        Ok(module)
    }

    pub fn load_pipeline(
        &self,
        library_name: LibraryName,
        func_name: String,
    ) -> TractResult<Arc<Function<'static>>> {
        let key = (library_name.clone(), func_name.to_string());

        // Check pipeline cache
        {   
            let cache = self.cached_pipelines.read().map_err(|e| anyhow!("{:?}", e))?;
            if let Some(f) = cache.get(&key) {
                return Ok(f.clone());
            }
        }

        // Load module + function
        let module = self.load_library(&library_name)?;
        let func = module
            .get_function(&func_name)
            .map_err(|e| anyhow!("{e}"))
            .with_context(|| {
                format!(
                    "Failed to load function `{func_name}` from library `{}`",
                    library_name.content()
                )
            })?;

        // SAFETY: Module is `Arc` and lives as long as the context 
        let static_func: Function<'static> = unsafe { std::mem::transmute(func) };
        let static_func = Arc::new(static_func);

        // Store in cache
        let mut cache = self.cached_pipelines.write().map_err(|e| anyhow!("{:?}", e))?;
        cache.insert(key, static_func.clone());

        Ok(static_func)
    }
}

impl DeviceContext for CudaContext {
    fn synchronize(&self) -> TractResult<()> {
        CUDA_STREAM.with_borrow(|stream| stream.wait_until_completed())
    }

    fn tensor_to_device(&self, tensor: TValue) -> TractResult<Box<dyn OwnedDeviceTensor>> {
        let data = tensor.as_bytes();
        static ZERO: [u8; 1] = [0];
        // Handle empty data
        let data = if data.is_empty() { &ZERO } else { data };

        Ok(Box::new(CudaTensor::from_bytes(
            data,
            tensor.datum_type(),
            tensor.shape(),
            tensor.strides(),
        )))
    }
}

#[derive(Debug)]
pub struct CudaStream {
    pub stream: Stream,
}

impl Default for CudaStream {
    fn default() -> Self {
        Self::new()
    }
}

impl CudaStream {
    pub fn new() -> Self {
        Self { stream: Stream::new(StreamFlags::NON_BLOCKING, None).unwrap() }
    }

    pub fn wait_until_completed(&self) -> TractResult<()> {
        self.stream.synchronize().map_err(|e| e.into())
    }
}
