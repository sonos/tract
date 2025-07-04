use cudarc::nvrtc::Ptx;
use tract_gpu::device::DeviceContext;
use tract_gpu::tensor::OwnedDeviceTensor;

use std::ops::Deref;
use std::sync::{OnceLock, RwLock};

use tract_core::internal::*;

use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaStream};

use crate::kernels::LibraryName;
use crate::tensor::CudaTensor;

thread_local! {
    pub static CUDA_STREAM: Arc<CudaStream> = cuda_context().default_stream();
}

pub fn cuda_context() -> TractCudaContext {
    static INSTANCE: OnceLock<TractCudaContext> = OnceLock::new();
    INSTANCE
        .get_or_init(|| {
            let ctxt = TractCudaContext::new().expect("Could not create CUDA context");
            tract_gpu::device::set_context(Box::new(ctxt.clone()))
                .expect("Could not set CUDA context");
            ctxt
        })
        .clone()
}

#[derive(Debug, Clone)]
pub struct TractCudaContext {
    inner: Arc<CudaContext>,
    cached_modules: Arc<RwLock<HashMap<LibraryName, Arc<CudaModule>>>>,
    #[allow(clippy::type_complexity)]
    cached_pipelines: Arc<RwLock<HashMap<(LibraryName, String), Arc<CudaFunction>>>>,
}

impl Deref for TractCudaContext {
    type Target = Arc<CudaContext>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl TractCudaContext {
    pub fn new() -> TractResult<Self> {
        let context =
            CudaContext::new(0).with_context(|| "Could not find system default CUDA device")?;

        let ctxt = Self {
            inner: context,
            cached_modules: Arc::new(RwLock::new(HashMap::new())),
            cached_pipelines: Arc::new(RwLock::new(HashMap::new())),
        };

        ctxt.preload_pipelines()?;
        Ok(ctxt)
    }

    pub fn preload_pipelines(&self) -> TractResult<()> {
        // TODO: Investigate CUDA lazy loading
        for ew_func in crate::kernels::UnaryOps::all_functions() {
            let _ = self.load_pipeline(LibraryName::UnaryOps, ew_func);
        }

        for bin_func in crate::kernels::BinOps::all_functions() {
            let _ = self.load_pipeline(LibraryName::BinaryOps, bin_func);
        }
        Ok(())
    }

    pub fn load_library(&self, name: &LibraryName) -> TractResult<Arc<CudaModule>> {
        {
            let cache = self.cached_modules.read().map_err(|e| anyhow!("{:?}", e))?;
            if let Some(module) = cache.get(name) {
                return Ok(module.clone());
            }
        }

        let module = self.inner.load_module(Ptx::from_src(name.content()))?;

        let mut cache = self.cached_modules.write().map_err(|e| anyhow!("{:?}", e))?;
        cache.insert(*name, module.clone());

        Ok(module)
    }

    pub fn load_pipeline(
        &self,
        library_name: LibraryName,
        func_name: String,
    ) -> TractResult<Arc<CudaFunction>> {
        // Check pipeline cache
        let key = (library_name, func_name.to_string());
        {
            let cache = self.cached_pipelines.read().map_err(|e| anyhow!("{:?}", e))?;
            if let Some(f) = cache.get(&key) {
                return Ok(f.clone());
            }
        }

        // Load module + function
        let module = self.load_library(&library_name)?;
        let func =
            module.load_function(&func_name).map_err(|e| anyhow!("{e}")).with_context(|| {
                format!(
                    "Failed to load function `{func_name}` from library `{}`",
                    library_name.content()
                )
            })?;

        let func = Arc::new(func);

        // Store in cache
        let mut cache = self.cached_pipelines.write().map_err(|e| anyhow!("{:?}", e))?;
        cache.insert(key, func.clone());

        Ok(func)
    }
}

impl DeviceContext for TractCudaContext {
    fn synchronize(&self) -> TractResult<()> {
        CUDA_STREAM.with(|stream| stream.synchronize().map_err(|e| e.into()))
    }

    fn tensor_to_device(&self, tensor: TValue) -> TractResult<Box<dyn OwnedDeviceTensor>> {
        Ok(Box::new(CudaTensor::from_tensor(tensor.view().tensor)))
    }
}
