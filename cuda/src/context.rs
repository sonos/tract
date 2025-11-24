use cudarc::cublas::CudaBlas;
use cudarc::nvrtc::Ptx;
use cudarc::runtime::result::device::get_device_prop;
use cudarc::runtime::sys::cudaDeviceProp;
use tract_gpu::device::DeviceContext;
use tract_gpu::tensor::{DeviceTensor, OwnedDeviceTensor};

use std::mem::MaybeUninit;
use std::ops::Deref;
use std::sync::{OnceLock, RwLock};

use tract_core::internal::*;

use cudarc::driver::{CudaContext, CudaFunction, CudaModule, CudaStream};

use crate::kernels::{COMMON_H, LibraryName, cubin_dir};
use crate::tensor::CudaTensor;

use cudarc::nvrtc::result::{compile_program, destroy_program, get_program_log};
use cudarc::nvrtc::sys::{
    nvrtcCreateProgram, nvrtcGetCUBIN, nvrtcGetCUBINSize, nvrtcProgram, nvrtcResult,
};
use std::ffi::{CStr, CString, c_char};
use std::path::{Path, PathBuf};

thread_local! {
    pub static CUDA_STREAM: TractCudaStream = TractCudaStream::new().expect("Could not create Cuda Stream");
}

pub fn cuda_context() -> &'static TractCudaContext {
    static INSTANCE: OnceLock<TractCudaContext> = OnceLock::new();
    INSTANCE.get_or_init(|| {
        let ctxt = TractCudaContext::new().expect("Could not create CUDA context");
        tract_gpu::device::set_context(Box::new(ctxt.clone())).expect("Could not set CUDA context");
        ctxt
    })
}

#[derive(Debug, Clone)]
pub struct TractCudaContext {
    inner: Arc<CudaContext>,
    device_properties: cudaDeviceProp,
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

        let prop = get_device_prop(0)?;
        let ctxt = Self {
            inner: context,
            device_properties: prop,
            cached_modules: Arc::new(RwLock::new(HashMap::new())),
            cached_pipelines: Arc::new(RwLock::new(HashMap::new())),
        };
        ctxt.compile_cubins()?;
        ctxt.preload_pipelines()?;
        Ok(ctxt)
    }

    pub fn properties(&self) -> &cudaDeviceProp {
        &self.device_properties
    }

    pub fn compile_cubins(&self) -> TractResult<()> {
        let cubin_dir = cubin_dir();
        if !cubin_dir.exists() {
            log::info!("Creating cache folder for CUDA cubins at {}", cubin_dir.display());
            std::fs::create_dir_all(cubin_dir)
                .with_context(|| format!("Failed to create {}", cubin_dir.display()))?;
        }

        let nvrtc_opts = self.build_nvrtc_opts()?;

        for lib in LibraryName::ALL {
            let out_path = lib.cubin_path();
            if out_path.exists() {
                continue;
            }

            log::info!("Compiling {:?} to {}…", lib, out_path.display());

            let mut input = lib.content().to_string();
            if input.contains("// liquid:true") {
                let parser = liquid::ParserBuilder::with_stdlib().build()?;
                let tmpl = parser.parse(lib.content())?;
                let globals = liquid::object!({});
                input = tmpl.render(&globals)?;
            }

            let c_src = CString::new(input).context("Failed to make CString from CUDA source")?;
            let prog = unsafe {
                let mut prog = MaybeUninit::uninit();
                nvrtcCreateProgram(
                    prog.as_mut_ptr(),
                    c_src.as_ptr(),
                    std::ptr::null(),
                    1,
                    &CString::new(COMMON_H)
                        .context("Failed to make CString from CUDA header")?
                        .as_ptr(),
                    &CString::new("common.cuh")
                        .context("Failed to make CString from CUDA header name")?
                        .as_ptr(),
                )
                .result()?;
                prog.assume_init()
            };

            if let Err(_e) = unsafe { compile_program::<String>(prog, &nvrtc_opts) } {
                let log = self.read_nvrtc_log(prog).unwrap_or_else(|_| "<no log>".into());
                let _ = unsafe { destroy_program(prog) };
                return Err(anyhow!("NVRTC compilation failed for {:?}:\n{}", lib, log));
            }

            let cubin = unsafe { self.get_cubin_bytes(prog) }
                .with_context(|| format!("Failed to extract CUBIN for {:?}", lib))?;

            unsafe { destroy_program(prog) }.context("nvrtcDestroyProgram failed")?;

            std::fs::write(&out_path, &cubin)
                .with_context(|| format!("Failed to write {:?}", out_path))?;
        }

        Ok(())
    }

    /// Build NVRTC options: include paths + GPU arch.
    fn build_nvrtc_opts(&self) -> TractResult<Vec<String>> {
        let cuda_home = std::env::var_os("CUDA_HOME")
            .map(PathBuf::from)
            .or_else(|| std::env::var_os("CUDA_PATH").map(PathBuf::from)) // Windows
            .or_else(|| {
                let p = Path::new("/usr/local/cuda");
                if p.exists() { Some(p.to_path_buf()) } else { None }
            })
            .or_else(|| {
                // Last resort: infer from nvcc location
                 std::process::Command::new("which").arg("nvcc").output().ok()
                    .and_then(|nvcc| Path::new(&String::from_utf8(nvcc.stdout).unwrap()).parent().and_then(|bin| bin.parent()).map(|p| p.to_path_buf()))
            });

        let cuda_inc = cuda_home.unwrap().join("include");
        if !cuda_inc.exists() {
            return Err(anyhow!("CUDA include dir not found at {:?}", cuda_inc));
        }

        let arch = format!(
            "--gpu-architecture=sm_{}{}",
            self.device_properties.major, self.device_properties.minor
        );

        Ok(vec!["--std=c++17".into(), arch, format!("-I{}", cuda_inc.display())])
    }

    /// Read the NVRTC program log as String.
    fn read_nvrtc_log(&self, prog: nvrtcProgram) -> TractResult<String> {
        let buf: Vec<c_char> =
            unsafe { get_program_log(prog).context("nvrtcGetProgramLog failed") }?;

        let bytes = unsafe { std::slice::from_raw_parts(buf.as_ptr() as *const u8, buf.len()) };

        match CStr::from_bytes_until_nul(bytes) {
            Ok(cstr) => Ok(cstr.to_string_lossy().into_owned()),
            Err(_) => Ok(String::from_utf8_lossy(bytes).into_owned()),
        }
    }

    /// Extract CUBIN bytes from an NVRTC program.
    unsafe fn get_cubin_bytes(&self, prog: nvrtcProgram) -> TractResult<Vec<u8>> {
        let mut len: usize = 0;
        let res = unsafe { nvrtcGetCUBINSize(prog, &mut len as *mut usize) };
        if res != nvrtcResult::NVRTC_SUCCESS {
            return Err(anyhow!("nvrtcGetCUBINSize failed ({:?})", res));
        }

        let mut cubin = vec![0u8; len];
        let res = unsafe { nvrtcGetCUBIN(prog, cubin.as_mut_ptr() as *mut c_char) };
        if res != nvrtcResult::NVRTC_SUCCESS {
            return Err(anyhow!("nvrtcGetCUBIN failed ({:?})", res));
        }
        Ok(cubin)
    }

    pub fn preload_pipelines(&self) -> TractResult<()> {
        for ew_func in crate::kernels::UnaryOps::all_functions() {
            let _ = self.load_pipeline(LibraryName::Unary, ew_func);
        }

        for bin_func in crate::kernels::BinOps::all_functions() {
            let _ = self.load_pipeline(LibraryName::Binary, bin_func);
        }

        for arr_func in crate::kernels::array::all_functions() {
            let _ = self.load_pipeline(LibraryName::Array, arr_func);
        }

        for nn_func in crate::kernels::nn::all_functions() {
            let _ = self.load_pipeline(LibraryName::NN, nn_func);
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

        let module = self.inner.load_module(Ptx::from_file(name.cubin_path()))?;

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
                    library_name.cubin_path().display()
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
        ensure!(DeviceTensor::is_supported_dt(tensor.datum_type()));
        Ok(Box::new(CudaTensor::from_tensor(tensor.view().tensor)?))
    }

    fn uninitialized_device_tensor(
        &self,
        shape: &[usize],
        dt: DatumType,
    ) -> TractResult<Box<dyn OwnedDeviceTensor>> {
        Ok(Box::new(CudaTensor::uninitialized_dt(shape, dt)?))
    }

    fn uninitialized_device_opaque_tensor(
        &self,
        opaque_fact: Box<dyn OpaqueFact>,
    ) -> TractResult<Box<dyn OwnedDeviceTensor>> {
        Ok(Box::new(CudaTensor::uninitialized_opaque(opaque_fact)?))
    }
}

pub struct TractCudaStream {
    inner: Arc<CudaStream>,
    cublas: CudaBlas,
}

impl TractCudaStream {
    fn new() -> TractResult<TractCudaStream> {
        let stream = cuda_context().default_stream();
        let cublas = CudaBlas::new(stream.clone())?;
        Ok(TractCudaStream { inner: stream, cublas })
    }

    pub fn cublas(&self) -> &CudaBlas {
        &self.cublas
    }
}

impl Deref for TractCudaStream {
    type Target = Arc<CudaStream>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}
