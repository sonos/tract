use cudarc::cublas::CudaBlas;
use cudarc::cudnn::Cudnn;
use cudarc::nvrtc::Ptx;
use cudarc::runtime::result::device::get_device_prop;
use cudarc::runtime::sys::cudaDeviceProp;
use tract_gpu::device::DeviceContext;
use tract_gpu::tensor::{DeviceTensor, OwnedDeviceTensor};

use std::cell::{Cell, RefCell};
use std::mem::MaybeUninit;
use std::ops::Deref;
use std::sync::{OnceLock, RwLock};

use tract_core::internal::*;

use cudarc::driver::{CudaContext, CudaEvent, CudaFunction, CudaModule, CudaStream};

use crate::kernels::{COMMON_H, LibraryName, cubin_dir};
use crate::tensor::CudaTensor;

use cudarc::nvrtc::result::{compile_program, destroy_program, get_program_log};
use cudarc::nvrtc::sys::{
    nvrtcCreateProgram, nvrtcGetCUBIN, nvrtcGetCUBINSize, nvrtcProgram, nvrtcResult,
};
use std::ffi::{CStr, CString, c_char};
use std::path::{Path, PathBuf};

pub fn cuda_context() -> &'static TractCudaContext {
    static INSTANCE: OnceLock<TractCudaContext> = OnceLock::new();
    INSTANCE.get_or_init(|| {
        let ctxt = TractCudaContext::new().expect("Could not create CUDA context");
        tract_gpu::device::set_context(Box::new(ctxt.clone())).expect("Could not set CUDA context");
        ctxt
    })
}

thread_local! {
    static CUDA_STREAM: TractCudaStream = TractCudaStream::new().expect("Could not create Cuda Stream");
}

pub fn with_cuda_stream<R>(f: impl FnOnce(&TractCudaStream) -> TractResult<R>) -> TractResult<R> {
    CUDA_STREAM.with(|cell| f(cell))
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

        // Log device identity so users diagnosing a bad deploy can tell what their code actually
        // found. cudaDeviceProp::name is a C char array; be defensive about odd bytes.
        let name_bytes: Vec<u8> =
            prop.name.iter().take_while(|c| **c != 0).map(|c| *c as u8).collect();
        let device_name = String::from_utf8_lossy(&name_bytes);
        log::info!(
            "tract-cuda: device 0 = {:?}, compute capability {}.{}, {} MiB global mem",
            device_name,
            prop.major,
            prop.minor,
            prop.totalGlobalMem / (1024 * 1024),
        );

        if let Ok(rt) = cudarc::runtime::result::version::get_runtime_version() {
            let (ma, mi) = (rt / 1000, (rt % 1000) / 10);
            log::info!("tract-cuda: CUDA runtime (cudart) version {ma}.{mi}");
        } else {
            log::warn!("tract-cuda: cudaRuntimeGetVersion failed");
        }

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
                let env = minijinja::Environment::new();
                let tmpl = env.template_from_str(lib.content())?;
                input = tmpl.render(())?;
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

    /// Build NVRTC options: GPU arch + include paths for the CUDA toolkit's host/device
    /// headers and the CCCL (libcudacxx) tree. Both are required because NVRTC in
    /// CUDA 13 does not intercept `<cuda_fp16.h>`, `<math_constants.h>`, or any of
    /// the libcudacxx headers — they have to be reachable on disk via `-I`. On
    /// Debian/Ubuntu the matching packages are `cuda-cccl-<ver>` and
    /// `cuda-cudart-dev-<ver>`.
    fn build_nvrtc_opts(&self) -> TractResult<Vec<String>> {
        let arch = format!(
            "--gpu-architecture=sm_{}{}",
            self.device_properties.major, self.device_properties.minor
        );
        log::info!("tract-cuda: NVRTC target architecture {arch}");

        let cuda_inc = resolve_toolkit_include_dir()?;
        log::info!("tract-cuda: toolkit include dir {}", cuda_inc.display());

        let cccl_root = find_cccl_root(&cuda_inc)?;
        log::info!("tract-cuda: CCCL root {}", cccl_root.display());

        // `cuda_fp16.h` is already checked as the sentinel in resolve_toolkit_include_dir.
        // Check the other non-CCCL header we rely on for the same fail-fast reason.
        let math_hdr = cuda_inc.join("math_constants.h");
        if !math_hdr.exists() {
            bail!(
                "Required header {} not found. Install cuda-cudart-dev-{}.",
                math_hdr.display(),
                cuda_pkg_suffix(),
            );
        }

        let opts = vec![
            "--std=c++17".into(),
            arch,
            format!("-I{}", cuda_inc.display()),
            format!("-I{}", cccl_root.display()),
        ];
        log::info!("tract-cuda: NVRTC opts = {opts:?}");
        Ok(opts)
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
        for ew_func in crate::kernels::element_wise::all_functions() {
            let _ = self.load_pipeline(LibraryName::ElementWise, ew_func);
        }

        for bin_func in crate::kernels::binary::all_functions() {
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
        with_cuda_stream(|stream| stream.synchronize().map_err(|e| e.into()))
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

    fn uninitialized_device_exotic_tensor(
        &self,
        exotic_fact: Box<dyn ExoticFact>,
    ) -> TractResult<Box<dyn OwnedDeviceTensor>> {
        Ok(Box::new(CudaTensor::uninitialized_exotic(exotic_fact)?))
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
        crate::kernels::array::cuda_copy_nd_dispatch(
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

/// A recorded GPU kernel timing entry: start/end events tagged with a node_id.
pub struct GpuProfileEntry {
    pub node_id: usize,
    pub start: CudaEvent,
    pub end: CudaEvent,
}

pub struct TractCudaStream {
    inner: Arc<CudaStream>,
    cublas: CudaBlas,
    cudnn: Arc<Cudnn>,
    /// When Some, kernel launches record start/end events here.
    profile_log: RefCell<Option<Vec<GpuProfileEntry>>>,
    /// The node_id currently being evaluated (set by the profiling harness).
    current_node_id: Cell<usize>,
}

impl TractCudaStream {
    fn new() -> TractResult<TractCudaStream> {
        let stream = cuda_context().default_stream();
        let cublas = CudaBlas::new(stream.clone()).context(
            "CudaBlas::new failed: libcublas loaded but the handle could not be created",
        )?;
        let cudnn = Cudnn::new(stream.clone()).context(
            "Cudnn::new failed: libcudnn loaded but the handle could not be created. \
             Version mismatch between libcudnn and the CUDA runtime is the most common cause.",
        )?;

        // Log once per stream — typically one per thread. Debug-level because the first stream's
        // INFO-level summary already captured the versions via info_once below.
        log_library_versions_once();

        Ok(TractCudaStream {
            inner: stream,
            cublas,
            cudnn,
            profile_log: RefCell::new(None),
            current_node_id: Cell::new(0),
        })
    }

    pub fn cublas(&self) -> &CudaBlas {
        &self.cublas
    }

    pub fn cudnn(&self) -> &Arc<Cudnn> {
        &self.cudnn
    }

    /// Enable GPU profiling. Kernel launches will record timing events.
    pub fn enable_profiling(&self) {
        *self.profile_log.borrow_mut() = Some(Vec::new());
    }

    /// Set the current node being evaluated (used by the profiling harness).
    pub fn set_current_node(&self, node_id: usize) {
        self.current_node_id.set(node_id);
    }

    /// Returns true if profiling is active.
    pub fn is_profiling(&self) -> bool {
        self.profile_log.borrow().is_some()
    }

    /// Record a start/end event pair around a kernel launch.
    /// Call this from `TractLaunchArgs::launch()` when profiling is active.
    pub fn record_profile_events(&self) -> TractResult<Option<(CudaEvent, CudaEvent)>> {
        if !self.is_profiling() {
            return Ok(None);
        }
        let flags = Some(cudarc::driver::sys::CUevent_flags::CU_EVENT_DEFAULT);
        let start = self.inner.record_event(flags)?;
        Ok(Some((start, self.inner.context().new_event(flags)?)))
    }

    /// Finish recording a profile entry (call after kernel launch).
    pub fn finish_profile_entry(&self, start: CudaEvent, end: CudaEvent) -> TractResult<()> {
        end.record(&self.inner)?;
        let node_id = self.current_node_id.get();
        self.profile_log.borrow_mut().as_mut().unwrap().push(GpuProfileEntry {
            node_id,
            start,
            end,
        });
        Ok(())
    }

    /// Drain all recorded profile entries. Caller should synchronize first.
    pub fn drain_profile(&self) -> Option<Vec<GpuProfileEntry>> {
        self.profile_log.borrow_mut().as_mut().map(|log| std::mem::take(log))
    }
}

impl Deref for TractCudaStream {
    type Target = Arc<CudaStream>;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

/// Resolve a CUDA toolkit root containing an `include/` directory, if any is available.
/// Returns `None` on runtime-only deploys where the toolkit isn't installed — callers are
/// expected to rely on NVRTC's built-in headers in that case and may still succeed.
fn resolve_cuda_home() -> Option<(&'static str, PathBuf)> {
    if let Some(v) = std::env::var_os("CUDA_HOME") {
        let p = PathBuf::from(v);
        log::debug!("tract-cuda: CUDA_HOME env var set, using {}", p.display());
        return Some(("CUDA_HOME env", p));
    }
    if let Some(v) = std::env::var_os("CUDA_PATH") {
        let p = PathBuf::from(v);
        log::debug!("tract-cuda: CUDA_PATH env var set, using {}", p.display());
        return Some(("CUDA_PATH env", p));
    }
    let default = Path::new("/usr/local/cuda");
    if default.exists() {
        log::debug!("tract-cuda: /usr/local/cuda exists, using that as CUDA_HOME");
        return Some(("default /usr/local/cuda", default.to_path_buf()));
    }
    if let Ok(nvcc) = std::process::Command::new("which").arg("nvcc").output()
        && nvcc.status.success()
    {
        let stdout = String::from_utf8_lossy(&nvcc.stdout);
        let p = Path::new(stdout.trim());
        if let Some(root) = p.parent().and_then(|bin| bin.parent()) {
            log::debug!(
                "tract-cuda: inferred CUDA_HOME {} from nvcc at {}",
                root.display(),
                p.display()
            );
            return Some(("inferred from `which nvcc`", root.to_path_buf()));
        }
    }
    log::debug!(
        "tract-cuda: no CUDA_HOME found (env vars unset, /usr/local/cuda absent, nvcc not in PATH)"
    );
    None
}

/// Resolve the toolkit include dir — the directory that holds `cuda_fp16.h`. CUDA 13
/// ships headers under `<root>/targets/<arch>-<os>/include/` and usually symlinks
/// `<root>/include/` to it, but minimal installs can skip that symlink; probe both.
fn resolve_toolkit_include_dir() -> TractResult<PathBuf> {
    let Some((_, cuda_home)) = resolve_cuda_home() else {
        let ver = cuda_pkg_suffix();
        bail!(
            "No CUDA toolkit root found. Tract's NVRTC kernels need the toolkit's \
             host/device headers (apt: cuda-cudart-dev-{ver}) plus CCCL (apt: \
             cuda-cccl-{ver}). Set CUDA_HOME if the toolkit is installed somewhere \
             other than /usr/local/cuda."
        );
    };

    let sentinel = "cuda_fp16.h";
    let mut candidates = vec![cuda_home.join("include")];
    let targets = cuda_home.join("targets");
    if targets.exists() {
        if let Ok(entries) = std::fs::read_dir(&targets) {
            for entry in entries.flatten() {
                candidates.push(entry.path().join("include"));
            }
        }
    }

    let mut tried: Vec<PathBuf> = Vec::new();
    for cand in &candidates {
        let probe = cand.join(sentinel);
        let found = probe.exists();
        log::debug!("tract-cuda: toolkit-inc probe {} exists={found}", probe.display());
        tried.push(probe);
        if found {
            return Ok(cand.clone());
        }
    }

    bail!(
        "Could not find a toolkit include dir containing {sentinel}. Tried:\n  - {}\nInstall \
         cuda-cudart-dev-{}.",
        tried.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join("\n  - "),
        cuda_pkg_suffix(),
    )
}

/// Locate the CCCL (libcudacxx) include root that contains `cuda/std/cstdint`. Probes
/// under the given toolkit include dir: `cccl/`, `libcudacxx/include/`, then the dir
/// itself (pre-13 layout).
fn find_cccl_root(cuda_inc: &Path) -> TractResult<PathBuf> {
    let sentinel = Path::new("cuda").join("std").join("cstdint");
    let mut tried: Vec<PathBuf> = Vec::new();

    for cand in
        [cuda_inc.join("cccl"), cuda_inc.join("libcudacxx").join("include"), cuda_inc.to_path_buf()]
    {
        let probe_path = cand.join(&sentinel);
        let found = probe_path.exists();
        log::debug!("tract-cuda: CCCL probe {} exists={found}", probe_path.display());
        tried.push(probe_path);
        if found {
            return Ok(cand);
        }
    }

    bail!(
        "CCCL/libcudacxx headers not found. Tract's NVRTC kernels need <cuda/std/cstdint> and \
         <cuda/std/type_traits>. Tried:\n  - {}\nInstall the CCCL headers (apt: cuda-cccl-{}).",
        tried.iter().map(|p| p.display().to_string()).collect::<Vec<_>>().join("\n  - "),
        cuda_pkg_suffix(),
    )
}

/// Package suffix used by the NVIDIA apt repo, e.g. `13-0` for cudart 13.0. We prefer the
/// runtime's actual cudart version when it's already loaded (via cudarc) — that aligns with
/// whatever is installed on this host — and fall back to the feature-gated minimum
/// (`REQUIRED_CUDA_API`) otherwise.
fn cuda_pkg_suffix() -> String {
    let v = cudarc::runtime::result::version::get_runtime_version()
        .unwrap_or(crate::utils::REQUIRED_CUDA_API);
    format!("{}-{}", v / 1000, (v % 1000) / 10)
}

/// Log cuBLAS / cuDNN / cudart versions and device property summary at INFO level, exactly once
/// per process. Called from every `TractCudaStream::new`, but the `OnceLock` guard ensures we
/// don't spam the log when user code creates one stream per thread.
fn log_library_versions_once() {
    static ONCE: OnceLock<()> = OnceLock::new();
    ONCE.get_or_init(|| {
        let cudnn_v = cudarc::cudnn::result::get_version();
        log::info!(
            "tract-cuda: cuDNN version {}.{}.{} (raw {cudnn_v})",
            cudnn_v / 1000,
            (cudnn_v % 1000) / 100,
            cudnn_v % 100,
        );

        // cuBLAS exposes cublasGetVersion_v2(handle, *int). We don't plumb a handle here, but
        // cublasGetCudartVersion is no-arg and reports the cudart version cuBLAS was built
        // against — which is the interesting diagnostic when debugging a cuBLAS/cudart mismatch.
        let cublas_cudart = unsafe { cudarc::cublas::sys::cublasGetCudartVersion() };
        log::info!(
            "tract-cuda: cuBLAS was built against cudart {}.{}",
            cublas_cudart / 1000,
            (cublas_cudart % 1000) / 10,
        );
    });
}
