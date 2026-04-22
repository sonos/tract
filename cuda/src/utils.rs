use std::ffi::c_int;
use std::sync::OnceLock;

use anyhow::Context;
use libloading::{Library, Symbol};

use tract_core::internal::*;
use tract_core::tract_linalg::block_quant::*;
use tract_gpu::tensor::DeviceTensor;

use crate::Q40_ROW_PADDING;
use crate::ops::GgmlQuantQ81Fact;

static CULIBS_MISSING: OnceLock<Option<&'static str>> = OnceLock::new();
static DEPENDENCIES_OK: OnceLock<()> = OnceLock::new();

// Ensure exactly cuda-13000 is enabled.
// Prevent accidental change of feature gate without
// updating this required API version used for compatibility check.
// please update the 3 references bellow if cudarc gate is updated to a newer version.
pub const REQUIRED_CUDA_API: i32 = 13000;
#[cfg(not(feature = "cuda-13000"))]
compile_error!(
    "Tract CUDA backend currently supports only cudarc feature 'cuda-13000'. \
        Enabled in Cargo features.",
);

/// CUDA Driver API status code type (CUresult is an enum, but it's ABI-compatible with int).
type CuResult = c_int;

type CuInitFn = unsafe extern "C" fn(flags: u32) -> CuResult;
type CuDriverGetVersionFn = unsafe extern "C" fn(version: *mut c_int) -> CuResult;

#[cfg(target_os = "linux")]
const CUDA_DRIVER_LIB_CANDIDATES: [&str; 2] = ["libcuda.so.1", "libcuda.so"];

#[cfg(target_os = "windows")]
const CUDA_DRIVER_LIB_CANDIDATES: [&str; 1] = ["nvcuda.dll"];

#[cfg(not(any(target_os = "linux", target_os = "windows")))]
const CUDA_DRIVER_LIB_CANDIDATES: [&str; 0] = [];

fn format_cuda_version(v: i32) -> (i32, i32) {
    (v / 1000, (v % 1000) / 10)
}

fn load_first_found_cuda_lib() -> TractResult<(&'static str, Library)> {
    for name in CUDA_DRIVER_LIB_CANDIDATES {
        match unsafe { Library::new(name) } {
            Ok(lib) => {
                log::debug!("tract-cuda: driver candidate {name:?} loaded");
                return Ok((name, lib));
            }
            Err(e) => log::debug!("tract-cuda: driver candidate {name:?} not loadable: {e}"),
        }
    }

    bail!(
        "CUDA driver library not found. Tried: {:?}. \
         Is an NVIDIA driver installed \
         (and, in containers, did you run with GPU passthrough)?",
        CUDA_DRIVER_LIB_CANDIDATES
    )
}

/// Checks that the installed CUDA driver is present and new enough to satisfy REQUIRED_CUDA_API.
///
/// IMPORTANT: call this before touching `cudarc::driver::sys` or any cudarc context creation,
/// otherwise cudarc may panic while eagerly binding symbols.
fn ensure_cuda_driver_compatible() -> TractResult<()> {
    // Load driver library without involving cudarc.
    let (lib_name, lib) = load_first_found_cuda_lib()?;
    let (req_major, req_minor) = format_cuda_version(REQUIRED_CUDA_API);

    unsafe {
        // Resolve symbols we need. If these are missing, the driver install is broken or not NVIDIA.
        let cu_init: Symbol<CuInitFn> = lib.get(b"cuInit\0").map_err(|e| {
            format_err!(
                "CUDA driver library loaded ({lib_name}), but symbol cuInit is missing. \
                 This does not look like a functional NVIDIA driver installation. Details: {e}"
            )
        })?;

        let cu_driver_get_version: Symbol<CuDriverGetVersionFn> =
            lib.get(b"cuDriverGetVersion\0").map_err(|e| {
                format_err!(
                    "CUDA driver library loaded ({lib_name}), but symbol cuDriverGetVersion is missing. \
                     Driver is too old or installation is corrupted. Details: {e}"
                )
            })?;

        // Initialize the driver. This also surfaces "no device / permission / container" issues early.
        let init_res = cu_init(0);
        if init_res != 0 {
            // Don't assume specific numeric codes here; just report the CUresult.
            bail!(
                "CUDA driver initialization failed (cuInit returned {}, via {lib_name}). \
                 Possible causes: no CUDA-capable device exposed to this process, \
                 missing /dev/nvidia* nodes (container), insufficient permissions, \
                 or a broken driver install.",
                init_res
            );
        }

        // Query driver API version.
        let mut version: c_int = 0;
        let ver_res = cu_driver_get_version(&mut version as *mut _);
        if ver_res != 0 {
            bail!(
                "cuDriverGetVersion failed (returned {}, via {lib_name}). \
                 NVIDIA driver may be corrupted or not functioning properly.",
                ver_res
            );
        }

        let (found_major, found_minor) = format_cuda_version(version);

        // Compare against required API based on feature gate.
        if version < REQUIRED_CUDA_API {
            bail!(
                "CUDA driver too old.\n\
                 Driver library: {lib_name}\n\
                 Built with cudarc feature cuda-{} (requires driver API >= {}.{}).\n\
                 Found driver API {}.{}.\n\
                 Fix: upgrade the NVIDIA driver, or rebuild tract-cuda with a lower cuda-XXXXX gate.",
                REQUIRED_CUDA_API,
                req_major,
                req_minor,
                found_major,
                found_minor
            );
        }

        log::info!(
            "tract-cuda: CUDA driver {lib_name} OK; API version {found_major}.{found_minor} (built for >= {req_major}.{req_minor})"
        );
    }

    Ok(())
}

/// Probe each cudarc-backed sub-library we rely on and return the name of the first one
/// that fails to load, or `None` if all are present. Cached across calls.
fn first_missing_cudarc_culib() -> Option<&'static str> {
    *CULIBS_MISSING.get_or_init(|| {
        // Names match the short labels cudarc uses in its dynamic-loading candidates.
        // SAFETY: cudarc's `is_culib_present` functions are documented dlopen probes with no
        // side effects beyond the library handle cache.
        let probe = |name: &'static str, ok: bool| -> Option<&'static str> {
            if ok {
                log::debug!("tract-cuda: cudarc probe for {name:?} succeeded");
                None
            } else {
                log::debug!("tract-cuda: cudarc probe for {name:?} failed");
                Some(name)
            }
        };
        probe("cuda (driver)", unsafe { cudarc::driver::sys::is_culib_present() })
            .or_else(|| {
                probe("cudart (runtime)", unsafe { cudarc::runtime::sys::is_culib_present() })
            })
            .or_else(|| probe("nvrtc", unsafe { cudarc::nvrtc::sys::is_culib_present() }))
            .or_else(|| probe("cublas", unsafe { cudarc::cublas::sys::is_culib_present() }))
            .or_else(|| probe("cudnn", unsafe { cudarc::cudnn::sys::is_culib_present() }))
    })
}

pub fn ensure_cuda_runtime_dependencies(context_msg: &'static str) -> TractResult<()> {
    // Fast path: if a previous call already validated the full chain, skip the whole probe.
    // CudaRuntime wires this into both `check()` and `prepare_with_options()`, so the second
    // caller would otherwise re-dlopen libcuda and re-log the init narrative.
    if DEPENDENCIES_OK.get().is_some() {
        return Ok(());
    }

    ensure_cuda_driver_compatible()
        .context("CUDA driver validation failed")
        .context(context_msg)?;

    if let Some(missing) = first_missing_cudarc_culib() {
        bail!(
            "{context_msg}: CUDA runtime sub-library {missing:?} could not be dlopen'd. \
             cudarc searches standard library names derived from the build-time CUDA \
             version; install the matching package or set LD_LIBRARY_PATH so the \
             loader can find it. Enable RUST_LOG=tract_cuda=debug to see each probe."
        );
    }

    log::info!(
        "tract-cuda: all required cudarc sub-libraries present (driver, cudart, nvrtc, cublas, cudnn)"
    );

    let _ = DEPENDENCIES_OK.set(());
    Ok(())
}

pub fn get_ggml_q81_fact(t: &DeviceTensor) -> Option<GgmlQuantQ81Fact> {
    if let DeviceTensor::Owned(t) = t {
        t.exotic_fact().and_then(|of| of.downcast_ref::<GgmlQuantQ81Fact>()).cloned()
    } else if let DeviceTensor::ArenaView(t) = t {
        t.exotic_fact().and_then(|of| of.downcast_ref::<GgmlQuantQ81Fact>()).cloned()
    } else {
        None
    }
}

pub fn pad_q40(bqs: &BlockQuantStorage, m: usize, k: usize) -> TractResult<BlockQuantStorage> {
    ensure!(k % 32 == 0);

    let to_pad = k.next_multiple_of(Q40_ROW_PADDING) - k;
    if to_pad == 0 {
        return Ok(bqs.clone()); // No padding needed
    }

    let row_bytes = k * Q4_0.block_bytes() / Q4_0.block_len();

    let pad_quant = Q4_0.quant_f32(&vec![0f32; to_pad])?;
    let pad_bytes = pad_quant.len();

    let mut new_data = Vec::with_capacity(m * (row_bytes + pad_bytes));
    let old_bytes = bqs.value().as_bytes();

    for row in 0..m {
        let start = row * row_bytes;
        new_data.extend_from_slice(&old_bytes[start..start + row_bytes]);
        new_data.extend_from_slice(&pad_quant);
    }

    BlockQuantStorage::new(
        tract_core::dyn_clone::clone_box(bqs.format()),
        m,
        k + to_pad,
        Arc::new(Blob::from_bytes(&new_data)?),
    )
}
