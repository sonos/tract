use std::ffi::c_int;
use std::sync::OnceLock;

use anyhow::Context;
use libloading::{Library, Symbol};

use tract_core::internal::tract_smallvec::ToSmallVec;
use tract_core::internal::*;
use tract_core::tract_linalg::block_quant::*;
use tract_gpu::tensor::DeviceTensor;

use crate::Q40_ROW_PADDING;
use crate::ops::GgmlQuantQ81Fact;

static CULIBS_PRESENT: OnceLock<bool> = OnceLock::new();

// Ensure exactly cuda-12060 is enabled.
// Prevent accidental change of feature gate without
// updating this required API version used for compatibility check.
// please update the 3 references bellow if cudarc gate is updated to a newer version.
pub const REQUIRED_CUDA_API: i32 = 12060;
#[cfg(not(feature = "cuda-12060"))]
compile_error!(
    "Tract CUDA backend currently supports only cudarc feature 'cuda-12060'. \
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

fn load_first_found_cuda_lib() -> TractResult<Library> {
    for name in CUDA_DRIVER_LIB_CANDIDATES {
        if let Ok(lib) = unsafe { Library::new(name) } {
            return Ok(lib);
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
    let lib = load_first_found_cuda_lib()?;

    unsafe {
        // Resolve symbols we need. If these are missing, the driver install is broken or not NVIDIA.
        let cu_init: Symbol<CuInitFn> = lib.get(b"cuInit\0").map_err(|e| {
            format_err!(
                "CUDA driver library loaded, but symbol cuInit is missing. \
                 This does not look like a functional NVIDIA driver installation. Details: {e}"
            )
        })?;

        let cu_driver_get_version: Symbol<CuDriverGetVersionFn> =
            lib.get(b"cuDriverGetVersion\0").map_err(|e| {
                format_err!(
                    "CUDA driver library loaded, but symbol cuDriverGetVersion is missing. \
                     Driver is too old or installation is corrupted. Details: {e}"
                )
            })?;

        // Initialize the driver. This also surfaces "no device / permission / container" issues early.
        let init_res = cu_init(0);
        if init_res != 0 {
            // Don't assume specific numeric codes here; just report the CUresult.
            bail!(
                "CUDA driver initialization failed (cuInit returned {}). \
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
                "cuDriverGetVersion failed (returned {}). \
                 NVIDIA driver may be corrupted or not functioning properly.",
                ver_res
            );
        }

        // Compare against required API based on feature gate.
        if version < REQUIRED_CUDA_API {
            let (req_major, req_minor) = format_cuda_version(REQUIRED_CUDA_API);
            let (found_major, found_minor) = format_cuda_version(version);

            bail!(
                "CUDA driver too old.\n\
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
    }

    Ok(())
}

fn are_cudarc_required_culibs_present() -> bool {
    *CULIBS_PRESENT.get_or_init(|| unsafe {
        cudarc::driver::sys::is_culib_present()
            && cudarc::runtime::sys::is_culib_present()
            && cudarc::nvrtc::sys::is_culib_present()
            && cudarc::cublas::sys::is_culib_present()
    })
}

pub fn ensure_cuda_runtime_dependencies(context_msg: &str) -> TractResult<()> {
    ensure_cuda_driver_compatible()
        .context("CUDA driver validation failed")
        .context(context_msg)?;

    ensure!(
        are_cudarc_required_culibs_present(),
        "{}: CUDA runtime sub-libraries missing/missaligned with cudarc.",
        context_msg
    );

    Ok(())
}

pub fn get_ggml_q81_fact(t: &DeviceTensor) -> Option<GgmlQuantQ81Fact> {
    if let DeviceTensor::Owned(t) = t {
        t.opaque_fact().and_then(|of| of.downcast_ref::<GgmlQuantQ81Fact>()).cloned()
    } else if let DeviceTensor::ArenaView(t) = t {
        t.opaque_fact().and_then(|of| of.downcast_ref::<GgmlQuantQ81Fact>()).cloned()
    } else {
        None
    }
}

pub fn pad_q40(q40_bqv: &BlobWithFact) -> TractResult<BlobWithFact> {
    let q40_bqf =
        q40_bqv.fact.downcast_ref::<BlockQuantFact>().context("Expected BlockQuantFact")?;
    let shape = q40_bqf.shape();
    ensure!(shape.len() >= 2);

    let k = *shape.last().unwrap();
    ensure!(k % 32 == 0);

    let to_pad = k.next_multiple_of(Q40_ROW_PADDING) - k;
    if to_pad == 0 {
        return Ok(q40_bqv.clone()); // No padding needed
    }

    let outer_rows: usize = shape[..shape.len() - 1].iter().product();
    let row_bytes = k * Q4_0.block_bytes() / Q4_0.block_len();

    let pad_quant = Q4_0.quant_f32(&vec![0f32; to_pad])?;
    let pad_bytes = pad_quant.len();

    let mut new_data = Vec::with_capacity(outer_rows * (row_bytes + pad_bytes));
    let old_bytes = q40_bqv.value.as_bytes();

    for row in 0..outer_rows {
        let start = row * row_bytes;
        new_data.extend_from_slice(&old_bytes[start..start + row_bytes]);
        new_data.extend_from_slice(&pad_quant);
    }
    let mut new_shape = shape.to_smallvec();
    *new_shape.last_mut().unwrap() += to_pad;

    Ok(BlobWithFact {
        fact: Box::new(BlockQuantFact::new(q40_bqf.format.clone(), new_shape)),
        value: Arc::new(Blob::from_bytes(&new_data)?),
    })
}
