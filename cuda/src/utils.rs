use std::ffi::c_int;
use std::sync::OnceLock;

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

unsafe extern "C" {
    unsafe fn cuDriverGetVersion(version: *mut c_int) -> c_int;
    unsafe fn cuInit(flags: u32) -> c_int;
}

pub fn ensure_cuda_driver_compatible() -> TractResult<()> {
    unsafe {
        // Step 1: Initialize driver
        let init_res = cuInit(0);
        if init_res != 0 {
            bail!(
                "CUDA driver initialization failed (cuInit returned {}). \
                 Is an NVIDIA driver installed and is a CUDA-capable GPU available?",
                init_res
            );
        }

        // Step 2: Query version
        let mut version: c_int = 0;
        let res = cuDriverGetVersion(&mut version as *mut _);

        if res != 0 {
            bail!(
                "cuDriverGetVersion failed with code {}. \
                 The NVIDIA driver may be corrupted or improperly installed.",
                res
            );
        }

        if version < REQUIRED_CUDA_API {
            let req_major = REQUIRED_CUDA_API / 1000;
            let req_minor = (REQUIRED_CUDA_API % 1000) / 10;

            let found_major = version / 1000;
            let found_minor = (version % 1000) / 10;

            bail!(
                "CUDA driver too old.\n\
                 Built with cudarc feature cuda-{} (requires >= {}.{}).\n\
                 Found driver API {}.{}.\n\
                 Upgrade NVIDIA driver.",
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

pub fn are_culibs_present() -> bool {
    *CULIBS_PRESENT.get_or_init(|| unsafe {
        cudarc::driver::sys::is_culib_present()
            && cudarc::runtime::sys::is_culib_present()
            && cudarc::nvrtc::sys::is_culib_present()
            && cudarc::cublas::sys::is_culib_present()
    })
}

pub fn ensure_cuda_runtime_dependencies() -> TractResult<()> {
    ensure_cuda_driver_compatible()?;
    ensure!(are_culibs_present(), "One or more required CUDA libraries are missing or too old.");
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
