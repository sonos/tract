/// Wasm SIMD implementation of `MatMatMulKer<f32>`
///
/// To run test, you need to install `wasmtime`
/// and export the following environment variables:
/// ```
/// > export RUSTFLAGS='-C target-feature=+simd128'
/// > export CARGO_TARGET_WASM32_WASI_RUNNER=wasmtime
/// > cargo test --target=wasm32-wasi
/// ```
use crate::mmm::candidate_named;
use crate::{DatumType, Ops};

#[cfg(target_feature = "relaxed-simd")]
use crate::frame::element_wise::ElementWiseKer;
use crate::frame::reduce::ReduceKer;

#[macro_use]
mod madd;

#[cfg(target_feature = "relaxed-simd")]
mod act;
#[cfg(all(test, target_arch = "wasm32", target_feature = "simd128"))]
mod dispatch_tests;
mod mmm_f32_gemm;
mod mmm_f32_gemv;
mod mmm_i32;
mod reduce;

#[cfg(target_feature = "relaxed-simd")]
pub use act::*;
pub use mmm_f32_gemm::*;
pub use mmm_f32_gemv::*;
pub use mmm_i32::*;

pub fn plug(ops: &mut Ops) {
    // Every kernel this policy names must be ManuallyOptimized: `preferred` hands the answer to
    // the candidate list, and a lesser tier would be dropped by retain_best_quality, leaving
    // the N>1 rule to pick max(nr*mr) among the surviving GEMV kernels — i.e. wasm_f32_32x1,
    // a matrix×vector kernel, for every GEMM.
    ops.overlay_mmm_policy(|prev, dt, query, candidates| match (dt, query.n) {
        // int8 -> i32 matmul: SIMD kernel (was generic scalar).
        (DatumType::I32, Some(1)) => prev(dt, query, candidates),
        (DatumType::I32, _) => candidate_named(candidates, &wasm_i32_4x4.name),
        // GEMV routes by M-band to the kernel whose MR fits. Bands derived from
        // benches/wasm.rs: at each edge, using the next-larger kernel beats halving outer
        // iterations of the smaller one (1 outer with ILP-absorbed padding > 2 outer with the
        // kernel preamble doubled). M=4/8/16 are exact tile fits at the lower edges; M=17/9/5
        // are the first values where the next-larger kernel wins.
        (DatumType::F32, Some(1)) => candidate_named(
            candidates,
            match query.m.unwrap_or(0) {
                0..=4 => &wasm_f32_4x1.name,
                5..=8 => &wasm_f32_8x1.name,
                9..=16 => &wasm_f32_16x1.name,
                _ => &wasm_f32_32x1.name,
            },
        ),
        (DatumType::F32, _) => candidate_named(candidates, &wasm_f32_8x8.name),
        _ => prev(dt, query, candidates),
    });
    // Relaxed-SIMD activation kernels (FMA path). Only installed when the
    // build has `+relaxed-simd`; otherwise the slots stay at the generic
    // scalar polynomial.
    #[cfg(target_feature = "relaxed-simd")]
    {
        ops.sigmoid_f32 = Box::new(|| WasmSigmoid4Relaxed::ew());
        ops.tanh_f32 = Box::new(|| WasmTanh4Relaxed::ew());
    }
    ops.max_f32 = Box::new(|| reduce::wasm_max_f32_32n::red());
    ops.min_f32 = Box::new(|| reduce::wasm_min_f32_32n::red());
    ops.sum_f32 = Box::new(|| reduce::wasm_sum_f32_32n::red());
    ops.rms_norm_f32 = Box::new(reduce::rms_norm_f32);
    ops.max_f16 = Box::new(|| reduce::wasm_max_f16_32n::red());
    ops.sum_f16 = Box::new(|| reduce::wasm_sum_f16_32n::red());
}

inventory::submit! {
    crate::platform::PlatformSelector {
        target: crate::platform::Target::Wasm32Simd128,
        plug,
    }
}
