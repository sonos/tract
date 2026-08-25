//! The wasm kernel tree, behind `simd128` and — for the activations — `relaxed-simd`.
//!
//! Running its tests needs `wasmtime` and the wasi target:
//!
//! ```text
//! RUSTFLAGS='-C target-feature=+simd128,+relaxed-simd' \
//! CARGO_TARGET_WASM32_WASIP1_RUNNER=wasmtime \
//! cargo test --target wasm32-wasip1 -p tract-linalg
//! ```
use crate::DatumType;

#[macro_use]
mod madd;

mod act;
#[cfg(all(test, target_arch = "wasm32", target_feature = "simd128"))]
mod dispatch_tests;
mod mmm_f32_gemm;
mod mmm_f32_gemv;
mod mmm_i32;
mod reduce;

pub use act::*;
pub use mmm_f32_gemm::*;
pub use mmm_f32_gemv::*;
pub use mmm_i32::*;

/// Every kernel this tier names must be ManuallyOptimized: its answer is held to the suitable
/// list, and a lesser quality would be dropped by retain_best_quality, leaving the N>1 rule to
/// pick max(nr*mr) among the surviving GEMV kernels — i.e. wasm_f32_32x1, a matrix×vector
/// kernel, for every GEMM.
fn preferred(
    _isa: &crate::isa::IsaSet,
    dt: DatumType,
    query: &crate::mmm::Query,
    _suitable: &[crate::mmm::Suitable],
) -> Option<&'static str> {
    match (dt, query.n) {
        // int8 -> i32 matmul: SIMD kernel (the generic scalar is the tier below).
        (DatumType::I32, Some(1)) => None,
        (DatumType::I32, _) => Some(wasm_i32_4x4.name.as_str()),
        // GEMV routes by M-band to the kernel whose MR fits. Bands derived from
        // benches/wasm.rs: at each edge, using the next-larger kernel beats halving outer
        // iterations of the smaller one (1 outer with ILP-absorbed padding > 2 outer with the
        // kernel preamble doubled). M=4/8/16 are exact tile fits at the lower edges; M=17/9/5
        // are the first values where the next-larger kernel wins.
        (DatumType::F32, Some(1)) => Some(match query.m.unwrap_or(0) {
            0..=4 => &wasm_f32_4x1.name,
            5..=8 => &wasm_f32_8x1.name,
            9..=16 => &wasm_f32_16x1.name,
            _ => &wasm_f32_32x1.name,
        }),
        (DatumType::F32, _) => Some(wasm_f32_8x8.name.as_str()),
        _ => None,
    }
}

inventory::submit! {
    crate::mmm_tiers::MmmTier {
        arch: Some(crate::isa::Arch::Wasm32Simd128),
        precedence: 1,
        name: "wasm-simd128",
        applies: |_| true,
        preferred,
    }
}

routine!(wasm32; F32, Sigmoid, WasmSigmoid4Relaxed, isa(Wasm32RelaxedSimd));
routine!(wasm32; F32, Tanh, WasmTanh4Relaxed, isa(Wasm32RelaxedSimd));

/// What this build offers, in the shared vocabulary. Unlike the other trees this is a build
/// question rather than a probe: wasm features are enabled at compile time and a module cannot
/// ask the engine what it got.
pub fn isa_set() -> crate::isa::IsaSet {
    use crate::isa::{Isa, IsaSet};
    let mut set = IsaSet::of_arch(crate::isa::Arch::Wasm32Simd128).with(Isa::Wasm32Simd128);
    if cfg!(target_feature = "relaxed-simd") {
        set = set.with(Isa::Wasm32RelaxedSimd);
    }
    set
}

routine!(wasm32; RmsNormF32, RmsNorm, "wasm_rms_norm_f32", reduce::rms_norm_f32);
