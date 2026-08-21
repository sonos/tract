/// Wasm SIMD implementation of `MatMatMulKer<f32>`
///
/// To run test, you need to install `wasmtime`
/// and export the following environment variables:
/// ```
/// > export RUSTFLAGS='-C target-feature=+simd128'
/// > export CARGO_TARGET_WASM32_WASI_RUNNER=wasmtime
/// > cargo test --target=wasm32-wasi
/// ```
use crate::Ops;

use crate::frame::element_wise::ElementWiseKer;
use crate::frame::reduce::{MapReduceKer, ReduceKer};

#[macro_use]
mod madd;

#[cfg(target_feature = "relaxed-simd")]
mod act;
mod act_f32;
#[cfg(test)]
mod dispatch_tests;
mod mmm_f32_gemm;
mod mmm_f32_gemv;
mod mmm_i32;
mod reduce;

#[cfg(target_feature = "relaxed-simd")]
pub use act::*;
pub use act_f32::*;
pub use mmm_f32_gemm::*;
pub use mmm_f32_gemv::*;
pub use mmm_i32::*;

pub fn plug(ops: &mut Ops) {
    ops.mmm_impls.push(wasm_f32_4x4.mmm());
    ops.mmm_impls.push(wasm_f32_4x1.mmm());
    ops.mmm_impls.push(wasm_f32_8x1.mmm());
    ops.mmm_impls.push(wasm_f32_16x1.mmm());
    ops.mmm_impls.push(wasm_f32_32x1.mmm());
    ops.mmm_impls.push(wasm_f32_8x8.mmm());
    // int8 -> i32 matmul: SIMD kernel (was generic scalar). ManuallyOptimized so
    // strategize's retain() keeps it over generic_i32_4x4 for i8 packing.
    ops.mmm_impls.push(wasm_i32_4x4.mmm());
    ops.qmmm_i32 = Box::new(|_, _, _| wasm_i32_4x4.mmm());
    // Selection paths. Both rely on kernel_selection::strategize honouring
    // the mmm_f32 / mmv_f32 callback, which it only does when the callback's
    // kernel is tagged ManuallyOptimized. Otherwise strategize falls through
    // to list_impls, whose retain() keeps only the top ImplementationQuality
    // and drops every TargetOptimized kernel.
    //   - N>1 (GEMM): mmm_f32 returns 8x8, so 8x8 MUST be ManuallyOptimized.
    //     If it were TargetOptimized it would be dropped by retain(), and the
    //     N>1 branch's max(nr*mr) over the surviving (ManuallyOptimized) GEMV
    //     kernels would pick wasm_f32_32x1 — a matrix×vector kernel — for
    //     every GEMM.
    //   - N=1 (GEMV): mmv_f32 routes by M-band to the kernel whose MR fits.
    //     The four GEMV kernels are ManuallyOptimized for the same reason —
    //     without the tag strategize discards the callback and picks
    //     max(mr)=32x1 for every M, leaving up to ~37% on the table for
    //     small-M GEMV.
    ops.mmm_f32 = Box::new(|_m, _k, _n| wasm_f32_8x8.mmm());
    // Bands derived from benches/wasm.rs. At each band edge, using
    // the next-larger kernel beats halving outer iterations of the smaller
    // one (1 outer with ILP-absorbed padding > 2 outer with kernel preamble
    // doubled). M=4/8/16 are exact tile fits at the lower edges; M=17/9/5
    // are the first values where the next-larger kernel wins.
    ops.mmv_f32 = Box::new(|m, _k| match m.unwrap_or(0) {
        0..=4 => wasm_f32_4x1.mmm(),
        5..=8 => wasm_f32_8x1.mmm(),
        9..=16 => wasm_f32_16x1.mmm(),
        _ => wasm_f32_32x1.mmm(),
    });
    ops.erf_f32 = Box::new(|| WasmErf4::ew());
    ops.silu_f32 = Box::new(|| WasmSilu4::ew());
    ops.gelu_f32 = Box::new(|| WasmGelu4::ew());
    // Sigmoid and tanh stay on the generic polynomial kernels, which LLVM
    // auto-vectorizes under +simd128; only the relaxed-simd FMA variants
    // beat them.
    #[cfg(target_feature = "relaxed-simd")]
    {
        ops.sigmoid_f32 = Box::new(|| WasmSigmoid4Relaxed::ew());
        ops.tanh_f32 = Box::new(|| WasmTanh4Relaxed::ew());
    }
    ops.max_f32 = Box::new(|| reduce::wasm_max_f32_32n::red());
    ops.min_f32 = Box::new(|| reduce::wasm_min_f32_32n::red());
    ops.sum_f32 = Box::new(|| reduce::wasm_sum_f32_32n::red());
    ops.softmax2_fastcompact_f32 = Box::new(|| reduce::wasm_softmax2_fastcompact_f32_32n::red());
    ops.rms_norm_f32 = Box::new(reduce::rms_norm_f32);
    ops.max_f16 = Box::new(|| reduce::wasm_max_f16_32n::red());
    ops.sum_f16 = Box::new(|| reduce::wasm_sum_f16_32n::red());
    ops.softmax2_fastcompact_f16 = Box::new(|| reduce::wasm_softmax2_fastcompact_f16_32n::red());
}
