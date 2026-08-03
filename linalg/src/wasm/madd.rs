// f32x4 mul+add → relaxed FMA when the build has +relaxed-simd, else explicit
// mul+add. Lets the MMM kernels emit f32x4.relaxed_madd without duplicating
// kernel source. Per PR #2199: LLVM does not auto-emit relaxed_madd from
// f32x4_add(f32x4_mul(...)) even with +relaxed-simd — hand emission is needed.
//
// Caller must have `use std::arch::wasm32::*;` in scope (every kernel does).
// Args are passed (acc, a, b); evaluation order differs between the two arms
// (acc-first in baseline, acc-last in FMA), so callers must pass simple
// variable names rather than expressions with side effects.
#[cfg(target_feature = "relaxed-simd")]
macro_rules! madd_f32x4 {
    ($acc:expr, $a:expr, $b:expr) => {
        f32x4_relaxed_madd($a, $b, $acc)
    };
}

#[cfg(not(target_feature = "relaxed-simd"))]
macro_rules! madd_f32x4 {
    ($acc:expr, $a:expr, $b:expr) => {
        f32x4_add($acc, f32x4_mul($a, $b))
    };
}

// Always-non-fused madd. Used by kernels with ≤4 SIMD accumulators per K-step
// (wasm_f32_4x1, _8x1, _16x1, _4x4), where the destructive `fmla.4s`
// emitted by +relaxed-simd creates a 4-cycle accumulator RAW recurrence
// that throttles throughput to 1 FMA/cycle even though Apple-class ARM64
// pipes can do 4. The separate `fmul.4s; fadd.4s` form gives each multiply
// a fresh destination register, letting the OoO renamer overlap the next
// iteration's multiply with the in-flight add. Measured: under
// +simd128,+relaxed-simd these kernels are 19-28% slower than under
// +simd128 when using the fused form on Apple M1 — both wasmtime
// (Cranelift) and Node 20 (V8) reproduce identically. Wider kernels
// (wasm_f32_32x1 with 8 accs, wasm_f32_8x8 with 16) keep the fused form
// because their pipe is saturated and FMA's 1-instruction-per-madd wins.
//
// Cross-check: XNNPACK only ships wasmrelaxedsimd-fma GEMM kernels at
// NR=8 (i.e. ≥8 accumulator-equivalents), independently arriving at the
// same threshold without writing it down.
macro_rules! madd_f32x4_nofma {
    ($acc:expr, $a:expr, $b:expr) => {
        f32x4_add($acc, f32x4_mul($a, $b))
    };
}
