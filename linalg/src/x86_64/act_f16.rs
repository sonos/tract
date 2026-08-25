//! AVX-512 f16 element-wise activations for cores without native f16 arithmetic.
//!
//! Each kernel round-trips through the matching f32 AVX-512 kernel via
//! `ew_kernel_via_f32!`: convert an f16 chunk into a 64-byte-aligned f32 scratch
//! (the f32 kernels assume 64-byte-aligned input), run the f32 kernel, convert
//! back.
//! Conversion is driven through `std::arch` intrinsics directly (see the
//! helpers below) because rustc + LLVM do not autovectorize the scalar
//! `f16::to_f32` / `f16::from_f32` loops.

use tract_data::internal::f16;

const CHUNK: usize = 256;

/// SiLU's f32 scratch length, wider than `CHUNK`.
///
/// Every call into an f32 kernel pays a fixed cost that does not shrink with the
/// length passed to it: MXCSR is saved, overwritten and restored, resynchronising
/// the FP pipeline at both ends. `avx512_silu_f32` needs a longer call than
/// `CHUNK` to amortise that and to get its four 64-lane groups in flight across
/// the divide; at `CHUNK` the ymm `fma_silu_f32`, whose groups are 16 lanes wide,
/// wins instead. Must stay a multiple of `nr`.
const SILU_CHUNK: usize = 1024;

// Vectorized f16 <-> f32 helpers using vcvtph2ps / vcvtps2ph. Rustc + LLVM
// do NOT autovectorize the scalar `.to_f32()` loop (the half crate's method
// has branches / function-call overhead), so we drive the conversion with
// intrinsics directly. Both helpers process 16 lanes per iteration; the tail
// (which only fires for the 1-15 leftover lanes inside a CHUNK-sized batch)
// falls back to scalar.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn cvt_f16_to_f32(src: &[f16], dst: &mut [f32]) {
    use core::arch::x86_64::*;
    let n = src.len();
    debug_assert!(dst.len() >= n);
    let chunks = n / 16;
    unsafe {
        for k in 0..chunks {
            let m = _mm256_loadu_si256(src.as_ptr().add(k * 16) as *const __m256i);
            let z = _mm512_cvtph_ps(m);
            _mm512_storeu_ps(dst.as_mut_ptr().add(k * 16), z);
        }
        for k in (chunks * 16)..n {
            *dst.get_unchecked_mut(k) = src.get_unchecked(k).to_f32();
        }
    }
}

bail_stub!(x86_64; unsafe fn cvt_f16_to_f32(&[f16], &mut [f32]));

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn cvt_f32_to_f16(src: &[f32], dst: &mut [f16]) {
    use core::arch::x86_64::*;
    let n = src.len();
    debug_assert!(dst.len() >= n);
    let chunks = n / 16;
    unsafe {
        for k in 0..chunks {
            let z = _mm512_loadu_ps(src.as_ptr().add(k * 16));
            // _MM_FROUND_TO_NEAREST_INT == 0 (round-to-nearest-even, matches f16::from_f32)
            let m = _mm512_cvtps_ph::<0>(z);
            _mm256_storeu_si256(dst.as_mut_ptr().add(k * 16) as *mut __m256i, m);
        }
        for k in (chunks * 16)..n {
            *dst.get_unchecked_mut(k) = f16::from_f32(*src.get_unchecked(k));
        }
    }
}

bail_stub!(x86_64; unsafe fn cvt_f32_to_f16(&[f32], &mut [f16]));

// hardswish_f16
routine_ew_via_f32!(x86_64;
    x86_64_avx512_hardswish_f16_64n,
    64,
    32,
    CHUNK,
    64,
    cvt_f16_to_f32,
    cvt_f32_to_f16,
    super::act::x86_64_avx512_hardswish_f32_64n,
    func(Hardswish),
    isa(X86_64Avx512f)
);

routine_ew_via_f32!(x86_64;
    x86_64_avx512_leaky_relu_f16_64n,
    64,
    32,
    CHUNK,
    64,
    cvt_f16_to_f32,
    cvt_f32_to_f16,
    super::act::x86_64_avx512_leaky_relu_f32_64n,
    func(LeakyRelu),
    param(alpha => alpha.to_f32()),
    isa(X86_64Avx512f)
);

routine_ew_via_f32!(x86_64;
    x86_64_avx512_sigmoid_f16_16n,
    16,
    16,
    CHUNK,
    64,
    cvt_f16_to_f32,
    cvt_f32_to_f16,
    super::avx512_sigmoid_f32,
    func(Sigmoid),
    isa(X86_64Avx512f)
);

routine_ew_via_f32!(x86_64;
    x86_64_avx512_tanh_f16_16n,
    16,
    16,
    CHUNK,
    64,
    cvt_f16_to_f32,
    cvt_f32_to_f16,
    super::avx512_tanh_f32,
    func(Tanh),
    isa(X86_64Avx512f)
);

routine_ew_via_f32!(x86_64;
    x86_64_avx512_silu_f16_16n,
    16,
    16,
    SILU_CHUNK,
    64,
    cvt_f16_to_f32,
    cvt_f32_to_f16,
    super::avx512_silu_f32,
    func(Silu),
    isa(X86_64Avx512f)
);

routine_ew_via_f32!(x86_64;
    x86_64_avx512_gelu_f16_16n,
    16,
    16,
    CHUNK,
    64,
    cvt_f16_to_f32,
    cvt_f32_to_f16,
    super::act::x86_64_avx512_gelu_f32_16n,
    func(Gelu),
    isa(X86_64Avx512f)
);
