use crate::Ops;

// `tract_sve` is set by build.rs only on aarch64-linux when the C compiler
// supports SVE intrinsics. The kernel registration + extern live behind it so
// non-SVE builds never reference the (absent) C symbol.
#[cfg(tract_sve)]
use crate::frame::mmm::ImplementationQuality::ManuallyOptimized;
#[cfg(tract_sve)]
use crate::mmm::*;
#[cfg(tract_sve)]
use crate::pack::PackedFormat;
// Explicit import so `f16` is tract's half::f16 (LADatum), not rustc's builtin
// primitive f16 — a glob import would not shadow the primitive.
#[cfg(tract_sve)]
use tract_data::prelude::f16;

// f32 SVE kernel can't do LeakyRelu or the i32 quantization ops (matches the
// arm64simd / SME f32 CAN_FUSE).
#[cfg(tract_sve)]
const CAN_FUSE: fn(&FusedSpec) -> bool = |f| {
    !matches!(
        f,
        FusedSpec::LeakyRelu(_)
            | FusedSpec::QScale(_, _, _)
            | FusedSpec::RoundingShiftRight(_, _)
            | FusedSpec::ShiftLeft(_)
    )
};

// The i32 quantized kernel keeps the quantization fuse ops (QScale /
// RoundingShiftRight / ShiftLeft) — they are the whole point of a quantized
// kernel — and excludes only LeakyRelu (matches arm64simd's i32 surface; i32
// LeakyRelu has no practical use and the C kernel does not implement it).
#[cfg(tract_sve)]
const CAN_FUSE_I32: fn(&FusedSpec) -> bool = |f| !matches!(f, FusedSpec::LeakyRelu(_));

#[cfg(tract_sve)]
const SVE2: fn() -> bool = has_sve2;

// The f16 kernels need FEAT_SVE2 AND FEAT_FP16 (native f16 arithmetic).
#[cfg(tract_sve)]
const SVE2_FP16: fn() -> bool = || has_sve2() && crate::arm64::has_fp16();

// The VLA SVE f32 GEMM kernel, implemented in C (arm64/sve/sve_mmm_f32.c) since
// Rust has no stable SVE intrinsics. Broadcast-A rank-1 update, N-tile walked in
// svcntw() chunks → correct and full-width at any VL.
#[cfg(tract_sve)]
mod sve_sys {
    use crate::frame::mmm::FusedKerSpec;
    use tract_data::prelude::f16;
    unsafe extern "C" {
        pub fn sve_mmm_f32_kernel(ops: *const FusedKerSpec<f32>) -> isize;
        pub fn sve_mmv_f32_64x1_kernel(ops: *const FusedKerSpec<f32>) -> isize;
        pub fn sve_mmm_i32_kernel(ops: *const FusedKerSpec<i32>) -> isize;
        pub fn sve_mmm_i32_64x1_kernel(ops: *const FusedKerSpec<i32>) -> isize;
        pub fn sve_mmm_f16_kernel(ops: *const FusedKerSpec<f16>) -> isize;
        pub fn sve_mmv_f16_64x1_kernel(ops: *const FusedKerSpec<f16>) -> isize;
        // VLA SVE2 fused row-wise RmsNorm (arm64/sve/sve_rms_norm.c). Plugged
        // into Ops::rms_norm_f32 when FEAT_SVE2 is present, overriding the
        // NEON 4-lane kernel with wider lanes (vl-dependent) and a predicated
        // tail (no scalar epilogue).
        pub fn sve_rms_norm_f32_kernel(buf: *mut f32, n: i64, eps: f32);
    }
}

/// Public Rust wrapper for the VLA SVE2 RmsNorm kernel. Matches the
/// `Box<dyn Fn(&mut [f32], f32)>` shape of `Ops::rms_norm_f32`.
#[cfg(tract_sve)]
pub fn sve_rms_norm_f32(buf: &mut [f32], eps: f32) {
    if buf.is_empty() {
        return;
    }
    unsafe { sve_sys::sve_rms_norm_f32_kernel(buf.as_mut_ptr(), buf.len() as i64, eps) }
}

#[cfg(tract_sve)]
MMMRustKernel!(sve_sys::sve_mmm_f32_kernel => sve_mmm_f32_8x8<f32>(8, 8)
    where(SVE2)
    can_fuse(CAN_FUSE)
    quality(ManuallyOptimized)
);

// The VLA SVE f32 GEMV kernel (arm64/sve/sve_mmv_f32_64x1.c), MR=64 NR=1,
// dispatched when N == 1 (matrix x f32 column vector). Wired to ops.mmv_f32.
#[cfg(tract_sve)]
MMMRustKernel!(sve_sys::sve_mmv_f32_64x1_kernel => sve_mmv_f32_64x1<f32>(64, 1)
    where(SVE2)
    can_fuse(CAN_FUSE)
    quality(ManuallyOptimized)
);

// The VLA SVE int8 -> int32 GEMM kernel (arm64/sve/sve_mmm_i32.c). Consumes
// tract's native i8i8 K-major packing via the widening rank-1 update (svld1sb +
// svmla), and supports the i32 quantization fuse ops. Wired to ops.qmmm_i32.
#[cfg(tract_sve)]
MMMRustKernel!(sve_sys::sve_mmm_i32_kernel => sve_mmm_i32_8x8<i32>(8, 8)
    where(SVE2)
    can_fuse(CAN_FUSE_I32)
    packing[1] = i8i8 => |k| k.with_packing(
        PackedFormat::new(DatumType::I8, 8, 16),
        PackedFormat::new(DatumType::I8, 8, 16),
    );
    quality(ManuallyOptimized)
    store(i8)
);

// The VLA SVE int8 -> int32 GEMV kernel (arm64/sve/sve_mmm_i32_64x1.c), MR=64
// NR=1, dispatched when N == 1. Same widening update vectorized over M. Wired to
// ops.qmmv_i32.
#[cfg(tract_sve)]
MMMRustKernel!(sve_sys::sve_mmm_i32_64x1_kernel => sve_mmm_i32_64x1<i32>(64, 1)
    where(SVE2)
    can_fuse(CAN_FUSE_I32)
    packing[1] = i8i8 => |k| k.with_packing(
        PackedFormat::new(DatumType::I8, 64, 16),
        PackedFormat::new(DatumType::I8, 1, 1),
    );
    quality(ManuallyOptimized)
    store(i8)
);

// The VLA SVE f16 GEMM kernel (arm64/sve/sve_mmm_f16.c), native f16 FMA, gated on
// SVE2 + FP16. Wired to ops.mmm_f16 when has_fp16().
#[cfg(tract_sve)]
MMMRustKernel!(sve_sys::sve_mmm_f16_kernel => sve_mmm_f16_8x8<f16>(8, 8)
    where(SVE2_FP16)
    can_fuse(CAN_FUSE)
    quality(ManuallyOptimized)
);

// The VLA SVE f16 GEMV kernel (arm64/sve/sve_mmv_f16_64x1.c), MR=64 NR=1,
// dispatched when N == 1. Wired to ops.mmv_f16 when has_fp16().
#[cfg(tract_sve)]
MMMRustKernel!(sve_sys::sve_mmv_f16_64x1_kernel => sve_mmv_f16_64x1<f16>(64, 1)
    where(SVE2_FP16)
    can_fuse(CAN_FUSE)
    quality(ManuallyOptimized)
);

// SVE / SVE2 backend.
//
// Unlike SME (Apple M4) and AMX (Apple), SVE/SVE2 is NOT present on any Apple
// silicon — it lives on ARMv9 server/mobile cores (Neoverse V1+/N2+, Cortex-X2+
// / A510+, Graviton 3/4). So detection is Linux-only in practice; macOS always
// returns false.
//
// The kernels are vector-length-agnostic (VLA): they read the vector width at
// runtime via `whilelt` predication and `svcntw()`, so a single kernel is
// correct at every VL (128..2048-bit). That means — unlike the SME kernels,
// which hardcoded SVL=512 and needed an RDSVL gate — the SVE kernels need NO
// vector-length gate for correctness. `rdvl_bytes()` is provided only for
// optional VL-matched dispatch (selecting a wider-tiled variant when the
// hardware VL is large), not for correctness.

#[cfg(target_os = "linux")]
pub fn has_sve() -> bool {
    if std::env::var_os("TRACT_SVE_DISABLE").is_some() {
        return false;
    }
    // HWCAP_SVE = 1 << 22 on aarch64 (kernel ABI).
    const HWCAP_SVE: u64 = 1 << 22;
    unsafe extern "C" {
        fn getauxval(t: u64) -> u64;
    }
    const AT_HWCAP: u64 = 16;
    unsafe { (getauxval(AT_HWCAP) & HWCAP_SVE) != 0 }
}

#[cfg(not(target_os = "linux"))]
pub fn has_sve() -> bool {
    // No Apple silicon implements SVE; no SVE on non-Linux targets we support.
    false
}

#[cfg(target_os = "linux")]
pub fn has_sve2() -> bool {
    if std::env::var_os("TRACT_SVE_DISABLE").is_some() {
        return false;
    }
    // HWCAP2_SVE2 = 1 << 1 on aarch64 (kernel ABI).
    const HWCAP2_SVE2: u64 = 1 << 1;
    unsafe extern "C" {
        fn getauxval(t: u64) -> u64;
    }
    const AT_HWCAP2: u64 = 26;
    unsafe { (getauxval(AT_HWCAP2) & HWCAP2_SVE2) != 0 }
}

#[cfg(not(target_os = "linux"))]
pub fn has_sve2() -> bool {
    false
}

/// SVE vector length in bytes, via `RDVL x0, #1` (encoding 0x04bf5020).
/// Legal whenever FEAT_SVE is implemented; callers MUST confirm `has_sve()`
/// first (RDVL is UNDEFINED without SVE and would SIGILL). Used only for
/// optional VL-matched kernel selection — VLA kernels do not need it.
#[cfg(target_os = "linux")]
#[allow(dead_code)]
pub fn rdvl_bytes() -> u64 {
    let vl: u64;
    unsafe {
        std::arch::asm!(
            ".inst 0x04bf5020", // rdvl x0, #1
            out("x0") vl,
            options(nomem, nostack, preserves_flags),
        );
    }
    vl
}

pub fn plug(ops: &mut Ops) {
    let _ = ops;
    if has_sve2() {
        #[cfg(target_os = "linux")]
        log::info!("SVE2 optimisation available (VL = {} bytes)", rdvl_bytes());
        #[cfg(tract_sve)]
        {
            // Force the SVE kernels for f32 mmm and i32 quantized mmm (mirrors the
            // SME backend) and also register them as candidates. TRACT_SVE_DISABLE=1
            // already turns the whole thing off via has_sve2().
            ops.mmm_f32 = Box::new(|_, _, _| sve_mmm_f32_8x8.mmm());
            ops.mmv_f32 = Box::new(|_, _| sve_mmv_f32_64x1.mmm());
            ops.qmmm_i32 = Box::new(|_, _, _| sve_mmm_i32_8x8.mmm());
            ops.qmmv_i32 = Box::new(|_, _| sve_mmm_i32_64x1.mmm());
            // RmsNorm: override the NEON-default plug from arm64::plug() with
            // the wider VLA SVE2 kernel. Same Box<Fn> shape as the linalg slot.
            ops.rms_norm_f32 = Box::new(sve_rms_norm_f32);
            ops.mmm_impls.extend_from_slice(&[
                sve_mmm_f32_8x8.mmm(),
                sve_mmv_f32_64x1.mmm(),
                sve_mmm_i32_8x8.mmm(),
                sve_mmm_i32_64x1.mmm(),
            ]);
            // f16 kernels additionally require FEAT_FP16.
            if crate::arm64::has_fp16() {
                ops.mmm_f16 = Box::new(|_, _, _| sve_mmm_f16_8x8.mmm());
                ops.mmv_f16 = Box::new(|_, _| sve_mmv_f16_64x1.mmm());
                ops.mmm_impls.extend_from_slice(&[sve_mmm_f16_8x8.mmm(), sve_mmv_f16_64x1.mmm()]);
            }
        }
    } else if has_sve() {
        log::info!("SVE (v1) present; SVE2 kernels not enabled");
    } else {
        log::info!("No SVE optimisation");
    }
}

#[cfg(all(test, tract_sve))]
mod rms_norm_tests {
    use super::*;

    fn scalar_ref(buf: &mut [f32], eps: f32) {
        let n = buf.len() as f32;
        let s: f32 = buf.iter().map(|x| x * x).sum();
        let inv_std = (s / n + eps).sqrt().recip();
        for x in buf.iter_mut() {
            *x *= inv_std;
        }
    }

    fn close_enough(got: &[f32], want: &[f32], n: usize) {
        // Tolerance scaled by sqrt(n) for FMA-reorder error in the sum_sq
        // reduction (different lane groupings vs scalar sequential add).
        let rel = 1e-5 + (n as f32).sqrt() * 1e-7;
        let abs = 1e-5;
        for (i, (g, w)) in got.iter().zip(want.iter()).enumerate() {
            let tol = (rel * w.abs().max(1.0)).max(abs);
            let diff = (g - w).abs();
            assert!(diff <= tol, "idx {i}: got {g} want {w} diff {diff} tol {tol}");
        }
    }

    fn check(n: usize, f: impl Fn(usize) -> f32) {
        if !has_sve2() {
            eprintln!("SVE2 not present, skipping (n={n})");
            return;
        }
        let mut sve_buf: Vec<f32> = (0..n).map(&f).collect();
        let mut ref_buf = sve_buf.clone();
        sve_rms_norm_f32(&mut sve_buf, 1e-5);
        scalar_ref(&mut ref_buf, 1e-5);
        close_enough(&sve_buf, &ref_buf, n);
    }

    #[test]
    fn empty_is_noop() {
        let mut x: Vec<f32> = vec![];
        sve_rms_norm_f32(&mut x, 1e-5);
        assert!(x.is_empty());
    }

    #[test]
    fn short_below_step() {
        // n < 4*vl on any vl (vl >= 4 lanes always) — exercise the predicated
        // tail entirely.
        for n in [1usize, 3, 7, 8, 15, 16, 17, 31, 32, 33] {
            check(n, |i| ((i as f32 * 0.13).sin() * 5.0) - 0.5);
        }
    }

    #[test]
    fn matches_reference_1024_with_tail() {
        check(1024 + 7, |i| (i as f32 * 0.07).cos() * 3.0);
    }

    #[test]
    fn matches_reference_4096() {
        check(4096, |i| ((i as f32 * 0.001).sin() * 4.0) + ((i as f32 * 0.013).cos() * 0.5));
    }

    #[test]
    fn all_zero() {
        // inv_std = 1/sqrt(eps), output = 0. Goes through check() for the
        // has_sve2() gate: calling the kernel directly SIGILLs on non-SVE2
        // hardware (e.g. the cortex-a53 qemu CI runner).
        check(256, |_| 0.0);
    }

    #[test]
    fn matches_neon_bit_close() {
        // Cross-check: SVE kernel ≈ NEON kernel (both should match scalar).
        // Reductions in different SIMD widths reorder differently, so not
        // bit-exact; close-enough tolerance.
        if !has_sve2() {
            eprintln!("SVE2 not present, skipping");
            return;
        }
        for n in [16usize, 64, 1024, 1024 + 7, 4096, 8192] {
            let x: Vec<f32> = (0..n).map(|i| (i as f32 * 0.11).sin() * 2.5).collect();
            let mut sve_out = x.clone();
            let mut neon_out = x.clone();
            sve_rms_norm_f32(&mut sve_out, 1e-5);
            crate::arm64::arm64simd_rms_norm_f32(&mut neon_out, 1e-5);
            close_enough(&sve_out, &neon_out, n);
        }
    }
}
