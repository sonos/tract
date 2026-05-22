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
    }
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
