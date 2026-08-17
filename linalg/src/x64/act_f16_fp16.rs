// AVX-512_FP16 native f16 element-wise activations.
//
// Sapphire Rapids (and later Intel) added the AVX-512 FP16 ISA: zmm-wide
// arithmetic on f16 directly (`vmulph`, `vfmadd*ph`, `vmaxph`, `vminph`,
// `vaddph`, `vsubph`, etc.). 32 f16 lanes per zmm — double the parallelism of
// the f32-roundtrip kernels in `act_f16.rs`, and zero conversion at the IO
// boundary.
//
// The kernels here mirror the algorithm of the f32 versions in `act.rs` and
// the f32-roundtrip f16 versions in `act_f16.rs`. Polynomials are evaluated
// directly in f16, accepting the lower mantissa precision (11 bits vs f32's
// 24) — the resulting tolerance fits inside the f16 activation tests'
// SuperApproximate band.
//
// Gated on `is_x86_feature_detected!("avx512fp16")` (the actual gating happens
// in `plug_avx512fp16` over in `x86_64_fma.rs`). Pre-FP16 AVX-512 hosts
// (Skylake-X, Cascade Lake, Ice Lake server prior to fp16 extension) keep
// using `act_f16.rs`'s f32-roundtrip versions.

use tract_data::internal::f16;

const FP16_TARGETS: &str = "avx512f,avx512fp16,avx512bw";

// hardswish(x) = x * clamp(x + 3, 0, 6) * (1/6).
// 128 f16 per iter (4 zmm × 32 lanes), 256 bytes / iter — same memory throughput
// as the f32 kernel's 64 f32 / iter.
ew_impl_wrap!(
    f16,
    x86_64_avx512fp16_hardswish_f16_128n,
    128,
    32,
    (),
    #[inline(never)]
    fn run(buf: &mut [f16], _: ()) {
        debug_assert!(buf.len() % Self::nr() == 0);
        debug_assert!(buf.as_ptr() as usize % Self::alignment_bytes() == 0);
        if buf.is_empty() {
            return;
        }
        unsafe { hardswish_f16_run(buf) }
    }
);

#[target_feature(enable = "avx512f,avx512fp16,avx512bw")]
unsafe fn hardswish_f16_run(buf: &mut [f16]) {
    let len = buf.len();
    let ptr = buf.as_ptr() as *mut u8;
    let three = f16::from_f32(3.0).to_bits();
    let six = f16::from_f32(6.0).to_bits();
    let recip6 = f16::from_f32(1.0 / 6.0).to_bits();
    unsafe {
        std::arch::asm!("
            vpbroadcastw zmm0, eax           // 3.0
            vpbroadcastw zmm1, ecx           // 6.0
            vpbroadcastw zmm2, edx           // 1/6
            vpxord       zmm3, zmm3, zmm3    // 0.0
            2:
                vmovdqa64 zmm4, [{ptr}]
                vmovdqa64 zmm5, [{ptr} + 64]
                vmovdqa64 zmm6, [{ptr} + 128]
                vmovdqa64 zmm7, [{ptr} + 192]

                vaddph  zmm8,  zmm4, zmm0
                vaddph  zmm9,  zmm5, zmm0
                vaddph  zmm10, zmm6, zmm0
                vaddph  zmm11, zmm7, zmm0

                vminph  zmm8,  zmm8,  zmm1
                vminph  zmm9,  zmm9,  zmm1
                vminph  zmm10, zmm10, zmm1
                vminph  zmm11, zmm11, zmm1

                vmaxph  zmm8,  zmm8,  zmm3
                vmaxph  zmm9,  zmm9,  zmm3
                vmaxph  zmm10, zmm10, zmm3
                vmaxph  zmm11, zmm11, zmm3

                vmulph  zmm8,  zmm8,  zmm4
                vmulph  zmm9,  zmm9,  zmm5
                vmulph  zmm10, zmm10, zmm6
                vmulph  zmm11, zmm11, zmm7

                vmulph  zmm8,  zmm8,  zmm2
                vmulph  zmm9,  zmm9,  zmm2
                vmulph  zmm10, zmm10, zmm2
                vmulph  zmm11, zmm11, zmm2

                vmovdqa64 [{ptr}],       zmm8
                vmovdqa64 [{ptr} + 64],  zmm9
                vmovdqa64 [{ptr} + 128], zmm10
                vmovdqa64 [{ptr} + 192], zmm11

                add {ptr}, 256
                sub {len}, 128
                jnz 2b
            ",
            len = inout(reg) len => _,
            ptr = inout(reg) ptr => _,
            in("eax") three as u32,
            in("ecx") six as u32,
            in("edx") recip6 as u32,
            out("zmm0") _, out("zmm1") _, out("zmm2") _, out("zmm3") _,
            out("zmm4") _, out("zmm5") _, out("zmm6") _, out("zmm7") _,
            out("zmm8") _, out("zmm9") _, out("zmm10") _, out("zmm11") _,
        );
    }
}

// leaky_relu(x, alpha) = x if x >= 0 else alpha*x
// For 0 <= alpha <= 1: leaky_relu(x, alpha) = max(x, alpha*x). For the typical
// alpha values used (0.01, 0.1, 0.2) this is exact.
//
// NOTE: This native fp16 version benched ~38% SLOWER than the f32-roundtrip
// version on Sapphire Rapids (9.44 Gelem/s f32-roundtrip vs 5.85 Gelem/s
// native, n=1024, single-thread). The two compute ops per element (vmulph +
// vmaxph) appear not to saturate Sapphire Rapids' FP16 execution port the
// same way f32 mul/max saturate the FP32 ports. The kernel is correct (passes
// proptest against the f16 reference) but is NOT plugged in — see the
// `plug_avx512fp16` comment in `x86_64_fma.rs`. Kept here in case a different
// AVX-512_FP16 uarch (Granite Rapids etc.) flips the comparison.
ew_impl_wrap!(
    f16,
    x86_64_avx512fp16_leaky_relu_f16_128n,
    128,
    32,
    f16,
    #[inline(never)]
    fn run(buf: &mut [f16], alpha: f16) {
        debug_assert!(buf.len() % Self::nr() == 0);
        debug_assert!(buf.as_ptr() as usize % Self::alignment_bytes() == 0);
        if buf.is_empty() {
            return;
        }
        unsafe { leaky_relu_f16_run(buf, alpha) }
    }
);

#[target_feature(enable = "avx512f,avx512fp16,avx512bw")]
unsafe fn leaky_relu_f16_run(buf: &mut [f16], alpha: f16) {
    let len = buf.len();
    let ptr = buf.as_ptr() as *mut u8;
    let alpha_bits = alpha.to_bits();
    unsafe {
        std::arch::asm!("
            vpbroadcastw zmm0, eax           // alpha
            2:
                vmovdqa64 zmm4, [{ptr}]
                vmovdqa64 zmm5, [{ptr} + 64]
                vmovdqa64 zmm6, [{ptr} + 128]
                vmovdqa64 zmm7, [{ptr} + 192]

                vmulph  zmm8,  zmm4, zmm0
                vmulph  zmm9,  zmm5, zmm0
                vmulph  zmm10, zmm6, zmm0
                vmulph  zmm11, zmm7, zmm0

                vmaxph  zmm8,  zmm8,  zmm4
                vmaxph  zmm9,  zmm9,  zmm5
                vmaxph  zmm10, zmm10, zmm6
                vmaxph  zmm11, zmm11, zmm7

                vmovdqa64 [{ptr}],       zmm8
                vmovdqa64 [{ptr} + 64],  zmm9
                vmovdqa64 [{ptr} + 128], zmm10
                vmovdqa64 [{ptr} + 192], zmm11

                add {ptr}, 256
                sub {len}, 128
                jnz 2b
            ",
            len = inout(reg) len => _,
            ptr = inout(reg) ptr => _,
            in("eax") alpha_bits as u32,
            out("zmm0") _,
            out("zmm4") _, out("zmm5") _, out("zmm6") _, out("zmm7") _,
            out("zmm8") _, out("zmm9") _, out("zmm10") _, out("zmm11") _,
        );
    }
}

#[cfg(test)]
pub mod test_x86_64_avx512fp16_hardswish {
    use super::*;
    crate::hardswish_frame_tests!(
        is_x86_feature_detected!("avx512fp16"),
        f16,
        x86_64_avx512fp16_hardswish_f16_128n
    );
}

#[cfg(test)]
pub mod test_x86_64_avx512fp16_leaky_relu {
    use super::*;
    crate::leaky_relu_frame_tests!(
        is_x86_feature_detected!("avx512fp16"),
        f16,
        x86_64_avx512fp16_leaky_relu_f16_128n
    );
}

// Suppress unused-const lint until we expand to more kernels.
#[allow(dead_code)]
const _UNUSED: &str = FP16_TARGETS;
