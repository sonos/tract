// AVX-512 (zmm, 16-wide) element-wise activation kernels with no FMA
// predecessor on x86: hardswish and leaky_relu. They mirror the aarch64 NEON
// kernels (arm64simd_hardswish_f32_8n / arm64simd_leaky_relu_f32_8n) but use
// 512-bit zmm registers, processing 64 f32 lanes per iteration. Validated
// against the generic scalar reference via the *_frame_tests! macros.

// hardswish(x) = x * relu6(x + 3) / 6
//              = x * max(0, min(6, x + 3)) * (1/6)
ew_impl_wrap!(
    f32,
    x86_64_avx512_hardswish_f32_64n,
    64,
    16,
    (),
    #[inline(never)]
    fn run(buf: &mut [f32], _: ()) {
        debug_assert!(buf.len() % Self::nr() == 0);
        debug_assert!(buf.as_ptr() as usize % Self::alignment_bytes() == 0);
        if buf.is_empty() {
            return;
        }
        unsafe { x86_64_avx512_hardswish_f32_64n_run(buf) }
    }
);

#[target_feature(enable = "avx512f")]
unsafe fn x86_64_avx512_hardswish_f32_64n_run(buf: &mut [f32]) {
    unsafe {
        let len = buf.len();
        let ptr = buf.as_ptr();
        std::arch::asm!("
            vbroadcastss zmm0, xmm0          // 3.0
            vbroadcastss zmm1, xmm1          // 6.0
            vbroadcastss zmm2, xmm2          // 1/6
            vpxord       zmm3, zmm3, zmm3    // 0.0
            2:
                vmovaps zmm4, [{ptr}]
                vmovaps zmm5, [{ptr} + 64]
                vmovaps zmm6, [{ptr} + 128]
                vmovaps zmm7, [{ptr} + 192]

                vaddps  zmm8,  zmm4, zmm0
                vaddps  zmm9,  zmm5, zmm0
                vaddps  zmm10, zmm6, zmm0
                vaddps  zmm11, zmm7, zmm0

                vminps  zmm8,  zmm8,  zmm1
                vminps  zmm9,  zmm9,  zmm1
                vminps  zmm10, zmm10, zmm1
                vminps  zmm11, zmm11, zmm1

                vmaxps  zmm8,  zmm8,  zmm3
                vmaxps  zmm9,  zmm9,  zmm3
                vmaxps  zmm10, zmm10, zmm3
                vmaxps  zmm11, zmm11, zmm3

                vmulps  zmm8,  zmm8,  zmm4
                vmulps  zmm9,  zmm9,  zmm5
                vmulps  zmm10, zmm10, zmm6
                vmulps  zmm11, zmm11, zmm7

                vmulps  zmm8,  zmm8,  zmm2
                vmulps  zmm9,  zmm9,  zmm2
                vmulps  zmm10, zmm10, zmm2
                vmulps  zmm11, zmm11, zmm2

                vmovaps [{ptr}],       zmm8
                vmovaps [{ptr} + 64],  zmm9
                vmovaps [{ptr} + 128], zmm10
                vmovaps [{ptr} + 192], zmm11

                add {ptr}, 256
                sub {len}, 64
                jnz 2b
            ",
        len = inout(reg) len => _,
        ptr = inout(reg) ptr => _,
        inout("xmm0") 3.0f32 => _,
        inout("xmm1") 6.0f32 => _,
        inout("xmm2") 1.0f32 / 6.0f32 => _,
        out("zmm3") _,
        out("zmm4") _, out("zmm5") _, out("zmm6") _, out("zmm7") _,
        out("zmm8") _, out("zmm9") _, out("zmm10") _, out("zmm11") _,
        );
    }
}

#[cfg(test)]
pub mod test_x86_64_avx512_hardswish_f32_64n {
    use super::*;
    hardswish_frame_tests!(
        is_x86_feature_detected!("avx512f"),
        f32,
        x86_64_avx512_hardswish_f32_64n
    );
}

// leaky_relu(x) = x > 0 ? x : alpha * x
ew_impl_wrap!(
    f32,
    x86_64_avx512_leaky_relu_f32_64n,
    64,
    16,
    f32,
    #[inline(never)]
    fn run(buf: &mut [f32], alpha: f32) {
        debug_assert!(buf.len() % Self::nr() == 0);
        debug_assert!(buf.as_ptr() as usize % Self::alignment_bytes() == 0);
        if buf.is_empty() {
            return;
        }
        unsafe { x86_64_avx512_leaky_relu_f32_64n_run(buf, alpha) }
    }
);

#[target_feature(enable = "avx512f")]
unsafe fn x86_64_avx512_leaky_relu_f32_64n_run(buf: &mut [f32], alpha: f32) {
    unsafe {
        let len = buf.len();
        let ptr = buf.as_ptr();
        std::arch::asm!("
            vbroadcastss zmm0, xmm0          // alpha
            vpxord       zmm1, zmm1, zmm1    // 0.0
            2:
                vmovaps zmm4, [{ptr}]
                vmovaps zmm5, [{ptr} + 64]
                vmovaps zmm6, [{ptr} + 128]
                vmovaps zmm7, [{ptr} + 192]

                // alpha * x in zmm8..11
                vmulps  zmm8,  zmm4, zmm0
                vmulps  zmm9,  zmm5, zmm0
                vmulps  zmm10, zmm6, zmm0
                vmulps  zmm11, zmm7, zmm0

                // mask = x > 0
                vcmpps  k1, zmm4, zmm1, 14
                vcmpps  k2, zmm5, zmm1, 14
                vcmpps  k3, zmm6, zmm1, 14
                vcmpps  k4, zmm7, zmm1, 14

                // where x > 0, overwrite alpha*x with x
                vmovaps zmm8{{k1}},  zmm4
                vmovaps zmm9{{k2}},  zmm5
                vmovaps zmm10{{k3}}, zmm6
                vmovaps zmm11{{k4}}, zmm7

                vmovaps [{ptr}],       zmm8
                vmovaps [{ptr} + 64],  zmm9
                vmovaps [{ptr} + 128], zmm10
                vmovaps [{ptr} + 192], zmm11

                add {ptr}, 256
                sub {len}, 64
                jnz 2b
            ",
        len = inout(reg) len => _,
        ptr = inout(reg) ptr => _,
        inout("xmm0") alpha => _,
        out("zmm1") _,
        out("zmm4") _, out("zmm5") _, out("zmm6") _, out("zmm7") _,
        out("zmm8") _, out("zmm9") _, out("zmm10") _, out("zmm11") _,
        out("k1") _, out("k2") _, out("k3") _, out("k4") _,
        );
    }
}

#[cfg(test)]
pub mod test_x86_64_avx512_leaky_relu_f32_64n {
    use super::*;
    leaky_relu_frame_tests!(
        is_x86_feature_detected!("avx512f"),
        f32,
        x86_64_avx512_leaky_relu_f32_64n
    );
}

// SiLU(x) = x * sigmoid(x). Composed at the kernel level (mirrors arm64): save
// the input chunk, run the AVX-512 sigmoid kernel in place, then multiply back
// by the saved original. nr() and CHUNK (256) are multiples of 16 so the
// sigmoid kernel always receives a 64-byte-aligned slice whose length is a
// multiple of 16.
ew_impl_wrap!(
    f32,
    x86_64_avx512_silu_f32_16n,
    16,
    16,
    (),
    #[inline(never)]
    fn run(buf: &mut [f32], _: ()) {
        debug_assert!(buf.len() % Self::nr() == 0);
        debug_assert!(buf.as_ptr() as usize % Self::alignment_bytes() == 0);
        const CHUNK: usize = 256;
        let mut scratch = [0f32; CHUNK];
        let mut start = 0;
        while start < buf.len() {
            let end = (start + CHUNK).min(buf.len());
            let chunk = &mut buf[start..end];
            let n = chunk.len();
            scratch[..n].copy_from_slice(chunk);
            super::avx512_sigmoid_f32::run(chunk, ());
            for i in 0..n {
                chunk[i] *= scratch[i];
            }
            start = end;
        }
    }
);

#[cfg(test)]
pub mod test_x86_64_avx512_silu_f32_16n {
    use super::*;
    silu_frame_tests!(is_x86_feature_detected!("avx512f"), f32, x86_64_avx512_silu_f32_16n);
}

// Tanh-form GELU (pow=3) matching tract's GeluApproximate:
//   gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// Composed at the kernel level (mirrors arm64): save the original x, compute
// the tanh argument in place, run the AVX-512 tanh kernel, then finish with the
// 0.5 * x * (1 + tanh) combine.
ew_impl_wrap!(
    f32,
    x86_64_avx512_gelu_f32_16n,
    16,
    16,
    (),
    #[inline(never)]
    fn run(buf: &mut [f32], _: ()) {
        debug_assert!(buf.len() % Self::nr() == 0);
        debug_assert!(buf.as_ptr() as usize % Self::alignment_bytes() == 0);
        const SQRT_2_OVER_PI: f32 = 0.7978845608028654;
        const COEF: f32 = 0.044715;
        const CHUNK: usize = 256;
        let mut scratch = [0f32; CHUNK];
        let mut start = 0;
        while start < buf.len() {
            let end = (start + CHUNK).min(buf.len());
            let chunk = &mut buf[start..end];
            let n = chunk.len();
            for i in 0..n {
                let x = chunk[i];
                scratch[i] = x;
                chunk[i] = SQRT_2_OVER_PI * (x + COEF * x * x * x);
            }
            super::avx512_tanh_f32::run(chunk, ());
            for i in 0..n {
                chunk[i] = 0.5 * scratch[i] * (1.0 + chunk[i]);
            }
            start = end;
        }
    }
);

#[cfg(test)]
pub mod test_x86_64_avx512_gelu_f32_16n {
    use super::*;
    gelu_frame_tests!(is_x86_feature_detected!("avx512f"), f32, x86_64_avx512_gelu_f32_16n);
}
