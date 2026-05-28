use crate::num_traits::Zero;
use tract_data::internal::f16;

map_reduce_impl_wrap!(
    f32,
    x86_64_fma_softmax2_fastcompact_f32_32n,
    32,
    8,
    f32,
    f32::MIN,
    0f32,
    #[inline(never)]
    fn run(buf: &mut [f32], max: f32) -> f32 {
        assert!(buf.len() % 32 == 0);
        assert!(buf.len() > 0);
        unsafe { x86_64_fma_softmax2_fastcompact_f32_32n_run(buf, max) }
    },
    #[inline(never)]
    fn reduce_two(a: f32, b: f32) -> f32 {
        a + b
    }
);

#[target_feature(enable = "avx,fma")]
unsafe fn x86_64_fma_softmax2_fastcompact_f32_32n_run(buf: &mut [f32], max: f32) -> f32 {
    unsafe {
        let len = buf.len();
        let ptr = buf.as_ptr();
        let mut acc = 0f32;
        const MLN2: f32 = 0.6931471805f32;
        const A: f32 = 8388608.0f32;
        const B: f32 = 1065353216.0f32;
        const C: f32 = 60801.0f32;
        const SLOPE: f32 = A / MLN2;
        const OFFSET: f32 = B - C;
        std::arch::asm!("
            vbroadcastss ymm0, xmm0
            vmovaps ymm1, ymm0
            vmovaps ymm2, ymm0
            vmovaps ymm3, ymm0

            vpxor   ymm12, ymm12, ymm12
            vbroadcastss ymm13, xmm13
            vbroadcastss ymm14, xmm14
            vbroadcastss ymm15, xmm15
            2:
                vmovaps ymm4, [{ptr}]
                vmovaps ymm5, [{ptr} + 32]
                vmovaps ymm6, [{ptr} + 64]
                vmovaps ymm7, [{ptr} + 96]

                vsubps ymm4, ymm4, ymm13
                vsubps ymm5, ymm5, ymm13
                vsubps ymm6, ymm6, ymm13
                vsubps ymm7, ymm7, ymm13

                vmovaps ymm8, ymm15
                vmovaps ymm9, ymm15
                vmovaps ymm10, ymm15
                vmovaps ymm11, ymm15

                vfmadd231ps ymm8, ymm4, ymm14
                vfmadd231ps ymm9, ymm5, ymm14
                vfmadd231ps ymm10, ymm6, ymm14
                vfmadd231ps ymm11, ymm7, ymm14

                vmaxps ymm8, ymm8, ymm12
                vmaxps ymm9, ymm9, ymm12
                vmaxps ymm10, ymm10, ymm12
                vmaxps ymm11, ymm11, ymm12

                vcvttps2dq ymm8, ymm8
                vcvttps2dq ymm9, ymm9
                vcvttps2dq ymm10, ymm10
                vcvttps2dq ymm11, ymm11

                vmovaps [{ptr}]     , ymm8
                vmovaps [{ptr} + 32], ymm9
                vmovaps [{ptr} + 64], ymm10
                vmovaps [{ptr} + 96], ymm11

                vaddps ymm0, ymm0, ymm8
                vaddps ymm1, ymm1, ymm9
                vaddps ymm2, ymm2, ymm10
                vaddps ymm3, ymm3, ymm11

                add {ptr}, 128
                sub {len}, 32
                jnz 2b

            vaddps ymm0, ymm0, ymm1
            vaddps ymm2, ymm2, ymm3
            vaddps ymm0, ymm0, ymm2
            vperm2f128 ymm1, ymm0, ymm0, 1
            vaddps xmm0, xmm0, xmm1
            vpermilps xmm1, xmm0, 2 + (3 << 2)
            vaddps xmm0, xmm0, xmm1
            vpermilps xmm1, xmm0, 1
            vaddps xmm0, xmm0, xmm1
            ",
        len = inout(reg) len => _,
        ptr = inout(reg) ptr => _,
        inout("ymm0") acc,
        out("ymm1") _, out("ymm2") _, out("ymm3") _,
        out("ymm4") _, out("ymm5") _, out("ymm6") _, out("ymm7") _,
        out("ymm8") _, out("ymm9") _, out("ymm10") _, out("ymm11") _,
        out("ymm12") _,
        inout("ymm13") max => _,
        inout("ymm14") SLOPE => _,
        inout("ymm15") OFFSET => _,
        );
        acc
    }
}

#[cfg(test)]
mod test_x86_64_fma_softmax2_fastcompact_f32_32n {
    use super::*;
    crate::softmax_l2_frame_tests!(
        is_x86_feature_detected!("fma"),
        f32,
        x86_64_fma_softmax2_fastcompact_f32_32n
    );
}

// AVX-512 version: processes 64 f32 per loop iteration (4 zmm registers of 16
// lanes each). Same fast-compact-exp algorithm as the FMA kernel above:
//   y = bitcast_u32(max(0, SLOPE*(x-max) + OFFSET))   (via vcvttps2dq)
// then writes y back and accumulates sum(y). Runtime-gated on avx512f (see
// x86_64_fma.rs::plug_avx512f); non-AVX512 CPUs keep using the FMA kernel.
// nr=64, 64-byte (16xf32) alignment.
map_reduce_impl_wrap!(
    f32,
    x86_64_avx512_softmax2_fastcompact_f32_64n,
    64,
    16,
    f32,
    f32::MIN,
    0f32,
    #[inline(never)]
    fn run(buf: &mut [f32], max: f32) -> f32 {
        assert!(buf.len() % 64 == 0);
        assert!(buf.len() > 0);
        unsafe { x86_64_avx512_softmax2_fastcompact_f32_64n_run(buf, max) }
    },
    #[inline(never)]
    fn reduce_two(a: f32, b: f32) -> f32 {
        a + b
    }
);

#[target_feature(enable = "avx512f")]
unsafe fn x86_64_avx512_softmax2_fastcompact_f32_64n_run(buf: &mut [f32], max: f32) -> f32 {
    unsafe {
        let len = buf.len();
        let ptr = buf.as_ptr();
        let mut acc = 0f32;
        const MLN2: f32 = 0.6931471805f32;
        const A: f32 = 8388608.0f32;
        const B: f32 = 1065353216.0f32;
        const C: f32 = 60801.0f32;
        const SLOPE: f32 = A / MLN2;
        const OFFSET: f32 = B - C;
        std::arch::asm!("
            vbroadcastss zmm0, xmm0
            vmovaps zmm1, zmm0
            vmovaps zmm2, zmm0
            vmovaps zmm3, zmm0

            vpxord  zmm28, zmm28, zmm28          // zero (clamp floor)
            vbroadcastss zmm29, xmm29            // max
            vbroadcastss zmm30, xmm30            // slope
            vbroadcastss zmm31, xmm31            // offset
            2:
                vmovaps zmm4, [{ptr}]
                vmovaps zmm5, [{ptr} + 64]
                vmovaps zmm6, [{ptr} + 128]
                vmovaps zmm7, [{ptr} + 192]

                vsubps zmm4, zmm4, zmm29
                vsubps zmm5, zmm5, zmm29
                vsubps zmm6, zmm6, zmm29
                vsubps zmm7, zmm7, zmm29

                vmovaps zmm8, zmm31
                vmovaps zmm9, zmm31
                vmovaps zmm10, zmm31
                vmovaps zmm11, zmm31

                vfmadd231ps zmm8, zmm4, zmm30
                vfmadd231ps zmm9, zmm5, zmm30
                vfmadd231ps zmm10, zmm6, zmm30
                vfmadd231ps zmm11, zmm7, zmm30

                vmaxps zmm8, zmm8, zmm28
                vmaxps zmm9, zmm9, zmm28
                vmaxps zmm10, zmm10, zmm28
                vmaxps zmm11, zmm11, zmm28

                vcvttps2dq zmm8, zmm8
                vcvttps2dq zmm9, zmm9
                vcvttps2dq zmm10, zmm10
                vcvttps2dq zmm11, zmm11

                vmovaps [{ptr}]      , zmm8
                vmovaps [{ptr} + 64] , zmm9
                vmovaps [{ptr} + 128], zmm10
                vmovaps [{ptr} + 192], zmm11

                vaddps zmm0, zmm0, zmm8
                vaddps zmm1, zmm1, zmm9
                vaddps zmm2, zmm2, zmm10
                vaddps zmm3, zmm3, zmm11

                add {ptr}, 256
                sub {len}, 64
                jnz 2b

            vaddps zmm0, zmm0, zmm1
            vaddps zmm2, zmm2, zmm3
            vaddps zmm0, zmm0, zmm2             // zmm0 holds 16 partial sums
            vextractf64x4 ymm1, zmm0, 1         // upper 256 bits (8xf32) -> ymm1 (avx512f)
            vaddps ymm0, ymm0, ymm1            // ymm0 holds 8 values
            vextractf128 xmm1, ymm0, 1          // upper 4xf32 -> xmm1
            vaddps xmm0, xmm0, xmm1            // xmm0 holds 4 values
            vpermilps xmm1, xmm0, 2 + (3 << 2)
            vaddps xmm0, xmm0, xmm1            // xmm0 holds 2 values
            vpermilps xmm1, xmm0, 1
            vaddps xmm0, xmm0, xmm1
            ",
        len = inout(reg) len => _,
        ptr = inout(reg) ptr => _,
        inout("zmm0") acc,
        out("zmm1") _, out("zmm2") _, out("zmm3") _,
        out("zmm4") _, out("zmm5") _, out("zmm6") _, out("zmm7") _,
        out("zmm8") _, out("zmm9") _, out("zmm10") _, out("zmm11") _,
        out("zmm28") _,
        inout("zmm29") max => _,
        inout("zmm30") SLOPE => _,
        inout("zmm31") OFFSET => _,
        );
        acc
    }
}

#[cfg(test)]
mod test_x86_64_avx512_softmax2_fastcompact_f32_64n {
    use super::*;
    crate::softmax_l2_frame_tests!(
        is_x86_feature_detected!("avx512f"),
        f32,
        x86_64_avx512_softmax2_fastcompact_f32_64n
    );
}

// AVX-512 f16 softmax_l2: same fast-compact-exp algorithm as the FMA f32
// kernel, with f16 <-> f32 conversion at the IO boundary. Each loop iteration
// handles 64 f16 (128 bytes) through 4× (ymm f16 load -> vcvtph2ps -> zmm f32
// compute -> vcvttps2dq -> vcvtps2ph -> ymm f16 store). The sum is accumulated
// in f32 across the loop (higher precision than the generic HSoftMaxL2 which
// accumulates in f16) and cast to f16 at return; the SuperApproximate test
// tolerance covers the precision delta.
// nr=64 (multiple of 4 ymm f16 loads); alignment_items=32 (64-byte aligned).
map_reduce_impl_wrap!(
    f16,
    x86_64_avx512_softmax2_fastcompact_f16_64n,
    64,
    32,
    f16,
    f16::MIN,
    f16::zero(),
    #[inline(never)]
    fn run(buf: &mut [f16], max: f16) -> f16 {
        assert!(buf.len() % 64 == 0);
        assert!(buf.len() > 0);
        unsafe { x86_64_avx512_softmax2_fastcompact_f16_64n_run(buf, max) }
    },
    #[inline(never)]
    fn reduce_two(a: f16, b: f16) -> f16 {
        a + b
    }
);

#[target_feature(enable = "avx512f")]
unsafe fn x86_64_avx512_softmax2_fastcompact_f16_64n_run(
    buf: &mut [tract_data::internal::f16],
    max: tract_data::internal::f16,
) -> tract_data::internal::f16 {
    unsafe {
        let len = buf.len();
        let ptr = buf.as_ptr();
        let max_f32: f32 = max.to_f32();
        let mut acc = 0f32;
        const MLN2: f32 = 0.6931471805f32;
        const A: f32 = 8388608.0f32;
        const B: f32 = 1065353216.0f32;
        const C: f32 = 60801.0f32;
        const SLOPE: f32 = A / MLN2;
        const OFFSET: f32 = B - C;
        std::arch::asm!("
            vbroadcastss zmm0, xmm0
            vmovaps zmm1, zmm0
            vmovaps zmm2, zmm0
            vmovaps zmm3, zmm0

            vpxord       zmm28, zmm28, zmm28      // 0 (clamp floor)
            vbroadcastss zmm29, xmm29             // max (f32)
            vbroadcastss zmm30, xmm30             // slope
            vbroadcastss zmm31, xmm31             // offset
            2:
                // load 4 ymm of f16 (16 f16 per ymm = 32 bytes), convert to zmm f32
                vcvtph2ps zmm4, [{ptr}]
                vcvtph2ps zmm5, [{ptr} + 32]
                vcvtph2ps zmm6, [{ptr} + 64]
                vcvtph2ps zmm7, [{ptr} + 96]

                // subtract max
                vsubps zmm4, zmm4, zmm29
                vsubps zmm5, zmm5, zmm29
                vsubps zmm6, zmm6, zmm29
                vsubps zmm7, zmm7, zmm29

                // OFFSET + SLOPE * (x - max)
                vmovaps zmm8,  zmm31
                vmovaps zmm9,  zmm31
                vmovaps zmm10, zmm31
                vmovaps zmm11, zmm31
                vfmadd231ps zmm8,  zmm4, zmm30
                vfmadd231ps zmm9,  zmm5, zmm30
                vfmadd231ps zmm10, zmm6, zmm30
                vfmadd231ps zmm11, zmm7, zmm30

                // max(0, ...)
                vmaxps zmm8,  zmm8,  zmm28
                vmaxps zmm9,  zmm9,  zmm28
                vmaxps zmm10, zmm10, zmm28
                vmaxps zmm11, zmm11, zmm28

                // fast-compact-exp trick: the truncated i32 has the same bit
                // pattern as the f32 ~exp(x), so accumulate AS f32 + store as f16
                vcvttps2dq zmm8,  zmm8
                vcvttps2dq zmm9,  zmm9
                vcvttps2dq zmm10, zmm10
                vcvttps2dq zmm11, zmm11

                vaddps zmm0, zmm0, zmm8
                vaddps zmm1, zmm1, zmm9
                vaddps zmm2, zmm2, zmm10
                vaddps zmm3, zmm3, zmm11

                // convert back to f16 and store (4th operand 0 = round to nearest even)
                vcvtps2ph [{ptr}],      zmm8,  0
                vcvtps2ph [{ptr} + 32], zmm9,  0
                vcvtps2ph [{ptr} + 64], zmm10, 0
                vcvtps2ph [{ptr} + 96], zmm11, 0

                add {ptr}, 128
                sub {len}, 64
                jnz 2b

            // reduce zmm0..3 to a scalar f32 in xmm0
            vaddps zmm0, zmm0, zmm1
            vaddps zmm2, zmm2, zmm3
            vaddps zmm0, zmm0, zmm2
            vextractf64x4 ymm1, zmm0, 1
            vaddps ymm0, ymm0, ymm1
            vextractf128 xmm1, ymm0, 1
            vaddps xmm0, xmm0, xmm1
            vpermilps xmm1, xmm0, 2 + (3 << 2)
            vaddps xmm0, xmm0, xmm1
            vpermilps xmm1, xmm0, 1
            vaddps xmm0, xmm0, xmm1
            ",
        len = inout(reg) len => _,
        ptr = inout(reg) ptr => _,
        inout("zmm0") acc,
        out("zmm1") _, out("zmm2") _, out("zmm3") _,
        out("zmm4") _, out("zmm5") _, out("zmm6") _, out("zmm7") _,
        out("zmm8") _, out("zmm9") _, out("zmm10") _, out("zmm11") _,
        out("zmm28") _,
        inout("zmm29") max_f32 => _,
        inout("zmm30") SLOPE => _,
        inout("zmm31") OFFSET => _,
        );
        f16::from_f32(acc)
    }
}

#[cfg(test)]
mod test_x86_64_avx512_softmax2_fastcompact_f16_64n {
    use super::*;
    use tract_data::internal::f16;
    crate::softmax_l2_frame_tests!(
        is_x86_feature_detected!("avx512f"),
        f16,
        x86_64_avx512_softmax2_fastcompact_f16_64n
    );
}
