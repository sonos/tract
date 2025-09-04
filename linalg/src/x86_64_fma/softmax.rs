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

#[target_feature(enable = "avx,fma")]
unsafe fn x86_64_fma_softmax2_fastcompact_f32_32n_run(buf: &mut [f32], max: f32) -> f32 {
    unsafe {
        let mut len = buf.len();
        let mut ptr = buf.as_ptr();
        let mut acc = 0f32;
        const MLN2: f32 = 0.6931471805f32;
        const A: f32 = 8388608.0f32;
        const B: f32 = 1065353216.0f32;
        const C: f32 = 60801.0f32;
        const SLOPE: f32 = A / MLN2;
        const OFFSET: f32 = B - C;
        std::arch::asm!(
            r#"
            vbroadcastss ymm0, xmm0
            vmovaps ymm1, ymm0
            vmovaps ymm2, ymm0
            vmovaps ymm3, ymm0

            vpxor ymm12, ymm12, ymm12
            vbroadcastss ymm13, xmm13
            vbroadcastss ymm14, xmm14
            vbroadcastss ymm15, xmm15
            2:
                vmovaps ymm4, [rdi]
                vmovaps ymm5, [rdi + 32]
                vmovaps ymm6, [rdi + 64]
                vmovaps ymm7, [rdi + 96]

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

                vmovaps [rdi]     , ymm8
                vmovaps [rdi + 32], ymm9
                vmovaps [rdi + 64], ymm10
                vmovaps [rdi + 96], ymm11

                vaddps ymm0, ymm0, ymm8
                vaddps ymm1, ymm1, ymm9
                vaddps ymm2, ymm2, ymm10
                vaddps ymm3, ymm3, ymm11

                add rdi, 128
                sub rsi, 32
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
            "#,
        inout("rsi") len => _,
        inout("rdi") ptr => _,
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

#[target_feature(enable = "avx512f")]
unsafe fn x86_64_avx512_softmax2_fastcompact_f32_64n_run(buf: &mut [f32], max: f32) -> f32 {
    unsafe {
        let mut len = buf.len();
        let mut ptr = buf.as_ptr();
        let mut acc = 0f32;
        const MLN2: f32 = 0.6931471805f32;
        const A: f32 = 8388608.0f32;
        const B: f32 = 1065353216.0f32;
        const C: f32 = 60801.0f32;
        const SLOPE: f32 = A / MLN2;
        const OFFSET: f32 = B - C;
        std::arch::asm!(
            r#"
            vbroadcastss zmm0, xmm0
            vmovaps zmm1, zmm0
            vmovaps zmm2, zmm0
            vmovaps zmm3, zmm0

            vpxorq   zmm12, zmm12, zmm12
            vbroadcastss zmm13, xmm13
            vbroadcastss zmm14, xmm14
            vbroadcastss zmm15, xmm15
            2:
                vmovaps zmm4, [rdi]
                vmovaps zmm5, [rdi + 64]
                vmovaps zmm6, [rdi + 128]
                vmovaps zmm7, [rdi + 192]

                vsubps zmm4, zmm4, zmm13
                vsubps zmm5, zmm5, zmm13
                vsubps zmm6, zmm6, zmm13
                vsubps zmm7, zmm7, zmm13

                vmovaps zmm8, zmm15
                vmovaps zmm9, zmm15
                vmovaps zmm10, zmm15
                vmovaps zmm11, zmm15

                vfmadd231ps zmm8, zmm4, zmm14
                vfmadd231ps zmm9, zmm5, zmm14
                vfmadd231ps zmm10, zmm6, zmm14
                vfmadd231ps zmm11, zmm7, zmm14

                vmaxps zmm8, zmm8, zmm12
                vmaxps zmm9, zmm9, zmm12
                vmaxps zmm10, zmm10, zmm12
                vmaxps zmm11, zmm11, zmm12

                vcvttps2dq zmm8, zmm8
                vcvttps2dq zmm9, zmm9
                vcvttps2dq zmm10, zmm10
                vcvttps2dq zmm11, zmm11

                vmovaps [rdi]      , zmm8
                vmovaps [rdi + 64] , zmm9
                vmovaps [rdi + 128], zmm10
                vmovaps [rdi + 192], zmm11

                vaddps zmm0, zmm0, zmm8
                vaddps zmm1, zmm1, zmm9
                vaddps zmm2, zmm2, zmm10
                vaddps zmm3, zmm3, zmm11

                add rdi, 256
                sub rsi, 64
                jnz 2b

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
            "#,
        inout("rsi") len => _,
        inout("rdi") ptr => _,
        inout("zmm0") acc,
        out("zmm1") _, out("zmm2") _, out("zmm3") _,
        out("zmm4") _, out("zmm5") _, out("zmm6") _, out("zmm7") _,
        out("zmm8") _, out("zmm9") _, out("zmm10") _, out("zmm11") _,
        out("zmm12") _,
        inout("zmm13") max => _,
        inout("zmm14") SLOPE => _,
        inout("zmm15") OFFSET => _,
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

#[cfg(test)]
mod test_x86_64_avx512_softmax2_fastcompact_f32_64n {
    use super::*;
    crate::softmax_l2_frame_tests!(
        is_x86_feature_detected!("avx512f"),
        f32,
        x86_64_avx512_softmax2_fastcompact_f32_64n
    );
}
