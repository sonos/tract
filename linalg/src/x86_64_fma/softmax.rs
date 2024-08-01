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

#[cfg(test)]
mod test_x86_64_fma_softmax2_fastcompact_f32_32n {
    use super::*;
    crate::softmax_l2_frame_tests!(is_x86_feature_detected!("fma"), f32, x86_64_fma_softmax2_fastcompact_f32_32n);
}
