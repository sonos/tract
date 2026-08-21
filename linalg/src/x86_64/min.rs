reduce_impl_wrap2!(x86_64;
    f32,
    x86_64_fma_min_f32_32n,
    32,
    8,
    (),
    f32::MAX,
    #[inline(never)]
    fn run(buf: &[f32], _: ()) -> f32 {
        assert!(buf.len() % 32 == 0);
        assert!(buf.len() > 0);
        unsafe { x86_64_fma_min_f32_32n_run(buf) }
    },
    #[inline(never)]
    fn reduce_two(a: f32, b: f32) -> f32 {
        a.min(b)
    }
);

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn x86_64_fma_min_f32_32n_run(buf: &[f32]) -> f32 {
    unsafe {
        let len = buf.len();
        let ptr = buf.as_ptr();
        let mut acc = f32::MAX;
        std::arch::asm!("
            // reg-source vbroadcastss needs avx2; this kernel must stay avx-safe
            vpermilps xmm0, xmm0, 0
            vinsertf128 ymm0, ymm0, xmm0, 1
            vmovaps ymm1, ymm0
            vmovaps ymm2, ymm0
            vmovaps ymm3, ymm0
            2:
                vmovaps ymm4, [{ptr}]
                vmovaps ymm5, [{ptr} + 32]
                vmovaps ymm6, [{ptr} + 64]
                vmovaps ymm7, [{ptr} + 96]
                vminps ymm0, ymm0, ymm4
                vminps ymm1, ymm1, ymm5
                vminps ymm2, ymm2, ymm6
                vminps ymm3, ymm3, ymm7
                add {ptr}, 128
                sub {len}, 32
                jnz 2b
            vminps ymm0, ymm0, ymm1
            vminps ymm2, ymm2, ymm3
            vminps ymm0, ymm0, ymm2
            vperm2f128 ymm1, ymm0, ymm0, 1      // copy second half (4xf32) of ymm0 to ymm1
            vminps xmm0, xmm0, xmm1             // xmm0 contains 4 values to min
            vpermilps xmm1, xmm0, 2 + (3 << 2)  // second 2x32 bit half moved to top
            vminps xmm0, xmm0, xmm1             // xmm0 containes 2 values
            vpermilps xmm1, xmm0, 1             // second f32 to top
            vminps xmm0, xmm0, xmm1
            ",
        len = inout(reg) len => _,
        ptr = inout(reg) ptr => _,
        inout("ymm0") acc,
        out("ymm1") _, out("ymm2") _, out("ymm3") _,
        out("ymm4") _, out("ymm5") _, out("ymm6") _, out("ymm7") _
        );
        acc
    }
}

#[cfg(all(test, target_arch = "x86_64"))]
mod test_x86_64_fma_min_f32_32n {
    use super::*;
    crate::min_frame_tests!(is_x86_feature_detected!("avx"), f32, x86_64_fma_min_f32_32n);
}
