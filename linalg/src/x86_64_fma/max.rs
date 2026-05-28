reduce_impl_wrap!(
    f32,
    x86_64_fma_max_f32_32n,
    32,
    8,
    (),
    f32::MIN,
    #[inline(never)]
    fn run(buf: &[f32], _: ()) -> f32 {
        assert!(buf.len() % 32 == 0);
        assert!(buf.len() > 0);
        unsafe { x86_64_fma_max_f32_32n_run(buf) }
    },
    #[inline(never)]
    fn reduce_two(a: f32, b: f32) -> f32 {
        a.max(b)
    }
);

#[target_feature(enable = "avx")]
unsafe fn x86_64_fma_max_f32_32n_run(buf: &[f32]) -> f32 {
    unsafe {
        let len = buf.len();
        let ptr = buf.as_ptr();
        let mut acc = f32::MIN;
        std::arch::asm!("
            vbroadcastss ymm0, xmm0
            vmovaps ymm1, ymm0
            vmovaps ymm2, ymm0
            vmovaps ymm3, ymm0
            2:
                vmovaps ymm4, [{ptr}]
                vmovaps ymm5, [{ptr} + 32]
                vmovaps ymm6, [{ptr} + 64]
                vmovaps ymm7, [{ptr} + 96]
                vmaxps ymm0, ymm0, ymm4
                vmaxps ymm1, ymm1, ymm5
                vmaxps ymm2, ymm2, ymm6
                vmaxps ymm3, ymm3, ymm7
                add {ptr}, 128
                sub {len}, 32
                jnz 2b
            vmaxps ymm0, ymm0, ymm1
            vmaxps ymm2, ymm2, ymm3
            vmaxps ymm0, ymm0, ymm2
            vperm2f128 ymm1, ymm0, ymm0, 1      // copy second half (4xf32) of ymm0 to ymm1
            vmaxps xmm0, xmm0, xmm1             // xmm0 contains 4 values to max
            vpermilps xmm1, xmm0, 2 + (3 << 2)  // second 2x32 bit half moved to top
            vmaxps xmm0, xmm0, xmm1             // xmm0 containes 2 values
            vpermilps xmm1, xmm0, 1             // second f32 to top
            vmaxps xmm0, xmm0, xmm1
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

#[cfg(test)]
mod test_x86_64_fma_max_f32_32n {
    use super::*;
    crate::max_frame_tests!(is_x86_feature_detected!("avx2"), f32, x86_64_fma_max_f32_32n);
}

// AVX-512 version: processes 64 f32 per loop iteration (4 zmm registers of 16
// lanes each). Runtime-gated on avx512f (see x86_64_fma.rs::plug_avx512f); on
// non-AVX512 CPUs this kernel is never registered and the FMA path above stays
// in use. nr=64, 64-byte (16xf32) alignment.
reduce_impl_wrap!(
    f32,
    x86_64_avx512_max_f32_64n,
    64,
    16,
    (),
    f32::MIN,
    #[inline(never)]
    fn run(buf: &[f32], _: ()) -> f32 {
        assert!(buf.len() % 64 == 0);
        assert!(buf.len() > 0);
        unsafe { x86_64_avx512_max_f32_64n_run(buf) }
    },
    #[inline(never)]
    fn reduce_two(a: f32, b: f32) -> f32 {
        a.max(b)
    }
);

#[target_feature(enable = "avx512f")]
unsafe fn x86_64_avx512_max_f32_64n_run(buf: &[f32]) -> f32 {
    unsafe {
        let len = buf.len();
        let ptr = buf.as_ptr();
        let mut acc = f32::MIN;
        std::arch::asm!("
            vbroadcastss zmm0, xmm0
            vmovaps zmm1, zmm0
            vmovaps zmm2, zmm0
            vmovaps zmm3, zmm0
            2:
                vmaxps zmm0, zmm0, [{ptr}]
                vmaxps zmm1, zmm1, [{ptr} + 64]
                vmaxps zmm2, zmm2, [{ptr} + 128]
                vmaxps zmm3, zmm3, [{ptr} + 192]
                add {ptr}, 256
                sub {len}, 64
                jnz 2b
            vmaxps zmm0, zmm0, zmm1
            vmaxps zmm2, zmm2, zmm3
            vmaxps zmm0, zmm0, zmm2             // zmm0 holds 16 partial maxima
            vextractf64x4 ymm1, zmm0, 1         // upper 256 bits (8xf32) of zmm0 -> ymm1 (avx512f)
            vmaxps ymm0, ymm0, ymm1            // ymm0 holds 8 values
            vextractf128 xmm1, ymm0, 1          // upper 4xf32 -> xmm1
            vmaxps xmm0, xmm0, xmm1            // xmm0 holds 4 values
            vpermilps xmm1, xmm0, 2 + (3 << 2)  // second 2x32 bit half moved to top
            vmaxps xmm0, xmm0, xmm1            // xmm0 holds 2 values
            vpermilps xmm1, xmm0, 1             // second f32 to top
            vmaxps xmm0, xmm0, xmm1
            ",
        len = inout(reg) len => _,
        ptr = inout(reg) ptr => _,
        inout("zmm0") acc,
        out("zmm1") _, out("zmm2") _, out("zmm3") _,
        );
        acc
    }
}

#[cfg(test)]
mod test_x86_64_avx512_max_f32_64n {
    use super::*;
    crate::max_frame_tests!(is_x86_feature_detected!("avx512f"), f32, x86_64_avx512_max_f32_64n);
}
