ew_impl_wrap!(
    f32,
    x86_64_avx_f32_mul_by_scalar_32n,
    32,
    8,
    f32,
    fn run(x: &mut [f32], s: f32) {
        debug_assert!(x.len() % Self::nr() == 0);
        debug_assert!(x.as_ptr() as usize % Self::alignment_bytes() == 0);
        unsafe { x86_64_avx_f32_mul_by_scalar_32n_run(x, s) }
    }
);

ew_impl_wrap!(
    f32,
    x86_64_avx512_f32_mul_by_scalar_64n,
    64,
    16,
    f32,
    fn run(x: &mut [f32], s: f32) {
        debug_assert!(x.len() % Self::nr() == 0);
        debug_assert!(x.as_ptr() as usize % Self::alignment_bytes() == 0);
        unsafe { x86_64_avx512_f32_mul_by_scalar_64n_run(x, s) }
    }
);

#[target_feature(enable = "avx")]
unsafe fn x86_64_avx_f32_mul_by_scalar_32n_run(buf: &mut [f32], scalar: f32) {
    unsafe {
        let len = buf.len();
        let ptr = buf.as_ptr();
        std::arch::asm!("
            vbroadcastss ymm0, xmm0
            2:
                vmovaps ymm4, [{ptr}]
                vmovaps ymm5, [{ptr} + 32]
                vmovaps ymm6, [{ptr} + 64]
                vmovaps ymm7, [{ptr} + 96]
                vmulps ymm4, ymm4, ymm0
                vmulps ymm5, ymm5, ymm0
                vmulps ymm6, ymm6, ymm0
                vmulps ymm7, ymm7, ymm0
                vmovaps [{ptr}], ymm4
                vmovaps [{ptr} + 32], ymm5
                vmovaps [{ptr} + 64], ymm6
                vmovaps [{ptr} + 96], ymm7
                add {ptr}, 128
                sub {len}, 32
                jnz 2b
            ",
        len = inout(reg) len => _,
        ptr = inout(reg) ptr => _,
        in("xmm0") scalar,
        out("ymm4") _, out("ymm5") _, out("ymm6") _, out("ymm7") _
        );
    }
}

#[target_feature(enable = "avx512f")]
unsafe fn x86_64_avx512_f32_mul_by_scalar_64n_run(buf: &mut [f32], scalar: f32) {
    unsafe {
        let len = buf.len();
        let ptr = buf.as_ptr();
        std::arch::asm!("
           vbroadcastss zmm0, xmm0
           2:
               vmovaps zmm4, [{ptr}]
               vmovaps zmm5, [{ptr} + 64]
               vmovaps zmm6, [{ptr} + 128]
               vmovaps zmm7, [{ptr} + 192]
               vmulps zmm4, zmm4, zmm0
               vmulps zmm5, zmm5, zmm0
               vmulps zmm6, zmm6, zmm0
               vmulps zmm7, zmm7, zmm0
               vmovaps [{ptr}], zmm4
               vmovaps [{ptr} + 64], zmm5
               vmovaps [{ptr} + 128], zmm6
               vmovaps [{ptr} + 192], zmm7
               add {ptr}, 256
               sub {len}, 64
               jnz 2b
           ",
        len = inout(reg) len => _,
        ptr = inout(reg) ptr => _,
        in("xmm0") scalar,
        out("zmm4") _, out("zmm5") _, out("zmm6") _, out("zmm7") _
        );
    }
}

#[cfg(test)]
#[macro_use]
pub mod test_x86_64_avx_f32_mul_by_scalar_32n {
    use super::*;
    by_scalar_frame_tests!(
        is_x86_feature_detected!("avx2"),
        f32,
        x86_64_avx_f32_mul_by_scalar_32n,
        |a, b| a * b
    );
}
