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

#[target_feature(enable = "avx")]
unsafe fn x86_64_avx_f32_mul_by_scalar_32n_run(buf: &mut [f32], scalar: f32) {
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
