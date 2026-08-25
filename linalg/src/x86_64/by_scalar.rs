routine_by_scalar_rust!(x86_64;
    f32,
    x86_64_avx_f32_mul_by_scalar_32n,
    32,
    8,
    fn run(x: &mut [f32], s: f32) {
        debug_assert!(x.len() % Self::nr() == 0);
        debug_assert!(x.as_ptr() as usize % Self::alignment_bytes() == 0);
        unsafe { x86_64_avx_f32_mul_by_scalar_32n_run(x, s) }
    },
    op(Mul),
    param(MulByScalar),
    isa(X86_64Avx)
);

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn x86_64_avx_f32_mul_by_scalar_32n_run(buf: &mut [f32], scalar: f32) {
    unsafe {
        let len = buf.len();
        let ptr = buf.as_ptr();
        std::arch::asm!("
            // reg-source vbroadcastss needs avx2; this kernel must stay avx-safe
            vpermilps xmm0, xmm0, 0
            vinsertf128 ymm0, ymm0, xmm0, 1
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
