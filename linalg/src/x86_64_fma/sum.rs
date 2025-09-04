use crate::num_traits::Zero;

reduce_impl_wrap!(
    f32,
    x86_64_fma_sum_f32_32n,
    32,
    8,
    (),
    f32::zero(),
    #[inline(never)]
    fn run(buf: &[f32], _: ()) -> f32 {
        assert!(buf.len() % 32 == 0);
        assert!(!buf.is_empty());
        unsafe { x86_64_fma_sum_f32_32n_run(buf) }
    },
    #[inline(never)]
    fn reduce_two(a: f32, b: f32) -> f32 {
        a + b
    }
);

#[target_feature(enable = "avx")]
unsafe fn x86_64_fma_sum_f32_32n_run(buf: &[f32]) -> f32 {
    let mut len = buf.len();
    let mut ptr = buf.as_ptr();
    let mut acc = 0f32;
    core::arch::asm!(
        r#"
        vbroadcastss ymm0, xmm0
        vmovaps ymm1, ymm0
        vmovaps ymm2, ymm0
        vmovaps ymm3, ymm0
        2:
            vmovaps ymm4, [rdi]
            vmovaps ymm5, [rdi + 32]
            vmovaps ymm6, [rdi + 64]
            vmovaps ymm7, [rdi + 96]
            vaddps ymm0, ymm0, ymm4
            vaddps ymm1, ymm1, ymm5
            vaddps ymm2, ymm2, ymm6
            vaddps ymm3, ymm3, ymm7
            add rdi, 128
            sub rsi, 32
            jnz 2b
        vaddps ymm0, ymm0, ymm1
        vaddps ymm2, ymm2, ymm3
        vaddps ymm0, ymm0, ymm2
        vperm2f128 ymm1, ymm0, ymm0, 1      // copy second half (4xf32) of ymm0 to ymm1
        vaddps xmm0, xmm0, xmm1              // xmm0 contains 4 values to sum
        vpermilps xmm1, xmm0, 2 + (3 << 2)  // second 2x32 bit half moved to top
        vaddps xmm0, xmm0, xmm1              // xmm0 containes 2 values
        vpermilps xmm1, xmm0, 1              // second f32 to top
        vaddps xmm0, xmm0, xmm1
        "#,
        inout("rsi") len => _,
        inout("rdi") ptr => _,
        inout("ymm0") acc,
        out("ymm1") _, out("ymm2") _, out("ymm3") _,
        out("ymm4") _, out("ymm5") _, out("ymm6") _, out("ymm7") _,
        options(nostack, preserves_flags)
    );
    acc
}

unsafe fn x86_64_avx512_sum_f32_64n_run(buf: &[f32]) -> f32 {
    let mut len = buf.len();
    let mut ptr = buf.as_ptr();
    let mut acc = 0f32;
    core::arch::asm!(
        r#"
       vxorps zmm0, zmm0, zmm0
       vxorps zmm1, zmm1, zmm1
       vxorps zmm2, zmm2, zmm2
       vxorps zmm3, zmm3, zmm3
       2:
           prefetchnta [rdi + 512]
           vmovaps zmm4, [rdi]
           vmovaps zmm5, [rdi + 64]
           vmovaps zmm6, [rdi + 128]
           vmovaps zmm7, [rdi + 192]
           vaddps zmm0, zmm0, zmm4
           vaddps zmm1, zmm1, zmm5
           vaddps zmm2, zmm2, zmm6
           vaddps zmm3, zmm3, zmm7
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
       vhaddps xmm0, xmm0, xmm0
       vhaddps xmm0, xmm0, xmm0
       "#,
        inout("rsi") len => _,
        inout("rdi") ptr => _,
        inout("zmm0") acc,
        out("zmm1") _, out("zmm2") _, out("zmm3") _,
        out("zmm4") _, out("zmm5") _, out("zmm6") _, out("zmm7") _,
        options(nostack, preserves_flags)
    );
    acc
}

reduce_impl_wrap!(
    f32,
    x86_64_avx512_sum_f32_64n,
    64,
    16,
    (),
    f32::zero(),
    #[inline(never)]
    fn run(buf: &[f32], _: ()) -> f32 {
        assert!(buf.len() % 64 == 0);
        assert!(!buf.is_empty());
        unsafe { x86_64_avx512_sum_f32_64n_run(buf) }
    },
    #[inline(never)]
    fn reduce_two(a: f32, b: f32) -> f32 {
        a + b
    }
);
