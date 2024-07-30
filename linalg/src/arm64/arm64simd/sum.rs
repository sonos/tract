use crate::num_traits::Zero;

reduce_impl_wrap!(
    f32,
    arm64simd_sum_f32_16n,
    16,
    4,
    (),
    f32::zero(),
    #[inline(never)]
    fn run(buf: &[f32], _: ()) -> f32 {
        assert!(buf.len() % 16 == 0);
        assert!(buf.len() > 0);
        unsafe fn run(buf: &[f32]) -> f32 {
            let len = buf.len();
            let ptr = buf.as_ptr();
            let mut out: u32;
            std::arch::asm!("
                movi v0.4s, #0
                movi v1.4s, #0
                movi v2.4s, #0
                movi v3.4s, #0
                2:
                    ld1 {{v4.4s, v5.4s, v6.4s, v7.4s}}, [{ptr}], 64
                    fadd v0.4s, v0.4s, v4.4s
                    fadd v1.4s, v1.4s, v5.4s
                    fadd v2.4s, v2.4s, v6.4s
                    fadd v3.4s, v3.4s, v7.4s

                    subs {len}, {len}, 16
                    bne 2b

                fadd v0.4s, v0.4s, v1.4s
                fadd v2.4s, v2.4s, v3.4s
                fadd v0.4s, v0.4s, v2.4s
                faddp v0.4s, v0.4s, v0.4s
                faddp v0.4s, v0.4s, v0.4s
                ",
                ptr = inout(reg) ptr => _,
                len = inout(reg) len => _,
                out("s0") out, out("v1") _, out("v2") _, out("v3") _,
                out("v4") _, out("v5") _, out("v6") _, out("v7") _,);
            f32::from_bits(out)
        }
        unsafe { run(buf) }
    },
    #[inline(never)]
    fn reduce_two(a: f32, b: f32) -> f32 {
        a + b
    }
);

#[cfg(test)]
mod test_arm64simd_sum_f32_16n {
    use super::*;
    crate::sum_frame_tests!(true, f32, arm64simd_sum_f32_16n);
}
