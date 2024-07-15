use tract_data::half::f16;
use crate::num_traits::Zero;

reduce_impl_wrap!(
    f16,
    arm64fp16_sum_f16_32n,
    32,
    8,
    (),
    f16::zero(),
    #[inline(never)]
    fn run(buf: &[f16], _: ()) -> f16 {
        assert!(buf.len() % 32 == 0);
        assert!(buf.len() > 0);
        #[target_feature(enable = "fp16")]
        unsafe fn run(buf: &[f16]) -> f16 {
            let len = buf.len();
            let ptr = buf.as_ptr();
            let mut out: u16;
            std::arch::asm!("
                movi v0.8h, #0
                movi v1.8h, #0
                movi v2.8h, #0
                movi v3.8h, #0
                2:
                    ld1 {{v4.8h, v5.8h, v6.8h, v7.8h}}, [{ptr}], 64
                    fadd v0.8h, v0.8h, v4.8h
                    fadd v1.8h, v1.8h, v5.8h
                    fadd v2.8h, v2.8h, v6.8h
                    fadd v3.8h, v3.8h, v7.8h

                    subs {len}, {len}, 32
                    bne 2b

                fadd v0.8h, v0.8h, v1.8h
                fadd v2.8h, v2.8h, v3.8h
                fadd v0.8h, v0.8h, v2.8h
                faddp v0.8h, v0.8h, v0.8h
                faddp v0.8h, v0.8h, v0.8h
                faddp v0.8h, v0.8h, v0.8h
                ",
                ptr = inout(reg) ptr => _,
                len = inout(reg) len => _,
                out("s0") out, out("v1") _, out("v2") _, out("v3") _,
                out("v4") _, out("v5") _, out("v6") _, out("v7") _,);
            f16::from_bits(out)
        }
        unsafe { run(buf) }
    },
    #[inline(never)]
    fn reduce_two(a: f16, b: f16) -> f16 {
        a + b
    }
);

#[cfg(test)]
mod test_arm64fp16_sum_f16_32n {
    use super::*;
    crate::sum_frame_tests!(crate::arm64::has_fp16(), f16, arm64fp16_sum_f16_32n);
}
