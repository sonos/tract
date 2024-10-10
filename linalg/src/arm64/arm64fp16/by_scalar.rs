use crate::f16;

by_scalar_impl_wrap!(
    f16,
    arm64fp16_mul_by_scalar_f16_32n,
    32,
    4,
    f16,
    fn run(buf: &mut [f16], s: f16) {
        assert!(buf.len() % 16 == 0);
        assert!(buf.len() > 0);
        #[target_feature(enable = "fp16")]
        unsafe fn run(buf: &mut[f16], s: f16) {
            let len = buf.len();
            let ptr = buf.as_ptr();
            std::arch::asm!("
            dup v0.8h, v0.h[0]
            2:
                ld1 {{v4.8h, v5.8h, v6.8h, v7.8h}}, [{ptr}]
                fmul v4.8h, v4.8h, v0.8h
                fmul v5.8h, v5.8h, v0.8h
                fmul v6.8h, v6.8h, v0.8h
                fmul v7.8h, v7.8h, v0.8h
                st1 {{v4.8h, v5.8h, v6.8h, v7.8h}}, [{ptr}], 64
                subs {len}, {len}, 32
                bne 2b
            ",
            len = inout(reg) len => _,
            ptr = inout(reg) ptr => _,
            in("v0") s.to_bits(),
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,);
        }
        unsafe { run(buf, s) }
    }
);

by_scalar_impl_wrap!(
    f16,
    arm64fp16_add_by_scalar_f16_32n,
    32,
    4,
    f16,
    fn run(buf: &mut [f16], s: f16) {
        assert!(buf.len() % 16 == 0);
        assert!(buf.len() > 0);
        #[target_feature(enable = "fp16")]
        unsafe fn run(buf: &mut[f16], s: f16) {
            let len = buf.len();
            let ptr = buf.as_ptr();
            std::arch::asm!("
            dup v0.8h, v0.h[0]
            2:
                ld1 {{v4.8h, v5.8h, v6.8h, v7.8h}}, [{ptr}]
                fadd v4.8h, v4.8h, v0.8h
                fadd v5.8h, v5.8h, v0.8h
                fadd v6.8h, v6.8h, v0.8h
                fadd v7.8h, v7.8h, v0.8h
                st1 {{v4.8h, v5.8h, v6.8h, v7.8h}}, [{ptr}], 64
                subs {len}, {len}, 32
                bne 2b
            ",
            len = inout(reg) len => _,
            ptr = inout(reg) ptr => _,
            in("v0") s.to_bits(),
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,);
        }
        unsafe { run(buf, s) }
    }
);

by_scalar_impl_wrap!(
    f16,
    arm64fp16_sub_by_scalar_f16_32n,
    32,
    4,
    f16,
    fn run(buf: &mut [f16], s: f16) {
        assert!(buf.len() % 16 == 0);
        assert!(buf.len() > 0);
        #[target_feature(enable = "fp16")]
        unsafe fn run(buf: &mut[f16], s: f16) {
            let len = buf.len();
            let ptr = buf.as_ptr();
            std::arch::asm!("
            dup v0.8h, v0.h[0]
            2:
                ld1 {{v4.8h, v5.8h, v6.8h, v7.8h}}, [{ptr}]
                fsub v4.8h, v4.8h, v0.8h
                fsub v5.8h, v5.8h, v0.8h
                fsub v6.8h, v6.8h, v0.8h
                fsub v7.8h, v7.8h, v0.8h
                st1 {{v4.8h, v5.8h, v6.8h, v7.8h}}, [{ptr}], 64
                subs {len}, {len}, 32
                bne 2b
            ",
            len = inout(reg) len => _,
            ptr = inout(reg) ptr => _,
            in("v0") s.to_bits(),
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,);
        }
        unsafe { run(buf, s) }
    }
);

by_scalar_impl_wrap!(
    f16,
    arm64fp16_subf_by_scalar_f16_32n,
    32,
    4,
    f16,
    fn run(buf: &mut [f16], s: f16) {
        assert!(buf.len() % 16 == 0);
        assert!(buf.len() > 0);
        #[target_feature(enable = "fp16")]
        unsafe fn run(buf: &mut[f16], s: f16) {
            let len = buf.len();
            let ptr = buf.as_ptr();
            std::arch::asm!("
            dup v0.8h, v0.h[0]
            2:
                ld1 {{v4.8h, v5.8h, v6.8h, v7.8h}}, [{ptr}]
                fsub v4.8h, v0.8h, v4.8h
                fsub v5.8h, v0.8h, v5.8h
                fsub v6.8h, v0.8h, v6.8h
                fsub v7.8h, v0.8h, v7.8h
                st1 {{v4.8h, v5.8h, v6.8h, v7.8h}}, [{ptr}], 64
                subs {len}, {len}, 32
                bne 2b
            ",
            len = inout(reg) len => _,
            ptr = inout(reg) ptr => _,
            in("v0") s.to_bits(),
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,);
        }
        unsafe { run(buf, s) }
    }
);

by_scalar_impl_wrap!(
    f16,
    arm64fp16_min_by_scalar_f16_32n,
    32,
    4,
    f16,
    fn run(buf: &mut [f16], s: f16) {
        assert!(buf.len() % 16 == 0);
        assert!(buf.len() > 0);
        #[target_feature(enable = "fp16")]
        unsafe fn run(buf: &mut[f16], s: f16) {
            let len = buf.len();
            let ptr = buf.as_ptr();
            std::arch::asm!("
            dup v0.8h, v0.h[0]
            2:
                ld1 {{v4.8h, v5.8h, v6.8h, v7.8h}}, [{ptr}]
                fmin v4.8h, v4.8h, v0.8h
                fmin v5.8h, v5.8h, v0.8h
                fmin v6.8h, v6.8h, v0.8h
                fmin v7.8h, v7.8h, v0.8h
                st1 {{v4.8h, v5.8h, v6.8h, v7.8h}}, [{ptr}], 64
                subs {len}, {len}, 32
                bne 2b
            ",
            len = inout(reg) len => _,
            ptr = inout(reg) ptr => _,
            in("v0") s.to_bits(),
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,);
        }
        unsafe { run(buf, s) }
    }
);

by_scalar_impl_wrap!(
    f16,
    arm64fp16_max_by_scalar_f16_32n,
    32,
    4,
    f16,
    fn run(buf: &mut [f16], s: f16) {
        assert!(buf.len() % 16 == 0);
        assert!(buf.len() > 0);
        #[target_feature(enable = "fp16")]
        unsafe fn run(buf: &mut[f16], s: f16) {
            let len = buf.len();
            let ptr = buf.as_ptr();
            std::arch::asm!("
            dup v0.8h, v0.h[0]
            2:
                ld1 {{v4.8h, v5.8h, v6.8h, v7.8h}}, [{ptr}]
                fmax v4.8h, v4.8h, v0.8h
                fmax v5.8h, v5.8h, v0.8h
                fmax v6.8h, v6.8h, v0.8h
                fmax v7.8h, v7.8h, v0.8h
                st1 {{v4.8h, v5.8h, v6.8h, v7.8h}}, [{ptr}], 64
                subs {len}, {len}, 32
                bne 2b
            ",
            len = inout(reg) len => _,
            ptr = inout(reg) ptr => _,
            in("v0") s.to_bits(),
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,);
        }
        unsafe { run(buf, s) }
    }
);

#[cfg(test)]
mod test_arm64fp16_mul_by_scalar_f16_32n {
    use super::*;
    by_scalar_frame_tests!(crate::arm64::has_fp16(), f16, arm64fp16_mul_by_scalar_f16_32n, |a, b| a * b);
    by_scalar_frame_tests!(crate::arm64::has_fp16(), f16, arm64fp16_add_by_scalar_f16_32n, |a, b| a + b);
    by_scalar_frame_tests!(crate::arm64::has_fp16(), f16, arm64fp16_sub_by_scalar_f16_32n, |a, b| a - b);
    by_scalar_frame_tests!(crate::arm64::has_fp16(), f16, arm64fp16_subf_by_scalar_f16_32n, |a, b| b - a);
    by_scalar_frame_tests!(crate::arm64::has_fp16(), f16, arm64fp16_min_by_scalar_f16_32n, |a, b| a.min(b));
    by_scalar_frame_tests!(crate::arm64::has_fp16(), f16, arm64fp16_max_by_scalar_f16_32n, |a, b| a.max(b));
}
