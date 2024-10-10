use tract_data::half::f16;

unicast_impl_wrap!(
    f16,
    arm64fp16_unicast_mul_f16_32n,
    32,
    8,
    #[inline(never)]
    fn run(a: &mut [f16], b: &[f16]) {
        assert!(a.len() == b.len());
        assert!(a.len() % 32 == 0);
        assert!(a.len() > 0);
        #[target_feature(enable = "fp16")]
        unsafe fn run(a: &mut [f16], b: &[f16]) {
            let len = a.len();
            let a_ptr = a.as_ptr();
            let b_ptr = b.as_ptr();
            std::arch::asm!("
                2:
                    ld1 {{v0.8h, v1.8h, v2.8h, v3.8h}}, [{a_ptr}]
                    ld1 {{v4.8h, v5.8h, v6.8h, v7.8h}}, [{b_ptr}], 64
                    fmul v0.8h, v0.8h, v4.8h
                    fmul v1.8h, v1.8h, v5.8h
                    fmul v2.8h, v2.8h, v6.8h
                    fmul v3.8h, v3.8h, v7.8h
                    st1 {{v0.8h, v1.8h, v2.8h, v3.8h}}, [{a_ptr}], 64
                    subs {len}, {len}, 32
                    bne 2b
            ",
            len = inout(reg) len => _,
            a_ptr = inout(reg) a_ptr => _,
            b_ptr = inout(reg) b_ptr => _,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,);
        }
        unsafe { run(a, b) }
    }
);

unicast_impl_wrap!(
    f16,
    arm64fp16_unicast_add_f16_32n,
    32,
    8,
    #[inline(never)]
    fn run(a: &mut [f16], b: &[f16]) {
        assert!(a.len() == b.len());
        assert!(a.len() % 32 == 0);
        assert!(a.len() > 0);
        #[target_feature(enable = "fp16")]
        unsafe fn run(a: &mut [f16], b: &[f16]) {
            let len = a.len();
            let a_ptr = a.as_ptr();
            let b_ptr = b.as_ptr();
            std::arch::asm!("
                2:
                    ld1 {{v0.8h, v1.8h, v2.8h, v3.8h}}, [{a_ptr}]
                    ld1 {{v4.8h, v5.8h, v6.8h, v7.8h}}, [{b_ptr}], 64
                    fadd v0.8h, v0.8h, v4.8h
                    fadd v1.8h, v1.8h, v5.8h
                    fadd v2.8h, v2.8h, v6.8h
                    fadd v3.8h, v3.8h, v7.8h
                    st1 {{v0.8h, v1.8h, v2.8h, v3.8h}}, [{a_ptr}], 64
                    subs {len}, {len}, 32
                    bne 2b
            ",
            len = inout(reg) len => _,
            a_ptr = inout(reg) a_ptr => _,
            b_ptr = inout(reg) b_ptr => _,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,);
        }
        unsafe { run(a, b) }
    }
);


unicast_impl_wrap!(
    f16,
    arm64fp16_unicast_sub_f16_32n,
    32,
    8,
    #[inline(never)]
    fn run(a: &mut [f16], b: &[f16]) {
        assert!(a.len() == b.len());
        assert!(a.len() % 32 == 0);
        assert!(a.len() > 0);
        #[target_feature(enable = "fp16")]
        unsafe fn run(a: &mut [f16], b: &[f16]) {
            let len = a.len();
            let a_ptr = a.as_ptr();
            let b_ptr = b.as_ptr();
            std::arch::asm!("
                2:
                    ld1 {{v0.8h, v1.8h, v2.8h, v3.8h}}, [{a_ptr}]
                    ld1 {{v4.8h, v5.8h, v6.8h, v7.8h}}, [{b_ptr}], 64
                    fsub v0.8h, v0.8h, v4.8h
                    fsub v1.8h, v1.8h, v5.8h
                    fsub v2.8h, v2.8h, v6.8h
                    fsub v3.8h, v3.8h, v7.8h
                    st1 {{v0.8h, v1.8h, v2.8h, v3.8h}}, [{a_ptr}], 64
                    subs {len}, {len}, 32
                    bne 2b
            ",
            len = inout(reg) len => _,
            a_ptr = inout(reg) a_ptr => _,
            b_ptr = inout(reg) b_ptr => _,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,);
        }
        unsafe { run(a, b) }
    }
);

unicast_impl_wrap!(
    f16,
    arm64fp16_unicast_subf_f16_32n,
    32,
    8,
    #[inline(never)]
    fn run(a: &mut [f16], b: &[f16]) {
        assert!(a.len() == b.len());
        assert!(a.len() % 32 == 0);
        assert!(a.len() > 0);
        #[target_feature(enable = "fp16")]
        unsafe fn run(a: &mut [f16], b: &[f16]) {
            let len = a.len();
            let a_ptr = a.as_ptr();
            let b_ptr = b.as_ptr();
            std::arch::asm!("
                2:
                    ld1 {{v0.8h, v1.8h, v2.8h, v3.8h}}, [{a_ptr}]
                    ld1 {{v4.8h, v5.8h, v6.8h, v7.8h}}, [{b_ptr}], 64
                    fsub v0.8h, v4.8h, v0.8h
                    fsub v1.8h, v5.8h, v1.8h
                    fsub v2.8h, v6.8h, v2.8h
                    fsub v3.8h, v7.8h, v3.8h
                    st1 {{v0.8h, v1.8h, v2.8h, v3.8h}}, [{a_ptr}], 64
                    subs {len}, {len}, 32
                    bne 2b
            ",
            len = inout(reg) len => _,
            a_ptr = inout(reg) a_ptr => _,
            b_ptr = inout(reg) b_ptr => _,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,);
        }
        unsafe { run(a, b) }
    }
);

unicast_impl_wrap!(
    f16,
    arm64fp16_unicast_min_f16_32n,
    32,
    8,
    #[inline(never)]
    fn run(a: &mut [f16], b: &[f16]) {
        assert!(a.len() == b.len());
        assert!(a.len() % 32 == 0);
        assert!(a.len() > 0);
        #[target_feature(enable = "fp16")]
        unsafe fn run(a: &mut [f16], b: &[f16]) {
            let len = a.len();
            let a_ptr = a.as_ptr();
            let b_ptr = b.as_ptr();
            std::arch::asm!("
                2:
                    ld1 {{v0.8h, v1.8h, v2.8h, v3.8h}}, [{a_ptr}]
                    ld1 {{v4.8h, v5.8h, v6.8h, v7.8h}}, [{b_ptr}], 64
                    fmin v0.8h, v0.8h, v4.8h
                    fmin v1.8h, v1.8h, v5.8h
                    fmin v2.8h, v2.8h, v6.8h
                    fmin v3.8h, v3.8h, v7.8h
                    st1 {{v0.8h, v1.8h, v2.8h, v3.8h}}, [{a_ptr}], 64
                    subs {len}, {len}, 32
                    bne 2b
            ",
            len = inout(reg) len => _,
            a_ptr = inout(reg) a_ptr => _,
            b_ptr = inout(reg) b_ptr => _,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,);
        }
        unsafe { run(a, b) }
    }
);

unicast_impl_wrap!(
    f16,
    arm64fp16_unicast_max_f16_32n,
    32,
    8,
    #[inline(never)]
    fn run(a: &mut [f16], b: &[f16]) {
        assert!(a.len() == b.len());
        assert!(a.len() % 32 == 0);
        assert!(a.len() > 0);
        #[target_feature(enable = "fp16")]
        unsafe fn run(a: &mut [f16], b: &[f16]) {
            let len = a.len();
            let a_ptr = a.as_ptr();
            let b_ptr = b.as_ptr();
            std::arch::asm!("
                2:
                    ld1 {{v0.8h, v1.8h, v2.8h, v3.8h}}, [{a_ptr}]
                    ld1 {{v4.8h, v5.8h, v6.8h, v7.8h}}, [{b_ptr}], 64
                    fmax v0.8h, v0.8h, v4.8h
                    fmax v1.8h, v1.8h, v5.8h
                    fmax v2.8h, v2.8h, v6.8h
                    fmax v3.8h, v3.8h, v7.8h
                    st1 {{v0.8h, v1.8h, v2.8h, v3.8h}}, [{a_ptr}], 64
                    subs {len}, {len}, 32
                    bne 2b
            ",
            len = inout(reg) len => _,
            a_ptr = inout(reg) a_ptr => _,
            b_ptr = inout(reg) b_ptr => _,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,);
        }
        unsafe { run(a, b) }
    }
);

#[cfg(test)]
mod test_arm64fp16_unicast_mul_f16_32n {
    use super::*;
    use proptest::strategy::Strategy;
    crate::unicast_frame_tests!(crate::arm64::has_fp16(), f16, arm64fp16_unicast_mul_f16_32n, |a, b| a * b);
    crate::unicast_frame_tests!(crate::arm64::has_fp16(), f16, arm64fp16_unicast_add_f16_32n, |a, b| a + b);
    crate::unicast_frame_tests!(crate::arm64::has_fp16(), f16, arm64fp16_unicast_sub_f16_32n, |a, b| a - b);
    crate::unicast_frame_tests!(crate::arm64::has_fp16(), f16, arm64fp16_unicast_subf_f16_32n, |a, b| b - a);
    crate::unicast_frame_tests!(crate::arm64::has_fp16(), f16, arm64fp16_unicast_min_f16_32n, |a, b| a.min(b));
    crate::unicast_frame_tests!(crate::arm64::has_fp16(), f16, arm64fp16_unicast_max_f16_32n, |a, b| a.max(b));
}
