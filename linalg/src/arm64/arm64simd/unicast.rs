
unicast_impl_wrap!(
    f32,
    arm64simd_unicast_mul_f32_16n,
    16,
    4,
    #[inline(never)]
    fn run(a: &mut [f32], b: &[f32]) {
        assert!(a.len() == b.len());
        assert!(a.len() % 16 == 0);
        assert!(a.len() > 0);
        unsafe fn run(a: &mut [f32], b: &[f32]) {
            let len = a.len();
            let a_ptr = a.as_ptr();
            let b_ptr = b.as_ptr();
            std::arch::asm!("
                2:
                    ld1 {{v0.4s, v1.4s, v2.4s, v3.4s}}, [{a_ptr}]
                    ld1 {{v4.4s, v5.4s, v6.4s, v7.4s}}, [{b_ptr}], 64
                    fmul v0.4s, v0.4s, v4.4s
                    fmul v1.4s, v1.4s, v5.4s
                    fmul v2.4s, v2.4s, v6.4s
                    fmul v3.4s, v3.4s, v7.4s
                    st1 {{v0.4s, v1.4s, v2.4s, v3.4s}}, [{a_ptr}], 64
                    subs {len}, {len}, 16
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
    f32,
    arm64simd_unicast_add_f32_16n,
    16,
    4,
    #[inline(never)]
    fn run(a: &mut [f32], b: &[f32]) {
        assert!(a.len() == b.len());
        assert!(a.len() % 16 == 0);
        assert!(a.len() > 0);
        unsafe fn run(a: &mut [f32], b: &[f32]) {
            let len = a.len();
            let a_ptr = a.as_ptr();
            let b_ptr = b.as_ptr();
            std::arch::asm!("
                2:
                    ld1 {{v0.4s, v1.4s, v2.4s, v3.4s}}, [{a_ptr}]
                    ld1 {{v4.4s, v5.4s, v6.4s, v7.4s}}, [{b_ptr}], 64
                    fadd v0.4s, v0.4s, v4.4s
                    fadd v1.4s, v1.4s, v5.4s
                    fadd v2.4s, v2.4s, v6.4s
                    fadd v3.4s, v3.4s, v7.4s
                    st1 {{v0.4s, v1.4s, v2.4s, v3.4s}}, [{a_ptr}], 64
                    subs {len}, {len}, 16
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
    f32,
    arm64simd_unicast_sub_f32_16n,
    16,
    4,
    #[inline(never)]
    fn run(a: &mut [f32], b: &[f32]) {
        assert!(a.len() == b.len());
        assert!(a.len() % 16 == 0);
        assert!(a.len() > 0);
        unsafe fn run(a: &mut [f32], b: &[f32]) {
            let len = a.len();
            let a_ptr = a.as_ptr();
            let b_ptr = b.as_ptr();
            std::arch::asm!("
                2:
                    ld1 {{v0.4s, v1.4s, v2.4s, v3.4s}}, [{a_ptr}]
                    ld1 {{v4.4s, v5.4s, v6.4s, v7.4s}}, [{b_ptr}], 64
                    fsub v0.4s, v0.4s, v4.4s
                    fsub v1.4s, v1.4s, v5.4s
                    fsub v2.4s, v2.4s, v6.4s
                    fsub v3.4s, v3.4s, v7.4s
                    st1 {{v0.4s, v1.4s, v2.4s, v3.4s}}, [{a_ptr}], 64
                    subs {len}, {len}, 16
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
    f32,
    arm64simd_unicast_subf_f32_16n,
    16,
    4,
    #[inline(never)]
    fn run(a: &mut [f32], b: &[f32]) {
        assert!(a.len() == b.len());
        assert!(a.len() % 16 == 0);
        assert!(a.len() > 0);
        unsafe fn run(a: &mut [f32], b: &[f32]) {
            let len = a.len();
            let a_ptr = a.as_ptr();
            let b_ptr = b.as_ptr();
            std::arch::asm!("
                2:
                    ld1 {{v0.4s, v1.4s, v2.4s, v3.4s}}, [{a_ptr}]
                    ld1 {{v4.4s, v5.4s, v6.4s, v7.4s}}, [{b_ptr}], 64
                    fsub v0.4s, v4.4s, v0.4s
                    fsub v1.4s, v5.4s, v1.4s
                    fsub v2.4s, v6.4s, v2.4s
                    fsub v3.4s, v7.4s, v3.4s
                    st1 {{v0.4s, v1.4s, v2.4s, v3.4s}}, [{a_ptr}], 64
                    subs {len}, {len}, 16
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
    f32,
    arm64simd_unicast_max_f32_16n,
    16,
    4,
    #[inline(never)]
    fn run(a: &mut [f32], b: &[f32]) {
        assert!(a.len() == b.len());
        assert!(a.len() % 16 == 0);
        assert!(a.len() > 0);
        unsafe fn run(a: &mut [f32], b: &[f32]) {
            let len = a.len();
            let a_ptr = a.as_ptr();
            let b_ptr = b.as_ptr();
            std::arch::asm!("
                2:
                    ld1 {{v0.4s, v1.4s, v2.4s, v3.4s}}, [{a_ptr}]
                    ld1 {{v4.4s, v5.4s, v6.4s, v7.4s}}, [{b_ptr}], 64
                    fmax v0.4s, v0.4s, v4.4s
                    fmax v1.4s, v1.4s, v5.4s
                    fmax v2.4s, v2.4s, v6.4s
                    fmax v3.4s, v3.4s, v7.4s
                    st1 {{v0.4s, v1.4s, v2.4s, v3.4s}}, [{a_ptr}], 64
                    subs {len}, {len}, 16
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
    f32,
    arm64simd_unicast_min_f32_16n,
    16,
    4,
    #[inline(never)]
    fn run(a: &mut [f32], b: &[f32]) {
        assert!(a.len() == b.len());
        assert!(a.len() % 16 == 0);
        assert!(a.len() > 0);
        unsafe fn run(a: &mut [f32], b: &[f32]) {
            let len = a.len();
            let a_ptr = a.as_ptr();
            let b_ptr = b.as_ptr();
            std::arch::asm!("
                2:
                    ld1 {{v0.4s, v1.4s, v2.4s, v3.4s}}, [{a_ptr}]
                    ld1 {{v4.4s, v5.4s, v6.4s, v7.4s}}, [{b_ptr}], 64
                    fmin v0.4s, v0.4s, v4.4s
                    fmin v1.4s, v1.4s, v5.4s
                    fmin v2.4s, v2.4s, v6.4s
                    fmin v3.4s, v3.4s, v7.4s
                    st1 {{v0.4s, v1.4s, v2.4s, v3.4s}}, [{a_ptr}], 64
                    subs {len}, {len}, 16
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
mod test_arm64simd_unicast_mul_f32_16n {
    use super::*;
    use proptest::strategy::Strategy;
    crate::unicast_frame_tests!(true, f32, arm64simd_unicast_mul_f32_16n, |a, b| a * b);
    crate::unicast_frame_tests!(true, f32, arm64simd_unicast_add_f32_16n, |a, b| a + b);
    crate::unicast_frame_tests!(true, f32, arm64simd_unicast_sub_f32_16n, |a, b| a - b);
    crate::unicast_frame_tests!(true, f32, arm64simd_unicast_subf_f32_16n, |a, b| b - a);
    crate::unicast_frame_tests!(true, f32, arm64simd_unicast_min_f32_16n, |a, b| a.min(b));
    crate::unicast_frame_tests!(true, f32, arm64simd_unicast_max_f32_16n, |a, b| a.max(b));
}
