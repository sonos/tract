by_scalar_impl_wrap!(
    f32,
    arm64simd_mul_by_scalar_f32_16n,
    16,
    4,
    f32,
    fn run(buf: &mut [f32], s: f32) {
        assert!(buf.len() % 16 == 0);
        assert!(buf.len() > 0);
        unsafe {
            let len = buf.len();
            let ptr = buf.as_ptr();
            std::arch::asm!("
            dup v0.4s, v0.s[0]
            2:
                ld1 {{v4.4s, v5.4s, v6.4s, v7.4s}}, [{ptr}]
                fmul v4.4s, v4.4s, v0.4s
                fmul v5.4s, v5.4s, v0.4s
                fmul v6.4s, v6.4s, v0.4s
                fmul v7.4s, v7.4s, v0.4s
                st1 {{v4.4s, v5.4s, v6.4s, v7.4s}}, [{ptr}], 64
                subs {len}, {len}, 16
                bne 2b
            ",
            len = inout(reg) len => _,
            ptr = inout(reg) ptr => _,
            in("v0") s,
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,);
        }
    }
);

by_scalar_impl_wrap!(
    f32,
    arm64simd_add_by_scalar_f32_16n,
    16,
    4,
    f32,
    fn run(buf: &mut [f32], s: f32) {
        assert!(buf.len() % 16 == 0);
        assert!(buf.len() > 0);
        unsafe {
            let len = buf.len();
            let ptr = buf.as_ptr();
            std::arch::asm!("
            dup v0.4s, v0.s[0]
            2:
                ld1 {{v4.4s, v5.4s, v6.4s, v7.4s}}, [{ptr}]
                fadd v4.4s, v4.4s, v0.4s
                fadd v5.4s, v5.4s, v0.4s
                fadd v6.4s, v6.4s, v0.4s
                fadd v7.4s, v7.4s, v0.4s
                st1 {{v4.4s, v5.4s, v6.4s, v7.4s}}, [{ptr}], 64
                subs {len}, {len}, 16
                bne 2b
            ",
            len = inout(reg) len => _,
            ptr = inout(reg) ptr => _,
            in("v0") s,
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,);
        }
    }
);

by_scalar_impl_wrap!(
    f32,
    arm64simd_sub_by_scalar_f32_16n,
    16,
    4,
    f32,
    fn run(buf: &mut [f32], s: f32) {
        assert!(buf.len() % 16 == 0);
        assert!(buf.len() > 0);
        unsafe {
            let len = buf.len();
            let ptr = buf.as_ptr();
            std::arch::asm!("
            dup v0.4s, v0.s[0]
            2:
                ld1 {{v4.4s, v5.4s, v6.4s, v7.4s}}, [{ptr}]
                fsub v4.4s, v4.4s, v0.4s
                fsub v5.4s, v5.4s, v0.4s
                fsub v6.4s, v6.4s, v0.4s
                fsub v7.4s, v7.4s, v0.4s
                st1 {{v4.4s, v5.4s, v6.4s, v7.4s}}, [{ptr}], 64
                subs {len}, {len}, 16
                bne 2b
            ",
            len = inout(reg) len => _,
            ptr = inout(reg) ptr => _,
            in("v0") s,
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,);
        }
    }
);

by_scalar_impl_wrap!(
    f32,
    arm64simd_subf_by_scalar_f32_16n,
    16,
    4,
    f32,
    fn run(buf: &mut [f32], s: f32) {
        assert!(buf.len() % 16 == 0);
        assert!(buf.len() > 0);
        unsafe {
            let len = buf.len();
            let ptr = buf.as_ptr();
            std::arch::asm!("
            dup v0.4s, v0.s[0]
            2:
                ld1 {{v4.4s, v5.4s, v6.4s, v7.4s}}, [{ptr}]
                fsub v4.4s, v0.4s, v4.4s
                fsub v5.4s, v0.4s, v5.4s
                fsub v6.4s, v0.4s, v6.4s
                fsub v7.4s, v0.4s, v7.4s
                st1 {{v4.4s, v5.4s, v6.4s, v7.4s}}, [{ptr}], 64
                subs {len}, {len}, 16
                bne 2b
            ",
            len = inout(reg) len => _,
            ptr = inout(reg) ptr => _,
            in("v0") s,
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,);
        }
    }
);

by_scalar_impl_wrap!(
    f32,
    arm64simd_min_by_scalar_f32_16n,
    16,
    4,
    f32,
    fn run(buf: &mut [f32], s: f32) {
        assert!(buf.len() % 16 == 0);
        assert!(buf.len() > 0);
        unsafe {
            let len = buf.len();
            let ptr = buf.as_ptr();
            std::arch::asm!("
            dup v0.4s, v0.s[0]
            2:
                ld1 {{v4.4s, v5.4s, v6.4s, v7.4s}}, [{ptr}]
                fmin v4.4s, v4.4s, v0.4s
                fmin v5.4s, v5.4s, v0.4s
                fmin v6.4s, v6.4s, v0.4s
                fmin v7.4s, v7.4s, v0.4s
                st1 {{v4.4s, v5.4s, v6.4s, v7.4s}}, [{ptr}], 64
                subs {len}, {len}, 16
                bne 2b
            ",
            len = inout(reg) len => _,
            ptr = inout(reg) ptr => _,
            in("v0") s,
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,);
        }
    }
);

by_scalar_impl_wrap!(
    f32,
    arm64simd_max_by_scalar_f32_16n,
    16,
    4,
    f32,
    fn run(buf: &mut [f32], s: f32) {
        assert!(buf.len() % 16 == 0);
        assert!(buf.len() > 0);
        unsafe {
            let len = buf.len();
            let ptr = buf.as_ptr();
            std::arch::asm!("
            dup v0.4s, v0.s[0]
            2:
                ld1 {{v4.4s, v5.4s, v6.4s, v7.4s}}, [{ptr}]
                fmax v4.4s, v4.4s, v0.4s
                fmax v5.4s, v5.4s, v0.4s
                fmax v6.4s, v6.4s, v0.4s
                fmax v7.4s, v7.4s, v0.4s
                st1 {{v4.4s, v5.4s, v6.4s, v7.4s}}, [{ptr}], 64
                subs {len}, {len}, 16
                bne 2b
            ",
            len = inout(reg) len => _,
            ptr = inout(reg) ptr => _,
            in("v0") s,
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,);
        }
    }
);

#[cfg(test)]
mod test_arm64simd_mul_by_scalar_f32_16n {
    use super::*;
    by_scalar_frame_tests!(true, f32, arm64simd_mul_by_scalar_f32_16n, |a, b| a * b);
    by_scalar_frame_tests!(true, f32, arm64simd_add_by_scalar_f32_16n, |a, b| a + b);
    by_scalar_frame_tests!(true, f32, arm64simd_sub_by_scalar_f32_16n, |a, b| a - b);
    by_scalar_frame_tests!(true, f32, arm64simd_subf_by_scalar_f32_16n, |a, b| b - a);
    by_scalar_frame_tests!(true, f32, arm64simd_min_by_scalar_f32_16n, |a, b| a.min(b));
    by_scalar_frame_tests!(true, f32, arm64simd_max_by_scalar_f32_16n, |a, b| a.max(b));
}
