ew_impl_wrap!(
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

#[cfg(test)]
mod test_arm64simd_mul_by_scalar_f32_16n {
    use super::*;
    mul_by_scalar_frame_tests!(true, f32, arm64simd_mul_by_scalar_f32_16n);
}
