ew_impl_wrap!(
    f32,
    arm64simd_leaky_relu_f32_8n,
    8,
    4,
    f32,
    #[inline(never)]
    fn run(buf: &mut [f32], alpha: f32) {
        assert!(buf.len() % 8 == 0);
        assert!(buf.len() > 0);
        unsafe {
            let len = buf.len();
            let ptr = buf.as_ptr();
            std::arch::asm!("
                dup v0.4s, {alpha:v}.s[0]
                dup v1.4s, {one:v}.s[0]
                2:
                    ldp q3, q4, [{ptr}]

                    fcmgt v5.4s, v3.4s, #0.0
                    fcmgt v6.4s, v4.4s, #0.0
                    bsl   v5.16b, v1.16b, v0.16b
                    bsl   v6.16b, v1.16b, v0.16b
                    fmul  v3.4s, v3.4s, v5.4s
                    fmul  v4.4s, v4.4s, v6.4s

                    stp q3, q4, [{ptr}], #32
                    subs {len}, {len}, 8
                    bne 2b
            ",
            one = in(vreg) 1.0f32,
            alpha = in(vreg) alpha,
            len = inout(reg) len => _,
            ptr = inout(reg) ptr => _,
            out("v0") _,
            out("v1") _,
            out("q3") _,
            out("q4") _,
            out("q5") _,
            out("q6") _,
            );
        }
    }
);

#[cfg(test)]
pub mod test_arm64simd_leaky_relu_f32_8n {
    use super::*;
    leaky_relu_frame_tests!(true, f32, arm64simd_leaky_relu_f32_8n);
}
