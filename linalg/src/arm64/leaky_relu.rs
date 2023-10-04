use tract_data::internal::f16;

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
                1:
                    ldp q3, q4, [{ptr}]

                    fcmgt v5.4s, v3.4s, #0.0
                    fcmgt v6.4s, v4.4s, #0.0
                    bsl   v5.16b, v1.16b, v0.16b
                    bsl   v6.16b, v1.16b, v0.16b
                    fmul  v3.4s, v3.4s, v5.4s
                    fmul  v4.4s, v4.4s, v6.4s

                    stp q3, q4, [{ptr}], #32
                    subs {len}, {len}, 8
                    bne 1b
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

ew_impl_wrap!(
    f16,
    arm64fp16_leaky_relu_f16_16n,
    16,
    8,
    f16,
    #[inline(never)]
    fn run(buf: &mut [f16], alpha: f16) {
        assert!(buf.len() % 8 == 0);
        assert!(buf.len() > 0);
        #[target_feature(enable = "fp16")]
        unsafe fn run(buf: &mut [f16], alpha: f16) {
            let len = buf.len();
            let ptr = buf.as_ptr();
            std::arch::asm!("
                    dup v0.8h, {alpha:v}.h[0]
                    dup v1.8h, {one:v}.h[0]
                    1:
                        ldp q3, q4, [{ptr}]

                        fcmgt v5.8h, v3.8h, #0.0
                        fcmgt v6.8h, v4.8h, #0.0
                        bsl   v5.16b, v1.16b, v0.16b
                        bsl   v6.16b, v1.16b, v0.16b
                        fmul  v3.8h, v3.8h, v5.8h
                        fmul  v4.8h, v4.8h, v6.8h

                        stp q3, q4, [{ptr}], #32
                        subs {len}, {len}, 16
                        bne 1b
                ",
            one = in(vreg) f16::from_f32(1.0f32).to_bits(),
            alpha = in(vreg) alpha.to_bits(),
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
        unsafe { run(buf, alpha) }
    }
);

#[cfg(test)]
pub mod test_arm64simd_leaky_relu_f32_8n {
    use super::*;
    leaky_relu_frame_tests!(true, f32, arm64simd_leaky_relu_f32_8n);
}

#[cfg(test)]
pub mod test_arm64simd_leaky_relu_f16_16n {
    use super::*;
    leaky_relu_frame_tests!(crate::arm64::has_fp16(), f16, arm64fp16_leaky_relu_f16_16n);
}
