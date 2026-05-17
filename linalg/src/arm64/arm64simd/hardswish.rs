ew_impl_wrap!(
    f32,
    arm64simd_hardswish_f32_8n,
    8,
    4,
    (),
    #[inline(never)]
    fn run(buf: &mut [f32], _: ()) {
        assert!(buf.len() % 8 == 0);
        assert!(buf.len() > 0);
        unsafe {
            let len = buf.len();
            let ptr = buf.as_ptr();
            std::arch::asm!("
                dup v0.4s, {three:v}.s[0]
                dup v1.4s, {six:v}.s[0]
                dup v2.4s, {inv6:v}.s[0]
                movi v3.4s, #0
                2:
                    ldp q4, q5, [{ptr}]

                    fadd v6.4s, v4.4s, v0.4s
                    fadd v7.4s, v5.4s, v0.4s

                    fmin v6.4s, v6.4s, v1.4s
                    fmin v7.4s, v7.4s, v1.4s

                    fmax v6.4s, v6.4s, v3.4s
                    fmax v7.4s, v7.4s, v3.4s

                    fmul v6.4s, v6.4s, v4.4s
                    fmul v7.4s, v7.4s, v5.4s

                    fmul v6.4s, v6.4s, v2.4s
                    fmul v7.4s, v7.4s, v2.4s

                    stp q6, q7, [{ptr}], #32
                    subs {len}, {len}, 8
                    bne 2b
            ",
            three = in(vreg) 3.0f32,
            six = in(vreg) 6.0f32,
            inv6 = in(vreg) 1.0f32 / 6.0f32,
            len = inout(reg) len => _,
            ptr = inout(reg) ptr => _,
            out("v0") _,
            out("v1") _,
            out("v2") _,
            out("v3") _,
            out("q4") _,
            out("q5") _,
            out("q6") _,
            out("q7") _,
            );
        }
    }
);

#[cfg(test)]
pub mod test_arm64simd_hardswish_f32_8n {
    use super::*;
    hardswish_frame_tests!(true, f32, arm64simd_hardswish_f32_8n);
}
