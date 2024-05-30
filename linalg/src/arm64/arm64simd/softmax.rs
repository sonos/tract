map_reduce_impl_wrap!(
    f32,
    arm64simd_softmax2_fastcompact_f32_16n,
    16,
    4,
    f32,
    f32::MIN,
    0f32,
    #[inline(never)]
    fn run(buf: &mut [f32], max: f32) -> f32 {
        assert!(buf.len() % 16 == 0);
        assert!(buf.len() > 0);
        let len = buf.len();
        let ptr = buf.as_ptr();
        let mut acc;
        const MLN2: f32 = 0.6931471805f32;
        const A: f32 = 8388608.0f32;
        const B: f32 = 1065353216.0f32;
        const C: f32 = 60801.0f32;
        const SLOPE: f32 = A / MLN2;
        const OFFSET: f32 = B - C;
        unsafe {
            std::arch::asm!("
            // v0-v3 sum acc
            eor v0.16b, v0.16b, v0.16b
            eor v1.16b, v1.16b, v1.16b
            eor v2.16b, v2.16b, v2.16b
            eor v3.16b, v3.16b, v3.16b

            dup v4.4s, v4.s[0] // max
            dup v5.4s, v5.s[0] // slope
            dup v6.4s, v6.s[0] // offset
            eor v7.16b, v7.16b, v7.16b // zero for max
            2:
                ld1 {{v8.4s, v9.4s, v10.4s, v11.4s}}, [{ptr}]

                fsub v8.4s, v8.4s, v4.4s
                fsub v9.4s, v9.4s, v4.4s
                fsub v10.4s, v10.4s, v4.4s
                fsub v11.4s, v11.4s, v4.4s

                fmul v8.4s, v8.4s, v5.4s
                fmul v9.4s, v9.4s, v5.4s
                fmul v10.4s, v10.4s, v5.4s
                fmul v11.4s, v11.4s, v5.4s

                fadd v8.4s, v8.4s, v6.4s
                fadd v9.4s, v9.4s, v6.4s
                fadd v10.4s, v10.4s, v6.4s
                fadd v11.4s, v11.4s, v6.4s

                fmax v8.4s, v8.4s, v7.4s
                fmax v9.4s, v9.4s, v7.4s
                fmax v10.4s, v10.4s, v7.4s
                fmax v11.4s, v11.4s, v7.4s

                fcvtnu v8.4s, v8.4s
                fcvtnu v9.4s, v9.4s
                fcvtnu v10.4s, v10.4s
                fcvtnu v11.4s, v11.4s

                fadd v0.4s, v0.4s, v8.4s
                fadd v1.4s, v1.4s, v9.4s
                fadd v2.4s, v2.4s, v10.4s
                fadd v3.4s, v3.4s, v11.4s

                st1 {{v8.4s, v9.4s, v10.4s, v11.4s}}, [{ptr}], 64
                subs {len}, {len}, 16
                bne 2b

            fadd v0.4s, v0.4s, v1.4s
            fadd v2.4s, v2.4s, v3.4s
            fadd v0.4s, v0.4s, v2.4s

            ext v1.16b, v0.16b, v0.16b, 4
            ext v2.16b, v0.16b, v0.16b, 8
            ext v3.16b, v0.16b, v0.16b, 12
            fadd v0.4s, v0.4s, v1.4s
            fadd v2.4s, v2.4s, v3.4s
            fadd v0.4s, v0.4s, v2.4s
            ",
            len = inout(reg) len => _,
            ptr = inout(reg) ptr => _,
            out("v0") acc,
            out("v1") _,
            out("v2") _,
            out("v3") _,
            inout("v4") max => _,
            inout("v5") SLOPE => _,
            inout("v6") OFFSET => _,
            out("v7") _,
            out("v8") _,
            out("v9") _,
            out("v10") _,
            out("v11") _,
            );
        }
        acc
    },
    #[inline(never)]
    fn reduce_two(a: f32, b: f32) -> f32 {
        a + b
    }
);

#[cfg(test)]
mod test_arm64simd_softmax2_fastcompact_f32_16n {
    use super::*;
    crate::softmax_l2_frame_tests!(true, f32, arm64simd_softmax2_fastcompact_f32_16n);
}
