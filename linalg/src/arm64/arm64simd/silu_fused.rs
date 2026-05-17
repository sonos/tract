// Fused SiLU: x * sigmoid(x).
// loop4 (16 lanes per iter) + loop1 (4-lane tail).
// Clones the sigmoid Padé polynomial from arm64simd_sigmoid_f32_4n.S.j2,
// with the input saved before clamp (in v8-v11) and multiplied back at the
// end. Single memory pass (load + store), no scratch buffer.

ew_impl_wrap!(
    f32,
    arm64simd_silu_f32_4n_fused,
    4,
    4,
    (),
    #[inline(never)]
    fn run(buf: &mut [f32], _: ()) {
        // Sigmoid Padé coefficients (matches arm64simd_sigmoid_f32_4n.S.j2).
        static COEFFS: [f32; 16] = [
            -18.6,
            18.6,
            -4.433153405e-18,
            1.169974371e-14,
            -1.875289645e-11,
            4.257889523e-8,
            0.00004811817576,
            0.008163842030,
            0.2499999971,
            3.922935744e-6,
            0.001524872358,
            0.1159886749,
            1.0,
            0.5,
            0.0,
            0.0,
        ];

        assert!(buf.len() % 4 == 0);
        if buf.is_empty() {
            return;
        }

        unsafe {
            let len = buf.len();
            let ptr = buf.as_mut_ptr();
            let coef_ptr = COEFFS.as_ptr();

            std::arch::asm!("
                ld1 {{ v0.4s, v1.4s, v2.4s, v3.4s }}, [{coef}]
                dup v5.4s, v0.s[0]
                dup v6.4s, v0.s[1]
                dup v7.4s, v3.s[1]

                cmp {len}, #16
                blt 9f

                1:
                    ld1 {{ v16.4s, v17.4s, v18.4s, v19.4s }}, [{ptr}]

                    mov v8.16b, v16.16b
                    mov v9.16b, v17.16b
                    mov v10.16b, v18.16b
                    mov v11.16b, v19.16b

                    fmax v16.4s, v16.4s, v5.4s
                    fmax v17.4s, v17.4s, v5.4s
                    fmax v18.4s, v18.4s, v5.4s
                    fmax v19.4s, v19.4s, v5.4s

                    fmin v16.4s, v16.4s, v6.4s
                    fmin v17.4s, v17.4s, v6.4s
                    fmin v18.4s, v18.4s, v6.4s
                    fmin v19.4s, v19.4s, v6.4s

                    fmul v20.4s, v16.4s, v16.4s
                    fmul v21.4s, v17.4s, v17.4s
                    fmul v22.4s, v18.4s, v18.4s
                    fmul v23.4s, v19.4s, v19.4s

                    dup v24.4s, v0.s[3]
                    fmla v24.4s, v20.4s, v0.s[2]
                    dup v25.4s, v0.s[3]
                    fmla v25.4s, v21.4s, v0.s[2]
                    dup v26.4s, v0.s[3]
                    fmla v26.4s, v22.4s, v0.s[2]
                    dup v27.4s, v0.s[3]
                    fmla v27.4s, v23.4s, v0.s[2]

                    dup v28.4s, v1.s[0]
                    fmla v28.4s, v20.4s, v24.4s
                    dup v29.4s, v1.s[0]
                    fmla v29.4s, v21.4s, v25.4s
                    dup v30.4s, v1.s[0]
                    fmla v30.4s, v22.4s, v26.4s
                    dup v31.4s, v1.s[0]
                    fmla v31.4s, v23.4s, v27.4s

                    dup v24.4s, v1.s[1]
                    fmla v24.4s, v20.4s, v28.4s
                    dup v25.4s, v1.s[1]
                    fmla v25.4s, v21.4s, v29.4s
                    dup v26.4s, v1.s[1]
                    fmla v26.4s, v22.4s, v30.4s
                    dup v27.4s, v1.s[1]
                    fmla v27.4s, v23.4s, v31.4s

                    dup v28.4s, v1.s[2]
                    fmla v28.4s, v20.4s, v24.4s
                    dup v29.4s, v1.s[2]
                    fmla v29.4s, v21.4s, v25.4s
                    dup v30.4s, v1.s[2]
                    fmla v30.4s, v22.4s, v26.4s
                    dup v31.4s, v1.s[2]
                    fmla v31.4s, v23.4s, v27.4s

                    dup v24.4s, v1.s[3]
                    fmla v24.4s, v20.4s, v28.4s
                    dup v25.4s, v1.s[3]
                    fmla v25.4s, v21.4s, v29.4s
                    dup v26.4s, v1.s[3]
                    fmla v26.4s, v22.4s, v30.4s
                    dup v27.4s, v1.s[3]
                    fmla v27.4s, v23.4s, v31.4s

                    dup v28.4s, v2.s[0]
                    fmla v28.4s, v20.4s, v24.4s
                    dup v29.4s, v2.s[0]
                    fmla v29.4s, v21.4s, v25.4s
                    dup v30.4s, v2.s[0]
                    fmla v30.4s, v22.4s, v26.4s
                    dup v31.4s, v2.s[0]
                    fmla v31.4s, v23.4s, v27.4s

                    fmul v16.4s, v16.4s, v28.4s
                    fmul v17.4s, v17.4s, v29.4s
                    fmul v18.4s, v18.4s, v30.4s
                    fmul v19.4s, v19.4s, v31.4s

                    dup v24.4s, v2.s[2]
                    fmla v24.4s, v20.4s, v2.s[1]
                    dup v25.4s, v2.s[2]
                    fmla v25.4s, v21.4s, v2.s[1]
                    dup v26.4s, v2.s[2]
                    fmla v26.4s, v22.4s, v2.s[1]
                    dup v27.4s, v2.s[2]
                    fmla v27.4s, v23.4s, v2.s[1]

                    dup v28.4s, v2.s[3]
                    fmla v28.4s, v20.4s, v24.4s
                    dup v29.4s, v2.s[3]
                    fmla v29.4s, v21.4s, v25.4s
                    dup v30.4s, v2.s[3]
                    fmla v30.4s, v22.4s, v26.4s
                    dup v31.4s, v2.s[3]
                    fmla v31.4s, v23.4s, v27.4s

                    dup v24.4s, v3.s[0]
                    fmla v24.4s, v20.4s, v28.4s
                    dup v25.4s, v3.s[0]
                    fmla v25.4s, v21.4s, v29.4s
                    dup v26.4s, v3.s[0]
                    fmla v26.4s, v22.4s, v30.4s
                    dup v27.4s, v3.s[0]
                    fmla v27.4s, v23.4s, v31.4s

                    fdiv v16.4s, v16.4s, v24.4s
                    fdiv v17.4s, v17.4s, v25.4s
                    fdiv v18.4s, v18.4s, v26.4s
                    fdiv v19.4s, v19.4s, v27.4s

                    fadd v16.4s, v16.4s, v7.4s
                    fadd v17.4s, v17.4s, v7.4s
                    fadd v18.4s, v18.4s, v7.4s
                    fadd v19.4s, v19.4s, v7.4s

                    fmul v16.4s, v16.4s, v8.4s
                    fmul v17.4s, v17.4s, v9.4s
                    fmul v18.4s, v18.4s, v10.4s
                    fmul v19.4s, v19.4s, v11.4s

                    st1 {{ v16.4s, v17.4s, v18.4s, v19.4s }}, [{ptr}], #64
                    sub {len}, {len}, #16
                    cmp {len}, #16
                    bge 1b

                9:
                cbz {len}, 3f

                2:
                    ld1 {{ v16.4s }}, [{ptr}]
                    mov v8.16b, v16.16b

                    fmax v16.4s, v16.4s, v5.4s
                    fmin v16.4s, v16.4s, v6.4s
                    fmul v20.4s, v16.4s, v16.4s

                    dup v24.4s, v0.s[3]
                    fmla v24.4s, v20.4s, v0.s[2]
                    dup v28.4s, v1.s[0]
                    fmla v28.4s, v20.4s, v24.4s
                    dup v24.4s, v1.s[1]
                    fmla v24.4s, v20.4s, v28.4s
                    dup v28.4s, v1.s[2]
                    fmla v28.4s, v20.4s, v24.4s
                    dup v24.4s, v1.s[3]
                    fmla v24.4s, v20.4s, v28.4s
                    dup v28.4s, v2.s[0]
                    fmla v28.4s, v20.4s, v24.4s
                    fmul v16.4s, v16.4s, v28.4s

                    dup v24.4s, v2.s[2]
                    fmla v24.4s, v20.4s, v2.s[1]
                    dup v28.4s, v2.s[3]
                    fmla v28.4s, v20.4s, v24.4s
                    dup v24.4s, v3.s[0]
                    fmla v24.4s, v20.4s, v28.4s

                    fdiv v16.4s, v16.4s, v24.4s
                    fadd v16.4s, v16.4s, v7.4s

                    fmul v16.4s, v16.4s, v8.4s

                    st1 {{ v16.4s }}, [{ptr}], #16
                    subs {len}, {len}, 4
                    bne 2b

                3:
            ",
            coef = in(reg) coef_ptr,
            ptr = inout(reg) ptr => _,
            len = inout(reg) len => _,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            out("v5") _, out("v6") _, out("v7") _,
            out("v8") _, out("v9") _, out("v10") _, out("v11") _,
            out("v16") _, out("v17") _, out("v18") _, out("v19") _,
            out("v20") _, out("v21") _, out("v22") _, out("v23") _,
            out("v24") _, out("v25") _, out("v26") _, out("v27") _,
            out("v28") _, out("v29") _, out("v30") _, out("v31") _,
            options(nostack),
            );
        }
    }
);

#[cfg(test)]
pub mod test_arm64simd_silu_f32_4n_fused {
    use super::*;
    silu_frame_tests!(true, f32, arm64simd_silu_f32_4n_fused);
}
