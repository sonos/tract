// Fused GELU (tanh-form, pow=3):
//   gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//
// loop4 (16 lanes per iter) + loop1 (4-lane tail). Clones the tanh Padé
// polynomial from arm64simd_tanh_f32_4n.S.j2, with pre-tanh argument
// computation up front and the final 0.5*x*(1+tanh) combined via fmla.
// Single memory pass (load + store), no scratch buffer.

ew_impl_wrap!(
    f32,
    arm64simd_gelu_f32_4n_fused,
    4,
    4,
    (),
    #[inline(never)]
    fn run(buf: &mut [f32], _: ()) {
        // Tanh Padé coefficients (matches arm64simd_tanh_f32_4n.S.j2) +
        // 3 GELU constants packed into the last vector lanes:
        //   index 13: 0.5
        //   index 14: sqrt(2/pi)              ≈ 0.7978846
        //   index 15: 0.044715 * sqrt(2/pi)   ≈ 0.0356774
        static COEFFS: [f32; 16] = [
            -8.9,
            8.9,
            -8.488492677e-14,
            5.277853000e-11,
            -2.022500419e-8,
            0.00001115424833,
            0.003103950131,
            0.1308400453,
            0.9999999934,
            0.0002546136580,
            0.02449515379,
            0.4641733162,
            1.0,
            0.5,
            0.7978845608028654,
            0.03567739613,
        ];

        assert!(buf.len() % 4 == 0);
        if buf.is_empty() {
            return;
        }

        unsafe {
            let len = buf.len();
            let ptr = buf.as_mut_ptr();
            let coef_ptr = COEFFS.as_ptr();

            // Register layout (loop4):
            //   v0-v3: coefficients
            //   v4:    sqrt(2/pi)  (broadcast of v3.s[2])
            //   v5:    tanh clamp low (-8.9, dup v0.s[0])
            //   v6:    tanh clamp high (8.9, dup v0.s[1])
            //   v7:    0.5 (broadcast of v3.s[1])
            //   v8-v11: 0.5 * original x (saved after load)
            //   v16-v19: working (load -> pre_tanh -> clamped -> numerator)
            //   v20-v23: x² for tanh polynomial
            //   v24-v27: polynomial intermediates (denominator at end)
            //   v28-v31: polynomial intermediates (also x³ temp before tanh)
            std::arch::asm!("
                ld1 {{ v0.4s, v1.4s, v2.4s, v3.4s }}, [{coef}]
                dup v5.4s, v0.s[0]
                dup v6.4s, v0.s[1]
                dup v7.4s, v3.s[1]
                dup v4.4s, v3.s[2]

                cmp {len}, #16
                blt 9f

                1:
                    ld1 {{ v16.4s, v17.4s, v18.4s, v19.4s }}, [{ptr}]

                    // Save 0.5 * original x into v8-v11
                    fmul v8.4s, v16.4s, v7.4s
                    fmul v9.4s, v17.4s, v7.4s
                    fmul v10.4s, v18.4s, v7.4s
                    fmul v11.4s, v19.4s, v7.4s

                    // Compute x^3 into v28-v31
                    fmul v28.4s, v16.4s, v16.4s
                    fmul v29.4s, v17.4s, v17.4s
                    fmul v30.4s, v18.4s, v18.4s
                    fmul v31.4s, v19.4s, v19.4s

                    fmul v28.4s, v28.4s, v16.4s
                    fmul v29.4s, v29.4s, v17.4s
                    fmul v30.4s, v30.4s, v18.4s
                    fmul v31.4s, v31.4s, v19.4s

                    // pre_tanh = sqrt(2/pi)*x + 0.0356774 * x^3
                    fmul v16.4s, v16.4s, v4.4s
                    fmul v17.4s, v17.4s, v4.4s
                    fmul v18.4s, v18.4s, v4.4s
                    fmul v19.4s, v19.4s, v4.4s

                    fmla v16.4s, v28.4s, v3.s[3]
                    fmla v17.4s, v29.4s, v3.s[3]
                    fmla v18.4s, v30.4s, v3.s[3]
                    fmla v19.4s, v31.4s, v3.s[3]

                    // Clamp pre_tanh argument
                    fmax v16.4s, v16.4s, v5.4s
                    fmax v17.4s, v17.4s, v5.4s
                    fmax v18.4s, v18.4s, v5.4s
                    fmax v19.4s, v19.4s, v5.4s

                    fmin v16.4s, v16.4s, v6.4s
                    fmin v17.4s, v17.4s, v6.4s
                    fmin v18.4s, v18.4s, v6.4s
                    fmin v19.4s, v19.4s, v6.4s

                    // Tanh Padé polynomial (cloned from arm64simd_tanh_f32_4n.S.j2)
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

                    // tanh(pre_arg) = num/denom
                    fdiv v16.4s, v16.4s, v24.4s
                    fdiv v17.4s, v17.4s, v25.4s
                    fdiv v18.4s, v18.4s, v26.4s
                    fdiv v19.4s, v19.4s, v27.4s

                    // result = 0.5*x * (1 + tanh) = (0.5*x) + (0.5*x) * tanh
                    fmla v8.4s, v8.4s, v16.4s
                    fmla v9.4s, v9.4s, v17.4s
                    fmla v10.4s, v10.4s, v18.4s
                    fmla v11.4s, v11.4s, v19.4s

                    st1 {{ v8.4s, v9.4s, v10.4s, v11.4s }}, [{ptr}], #64
                    sub {len}, {len}, #16
                    cmp {len}, #16
                    bge 1b

                9:
                cbz {len}, 3f

                2:
                    ld1 {{ v16.4s }}, [{ptr}]
                    fmul v8.4s, v16.4s, v7.4s

                    fmul v28.4s, v16.4s, v16.4s
                    fmul v28.4s, v28.4s, v16.4s

                    fmul v16.4s, v16.4s, v4.4s
                    fmla v16.4s, v28.4s, v3.s[3]

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

                    fmla v8.4s, v8.4s, v16.4s

                    st1 {{ v8.4s }}, [{ptr}], #16
                    subs {len}, {len}, 4
                    bne 2b

                3:
            ",
            coef = in(reg) coef_ptr,
            ptr = inout(reg) ptr => _,
            len = inout(reg) len => _,
            out("v0") _, out("v1") _, out("v2") _, out("v3") _,
            out("v4") _, out("v5") _, out("v6") _, out("v7") _,
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
pub mod test_arm64simd_gelu_f32_4n_fused {
    use super::*;
    gelu_frame_tests!(true, f32, arm64simd_gelu_f32_4n_fused);
}
