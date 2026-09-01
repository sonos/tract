use crate::frame::element_wise::ElementWiseKer;

// f32 SIMD activation kernels for WASM +simd128 (no relaxed-simd required).
//
// These are fused single-pass kernels that compute the full activation
// in-place on 4-wide f32 vectors, matching the generic scalar reference's
// coefficients. They use plain f32x4_mul/add (not relaxed_madd) so they
// run on any WASM SIMD host. Only activations whose generic kernel goes
// through libm (or an inlined transcendental) per element live here;
// sigmoid and tanh deliberately have no simd128 kernel, as their generic
// polynomial loops already auto-vectorize under +simd128 and a manual
// kernel is slower. Their relaxed-simd FMA variants in act.rs remain the
// only WASM overrides.

#[derive(Clone, Debug)]
pub struct WasmGelu4;

impl ElementWiseKer<f32> for WasmGelu4 {
    fn name() -> &'static str {
        "wasm_simd128"
    }

    fn alignment_bytes() -> usize {
        16
    }

    fn alignment_items() -> usize {
        4
    }

    fn nr() -> usize {
        4
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    fn run(buf: &mut [f32], _: ()) {
        use std::arch::wasm32::*;

        debug_assert!(buf.len() % Self::nr() == 0);
        debug_assert!(buf.as_ptr() as usize % Self::alignment_bytes() == 0);

        const SQRT_2_OVER_PI: f32 = 0.7978845608028654;
        const COEF: f32 = 0.044715;

        let lo = f32x4_splat(-8.9);
        let hi = f32x4_splat(8.9);
        let sqrt_2_over_pi = f32x4_splat(SQRT_2_OVER_PI);
        let coef = f32x4_splat(COEF);
        let half = f32x4_splat(0.5);
        let one = f32x4_splat(1.0);
        let neg_one = f32x4_splat(-1.0);

        let a13 = f32x4_splat(-8.488492677e-14);
        let a11 = f32x4_splat(5.277853000e-11);
        let a9 = f32x4_splat(-2.022500419e-8);
        let a7 = f32x4_splat(0.00001115424833);
        let a5 = f32x4_splat(0.003103950131);
        let a3 = f32x4_splat(0.1308400453);
        let a1 = f32x4_splat(0.9999999934);
        let b6 = f32x4_splat(0.0002546136580);
        let b4 = f32x4_splat(0.02449515379);
        let b2 = f32x4_splat(0.4641733162);
        let b0 = f32x4_splat(1.0);

        unsafe {
            let mut p = buf.as_mut_ptr();
            let end = p.add(buf.len());
            while p < end {
                let v = v128_load(p as *const v128);
                let orig = v;

                let x2 = f32x4_mul(v, v);
                let x3 = f32x4_mul(x2, v);

                let inner = f32x4_mul(sqrt_2_over_pi, f32x4_add(v, f32x4_mul(coef, x3)));
                let clamped = f32x4_max(lo, f32x4_min(hi, inner));

                let x2c = f32x4_mul(clamped, clamped);
                let mut pn = a13;
                pn = f32x4_add(f32x4_mul(x2c, pn), a11);
                pn = f32x4_add(f32x4_mul(x2c, pn), a9);
                pn = f32x4_add(f32x4_mul(x2c, pn), a7);
                pn = f32x4_add(f32x4_mul(x2c, pn), a5);
                pn = f32x4_add(f32x4_mul(x2c, pn), a3);
                pn = f32x4_add(f32x4_mul(x2c, pn), a1);
                let pn = f32x4_mul(pn, clamped);

                let mut qn = b6;
                qn = f32x4_add(f32x4_mul(x2c, qn), b4);
                qn = f32x4_add(f32x4_mul(x2c, qn), b2);
                qn = f32x4_add(f32x4_mul(x2c, qn), b0);

                // Lanes pinned to the low clamp: tanh is -1 there to f32
                // precision, but the polynomial lands one ulp short, so
                // 1 + tanh never cancels and the error grows with |x|.
                let pinned_low = f32x4_eq(clamped, lo);
                let tanh = f32x4_div(pn, qn);
                let tanh = v128_bitselect(neg_one, tanh, pinned_low);
                let result = f32x4_mul(f32x4_mul(half, orig), f32x4_add(one, tanh));

                v128_store(p as *mut v128, result);
                p = p.add(4);
            }
        }
    }
    bail_stub!(wasm32; fn run(&mut [f32], ()));
}

#[derive(Clone, Debug)]
pub struct WasmSilu4;

impl ElementWiseKer<f32> for WasmSilu4 {
    fn name() -> &'static str {
        "wasm_simd128"
    }

    fn alignment_bytes() -> usize {
        16
    }

    fn alignment_items() -> usize {
        4
    }

    fn nr() -> usize {
        4
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    fn run(buf: &mut [f32], _: ()) {
        use std::arch::wasm32::*;

        debug_assert!(buf.len() % Self::nr() == 0);
        debug_assert!(buf.as_ptr() as usize % Self::alignment_bytes() == 0);

        const LOW: f32 = -18.6;
        const HIGH: f32 = 18.6;

        const ALPHA_13: f32 = -4.433153405e-18;
        const ALPHA_11: f32 = 1.169974371e-14;
        const ALPHA_9: f32 = -1.875289645e-11;
        const ALPHA_7: f32 = 4.257889523e-8;
        const ALPHA_5: f32 = 0.00004811817576;
        const ALPHA_3: f32 = 0.008163842030;
        const ALPHA_1: f32 = 0.2499999971;

        const BETA_6: f32 = 3.922935744e-6;
        const BETA_4: f32 = 0.001524872358;
        const BETA_2: f32 = 0.1159886749;
        const BETA_0: f32 = 1.0;

        unsafe {
            let lo = f32x4_splat(LOW);
            let hi = f32x4_splat(HIGH);

            let a13 = f32x4_splat(ALPHA_13);
            let a11 = f32x4_splat(ALPHA_11);
            let a9 = f32x4_splat(ALPHA_9);
            let a7 = f32x4_splat(ALPHA_7);
            let a5 = f32x4_splat(ALPHA_5);
            let a3 = f32x4_splat(ALPHA_3);
            let a1 = f32x4_splat(ALPHA_1);

            let b6 = f32x4_splat(BETA_6);
            let b4 = f32x4_splat(BETA_4);
            let b2 = f32x4_splat(BETA_2);
            let b0 = f32x4_splat(BETA_0);

            let mut p = buf.as_mut_ptr();
            let end = p.add(buf.len());
            while p < end {
                let v = v128_load(p as *const v128);
                let orig = v;

                let x = f32x4_min(hi, f32x4_max(lo, v));
                let x2 = f32x4_mul(x, x);

                let mut pn = a13;
                pn = f32x4_add(f32x4_mul(x2, pn), a11);
                pn = f32x4_add(f32x4_mul(x2, pn), a9);
                pn = f32x4_add(f32x4_mul(x2, pn), a7);
                pn = f32x4_add(f32x4_mul(x2, pn), a5);
                pn = f32x4_add(f32x4_mul(x2, pn), a3);
                pn = f32x4_add(f32x4_mul(x2, pn), a1);
                let pn = f32x4_mul(pn, x);

                let mut qn = b6;
                qn = f32x4_add(f32x4_mul(x2, qn), b4);
                qn = f32x4_add(f32x4_mul(x2, qn), b2);
                qn = f32x4_add(f32x4_mul(x2, qn), b0);

                let sig = f32x4_add(f32x4_div(pn, qn), f32x4_splat(0.5));
                let result = f32x4_mul(orig, sig);

                v128_store(p as *mut v128, result);
                p = p.add(4);
            }
        }
    }
    bail_stub!(wasm32; fn run(&mut [f32], ()));
}

#[derive(Clone, Debug)]
pub struct WasmErf4;

impl ElementWiseKer<f32> for WasmErf4 {
    fn name() -> &'static str {
        "wasm_simd128"
    }

    fn alignment_bytes() -> usize {
        16
    }

    fn alignment_items() -> usize {
        4
    }

    fn nr() -> usize {
        4
    }

    #[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
    fn run(buf: &mut [f32], _: ()) {
        use std::arch::wasm32::*;

        debug_assert!(buf.len() % Self::nr() == 0);
        debug_assert!(buf.as_ptr() as usize % Self::alignment_bytes() == 0);

        const A1: f32 = 0.0705230784;
        const A2: f32 = 0.0422820123;
        const A3: f32 = 0.0092705272;
        const A4: f32 = 0.0001520143;
        const A5: f32 = 0.0002765672;
        const A6: f32 = 0.0000430638;

        unsafe {
            let a1 = f32x4_splat(A1);
            let a2 = f32x4_splat(A2);
            let a3 = f32x4_splat(A3);
            let a4 = f32x4_splat(A4);
            let a5 = f32x4_splat(A5);
            let a6 = f32x4_splat(A6);
            let one = f32x4_splat(1.0);

            let mut p = buf.as_mut_ptr();
            let end = p.add(buf.len());
            while p < end {
                let v = v128_load(p as *const v128);
                let sign = v;

                let abs = f32x4_abs(v);
                let mut y = f32x4_mul(a6, abs);
                y = f32x4_mul(f32x4_add(y, a5), abs);
                y = f32x4_mul(f32x4_add(y, a4), abs);
                y = f32x4_mul(f32x4_add(y, a3), abs);
                y = f32x4_mul(f32x4_add(y, a2), abs);
                y = f32x4_mul(f32x4_add(y, a1), abs);

                let y_plus_1 = f32x4_add(y, one);
                let mut recip = y_plus_1;
                recip = f32x4_mul(recip, recip);
                recip = f32x4_mul(recip, recip);
                recip = f32x4_mul(recip, recip);
                recip = f32x4_mul(recip, recip);
                recip = f32x4_div(one, recip);

                let result = f32x4_sub(one, recip);
                let sign_bits = v128_and(sign, f32x4_splat(-0.0));
                let result = v128_or(result, sign_bits);

                v128_store(p as *mut v128, result);
                p = p.add(4);
            }
        }
    }
    bail_stub!(wasm32; fn run(&mut [f32], ()));
}

submit_routine!(wasm32; F32, Gelu, WasmGelu4);
submit_routine!(wasm32; F32, Silu, WasmSilu4);
submit_routine!(wasm32; F32, Erf, WasmErf4);

#[cfg(all(test, target_feature = "simd128"))]
#[macro_use]
mod test_wasm_gelu {
    gelu_frame_tests!(true, f32, crate::wasm::WasmGelu4);
}

#[cfg(all(test, target_feature = "simd128"))]
#[macro_use]
mod test_wasm_silu {
    silu_frame_tests!(true, f32, crate::wasm::WasmSilu4);
}

#[cfg(all(test, target_feature = "simd128"))]
#[macro_use]
mod test_wasm_erf {
    crate::erf_frame_tests!(true, f32, crate::wasm::WasmErf4);
}

#[cfg(all(test, target_feature = "simd128"))]
mod test_wasm_gelu_tail {
    #[test]
    fn gelu_saturates_to_zero_below_the_tanh_clamp() {
        crate::frame::element_wise::test::test_element_wise::<crate::wasm::WasmGelu4, f32, _>(
            &[-1e6, -1e3, -100.0, -10.0, 10.0, 1e6],
            |x| 0.5 * x * (1.0 + (0.7978845608028654 * (x + 0.044715 * x * x * x)).tanh()),
        )
        .unwrap();
    }
}
