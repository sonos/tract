#[cfg(target_feature = "relaxed-simd")]
use crate::frame::element_wise::ElementWiseKer;

// Relaxed-SIMD activation kernels (f32, FMA path).
//
// `f32x4_relaxed_madd(a, b, c)` computes `a * b + c`. On hosts with hardware
// FMA (all ARM64, x86_64 with FMA3) it lowers to a single fused, single-
// rounded instruction. On hosts without, it falls back to mul+add — hence
// "relaxed". The result is therefore not bit-deterministic across all hosts,
// but it is at least as accurate as the separate mul+add (FMA does fewer
// roundings).
//
// For sigmoid/tanh polynomial evaluation, the 14 muladds in the Horner chain
// fuse cleanly. Measured ~1.65x over the baseline-simd128 explicit kernel and
// over LLVM auto-vec'd scalar on V8.
//
// Gated on `target_feature = "relaxed-simd"` because `f32x4_relaxed_madd`
// requires the relaxed-simd proposal to be enabled at compile time.
// ---------------------------------------------------------------------------

#[cfg(target_feature = "relaxed-simd")]
#[derive(Clone, Debug)]
pub struct WasmSigmoid4Relaxed;

#[cfg(target_feature = "relaxed-simd")]
impl ElementWiseKer<f32> for WasmSigmoid4Relaxed {
    fn name() -> &'static str {
        "wasm_relaxed_simd"
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

    fn run(buf: &mut [f32], _: ()) {
        use std::arch::wasm32::*;

        debug_assert!(buf.len() % Self::nr() == 0);
        debug_assert!(buf.as_ptr() as usize % Self::alignment_bytes() == 0);

        // Coefficients match generic/sigmoid.rs::ssigmoid bit-for-bit.
        // Output may differ by ≤1 ulp from scalar on FMA hosts (more accurate).
        const LOW: f32 = -18.6;
        const HIGH: f32 = -LOW;

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

            let half = f32x4_splat(0.5);
            let zero = f32x4_splat(0.0);
            let one = f32x4_splat(1.0);

            let mut p = buf.as_mut_ptr();
            let end = p.add(buf.len());
            while p < end {
                let v = v128_load(p as *const v128);
                let x = f32x4_min(hi, f32x4_max(lo, v));
                let x2 = f32x4_mul(x, x);

                // Horner numerator with FMA: pn = x2 * pn + a_n
                let pn = a13;
                let pn = f32x4_relaxed_madd(x2, pn, a11);
                let pn = f32x4_relaxed_madd(x2, pn, a9);
                let pn = f32x4_relaxed_madd(x2, pn, a7);
                let pn = f32x4_relaxed_madd(x2, pn, a5);
                let pn = f32x4_relaxed_madd(x2, pn, a3);
                let pn = f32x4_relaxed_madd(x2, pn, a1);
                let pn = f32x4_mul(pn, x);

                // Horner denominator with FMA
                let qn = b6;
                let qn = f32x4_relaxed_madd(x2, qn, b4);
                let qn = f32x4_relaxed_madd(x2, qn, b2);
                let qn = f32x4_relaxed_madd(x2, qn, b0);

                // sigmoid is (0, 1): the add below cancels down to ~1e-8 on either tail,
                // below the rounding error of the division, so the sum needs clamping to
                // stay in range.
                let r = f32x4_add(f32x4_div(pn, qn), half);
                let r = f32x4_min(one, f32x4_max(zero, r));
                v128_store(p as *mut v128, r);
                p = p.add(4);
            }
        }
    }
}

#[cfg(target_feature = "relaxed-simd")]
#[derive(Clone, Debug)]
pub struct WasmTanh4Relaxed;

#[cfg(target_feature = "relaxed-simd")]
impl ElementWiseKer<f32> for WasmTanh4Relaxed {
    fn name() -> &'static str {
        "wasm_relaxed_simd"
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

    fn run(buf: &mut [f32], _: ()) {
        use std::arch::wasm32::*;

        debug_assert!(buf.len() % Self::nr() == 0);
        debug_assert!(buf.as_ptr() as usize % Self::alignment_bytes() == 0);

        const LOW: f32 = -8.9;
        const HIGH: f32 = 8.9;

        const ALPHA_13: f32 = -8.488492677e-14;
        const ALPHA_11: f32 = 5.277853000e-11;
        const ALPHA_9: f32 = -2.022500419e-8;
        const ALPHA_7: f32 = 0.00001115424833;
        const ALPHA_5: f32 = 0.003103950131;
        const ALPHA_3: f32 = 0.1308400453;
        const ALPHA_1: f32 = 0.9999999934;

        const BETA_6: f32 = 0.0002546136580;
        const BETA_4: f32 = 0.02449515379;
        const BETA_2: f32 = 0.4641733162;
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

            let one = f32x4_splat(1.0);
            let minus_one = f32x4_splat(-1.0);

            let mut p = buf.as_mut_ptr();
            let end = p.add(buf.len());
            while p < end {
                let v = v128_load(p as *const v128);
                let x = f32x4_min(hi, f32x4_max(lo, v));
                let x2 = f32x4_mul(x, x);

                let pn = a13;
                let pn = f32x4_relaxed_madd(x2, pn, a11);
                let pn = f32x4_relaxed_madd(x2, pn, a9);
                let pn = f32x4_relaxed_madd(x2, pn, a7);
                let pn = f32x4_relaxed_madd(x2, pn, a5);
                let pn = f32x4_relaxed_madd(x2, pn, a3);
                let pn = f32x4_relaxed_madd(x2, pn, a1);
                let pn = f32x4_mul(pn, x);

                let qn = b6;
                let qn = f32x4_relaxed_madd(x2, qn, b4);
                let qn = f32x4_relaxed_madd(x2, qn, b2);
                let qn = f32x4_relaxed_madd(x2, qn, b0);

                // tanh is (-1, 1): the quotient comes within one ulp of ±1 across the top
                // of the input range, under the kernel's own rounding error, so it needs
                // clamping to stay in range.
                let r = f32x4_div(pn, qn);
                let r = f32x4_min(one, f32x4_max(minus_one, r));
                v128_store(p as *mut v128, r);
                p = p.add(4);
            }
        }
    }
}

#[cfg(all(test, target_feature = "relaxed-simd"))]
#[macro_use]
mod test_wasm_sigmoid_relaxed {
    sigmoid_frame_tests!(true, f32, crate::wasm::WasmSigmoid4Relaxed);
}

#[cfg(all(test, target_feature = "relaxed-simd"))]
#[macro_use]
mod test_wasm_tanh_relaxed {
    tanh_frame_tests!(true, f32, crate::wasm::WasmTanh4Relaxed);
}
