//! simd128 f32 exp: the [`crate::generic::exp`] reduction and fit, four lanes at a time.

routine_ew_rust!(wasm32;
    f32,
    wasm_exp_f32_16n,
    16,
    4,
    #[inline(never)]
    fn run(buf: &mut [f32], _: ()) {
        use crate::generic::exp::{HIGH, LN2_HI, LN2_LO, LOG2E, LOW, POLY, SCALE_BIAS};
        use std::arch::wasm32::*;
        debug_assert!(buf.len() % Self::nr() == 0);
        debug_assert!(buf.as_ptr() as usize % Self::alignment_bytes() == 0);
        #[inline(always)]
        fn exp4(x: v128) -> v128 {
            use std::arch::wasm32::*;
            // Clamped by selection rather than by f32x4_min and f32x4_max: those answer with
            // the bound where a lane is NaN, which is what must propagate instead.
            let high = f32x4_splat(HIGH);
            let low = f32x4_splat(LOW);
            let x = v128_bitselect(high, x, f32x4_gt(x, high));
            let x = v128_bitselect(low, x, f32x4_lt(x, low));
            let kf = f32x4_nearest(f32x4_mul(x, f32x4_splat(LOG2E)));
            let mut r = madd_f32x4!(x, kf, f32x4_splat(-LN2_HI));
            r = madd_f32x4!(r, kf, f32x4_splat(-LN2_LO));
            let mut q = f32x4_splat(POLY[0]);
            for c in &POLY[1..] {
                let c = f32x4_splat(*c);
                q = madd_f32x4!(c, q, r);
            }
            let k = i32x4_trunc_sat_f32x4(kf);
            let half = i32x4_shr(k, 1);
            let rest = i32x4_sub(k, half);
            let scale = |k| i32x4_shl(i32x4_add(k, i32x4_splat(SCALE_BIAS)), 23);
            f32x4_mul(f32x4_mul(q, scale(half)), scale(rest))
        }
        unsafe {
            let mut p = buf.as_mut_ptr();
            let end = p.add(buf.len());
            while p < end {
                v128_store(p as *mut v128, exp4(v128_load(p as *const v128)));
                p = p.add(4);
            }
        }
    },
    func(Exp)
);
