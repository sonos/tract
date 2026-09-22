//! simd128 f32 ln: the [`crate::generic::ln`] fit, four lanes at a time.

routine_ew_rust!(wasm32;
    f32,
    wasm_ln_f32_16n,
    16,
    4,
    #[inline(never)]
    fn run(buf: &mut [f32], _: ()) {
        use crate::generic::ln::{LN2_HI, LN2_LO, POLY, SPLIT, SUBNORMAL_SCALE, SUBNORMAL_SHIFT};
        use std::arch::wasm32::*;
        debug_assert!(buf.len() % Self::nr() == 0);
        debug_assert!(buf.as_ptr() as usize % Self::alignment_bytes() == 0);
        #[inline(always)]
        fn ln4(x: v128) -> v128 {
            use std::arch::wasm32::*;
            let one = f32x4_splat(1.0);
            let zero = f32x4_splat(0.0);
            let subnormal = f32x4_lt(x, f32x4_splat(f32::MIN_POSITIVE));
            let scaled =
                v128_bitselect(f32x4_mul(x, f32x4_splat(SUBNORMAL_SCALE)), x, subnormal);
            let mut exponent = i32x4_sub(u32x4_shr(scaled, 23), i32x4_splat(127));
            exponent =
                i32x4_sub(exponent, v128_and(subnormal, i32x4_splat(SUBNORMAL_SHIFT)));
            let mantissa =
                v128_or(v128_and(scaled, i32x4_splat(0x007fffff)), i32x4_splat(0x3f800000));
            let split = f32x4_gt(mantissa, f32x4_splat(SPLIT));
            let mantissa =
                v128_bitselect(f32x4_mul(mantissa, f32x4_splat(0.5)), mantissa, split);
            exponent = i32x4_add(exponent, v128_and(split, i32x4_splat(1)));
            let e = f32x4_convert_i32x4(exponent);
            let f = f32x4_sub(mantissa, one);
            let mut p = f32x4_splat(POLY[0]);
            for c in &POLY[1..] {
                let c = f32x4_splat(*c);
                p = madd_f32x4!(c, p, f);
            }
            let f2 = f32x4_mul(f, f);
            let mut y = f32x4_mul(f32x4_mul(p, f2), f);
            y = madd_f32x4!(y, e, f32x4_splat(LN2_LO));
            y = f32x4_add(madd_f32x4!(y, f32x4_splat(-0.5), f2), f);
            y = madd_f32x4!(y, e, f32x4_splat(LN2_HI));
            // A NaN compares false whichever way round, so `not greater than zero` is what
            // gathers the negatives, the zeros and the NaNs in one mask.
            let outside = v128_not(f32x4_gt(x, zero));
            let special = v128_bitselect(
                f32x4_splat(f32::NEG_INFINITY),
                f32x4_splat(f32::NAN),
                f32x4_eq(x, zero),
            );
            y = v128_bitselect(special, y, outside);
            v128_bitselect(f32x4_splat(f32::INFINITY), y, f32x4_eq(x, f32x4_splat(f32::INFINITY)))
        }
        unsafe {
            let mut p = buf.as_mut_ptr();
            let end = p.add(buf.len());
            while p < end {
                v128_store(p as *mut v128, ln4(v128_load(p as *const v128)));
                p = p.add(4);
            }
        }
    },
    func(Ln)
);
