//! NEON f32 ln: the [`crate::generic::ln`] fit, four lanes at a time.

routine_ew_rust!(aarch64;
    f32,
    arm64simd_ln_f32_16n,
    16,
    4,
    #[inline(never)]
    fn run(buf: &mut [f32], _: ()) {
        debug_assert!(buf.len() % Self::nr() == 0);
        debug_assert!(buf.as_ptr() as usize % Self::alignment_bytes() == 0);
        unsafe { arm64simd_ln_f32_16n_run(buf) }
    },
    func(Ln)
);

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn arm64simd_ln_f32_16n_run(buf: &mut [f32]) {
    use crate::generic::ln::{LN2_HI, LN2_LO, POLY, SPLIT, SUBNORMAL_SCALE, SUBNORMAL_SHIFT};
    use std::arch::aarch64::*;
    #[inline(always)]
    unsafe fn ln4(x: float32x4_t) -> float32x4_t {
        unsafe {
            let one = vdupq_n_f32(1.0);
            let zero = vdupq_n_f32(0.0);
            let subnormal = vcltq_f32(x, vdupq_n_f32(f32::MIN_POSITIVE));
            let scaled = vbslq_f32(subnormal, vmulq_f32(x, vdupq_n_f32(SUBNORMAL_SCALE)), x);
            let bits = vreinterpretq_u32_f32(scaled);
            let mut exponent =
                vsubq_s32(vreinterpretq_s32_u32(vshrq_n_u32::<23>(bits)), vdupq_n_s32(127));
            exponent = vsubq_s32(
                exponent,
                vandq_s32(vreinterpretq_s32_u32(subnormal), vdupq_n_s32(SUBNORMAL_SHIFT)),
            );
            let mantissa = vreinterpretq_f32_u32(vorrq_u32(
                vandq_u32(bits, vdupq_n_u32(0x007fffff)),
                vdupq_n_u32(0x3f800000),
            ));
            let split = vcgtq_f32(mantissa, vdupq_n_f32(SPLIT));
            let mantissa = vbslq_f32(split, vmulq_f32(mantissa, vdupq_n_f32(0.5)), mantissa);
            exponent = vaddq_s32(exponent, vandq_s32(vreinterpretq_s32_u32(split), vdupq_n_s32(1)));
            let e = vcvtq_f32_s32(exponent);
            let f = vsubq_f32(mantissa, one);
            let mut p = vdupq_n_f32(POLY[0]);
            for c in &POLY[1..] {
                p = vfmaq_f32(vdupq_n_f32(*c), p, f);
            }
            let f2 = vmulq_f32(f, f);
            let mut y = vmulq_f32(vmulq_f32(p, f2), f);
            y = vfmaq_f32(y, e, vdupq_n_f32(LN2_LO));
            y = vaddq_f32(vfmsq_f32(y, vdupq_n_f32(0.5), f2), f);
            y = vfmaq_f32(y, e, vdupq_n_f32(LN2_HI));
            // A NaN compares false whichever way round, so `not greater than zero` is what
            // gathers the negatives, the zeros and the NaNs in one mask.
            let outside = vmvnq_u32(vcgtq_f32(x, zero));
            let special = vbslq_f32(
                vceqq_f32(x, zero),
                vdupq_n_f32(f32::NEG_INFINITY),
                vdupq_n_f32(f32::NAN),
            );
            y = vbslq_f32(outside, special, y);
            vbslq_f32(vceqq_f32(x, vdupq_n_f32(f32::INFINITY)), vdupq_n_f32(f32::INFINITY), y)
        }
    }
    unsafe {
        let p = buf.as_mut_ptr();
        for i in (0..buf.len()).step_by(4) {
            vst1q_f32(p.add(i), ln4(vld1q_f32(p.add(i))));
        }
    }
}
