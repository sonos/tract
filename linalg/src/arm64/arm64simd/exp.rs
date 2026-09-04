//! NEON f32 exp: the [`crate::generic::exp`] reduction and fit, four lanes at a time.

routine_ew_rust!(aarch64;
    f32,
    arm64simd_exp_f32_16n,
    16,
    4,
    #[inline(never)]
    fn run(buf: &mut [f32], _: ()) {
        debug_assert!(buf.len() % Self::nr() == 0);
        debug_assert!(buf.as_ptr() as usize % Self::alignment_bytes() == 0);
        unsafe { arm64simd_exp_f32_16n_run(buf) }
    },
    func(Exp)
);

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "neon")]
unsafe fn arm64simd_exp_f32_16n_run(buf: &mut [f32]) {
    use crate::generic::exp::{HIGH, LN2_HI, LN2_LO, LOG2E, LOW, POLY, SCALE_BIAS};
    use std::arch::aarch64::*;
    #[inline(always)]
    unsafe fn exp4(x: float32x4_t) -> float32x4_t {
        unsafe {
            // Clamped by selection rather than by fmin and fmax: those answer with the
            // bound where a lane is NaN, which is what must propagate instead.
            let high = vdupq_n_f32(HIGH);
            let low = vdupq_n_f32(LOW);
            let x = vbslq_f32(vcgtq_f32(x, high), high, x);
            let x = vbslq_f32(vcltq_f32(x, low), low, x);
            let kf = vrndnq_f32(vmulq_f32(x, vdupq_n_f32(LOG2E)));
            let mut r = vfmsq_f32(x, kf, vdupq_n_f32(LN2_HI));
            r = vfmsq_f32(r, kf, vdupq_n_f32(LN2_LO));
            let mut q = vdupq_n_f32(POLY[0]);
            for c in &POLY[1..] {
                q = vfmaq_f32(vdupq_n_f32(*c), q, r);
            }
            let k = vcvtq_s32_f32(kf);
            let half = vshrq_n_s32::<1>(k);
            let rest = vsubq_s32(k, half);
            let scale =
                |k| vreinterpretq_f32_s32(vshlq_n_s32::<23>(vaddq_s32(k, vdupq_n_s32(SCALE_BIAS))));
            vmulq_f32(vmulq_f32(q, scale(half)), scale(rest))
        }
    }
    unsafe {
        let p = buf.as_mut_ptr();
        for i in (0..buf.len()).step_by(4) {
            vst1q_f32(p.add(i), exp4(vld1q_f32(p.add(i))));
        }
    }
}
