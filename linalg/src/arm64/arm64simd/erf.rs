// Abramowitz & Stegun 7.1.26, same coefficients as generic/erf.rs::serf.
// Vectorised 8-wide so GELU's Erf is not a scalar `powi(16)` per element.

routine_ew_rust!(aarch64;
    f32,
    arm64simd_erf_f32_8n,
    8,
    4,
    #[inline(never)]
    fn run(buf: &mut [f32], _: ()) {
        debug_assert_eq!(buf.len() % 8, 0);
        unsafe { erf_neon_8(buf) }
    },
    func(Erf)
);

unsafe fn erf_neon_8(buf: &mut [f32]) {
    unsafe {
        use std::arch::aarch64::*;
        const A1: f32 = 0.0705230784;
        const A2: f32 = 0.0422820123;
        const A3: f32 = 0.0092705272;
        const A4: f32 = 0.0001520143;
        const A5: f32 = 0.0002765672;
        const A6: f32 = 0.0000430638;
        let a1 = vdupq_n_f32(A1);
        let a2 = vdupq_n_f32(A2);
        let a3 = vdupq_n_f32(A3);
        let a4 = vdupq_n_f32(A4);
        let a5 = vdupq_n_f32(A5);
        let a6 = vdupq_n_f32(A6);
        let one = vdupq_n_f32(1.0);
        let two = vdupq_n_f32(2.0);
        let mut p = buf.as_mut_ptr();
        let end = p.add(buf.len());
        while p < end {
            let x0 = vld1q_f32(p);
            let x1 = vld1q_f32(p.add(4));
            let y0 = erf4(x0, a1, a2, a3, a4, a5, a6, one, two);
            let y1 = erf4(x1, a1, a2, a3, a4, a5, a6, one, two);
            vst1q_f32(p, y0);
            vst1q_f32(p.add(4), y1);
            p = p.add(8);
        }
    }
}

#[inline(always)]
unsafe fn erf4(
    x: std::arch::aarch64::float32x4_t,
    a1: std::arch::aarch64::float32x4_t,
    a2: std::arch::aarch64::float32x4_t,
    a3: std::arch::aarch64::float32x4_t,
    a4: std::arch::aarch64::float32x4_t,
    a5: std::arch::aarch64::float32x4_t,
    a6: std::arch::aarch64::float32x4_t,
    one: std::arch::aarch64::float32x4_t,
    two: std::arch::aarch64::float32x4_t,
) -> std::arch::aarch64::float32x4_t {
    unsafe {
        use std::arch::aarch64::*;
        let abs = vabsq_f32(x);
        let mut y = a6;
        y = vfmaq_f32(a5, y, abs);
        y = vfmaq_f32(a4, y, abs);
        y = vfmaq_f32(a3, y, abs);
        y = vfmaq_f32(a2, y, abs);
        y = vfmaq_f32(a1, y, abs);
        y = vmulq_f32(y, abs);
        let t = vaddq_f32(y, one);
        let mut p = vmulq_f32(t, t);
        p = vmulq_f32(p, p);
        p = vmulq_f32(p, p);
        p = vmulq_f32(p, p);
        let mut r = vrecpeq_f32(p);
        r = vmulq_f32(r, vmlsq_f32(two, p, r));
        r = vmulq_f32(r, vmlsq_f32(two, p, r));
        let mag = vsubq_f32(one, r);
        vbslq_f32(vcltq_f32(x, vdupq_n_f32(0.0)), vnegq_f32(mag), mag)
    }
}
