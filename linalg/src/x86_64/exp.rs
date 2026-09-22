//! f32 exp kernels: the [`crate::generic::exp`] reduction and fit, eight or sixteen lanes
//! at a time.

routine_ew_rust!(x86_64;
    f32,
    x86_64_fma_exp_f32_32n,
    32,
    8,
    #[inline(never)]
    fn run(buf: &mut [f32], _: ()) {
        debug_assert!(buf.len() % Self::nr() == 0);
        debug_assert!(buf.as_ptr() as usize % Self::alignment_bytes() == 0);
        unsafe { x86_64_fma_exp_f32_32n_run(buf) }
    },
    func(Exp),
    isa(X86_64Avx2, X86_64Fma)
);

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn x86_64_fma_exp_f32_32n_run(buf: &mut [f32]) {
    use crate::generic::exp::{HIGH, LN2_HI, LN2_LO, LOG2E, LOW, POLY, SCALE_BIAS};
    use std::arch::x86_64::*;
    #[inline(always)]
    unsafe fn exp8(x: __m256) -> __m256 {
        unsafe {
            // Clamped by selection rather than by min and max: a NaN is neither, and both
            // instructions would answer with the bound instead of propagating it.
            let high = _mm256_set1_ps(HIGH);
            let low = _mm256_set1_ps(LOW);
            let x = _mm256_blendv_ps(x, high, _mm256_cmp_ps::<_CMP_GT_OQ>(x, high));
            let x = _mm256_blendv_ps(x, low, _mm256_cmp_ps::<_CMP_LT_OQ>(x, low));
            let k = _mm256_cvtps_epi32(_mm256_mul_ps(x, _mm256_set1_ps(LOG2E)));
            let kf = _mm256_cvtepi32_ps(k);
            let mut r = _mm256_fnmadd_ps(kf, _mm256_set1_ps(LN2_HI), x);
            r = _mm256_fnmadd_ps(kf, _mm256_set1_ps(LN2_LO), r);
            let mut q = _mm256_set1_ps(POLY[0]);
            for c in &POLY[1..] {
                q = _mm256_fmadd_ps(q, r, _mm256_set1_ps(*c));
            }
            let bias = _mm256_set1_epi32(SCALE_BIAS);
            let half = _mm256_srai_epi32::<1>(k);
            let rest = _mm256_sub_epi32(k, half);
            let scale = |k| _mm256_castsi256_ps(_mm256_slli_epi32::<23>(_mm256_add_epi32(k, bias)));
            _mm256_mul_ps(_mm256_mul_ps(q, scale(half)), scale(rest))
        }
    }
    unsafe {
        let p = buf.as_mut_ptr();
        for i in (0..buf.len()).step_by(8) {
            _mm256_store_ps(p.add(i), exp8(_mm256_load_ps(p.add(i))));
        }
    }
}

routine_ew_rust!(x86_64;
    f32,
    x86_64_avx512_exp_f32_64n,
    64,
    16,
    #[inline(never)]
    fn run(buf: &mut [f32], _: ()) {
        debug_assert!(buf.len() % Self::nr() == 0);
        debug_assert!(buf.as_ptr() as usize % Self::alignment_bytes() == 0);
        unsafe { x86_64_avx512_exp_f32_64n_run(buf) }
    },
    func(Exp),
    isa(X86_64Avx512f)
);

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn x86_64_avx512_exp_f32_64n_run(buf: &mut [f32]) {
    use crate::generic::exp::{HIGH, LN2_HI, LN2_LO, LOG2E, LOW, POLY, SCALE_BIAS};
    use std::arch::x86_64::*;
    #[inline(always)]
    unsafe fn exp16(x: __m512) -> __m512 {
        unsafe {
            let high = _mm512_set1_ps(HIGH);
            let low = _mm512_set1_ps(LOW);
            let x = _mm512_mask_blend_ps(_mm512_cmp_ps_mask::<_CMP_GT_OQ>(x, high), x, high);
            let x = _mm512_mask_blend_ps(_mm512_cmp_ps_mask::<_CMP_LT_OQ>(x, low), x, low);
            let k = _mm512_cvtps_epi32(_mm512_mul_ps(x, _mm512_set1_ps(LOG2E)));
            let kf = _mm512_cvtepi32_ps(k);
            let mut r = _mm512_fnmadd_ps(kf, _mm512_set1_ps(LN2_HI), x);
            r = _mm512_fnmadd_ps(kf, _mm512_set1_ps(LN2_LO), r);
            let mut q = _mm512_set1_ps(POLY[0]);
            for c in &POLY[1..] {
                q = _mm512_fmadd_ps(q, r, _mm512_set1_ps(*c));
            }
            let bias = _mm512_set1_epi32(SCALE_BIAS);
            let half = _mm512_srai_epi32::<1>(k);
            let rest = _mm512_sub_epi32(k, half);
            let scale = |k| _mm512_castsi512_ps(_mm512_slli_epi32::<23>(_mm512_add_epi32(k, bias)));
            _mm512_mul_ps(_mm512_mul_ps(q, scale(half)), scale(rest))
        }
    }
    unsafe {
        let p = buf.as_mut_ptr();
        for i in (0..buf.len()).step_by(16) {
            _mm512_store_ps(p.add(i), exp16(_mm512_load_ps(p.add(i))));
        }
    }
}
