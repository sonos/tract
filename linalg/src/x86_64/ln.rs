//! f32 ln kernels: the [`crate::generic::ln`] fit, eight or sixteen lanes at a time.

routine_ew_rust!(x86_64;
    f32,
    x86_64_fma_ln_f32_32n,
    32,
    8,
    #[inline(never)]
    fn run(buf: &mut [f32], _: ()) {
        debug_assert!(buf.len() % Self::nr() == 0);
        debug_assert!(buf.as_ptr() as usize % Self::alignment_bytes() == 0);
        unsafe { x86_64_fma_ln_f32_32n_run(buf) }
    },
    func(Ln),
    isa(X86_64Avx2, X86_64Fma)
);

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn x86_64_fma_ln_f32_32n_run(buf: &mut [f32]) {
    use crate::generic::ln::{LN2_HI, LN2_LO, POLY, SPLIT, SUBNORMAL_SCALE, SUBNORMAL_SHIFT};
    use std::arch::x86_64::*;
    #[inline(always)]
    unsafe fn ln8(x: __m256) -> __m256 {
        unsafe {
            let one = _mm256_set1_ps(1.0);
            let zero = _mm256_setzero_ps();
            let subnormal = _mm256_cmp_ps::<_CMP_LT_OQ>(x, _mm256_set1_ps(f32::MIN_POSITIVE));
            let scaled =
                _mm256_blendv_ps(x, _mm256_mul_ps(x, _mm256_set1_ps(SUBNORMAL_SCALE)), subnormal);
            let bits = _mm256_castps_si256(scaled);
            let mut exponent =
                _mm256_sub_epi32(_mm256_srli_epi32::<23>(bits), _mm256_set1_epi32(127));
            exponent = _mm256_add_epi32(
                exponent,
                _mm256_and_si256(
                    _mm256_castps_si256(subnormal),
                    _mm256_set1_epi32(-SUBNORMAL_SHIFT),
                ),
            );
            let mantissa = _mm256_or_ps(
                _mm256_and_ps(scaled, _mm256_castsi256_ps(_mm256_set1_epi32(0x007fffff))),
                one,
            );
            let split = _mm256_cmp_ps::<_CMP_GT_OQ>(mantissa, _mm256_set1_ps(SPLIT));
            let mantissa =
                _mm256_blendv_ps(mantissa, _mm256_mul_ps(mantissa, _mm256_set1_ps(0.5)), split);
            exponent = _mm256_add_epi32(
                exponent,
                _mm256_and_si256(_mm256_castps_si256(split), _mm256_set1_epi32(1)),
            );
            let e = _mm256_cvtepi32_ps(exponent);
            let f = _mm256_sub_ps(mantissa, one);
            let mut p = _mm256_set1_ps(POLY[0]);
            for c in &POLY[1..] {
                p = _mm256_fmadd_ps(p, f, _mm256_set1_ps(*c));
            }
            let f2 = _mm256_mul_ps(f, f);
            let mut y = _mm256_mul_ps(_mm256_mul_ps(p, f2), f);
            y = _mm256_fmadd_ps(e, _mm256_set1_ps(LN2_LO), y);
            y = _mm256_add_ps(_mm256_fnmadd_ps(_mm256_set1_ps(0.5), f2, y), f);
            y = _mm256_fmadd_ps(e, _mm256_set1_ps(LN2_HI), y);
            // A NaN compares false whichever way round, so `not greater than zero` is what
            // gathers the negatives, the zeros and the NaNs in one mask.
            let outside = _mm256_cmp_ps::<_CMP_NGT_UQ>(x, zero);
            let special = _mm256_blendv_ps(
                _mm256_set1_ps(f32::NAN),
                _mm256_set1_ps(f32::NEG_INFINITY),
                _mm256_cmp_ps::<_CMP_EQ_OQ>(x, zero),
            );
            y = _mm256_blendv_ps(y, special, outside);
            _mm256_blendv_ps(
                y,
                _mm256_set1_ps(f32::INFINITY),
                _mm256_cmp_ps::<_CMP_EQ_OQ>(x, _mm256_set1_ps(f32::INFINITY)),
            )
        }
    }
    unsafe {
        let p = buf.as_mut_ptr();
        for i in (0..buf.len()).step_by(8) {
            _mm256_store_ps(p.add(i), ln8(_mm256_load_ps(p.add(i))));
        }
    }
}

routine_ew_rust!(x86_64;
    f32,
    x86_64_avx512_ln_f32_64n,
    64,
    16,
    #[inline(never)]
    fn run(buf: &mut [f32], _: ()) {
        debug_assert!(buf.len() % Self::nr() == 0);
        debug_assert!(buf.as_ptr() as usize % Self::alignment_bytes() == 0);
        unsafe { x86_64_avx512_ln_f32_64n_run(buf) }
    },
    func(Ln),
    isa(X86_64Avx512f)
);

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn x86_64_avx512_ln_f32_64n_run(buf: &mut [f32]) {
    use crate::generic::ln::{LN2_HI, LN2_LO, POLY, SPLIT, SUBNORMAL_SCALE, SUBNORMAL_SHIFT};
    use std::arch::x86_64::*;
    #[inline(always)]
    unsafe fn ln16(x: __m512) -> __m512 {
        unsafe {
            let one = _mm512_set1_ps(1.0);
            let zero = _mm512_setzero_ps();
            let subnormal = _mm512_cmp_ps_mask::<_CMP_LT_OQ>(x, _mm512_set1_ps(f32::MIN_POSITIVE));
            let scaled = _mm512_mask_mul_ps(x, subnormal, x, _mm512_set1_ps(SUBNORMAL_SCALE));
            let bits = _mm512_castps_si512(scaled);
            let mut exponent =
                _mm512_sub_epi32(_mm512_srli_epi32::<23>(bits), _mm512_set1_epi32(127));
            exponent = _mm512_mask_sub_epi32(
                exponent,
                subnormal,
                exponent,
                _mm512_set1_epi32(SUBNORMAL_SHIFT),
            );
            let mantissa = _mm512_or_ps(
                _mm512_and_ps(scaled, _mm512_castsi512_ps(_mm512_set1_epi32(0x007fffff))),
                one,
            );
            let split = _mm512_cmp_ps_mask::<_CMP_GT_OQ>(mantissa, _mm512_set1_ps(SPLIT));
            let mantissa = _mm512_mask_mul_ps(mantissa, split, mantissa, _mm512_set1_ps(0.5));
            exponent = _mm512_mask_add_epi32(exponent, split, exponent, _mm512_set1_epi32(1));
            let e = _mm512_cvtepi32_ps(exponent);
            let f = _mm512_sub_ps(mantissa, one);
            let mut p = _mm512_set1_ps(POLY[0]);
            for c in &POLY[1..] {
                p = _mm512_fmadd_ps(p, f, _mm512_set1_ps(*c));
            }
            let f2 = _mm512_mul_ps(f, f);
            let mut y = _mm512_mul_ps(_mm512_mul_ps(p, f2), f);
            y = _mm512_fmadd_ps(e, _mm512_set1_ps(LN2_LO), y);
            y = _mm512_add_ps(_mm512_fnmadd_ps(_mm512_set1_ps(0.5), f2, y), f);
            y = _mm512_fmadd_ps(e, _mm512_set1_ps(LN2_HI), y);
            let outside = _mm512_cmp_ps_mask::<_CMP_NGT_UQ>(x, zero);
            let special = _mm512_mask_blend_ps(
                _mm512_cmp_ps_mask::<_CMP_EQ_OQ>(x, zero),
                _mm512_set1_ps(f32::NAN),
                _mm512_set1_ps(f32::NEG_INFINITY),
            );
            y = _mm512_mask_blend_ps(outside, y, special);
            _mm512_mask_blend_ps(
                _mm512_cmp_ps_mask::<_CMP_EQ_OQ>(x, _mm512_set1_ps(f32::INFINITY)),
                y,
                _mm512_set1_ps(f32::INFINITY),
            )
        }
    }
    unsafe {
        let p = buf.as_mut_ptr();
        for i in (0..buf.len()).step_by(16) {
            _mm512_store_ps(p.add(i), ln16(_mm512_load_ps(p.add(i))));
        }
    }
}
