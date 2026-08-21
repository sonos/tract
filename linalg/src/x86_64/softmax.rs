// Accurate f32 softmax_l2: Cody-Waite argument reduction and a degree-5 minimax
// e^r fit, identical arithmetic to the aarch64 NEON body in generic/reduce.rs
// so every SIMD target agrees. `map_neutral` is NEG_INFINITY so a padded lane
// underflows to zero even on a fully masked row, where the row max equals the
// neutral.
// nr=32 (4x ymm of 8), 32-byte aligned.
map_reduce_impl_wrap!(
    f32,
    x86_64_fma_softmax2_f32_32n,
    32,
    8,
    f32,
    f32::NEG_INFINITY,
    0f32,
    #[inline(never)]
    fn run(buf: &mut [f32], max: f32) -> f32 {
        assert!(buf.len() % 32 == 0);
        unsafe { x86_64_fma_softmax2_f32_32n_run(buf, max) }
    },
    #[inline(never)]
    fn reduce_two(a: f32, b: f32) -> f32 {
        a + b
    }
);

#[target_feature(enable = "avx2,fma")]
unsafe fn x86_64_fma_softmax2_f32_32n_run(buf: &mut [f32], max: f32) -> f32 {
    use std::arch::x86_64::*;
    // exp(x) for x = row_value - max, i.e. never positive off the padding lanes.
    #[inline(always)]
    unsafe fn exp8(x: __m256) -> __m256 {
        unsafe {
            let k = _mm256_cvtps_epi32(_mm256_mul_ps(x, _mm256_set1_ps(1.442_695_04)));
            let kf = _mm256_cvtepi32_ps(k);
            let mut rr = _mm256_fnmadd_ps(kf, _mm256_set1_ps(0.693_145_75), x);
            rr = _mm256_fnmadd_ps(kf, _mm256_set1_ps(1.428_606_8e-6), rr);
            let mut q = _mm256_set1_ps(8.297653546e-03);
            q = _mm256_fmadd_ps(q, rr, _mm256_set1_ps(4.191538191e-02));
            q = _mm256_fmadd_ps(q, rr, _mm256_set1_ps(1.666757475e-01));
            q = _mm256_fmadd_ps(q, rr, _mm256_set1_ps(4.999889485e-01));
            q = _mm256_fmadd_ps(q, rr, _mm256_set1_ps(9.999996920e-01));
            q = _mm256_fmadd_ps(q, rr, _mm256_set1_ps(1.000000072e+00));
            // Biased exponent must stay a valid field: k + 127 goes non-positive
            // around x = -88, and shifting that in would build -inf.
            let biased = _mm256_max_epi32(
                _mm256_min_epi32(
                    _mm256_add_epi32(k, _mm256_set1_epi32(127)),
                    _mm256_set1_epi32(254),
                ),
                _mm256_set1_epi32(1),
            );
            let scale = _mm256_castsi256_ps(_mm256_slli_epi32::<23>(biased));
            let out = _mm256_or_ps(
                _mm256_cmp_ps::<_CMP_LT_OQ>(x, _mm256_set1_ps(-103.0)),
                _mm256_cmp_ps::<_CMP_GT_OQ>(x, _mm256_set1_ps(0.0)),
            );
            _mm256_blendv_ps(_mm256_mul_ps(q, scale), _mm256_setzero_ps(), out)
        }
    }
    unsafe {
        let vm = _mm256_set1_ps(max);
        let mut a0 = _mm256_setzero_ps();
        let mut a1 = _mm256_setzero_ps();
        let mut a2 = _mm256_setzero_ps();
        let mut a3 = _mm256_setzero_ps();
        let p = buf.as_mut_ptr();
        let n = buf.len();
        let mut i = 0;
        while i + 32 <= n {
            let y0 = exp8(_mm256_sub_ps(_mm256_load_ps(p.add(i)), vm));
            let y1 = exp8(_mm256_sub_ps(_mm256_load_ps(p.add(i + 8)), vm));
            let y2 = exp8(_mm256_sub_ps(_mm256_load_ps(p.add(i + 16)), vm));
            let y3 = exp8(_mm256_sub_ps(_mm256_load_ps(p.add(i + 24)), vm));
            _mm256_store_ps(p.add(i), y0);
            _mm256_store_ps(p.add(i + 8), y1);
            _mm256_store_ps(p.add(i + 16), y2);
            _mm256_store_ps(p.add(i + 24), y3);
            a0 = _mm256_add_ps(a0, y0);
            a1 = _mm256_add_ps(a1, y1);
            a2 = _mm256_add_ps(a2, y2);
            a3 = _mm256_add_ps(a3, y3);
            i += 32;
        }
        let acc = _mm256_add_ps(_mm256_add_ps(a0, a1), _mm256_add_ps(a2, a3));
        let mut s = _mm_add_ps(_mm256_castps256_ps128(acc), _mm256_extractf128_ps::<1>(acc));
        s = _mm_add_ps(s, _mm_movehl_ps(s, s));
        s = _mm_add_ss(s, _mm_shuffle_ps::<1>(s, s));
        _mm_cvtss_f32(s)
    }
}

#[cfg(test)]
mod test_x86_64_fma_softmax2_f32_32n {
    use super::*;
    crate::softmax_l2_frame_tests!(
        is_x86_feature_detected!("fma"),
        f32,
        x86_64_fma_softmax2_f32_32n
    );
}

// AVX-512 accurate f32 softmax_l2: same arithmetic as the FMA kernel, 64 f32
// (4x zmm of 16) per iteration. Runtime-gated on avx512f in plug_avx512f.
// nr=64, 64-byte aligned.
map_reduce_impl_wrap!(
    f32,
    x86_64_avx512_softmax2_f32_64n,
    64,
    16,
    f32,
    f32::NEG_INFINITY,
    0f32,
    #[inline(never)]
    fn run(buf: &mut [f32], max: f32) -> f32 {
        assert!(buf.len() % 64 == 0);
        unsafe { x86_64_avx512_softmax2_f32_64n_run(buf, max) }
    },
    #[inline(never)]
    fn reduce_two(a: f32, b: f32) -> f32 {
        a + b
    }
);

#[target_feature(enable = "avx512f")]
unsafe fn x86_64_avx512_softmax2_f32_64n_run(buf: &mut [f32], max: f32) -> f32 {
    use std::arch::x86_64::*;
    #[inline(always)]
    unsafe fn exp16(x: __m512) -> __m512 {
        unsafe {
            let k = _mm512_cvtps_epi32(_mm512_mul_ps(x, _mm512_set1_ps(1.442_695_04)));
            let kf = _mm512_cvtepi32_ps(k);
            let mut rr = _mm512_fnmadd_ps(kf, _mm512_set1_ps(0.693_145_75), x);
            rr = _mm512_fnmadd_ps(kf, _mm512_set1_ps(1.428_606_8e-6), rr);
            let mut q = _mm512_set1_ps(8.297653546e-03);
            q = _mm512_fmadd_ps(q, rr, _mm512_set1_ps(4.191538191e-02));
            q = _mm512_fmadd_ps(q, rr, _mm512_set1_ps(1.666757475e-01));
            q = _mm512_fmadd_ps(q, rr, _mm512_set1_ps(4.999889485e-01));
            q = _mm512_fmadd_ps(q, rr, _mm512_set1_ps(9.999996920e-01));
            q = _mm512_fmadd_ps(q, rr, _mm512_set1_ps(1.000000072e+00));
            let biased = _mm512_max_epi32(
                _mm512_min_epi32(
                    _mm512_add_epi32(k, _mm512_set1_epi32(127)),
                    _mm512_set1_epi32(254),
                ),
                _mm512_set1_epi32(1),
            );
            let scale = _mm512_castsi512_ps(_mm512_slli_epi32::<23>(biased));
            let out = _mm512_cmp_ps_mask::<_CMP_LT_OQ>(x, _mm512_set1_ps(-103.0))
                | _mm512_cmp_ps_mask::<_CMP_GT_OQ>(x, _mm512_set1_ps(0.0));
            _mm512_mask_blend_ps(out, _mm512_mul_ps(q, scale), _mm512_setzero_ps())
        }
    }
    unsafe {
        let vm = _mm512_set1_ps(max);
        let mut a0 = _mm512_setzero_ps();
        let mut a1 = _mm512_setzero_ps();
        let mut a2 = _mm512_setzero_ps();
        let mut a3 = _mm512_setzero_ps();
        let p = buf.as_mut_ptr();
        let n = buf.len();
        let mut i = 0;
        while i + 64 <= n {
            let y0 = exp16(_mm512_sub_ps(_mm512_load_ps(p.add(i)), vm));
            let y1 = exp16(_mm512_sub_ps(_mm512_load_ps(p.add(i + 16)), vm));
            let y2 = exp16(_mm512_sub_ps(_mm512_load_ps(p.add(i + 32)), vm));
            let y3 = exp16(_mm512_sub_ps(_mm512_load_ps(p.add(i + 48)), vm));
            _mm512_store_ps(p.add(i), y0);
            _mm512_store_ps(p.add(i + 16), y1);
            _mm512_store_ps(p.add(i + 32), y2);
            _mm512_store_ps(p.add(i + 48), y3);
            a0 = _mm512_add_ps(a0, y0);
            a1 = _mm512_add_ps(a1, y1);
            a2 = _mm512_add_ps(a2, y2);
            a3 = _mm512_add_ps(a3, y3);
            i += 64;
        }
        _mm512_reduce_add_ps(_mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(a2, a3)))
    }
}

#[cfg(test)]
mod test_x86_64_avx512_softmax2_f32_64n {
    use super::*;
    crate::softmax_l2_frame_tests!(
        is_x86_feature_detected!("avx512f"),
        f32,
        x86_64_avx512_softmax2_f32_64n
    );
}
