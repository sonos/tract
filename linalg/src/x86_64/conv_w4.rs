#[cfg(target_arch = "x86_64")]
use std::arch::x86_64::*;

/// The inputs of eight consecutive output points, `stride` elements apart. Stride 2 loads the
/// sixteen elements from `p`, one more than it keeps.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
unsafe fn load8(p: *const f32, stride: isize) -> __m256 {
    unsafe {
        match stride {
            1 => _mm256_loadu_ps(p),
            2 => {
                let a = _mm256_loadu_ps(p);
                let b = _mm256_loadu_ps(p.add(8));
                let lo = _mm256_permute2f128_ps(a, b, 0x20);
                let hi = _mm256_permute2f128_ps(a, b, 0x31);
                _mm256_shuffle_ps(lo, hi, 0x88)
            }
            _ => {
                let s = stride as i32;
                let idx = _mm256_setr_epi32(0, s, 2 * s, 3 * s, 4 * s, 5 * s, 6 * s, 7 * s);
                _mm256_i32gather_ps(p, idx, 4)
            }
        }
    }
}

/// Four channels of a convolution along W, as [`crate::routines::ConvW4F32`] describes, eight
/// output points to a vector. Input strides 1, 2 and 3 are vectorised; any other stride, and the
/// last few points of a run, are computed one point at a time. No load reaches past
/// `offsets[t] + (len - 1) * in_stride`.
///
/// # Safety
/// As [`crate::routines::ConvW4F32`], on a machine with AVX2 and FMA.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2,fma")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn conv_w4_f32(
    input: *const f32,
    output: *mut f32,
    taps: &[f32],
    offsets: &[isize],
    bias: &[f32; 4],
    len: usize,
    in_stride: isize,
    oc_stride: isize,
) {
    unsafe {
        let n = offsets.len();
        let mut i = 0usize;
        if (1..=3).contains(&in_stride) {
            let b0 = _mm256_set1_ps(bias[0]);
            let b1 = _mm256_set1_ps(bias[1]);
            let b2 = _mm256_set1_ps(bias[2]);
            let b3 = _mm256_set1_ps(bias[3]);
            // load8 reads one element past its last point at stride 2, which belongs to the
            // next output point, so that point has to be in the run too.
            let block = if in_stride == 2 { 9 } else { 8 };
            while i + 8 + block <= len {
                let (mut a0, mut a1, mut a2, mut a3) = (b0, b1, b2, b3);
                let (mut a4, mut a5, mut a6, mut a7) = (b0, b1, b2, b3);
                for t in 0..n {
                    let p = input.offset(offsets[t] + i as isize * in_stride);
                    let x = load8(p, in_stride);
                    let y = load8(p.offset(8 * in_stride), in_stride);
                    let k0 = _mm256_set1_ps(taps[t]);
                    let k1 = _mm256_set1_ps(taps[n + t]);
                    let k2 = _mm256_set1_ps(taps[2 * n + t]);
                    let k3 = _mm256_set1_ps(taps[3 * n + t]);
                    a0 = _mm256_fmadd_ps(x, k0, a0);
                    a1 = _mm256_fmadd_ps(x, k1, a1);
                    a2 = _mm256_fmadd_ps(x, k2, a2);
                    a3 = _mm256_fmadd_ps(x, k3, a3);
                    a4 = _mm256_fmadd_ps(y, k0, a4);
                    a5 = _mm256_fmadd_ps(y, k1, a5);
                    a6 = _mm256_fmadd_ps(y, k2, a6);
                    a7 = _mm256_fmadd_ps(y, k3, a7);
                }
                _mm256_storeu_ps(output.add(i), a0);
                _mm256_storeu_ps(output.offset(oc_stride).add(i), a1);
                _mm256_storeu_ps(output.offset(2 * oc_stride).add(i), a2);
                _mm256_storeu_ps(output.offset(3 * oc_stride).add(i), a3);
                _mm256_storeu_ps(output.add(i + 8), a4);
                _mm256_storeu_ps(output.offset(oc_stride).add(i + 8), a5);
                _mm256_storeu_ps(output.offset(2 * oc_stride).add(i + 8), a6);
                _mm256_storeu_ps(output.offset(3 * oc_stride).add(i + 8), a7);
                i += 16;
            }
            while i + block <= len {
                let (mut a0, mut a1, mut a2, mut a3) = (b0, b1, b2, b3);
                for t in 0..n {
                    let x = load8(input.offset(offsets[t] + i as isize * in_stride), in_stride);
                    a0 = _mm256_fmadd_ps(x, _mm256_set1_ps(taps[t]), a0);
                    a1 = _mm256_fmadd_ps(x, _mm256_set1_ps(taps[n + t]), a1);
                    a2 = _mm256_fmadd_ps(x, _mm256_set1_ps(taps[2 * n + t]), a2);
                    a3 = _mm256_fmadd_ps(x, _mm256_set1_ps(taps[3 * n + t]), a3);
                }
                _mm256_storeu_ps(output.add(i), a0);
                _mm256_storeu_ps(output.offset(oc_stride).add(i), a1);
                _mm256_storeu_ps(output.offset(2 * oc_stride).add(i), a2);
                _mm256_storeu_ps(output.offset(3 * oc_stride).add(i), a3);
                i += 8;
            }
        }
        scalar(input, output, taps, offsets, bias, i, len, in_stride, oc_stride);
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
unsafe fn scalar(
    input: *const f32,
    output: *mut f32,
    taps: &[f32],
    offsets: &[isize],
    bias: &[f32; 4],
    from: usize,
    len: usize,
    in_stride: isize,
    oc_stride: isize,
) {
    unsafe {
        let n = offsets.len();
        for i in from..len {
            for o in 0..4 {
                let mut sum = bias[o];
                for t in 0..n {
                    sum += taps[o * n + t] * *input.offset(offsets[t] + i as isize * in_stride);
                }
                *output.offset(o as isize * oc_stride).add(i) = sum;
            }
        }
    }
}

bail_stub!(x86_64; pub unsafe fn conv_w4_f32(
    *const f32, *mut f32, &[f32], &[isize], &[f32; 4], usize, isize, isize
));

submit_routine!(x86_64; ConvW4F32, ConvW4, "x86_64_fma_conv_w4_f32", conv_w4_f32,
    isa(X86_64Avx2, X86_64Fma));

/// Which elements of each of the up to three vectors [`load16`] reads are loaded for `points`
/// output points, from 1 to 16, so the last loaded element is the last point's own. Strides
/// above 3 are gathered, and the first mask then selects points rather than elements.
#[cfg(target_arch = "x86_64")]
fn load_masks(stride: isize, points: usize) -> [__mmask16; 3] {
    if stride > 3 {
        return [((1u32 << points) - 1) as __mmask16, 0, 0];
    }
    let needed = (points - 1) * stride as usize + 1;
    std::array::from_fn(|j| {
        let k = needed.saturating_sub(16 * j).min(16);
        ((1u32 << k) - 1) as __mmask16
    })
}

/// The inputs of sixteen consecutive output points, `stride` elements apart, zero for the points
/// `masks` (from [`load_masks`]) leaves out. Masked-off elements are never read.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn load16(p: *const f32, stride: isize, masks: &[__mmask16; 3]) -> __m512 {
    unsafe {
        match stride {
            1 => _mm512_maskz_loadu_ps(masks[0], p),
            2 => {
                let a = _mm512_maskz_loadu_ps(masks[0], p);
                let b = _mm512_maskz_loadu_ps(masks[1], p.wrapping_add(16));
                let even =
                    _mm512_setr_epi32(0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24, 26, 28, 30);
                _mm512_permutex2var_ps(a, even, b)
            }
            3 => {
                let a = _mm512_maskz_loadu_ps(masks[0], p);
                let b = _mm512_maskz_loadu_ps(masks[1], p.wrapping_add(16));
                let c = _mm512_maskz_loadu_ps(masks[2], p.wrapping_add(32));
                let from_ab =
                    _mm512_setr_epi32(0, 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 0, 0, 0, 0, 0);
                let then_c =
                    _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 17, 20, 23, 26, 29);
                _mm512_permutex2var_ps(_mm512_permutex2var_ps(a, from_ab, b), then_c, c)
            }
            _ => {
                let lanes = _mm512_setr_epi32(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15);
                let idx = _mm512_mullo_epi32(lanes, _mm512_set1_epi32(stride as i32));
                _mm512_mask_i32gather_ps::<4>(_mm512_setzero_ps(), masks[0], idx, p)
            }
        }
    }
}

/// Four channels of a convolution along W, as [`crate::routines::ConvW4F32`] describes, sixteen
/// output points to a vector. Every positive input stride is vectorised, 1 to 3 by loads and
/// shuffles and the others by gathers, and the last points of a run by masked loads and stores;
/// only a non-positive stride, or one too wide for a 32-bit gather index, runs one point at a
/// time. No load reaches past `offsets[t] + (len - 1) * in_stride`.
///
/// # Safety
/// As [`crate::routines::ConvW4F32`], on a machine with AVX-512F.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn conv_w4_f32_avx512(
    input: *const f32,
    output: *mut f32,
    taps: &[f32],
    offsets: &[isize],
    bias: &[f32; 4],
    len: usize,
    in_stride: isize,
    oc_stride: isize,
) {
    unsafe {
        if !(1..=i32::MAX as isize / 16).contains(&in_stride) {
            return scalar(input, output, taps, offsets, bias, 0, len, in_stride, oc_stride);
        }
        let n = offsets.len();
        let b0 = _mm512_set1_ps(bias[0]);
        let b1 = _mm512_set1_ps(bias[1]);
        let b2 = _mm512_set1_ps(bias[2]);
        let b3 = _mm512_set1_ps(bias[3]);
        let full = load_masks(in_stride, 16);
        let mut i = 0usize;
        while i + 32 <= len {
            let (mut a0, mut a1, mut a2, mut a3) = (b0, b1, b2, b3);
            let (mut a4, mut a5, mut a6, mut a7) = (b0, b1, b2, b3);
            for t in 0..n {
                let p = input.offset(offsets[t] + i as isize * in_stride);
                let x = load16(p, in_stride, &full);
                let y = load16(p.offset(16 * in_stride), in_stride, &full);
                let k0 = _mm512_set1_ps(taps[t]);
                let k1 = _mm512_set1_ps(taps[n + t]);
                let k2 = _mm512_set1_ps(taps[2 * n + t]);
                let k3 = _mm512_set1_ps(taps[3 * n + t]);
                a0 = _mm512_fmadd_ps(x, k0, a0);
                a1 = _mm512_fmadd_ps(x, k1, a1);
                a2 = _mm512_fmadd_ps(x, k2, a2);
                a3 = _mm512_fmadd_ps(x, k3, a3);
                a4 = _mm512_fmadd_ps(y, k0, a4);
                a5 = _mm512_fmadd_ps(y, k1, a5);
                a6 = _mm512_fmadd_ps(y, k2, a6);
                a7 = _mm512_fmadd_ps(y, k3, a7);
            }
            _mm512_storeu_ps(output.add(i), a0);
            _mm512_storeu_ps(output.offset(oc_stride).add(i), a1);
            _mm512_storeu_ps(output.offset(2 * oc_stride).add(i), a2);
            _mm512_storeu_ps(output.offset(3 * oc_stride).add(i), a3);
            _mm512_storeu_ps(output.add(i + 16), a4);
            _mm512_storeu_ps(output.offset(oc_stride).add(i + 16), a5);
            _mm512_storeu_ps(output.offset(2 * oc_stride).add(i + 16), a6);
            _mm512_storeu_ps(output.offset(3 * oc_stride).add(i + 16), a7);
            i += 32;
        }
        while i < len {
            let points = (len - i).min(16);
            let masks = load_masks(in_stride, points);
            let store = ((1u32 << points) - 1) as __mmask16;
            let (mut a0, mut a1, mut a2, mut a3) = (b0, b1, b2, b3);
            for t in 0..n {
                let x =
                    load16(input.offset(offsets[t] + i as isize * in_stride), in_stride, &masks);
                a0 = _mm512_fmadd_ps(x, _mm512_set1_ps(taps[t]), a0);
                a1 = _mm512_fmadd_ps(x, _mm512_set1_ps(taps[n + t]), a1);
                a2 = _mm512_fmadd_ps(x, _mm512_set1_ps(taps[2 * n + t]), a2);
                a3 = _mm512_fmadd_ps(x, _mm512_set1_ps(taps[3 * n + t]), a3);
            }
            _mm512_mask_storeu_ps(output.add(i), store, a0);
            _mm512_mask_storeu_ps(output.offset(oc_stride).add(i), store, a1);
            _mm512_mask_storeu_ps(output.offset(2 * oc_stride).add(i), store, a2);
            _mm512_mask_storeu_ps(output.offset(3 * oc_stride).add(i), store, a3);
            i += points;
        }
    }
}

bail_stub!(x86_64; pub unsafe fn conv_w4_f32_avx512(
    *const f32, *mut f32, &[f32], &[isize], &[f32; 4], usize, isize, isize
));

submit_routine!(x86_64; ConvW4F32, ConvW4, "x86_64_avx512_conv_w4_f32", conv_w4_f32_avx512,
    isa(X86_64Avx512f));

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;

    const SENTINEL: f32 = -1234.5;

    fn check(
        conv: crate::routines::ConvW4F32,
        in_stride: isize,
        len: usize,
        channels: usize,
        kernel: usize,
    ) {
        let row = 64 + len * in_stride as usize;
        let offsets: Vec<isize> =
            (0..channels).flat_map(|c| (0..kernel).map(move |k| (c * row + k) as isize)).collect();
        let n = offsets.len();
        let extent = offsets[n - 1] as usize + len.saturating_sub(1) * in_stride as usize + 1;
        let input: Vec<f32> = (0..extent).map(|i| ((i * 7919 % 211) as f32 - 105.) / 50.).collect();
        let taps: Vec<f32> = (0..4 * n).map(|t| ((t * 104729 % 97) as f32 - 48.) / 30.).collect();
        let bias = [0.5, -1.25, 2.0, 0.0];
        let oc_stride = len + 5;
        let mut output = vec![SENTINEL; 4 * oc_stride];
        unsafe {
            conv(
                input.as_ptr(),
                output.as_mut_ptr(),
                &taps,
                &offsets,
                &bias,
                len,
                in_stride,
                oc_stride as isize,
            )
        };
        for o in 0..4 {
            for i in 0..oc_stride {
                let got = output[o * oc_stride + i];
                if i >= len {
                    assert_eq!(got, SENTINEL, "wrote past the run: o={o} i={i}");
                    continue;
                }
                let want = (0..n).fold(bias[o], |sum, t| {
                    sum + taps[o * n + t] * input[(offsets[t] + (i as isize) * in_stride) as usize]
                });
                let tol = 1e-5 * (1. + want.abs());
                assert!(
                    (got - want).abs() <= tol,
                    "stride {in_stride} len {len} o={o} i={i}: got {got}, want {want}"
                );
            }
        }
    }

    #[test]
    fn matches_reference() {
        if !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma")) {
            return;
        }
        for in_stride in 1..=4 {
            for len in 0..40 {
                check(conv_w4_f32, in_stride, len, 3, 3);
                check(conv_w4_f32, in_stride, len, 1, 2);
            }
        }
    }

    #[test]
    fn avx512_matches_reference() {
        if !is_x86_feature_detected!("avx512f") {
            return;
        }
        for in_stride in 1..=5 {
            for len in 0..80 {
                check(conv_w4_f32_avx512, in_stride, len, 3, 3);
                check(conv_w4_f32_avx512, in_stride, len, 1, 2);
            }
        }
    }
}
