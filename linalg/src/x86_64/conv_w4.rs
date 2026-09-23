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

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;

    const SENTINEL: f32 = -1234.5;

    fn check(in_stride: isize, len: usize, channels: usize, kernel: usize) {
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
            conv_w4_f32(
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
                check(in_stride, len, 3, 3);
                check(in_stride, len, 1, 2);
            }
        }
    }
}
