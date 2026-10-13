/// Depthwise convolution along an axis the output is contiguous on: `len` output points, each
/// the bias plus every `taps[t] * input[offsets[t] + i * in_stride]`.
///
/// Same contract as the aarch64 and wasm kernels. The arithmetic is the scalar walk's, in the
/// same order: the bias, then one multiply and one add per tap. It uses no FMA, so every output
/// is bit-identical to the scalar path and a plan gives the same results with or without this
/// kernel. Input strides 1 and 2 are vectorised at any tap count; other strides stay scalar.
///
/// # Safety
/// `input.offset(offsets[t] + i * in_stride)` must be readable for every tap and every `i` below
/// `len`, and `len` output points writable from `output`. The CPU must support AVX.
#[cfg(target_arch = "x86_64")]
pub unsafe fn depthwise_w_f32(
    input: *const f32,
    output: *mut f32,
    taps: &[f32],
    offsets: &[isize],
    bias: f32,
    len: usize,
    in_stride: isize,
) {
    unsafe {
        match in_stride {
            1 => vectorized::<1>(input, output, taps, offsets, bias, len),
            2 => vectorized::<2>(input, output, taps, offsets, bias, len),
            _ => scalar(input, output, taps, offsets, bias, 0, len, in_stride),
        }
    }
}

/// Loads eight points at input stride 1 or 2. Stride 2 reads one unused element after
/// its last point, so its caller must retain a following point inside the input run.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn load8<const STRIDE: isize>(p: *const f32) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;
    unsafe {
        if STRIDE == 1 {
            _mm256_loadu_ps(p)
        } else {
            let a = _mm256_loadu_ps(p);
            let b = _mm256_loadu_ps(p.add(8));
            let lo = _mm256_permute2f128_ps(a, b, 0x20);
            let hi = _mm256_permute2f128_ps(a, b, 0x31);
            _mm256_shuffle_ps(lo, hi, 0x88)
        }
    }
}

/// Input strides 1 and 2, with the tap count taken at runtime.
///
/// Taps are the outer loop and 32 output points the inner one: four 8-lane accumulators give
/// the adder independent chains, and each tap's broadcast is paid once per 32 points. Real 3x3,
/// 7x7 or 9x9 kernels reach here with 9, 49 or 81 taps.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx")]
unsafe fn vectorized<const STRIDE: isize>(
    input: *const f32,
    output: *mut f32,
    taps: &[f32],
    offsets: &[isize],
    bias: f32,
    len: usize,
) {
    use std::arch::x86_64::*;
    unsafe {
        let biasv = _mm256_set1_ps(bias);
        let in_stride = STRIDE;
        let extra = usize::from(STRIDE == 2);
        let mut i = 0usize;
        while i + 32 + extra <= len {
            let mut a0 = biasv;
            let mut a1 = biasv;
            let mut a2 = biasv;
            let mut a3 = biasv;
            for (tap, offset) in taps.iter().zip(offsets) {
                let k = _mm256_set1_ps(*tap);
                let p = input.offset(*offset + i as isize * in_stride);
                a0 = _mm256_add_ps(a0, _mm256_mul_ps(load8::<STRIDE>(p), k));
                a1 = _mm256_add_ps(a1, _mm256_mul_ps(load8::<STRIDE>(p.offset(8 * in_stride)), k));
                a2 = _mm256_add_ps(a2, _mm256_mul_ps(load8::<STRIDE>(p.offset(16 * in_stride)), k));
                a3 = _mm256_add_ps(a3, _mm256_mul_ps(load8::<STRIDE>(p.offset(24 * in_stride)), k));
            }
            _mm256_storeu_ps(output.add(i), a0);
            _mm256_storeu_ps(output.add(i + 8), a1);
            _mm256_storeu_ps(output.add(i + 16), a2);
            _mm256_storeu_ps(output.add(i + 24), a3);
            i += 32;
        }
        while i + 8 + extra <= len {
            let mut acc = biasv;
            for (tap, offset) in taps.iter().zip(offsets) {
                let x = load8::<STRIDE>(input.offset(*offset + i as isize * in_stride));
                acc = _mm256_add_ps(acc, _mm256_mul_ps(x, _mm256_set1_ps(*tap)));
            }
            _mm256_storeu_ps(output.add(i), acc);
            i += 8;
        }
        scalar(input, output, taps, offsets, bias, i, len, in_stride);
    }
}

#[cfg(target_arch = "x86_64")]
#[allow(clippy::too_many_arguments)]
unsafe fn scalar(
    input: *const f32,
    output: *mut f32,
    taps: &[f32],
    offsets: &[isize],
    bias: f32,
    from: usize,
    len: usize,
    in_stride: isize,
) {
    unsafe {
        for i in from..len {
            let mut sum = bias;
            for (tap, offset) in taps.iter().zip(offsets) {
                sum += tap * *input.offset(*offset + i as isize * in_stride);
            }
            *output.add(i) = sum;
        }
    }
}

bail_stub!(x86_64; pub unsafe fn depthwise_w_f32(
    *const f32, *mut f32, &[f32], &[isize], f32, usize, isize
));

submit_routine!(x86_64; DepthwiseWF32, DepthwiseW, "x86_64_avx_depthwise_w_f32", depthwise_w_f32,
    isa(X86_64Avx));

#[cfg(all(test, target_arch = "x86_64"))]
mod tests {
    use super::*;

    fn reference(
        input: &[f32],
        taps: &[f32],
        offsets: &[isize],
        bias: f32,
        len: usize,
        in_stride: isize,
        center: usize,
    ) -> Vec<f32> {
        (0..len)
            .map(|i| {
                taps.iter().zip(offsets).fold(bias, |sum, (tap, offset)| {
                    sum + tap * input[(center as isize + offset + i as isize * in_stride) as usize]
                })
            })
            .collect()
    }

    fn compare(taps: usize, len: usize, in_stride: isize) {
        if !std::is_x86_feature_detected!("avx") {
            return;
        }
        let k: Vec<f32> = (0..taps).map(|t| (t as f32 * 0.7).sin()).collect();
        let offsets: Vec<isize> = (0..taps as isize).map(|t| t - taps as isize / 2).collect();
        let center = taps;
        let extent = if len == 0 {
            center * 2
        } else {
            (center as isize + offsets.last().unwrap() + (len - 1) as isize * in_stride + 1)
                as usize
        };
        let input: Vec<f32> = (0..extent).map(|i| (i as f32 * 0.37).cos()).collect();
        let expected = reference(&input, &k, &offsets, 0.25, len, in_stride, center);
        let mut output = vec![0f32; len];
        unsafe {
            depthwise_w_f32(
                input.as_ptr().add(center),
                output.as_mut_ptr(),
                &k,
                &offsets,
                0.25,
                len,
                in_stride,
            )
        };
        // Same operations in the same order: the results must match bit for bit.
        let bits = |v: &[f32]| v.iter().map(|x| x.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&output), bits(&expected), "taps {taps} len {len} stride {in_stride}");
    }

    #[test]
    fn matches_scalar_bit_for_bit() {
        for taps in [1, 2, 3, 4, 5, 9, 25, 49, 81] {
            for len in [0, 1, 7, 8, 9, 15, 16, 17, 31, 32, 33, 64, 100, 224] {
                compare(taps, len, 1);
                compare(taps, len, 2);
                compare(taps, len, 3);
            }
        }
    }
}
