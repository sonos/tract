/// Four channels of a convolution along W, as [`crate::routines::ConvW4F32`] describes, four
/// output points to a vector and sixteen to a pass, so the sixteen accumulators cover the FMA
/// latency of a wide core. Input strides 1 to 4 are vectorised, stride 2 de-interleaved by `uzp1`
/// over two plain loads and strides 3 and 4 by `vld3q` and `vld4q`; any other stride, and the
/// last few points of a run, are computed one point at a time. No load reaches past
/// `offsets[t] + (len - 1) * in_stride`.
///
/// # Safety
/// As [`crate::routines::ConvW4F32`].
#[cfg(target_arch = "aarch64")]
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
        let from = match in_stride {
            1 => strided::<1>(input, output, taps, offsets, bias, len, oc_stride),
            2 => strided::<2>(input, output, taps, offsets, bias, len, oc_stride),
            3 => strided::<3>(input, output, taps, offsets, bias, len, oc_stride),
            4 => strided::<4>(input, output, taps, offsets, bias, len, oc_stride),
            _ => 0,
        };
        scalar(input, output, taps, offsets, bias, from, len, in_stride, oc_stride);
    }
}

/// The inputs of four consecutive output points, `S` elements apart. Reads `4 * S` elements
/// from `p`, `S - 1` more than it keeps.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn load4<const S: usize>(p: *const f32) -> std::arch::aarch64::float32x4_t {
    unsafe {
        use std::arch::aarch64::*;
        match S {
            1 => vld1q_f32(p),
            2 => vuzp1q_f32(vld1q_f32(p), vld1q_f32(p.add(4))),
            3 => vld3q_f32(p).0,
            _ => vld4q_f32(p).0,
        }
    }
}

/// The vectorised part of [`conv_w4_f32`] at input stride `S`, returning the first point it
/// left for the scalar loop.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
#[allow(clippy::too_many_arguments)]
unsafe fn strided<const S: usize>(
    input: *const f32,
    output: *mut f32,
    taps: &[f32],
    offsets: &[isize],
    bias: &[f32; 4],
    len: usize,
    oc_stride: isize,
) -> usize {
    unsafe {
        use std::arch::aarch64::*;
        let n = offsets.len();
        let b = [
            vdupq_n_f32(bias[0]),
            vdupq_n_f32(bias[1]),
            vdupq_n_f32(bias[2]),
            vdupq_n_f32(bias[3]),
        ];
        // load4 reads past its last point at strides above 1, into the next output point's
        // inputs, so that point has to be in the run too.
        let spare = (S > 1) as usize;
        let mut i = 0usize;
        while i + 16 + spare <= len {
            let mut acc = [b; 4];
            for t in 0..n {
                let p = input.offset(offsets[t]).add(i * S);
                let k = [
                    vdupq_n_f32(taps[t]),
                    vdupq_n_f32(taps[n + t]),
                    vdupq_n_f32(taps[2 * n + t]),
                    vdupq_n_f32(taps[3 * n + t]),
                ];
                for (v, row) in acc.iter_mut().enumerate() {
                    let x = load4::<S>(p.add(4 * v * S));
                    for (a, k) in row.iter_mut().zip(k) {
                        *a = vfmaq_f32(*a, x, k);
                    }
                }
            }
            for (v, row) in acc.iter().enumerate() {
                for (o, a) in row.iter().enumerate() {
                    vst1q_f32(output.offset(o as isize * oc_stride).add(i + 4 * v), *a);
                }
            }
            i += 16;
        }
        while i + 4 + spare <= len {
            let mut acc = b;
            for t in 0..n {
                let x = load4::<S>(input.offset(offsets[t]).add(i * S));
                for (o, a) in acc.iter_mut().enumerate() {
                    *a = vfmaq_f32(*a, x, vdupq_n_f32(taps[o * n + t]));
                }
            }
            for (o, a) in acc.iter().enumerate() {
                vst1q_f32(output.offset(o as isize * oc_stride).add(i), *a);
            }
            i += 4;
        }
        i
    }
}

#[cfg(target_arch = "aarch64")]
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

bail_stub!(aarch64; pub unsafe fn conv_w4_f32(
    *const f32, *mut f32, &[f32], &[isize], &[f32; 4], usize, isize, isize
));

submit_routine!(aarch64; ConvW4F32, ConvW4, "arm64simd_conv_w4_f32", conv_w4_f32);

#[cfg(all(test, target_arch = "aarch64"))]
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
        for in_stride in 1..=5 {
            for len in 0..80 {
                check(in_stride, len, 3, 3);
                check(in_stride, len, 1, 2);
            }
        }
    }
}
