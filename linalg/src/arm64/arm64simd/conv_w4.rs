/// Four channels of a convolution along W, as [`crate::routines::ConvW4F32`] describes, four
/// output points to a vector. Input strides 1 and 2 are vectorised, stride 2 de-interleaved by
/// `vld2q`; any other stride, and the last few points of a run, are computed one point at a time.
/// No load reaches past `offsets[t] + (len - 1) * in_stride`.
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
        use std::arch::aarch64::*;
        let n = offsets.len();
        let b0 = vdupq_n_f32(bias[0]);
        let b1 = vdupq_n_f32(bias[1]);
        let b2 = vdupq_n_f32(bias[2]);
        let b3 = vdupq_n_f32(bias[3]);
        let store4 = |i: usize, a0, a1, a2, a3| {
            vst1q_f32(output.add(i), a0);
            vst1q_f32(output.offset(oc_stride).add(i), a1);
            vst1q_f32(output.offset(2 * oc_stride).add(i), a2);
            vst1q_f32(output.offset(3 * oc_stride).add(i), a3);
        };
        let mut i = 0usize;
        if in_stride == 1 {
            while i + 8 <= len {
                let (mut a0, mut a1, mut a2, mut a3) = (b0, b1, b2, b3);
                let (mut a4, mut a5, mut a6, mut a7) = (b0, b1, b2, b3);
                for t in 0..n {
                    let p = input.offset(offsets[t]).add(i);
                    let x = vld1q_f32(p);
                    let y = vld1q_f32(p.add(4));
                    let k0 = vdupq_n_f32(taps[t]);
                    let k1 = vdupq_n_f32(taps[n + t]);
                    let k2 = vdupq_n_f32(taps[2 * n + t]);
                    let k3 = vdupq_n_f32(taps[3 * n + t]);
                    a0 = vfmaq_f32(a0, x, k0);
                    a1 = vfmaq_f32(a1, x, k1);
                    a2 = vfmaq_f32(a2, x, k2);
                    a3 = vfmaq_f32(a3, x, k3);
                    a4 = vfmaq_f32(a4, y, k0);
                    a5 = vfmaq_f32(a5, y, k1);
                    a6 = vfmaq_f32(a6, y, k2);
                    a7 = vfmaq_f32(a7, y, k3);
                }
                store4(i, a0, a1, a2, a3);
                store4(i + 4, a4, a5, a6, a7);
                i += 8;
            }
            while i + 4 <= len {
                let (mut a0, mut a1, mut a2, mut a3) = (b0, b1, b2, b3);
                for t in 0..n {
                    let x = vld1q_f32(input.offset(offsets[t]).add(i));
                    a0 = vfmaq_f32(a0, x, vdupq_n_f32(taps[t]));
                    a1 = vfmaq_f32(a1, x, vdupq_n_f32(taps[n + t]));
                    a2 = vfmaq_f32(a2, x, vdupq_n_f32(taps[2 * n + t]));
                    a3 = vfmaq_f32(a3, x, vdupq_n_f32(taps[3 * n + t]));
                }
                store4(i, a0, a1, a2, a3);
                i += 4;
            }
        } else if in_stride == 2 {
            // vld2q reads one element past its last point, which belongs to the next output
            // point, so that point has to be in the run too.
            while i + 9 <= len {
                let (mut a0, mut a1, mut a2, mut a3) = (b0, b1, b2, b3);
                let (mut a4, mut a5, mut a6, mut a7) = (b0, b1, b2, b3);
                for t in 0..n {
                    let p = input.offset(offsets[t]).offset(i as isize * 2);
                    let x = vld2q_f32(p).0;
                    let y = vld2q_f32(p.add(8)).0;
                    let k0 = vdupq_n_f32(taps[t]);
                    let k1 = vdupq_n_f32(taps[n + t]);
                    let k2 = vdupq_n_f32(taps[2 * n + t]);
                    let k3 = vdupq_n_f32(taps[3 * n + t]);
                    a0 = vfmaq_f32(a0, x, k0);
                    a1 = vfmaq_f32(a1, x, k1);
                    a2 = vfmaq_f32(a2, x, k2);
                    a3 = vfmaq_f32(a3, x, k3);
                    a4 = vfmaq_f32(a4, y, k0);
                    a5 = vfmaq_f32(a5, y, k1);
                    a6 = vfmaq_f32(a6, y, k2);
                    a7 = vfmaq_f32(a7, y, k3);
                }
                store4(i, a0, a1, a2, a3);
                store4(i + 4, a4, a5, a6, a7);
                i += 8;
            }
            while i + 5 <= len {
                let (mut a0, mut a1, mut a2, mut a3) = (b0, b1, b2, b3);
                for t in 0..n {
                    let x = vld2q_f32(input.offset(offsets[t]).offset(i as isize * 2)).0;
                    a0 = vfmaq_f32(a0, x, vdupq_n_f32(taps[t]));
                    a1 = vfmaq_f32(a1, x, vdupq_n_f32(taps[n + t]));
                    a2 = vfmaq_f32(a2, x, vdupq_n_f32(taps[2 * n + t]));
                    a3 = vfmaq_f32(a3, x, vdupq_n_f32(taps[3 * n + t]));
                }
                store4(i, a0, a1, a2, a3);
                i += 4;
            }
        }
        scalar(input, output, taps, offsets, bias, i, len, in_stride, oc_stride);
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
        for in_stride in 1..=3 {
            for len in 0..40 {
                check(in_stride, len, 3, 3);
                check(in_stride, len, 1, 2);
            }
        }
    }
}
