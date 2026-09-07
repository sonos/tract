/// Depthwise convolution along an axis the output is contiguous on: `len` output points, each
/// the bias plus every `taps[t] * input[offsets[t] + i * in_stride]`.
///
/// `taps` and `offsets` hold one kernel tap each and must be the same length. `in_stride` is the
/// input step one output point costs, in elements: 1 is a plain load, 2 and 3 are de-interleaved
/// by `vld2q`/`vld3q`, and anything else stays scalar. The vector loops stop early enough that
/// no lane reads past `offsets[t] + (len - 1) * in_stride`, so a run ending on the tensor's last
/// element is safe.
///
/// # Safety
/// `input.offset(offsets[t] + i * in_stride)` must be readable for every tap and every `i` below
/// `len`, and `len` output points writable from `output`.
#[cfg(target_arch = "aarch64")]
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
        match taps.len() {
            1 => vectorised::<1>(input, output, taps, offsets, bias, len, in_stride),
            2 => vectorised::<2>(input, output, taps, offsets, bias, len, in_stride),
            3 => vectorised::<3>(input, output, taps, offsets, bias, len, in_stride),
            4 => vectorised::<4>(input, output, taps, offsets, bias, len, in_stride),
            _ => scalar(input, output, taps, offsets, bias, 0, len, in_stride),
        }
    }
}

#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
unsafe fn vectorised<const N: usize>(
    input: *const f32,
    output: *mut f32,
    taps: &[f32],
    offsets: &[isize],
    bias: f32,
    len: usize,
    in_stride: isize,
) {
    unsafe {
        use std::arch::aarch64::*;
        let mut k = [0f32; N];
        k.copy_from_slice(&taps[..N]);
        let mut off = [0isize; N];
        off.copy_from_slice(&offsets[..N]);
        let biasv = vdupq_n_f32(bias);
        let mut i = 0usize;
        if in_stride == 1 {
            while i + 8 <= len {
                let mut acc0 = biasv;
                let mut acc1 = biasv;
                for n in 0..N {
                    let kn = vdupq_n_f32(k[n]);
                    let p = input.offset(off[n]).add(i);
                    acc0 = vfmaq_f32(acc0, vld1q_f32(p), kn);
                    acc1 = vfmaq_f32(acc1, vld1q_f32(p.add(4)), kn);
                }
                vst1q_f32(output.add(i), acc0);
                vst1q_f32(output.add(i + 4), acc1);
                i += 8;
            }
            while i + 4 <= len {
                let mut acc = biasv;
                for n in 0..N {
                    let kn = vdupq_n_f32(k[n]);
                    acc = vfmaq_f32(acc, vld1q_f32(input.offset(off[n]).add(i)), kn);
                }
                vst1q_f32(output.add(i), acc);
                i += 4;
            }
        } else if in_stride == 2 {
            while i + 9 <= len {
                let mut acc0 = biasv;
                let mut acc1 = biasv;
                for n in 0..N {
                    let kn = vdupq_n_f32(k[n]);
                    let p = input.offset(off[n]).offset(i as isize * 2);
                    acc0 = vfmaq_f32(acc0, vld2q_f32(p).0, kn);
                    acc1 = vfmaq_f32(acc1, vld2q_f32(p.add(8)).0, kn);
                }
                vst1q_f32(output.add(i), acc0);
                vst1q_f32(output.add(i + 4), acc1);
                i += 8;
            }
            while i + 5 <= len {
                let mut acc = biasv;
                for n in 0..N {
                    let kn = vdupq_n_f32(k[n]);
                    acc = vfmaq_f32(
                        acc,
                        vld2q_f32(input.offset(off[n]).offset(i as isize * 2)).0,
                        kn,
                    );
                }
                vst1q_f32(output.add(i), acc);
                i += 4;
            }
        } else if in_stride == 3 {
            while i + 9 <= len {
                let mut acc0 = biasv;
                let mut acc1 = biasv;
                for n in 0..N {
                    let kn = vdupq_n_f32(k[n]);
                    let p = input.offset(off[n]).offset(i as isize * 3);
                    acc0 = vfmaq_f32(acc0, vld3q_f32(p).0, kn);
                    acc1 = vfmaq_f32(acc1, vld3q_f32(p.add(12)).0, kn);
                }
                vst1q_f32(output.add(i), acc0);
                vst1q_f32(output.add(i + 4), acc1);
                i += 8;
            }
            while i + 5 <= len {
                let mut acc = biasv;
                for n in 0..N {
                    let kn = vdupq_n_f32(k[n]);
                    acc = vfmaq_f32(
                        acc,
                        vld3q_f32(input.offset(off[n]).offset(i as isize * 3)).0,
                        kn,
                    );
                }
                vst1q_f32(output.add(i), acc);
                i += 4;
            }
        }
        scalar(input, output, taps, offsets, bias, i, len, in_stride);
    }
}

#[cfg(target_arch = "aarch64")]
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
                sum += tap * *input.offset(offset + i as isize * in_stride);
            }
            *output.add(i) = sum;
        }
    }
}

bail_stub!(aarch64; pub unsafe fn depthwise_w_f32(
    *const f32, *mut f32, &[f32], &[isize], f32, usize, isize
));

submit_routine!(aarch64; DepthwiseWF32, DepthwiseW, "arm64simd_depthwise_w_f32", depthwise_w_f32);

#[cfg(all(test, target_arch = "aarch64"))]
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
        let k: Vec<f32> = (0..taps).map(|t| (t as f32 * 0.7).sin()).collect();
        let offsets: Vec<isize> = (0..taps as isize).map(|t| t - taps as isize / 2).collect();
        let center = taps;
        let input: Vec<f32> = (0..center + (len - 1) * in_stride as usize + center)
            .map(|i| (i as f32 * 0.11).cos())
            .collect();
        let want = reference(&input, &k, &offsets, 0.25, len, in_stride, center);
        let mut got = vec![0f32; len];
        unsafe {
            depthwise_w_f32(
                input.as_ptr().add(center),
                got.as_mut_ptr(),
                &k,
                &offsets,
                0.25,
                len,
                in_stride,
            )
        };
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() < 1e-5,
                "taps={taps} len={len} stride={in_stride} at {i}: got {g}, want {w}"
            );
        }
    }

    #[test]
    fn matches_reference() {
        for taps in [1usize, 2, 3, 4, 5] {
            for in_stride in [1isize, 2, 3, 4] {
                for len in [1usize, 3, 4, 7, 8, 9, 15, 16, 33, 481] {
                    compare(taps, len, in_stride);
                }
            }
        }
    }
}
