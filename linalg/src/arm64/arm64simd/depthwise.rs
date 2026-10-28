/// Depthwise convolution along an axis the output is contiguous on: `len` output points, each
/// the bias plus every `taps[t] * input[offsets[t] + i * in_stride]`.
///
/// `taps` and `offsets` hold one kernel tap each and must be the same length. `in_stride` is the
/// input step one output point costs, in elements. Strides 1 to 4 are vectorised for any tap
/// count, and anything else stays scalar: one to four taps at strides 1 to 3 take a loop with
/// the count fixed at compile time, every other case [`wide`]. The vector loops stop early
/// enough that no lane reads past `offsets[t] + (len - 1) * in_stride`, so a run ending on the
/// tensor's last element is safe.
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
        match (taps.len(), in_stride) {
            (1, 1..=3) => vectorised::<1>(input, output, taps, offsets, bias, len, in_stride),
            (2, 1..=3) => vectorised::<2>(input, output, taps, offsets, bias, len, in_stride),
            (3, 1..=3) => vectorised::<3>(input, output, taps, offsets, bias, len, in_stride),
            (4, 1..=3) => vectorised::<4>(input, output, taps, offsets, bias, len, in_stride),
            (_, 1) => wide::<1>(input, output, taps, offsets, bias, len),
            (_, 2) => wide::<2>(input, output, taps, offsets, bias, len),
            (_, 3) => wide::<3>(input, output, taps, offsets, bias, len),
            (_, 4) => wide::<4>(input, output, taps, offsets, bias, len),
            _ => scalar(input, output, taps, offsets, bias, 0, len, in_stride),
        }
    }
}

/// Any tap count at input stride `S`, with the count taken at runtime. Taps are the outer loop
/// and up to 64 output points the inner one, so a long run gives the FMA unit sixteen
/// independent chains, and each tap's broadcast is paid once per pass rather than once per
/// vector. The rest of a run takes passes of 32, 16 and 4 points.
#[cfg(target_arch = "aarch64")]
unsafe fn wide<const S: usize>(
    input: *const f32,
    output: *mut f32,
    taps: &[f32],
    offsets: &[isize],
    bias: f32,
    len: usize,
) {
    unsafe {
        // load4 reads past its last point at strides above 1, into the next output point's
        // inputs, so that point has to be in the run too.
        let spare = (S > 1) as usize;
        let mut i = 0usize;
        while i + 64 + spare <= len {
            pass::<S, 16>(input, output, taps, offsets, bias, i);
            i += 64;
        }
        if i + 32 + spare <= len {
            pass::<S, 8>(input, output, taps, offsets, bias, i);
            i += 32;
        }
        if i + 16 + spare <= len {
            pass::<S, 4>(input, output, taps, offsets, bias, i);
            i += 16;
        }
        while i + 4 + spare <= len {
            pass::<S, 1>(input, output, taps, offsets, bias, i);
            i += 4;
        }
        scalar(input, output, taps, offsets, bias, i, len, S as isize);
    }
}

/// `4 * V` output points from point `i` of [`wide`], one accumulator per vector.
#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn pass<const S: usize, const V: usize>(
    input: *const f32,
    output: *mut f32,
    taps: &[f32],
    offsets: &[isize],
    bias: f32,
    i: usize,
) {
    unsafe {
        use std::arch::aarch64::*;
        let mut acc = [vdupq_n_f32(bias); V];
        for (&tap, &offset) in taps.iter().zip(offsets) {
            let kn = vdupq_n_f32(tap);
            let p = input.offset(offset).add(i * S);
            for (v, a) in acc.iter_mut().enumerate() {
                *a = vfmaq_f32(*a, load4::<S>(p.add(4 * v * S)), kn);
            }
        }
        for (v, a) in acc.iter().enumerate() {
            vst1q_f32(output.add(i + 4 * v), *a);
        }
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

/// Depthwise convolution with the channel as the contiguous dim: `len` output positions of
/// `channels` channels each. Point `i`, channel `c`, is `bias[c]` plus every
/// `kernel[tap_indices[t] * k_stride + c] * input[i * in_stride + offsets[t] + c]`.
///
/// `k_stride` is the kernel's per-tap stride, at least `channels`: a caller can run a block of
/// channels by offsetting every pointer to the block's first channel. `tap_indices` and
/// `offsets` hold one kernel tap each and must be the same length. `in_stride`
/// and `out_stride` are the input and output steps one output position costs, in elements; both
/// step over whole channel vectors, so every tap stays contiguous in `c`.
///
/// # Safety
/// `input.offset(i * in_stride + offsets[t] + c)` and `kernel.add(tap_indices[t] * k_stride + c)`
/// must be readable for every tap, every `i` below `len` and every `c` below `channels`;
/// `channels` values must be readable from `bias`, and `channels` values writable from
/// `output.offset(i * out_stride)` for every `i` below `len`.
#[cfg(target_arch = "aarch64")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn depthwise_c_f32(
    input: *const f32,
    output: *mut f32,
    kernel: *const f32,
    tap_indices: &[usize],
    offsets: &[isize],
    bias: *const f32,
    channels: usize,
    k_stride: usize,
    len: usize,
    in_stride: isize,
    out_stride: isize,
) {
    unsafe {
        use std::arch::aarch64::*;
        for i in 0..len {
            let iptr = input.offset(i as isize * in_stride);
            let optr = output.offset(i as isize * out_stride);
            let mut c = 0usize;
            while c + 16 <= channels {
                let mut acc0 = vld1q_f32(bias.add(c));
                let mut acc1 = vld1q_f32(bias.add(c + 4));
                let mut acc2 = vld1q_f32(bias.add(c + 8));
                let mut acc3 = vld1q_f32(bias.add(c + 12));
                for (&kix, &off) in tap_indices.iter().zip(offsets) {
                    let kt = kernel.add(kix * k_stride + c);
                    let xt = iptr.offset(off).add(c);
                    acc0 = vfmaq_f32(acc0, vld1q_f32(xt), vld1q_f32(kt));
                    acc1 = vfmaq_f32(acc1, vld1q_f32(xt.add(4)), vld1q_f32(kt.add(4)));
                    acc2 = vfmaq_f32(acc2, vld1q_f32(xt.add(8)), vld1q_f32(kt.add(8)));
                    acc3 = vfmaq_f32(acc3, vld1q_f32(xt.add(12)), vld1q_f32(kt.add(12)));
                }
                vst1q_f32(optr.add(c), acc0);
                vst1q_f32(optr.add(c + 4), acc1);
                vst1q_f32(optr.add(c + 8), acc2);
                vst1q_f32(optr.add(c + 12), acc3);
                c += 16;
            }
            while c + 4 <= channels {
                let mut acc = vld1q_f32(bias.add(c));
                for (&kix, &off) in tap_indices.iter().zip(offsets) {
                    acc = vfmaq_f32(
                        acc,
                        vld1q_f32(iptr.offset(off).add(c)),
                        vld1q_f32(kernel.add(kix * k_stride + c)),
                    );
                }
                vst1q_f32(optr.add(c), acc);
                c += 4;
            }
            while c < channels {
                let mut sum = *bias.add(c);
                for (&kix, &off) in tap_indices.iter().zip(offsets) {
                    sum += *kernel.add(kix * k_stride + c) * *iptr.offset(off).add(c);
                }
                *optr.add(c) = sum;
                c += 1;
            }
        }
    }
}

bail_stub!(aarch64; pub unsafe fn depthwise_c_f32(
    *const f32, *mut f32, *const f32, &[usize], &[isize], *const f32, usize, usize, usize, isize,
    isize
));

submit_routine!(aarch64; DepthwiseWF32, DepthwiseW, "arm64simd_depthwise_w_f32", depthwise_w_f32);
submit_routine!(aarch64; DepthwiseCF32, DepthwiseC, "arm64simd_depthwise_c_f32", depthwise_c_f32);

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
        for taps in [1usize, 2, 3, 4, 5, 6, 7, 9, 15, 16, 25, 49] {
            for in_stride in [1isize, 2, 3, 4] {
                for len in [1usize, 3, 4, 7, 8, 9, 15, 16, 33, 65, 80, 113, 481] {
                    compare(taps, len, in_stride);
                }
            }
        }
    }

    fn compare_c(
        taps: usize,
        channels: usize,
        block: std::ops::Range<usize>,
        len: usize,
        s: isize,
    ) {
        let kernel: Vec<f32> = (0..taps * channels).map(|i| (i as f32 * 0.7).sin()).collect();
        let tap_indices: Vec<usize> = (0..taps).rev().collect();
        let offsets: Vec<isize> =
            (0..taps as isize).map(|t| (t - taps as isize / 2) * channels as isize).collect();
        let bias: Vec<f32> = (0..channels).map(|i| i as f32 * 0.05 - 0.1).collect();
        let in_stride = channels as isize * s;
        let out_stride = channels as isize;
        let center = (taps / 2) * channels;
        let input: Vec<f32> =
            (0..center + (len - 1) * in_stride as usize + (taps - taps / 2) * channels)
                .map(|i| (i as f32 * 0.11).cos())
                .collect();
        let out_len = (len - 1) * out_stride as usize + channels;
        let mut want = vec![0f32; out_len];
        for i in 0..len {
            for c in 0..channels {
                let mut sum = bias[c];
                for (t, &kix) in tap_indices.iter().enumerate() {
                    sum += kernel[kix * channels + c]
                        * input
                            [(center as isize + i as isize * in_stride + offsets[t]) as usize + c];
                }
                want[i * out_stride as usize + c] = sum;
            }
        }
        let mut got = want.clone();
        for i in 0..len {
            got[i * out_stride as usize..][block.clone()].fill(f32::NAN);
        }
        unsafe {
            depthwise_c_f32(
                input.as_ptr().add(center + block.start),
                got.as_mut_ptr().add(block.start),
                kernel.as_ptr().add(block.start),
                &tap_indices,
                &offsets,
                bias.as_ptr().add(block.start),
                block.len(),
                channels,
                len,
                in_stride,
                out_stride,
            )
        };
        for (i, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() < 1e-5,
                "taps={taps} ch={channels} len={len} s={s} at {i}: got {g}, want {w}"
            );
        }
    }

    #[test]
    fn depthwise_c_matches_reference() {
        for taps in [1usize, 2, 3, 4, 5, 9] {
            for channels in [1usize, 3, 4, 7, 8, 15, 16, 17, 64] {
                for len in [1usize, 5, 8] {
                    for s in [1isize, 2, 3] {
                        compare_c(taps, channels, 0..channels, len, s);
                    }
                }
            }
        }
    }

    #[test]
    fn depthwise_c_block_matches_reference() {
        for taps in [1usize, 3, 9] {
            for block in [0..16, 16..32, 16..48, 32..64, 5..27, 63..64] {
                for len in [1usize, 5] {
                    compare_c(taps, 64, block.clone(), len, 1);
                }
            }
        }
    }
}
