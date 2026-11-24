//! f16 depthwise kernels: the f32 `arm64simd` depthwise contracts on eight-lane halves,
//! accumulating in f16. Loads and stores go through the u16 NEON intrinsics and only the fused
//! multiply-add is assembly, the f16 NEON intrinsics not being available on stable Rust.

use tract_data::half::f16;

/// `acc + a * b`, lane by lane, on eight f16 lanes carried as u16.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "fp16")]
#[inline]
unsafe fn fma(
    acc: std::arch::aarch64::uint16x8_t,
    a: std::arch::aarch64::uint16x8_t,
    b: std::arch::aarch64::uint16x8_t,
) -> std::arch::aarch64::uint16x8_t {
    let mut acc = acc;
    unsafe {
        std::arch::asm!(
            "fmla {acc:v}.8h, {a:v}.8h, {b:v}.8h",
            acc = inout(vreg) acc,
            a = in(vreg) a,
            b = in(vreg) b,
            options(pure, nomem, nostack, preserves_flags)
        );
    }
    acc
}

/// `acc + a * b`, on one f16.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "fp16")]
#[inline]
unsafe fn fma_scalar(acc: f16, a: f16, b: f16) -> f16 {
    let mut bits = acc.to_bits();
    unsafe {
        std::arch::asm!(
            "fmadd {acc:h}, {a:h}, {b:h}, {acc:h}",
            acc = inout(vreg) bits,
            a = in(vreg) a.to_bits(),
            b = in(vreg) b.to_bits(),
            options(pure, nomem, nostack, preserves_flags)
        );
    }
    f16::from_bits(bits)
}

/// [`crate::routines::DepthwiseWF16`]: input strides 1 and 2 are vectorised, eight output
/// points per vector, the last vector of a run overlapping the one before it. Every other
/// stride, and the points a vector cannot reach, stay scalar. No lane reads past
/// `offsets[t] + (len - 1) * in_stride`.
///
/// # Safety
/// As [`crate::routines::DepthwiseWF32`].
#[cfg(target_arch = "aarch64")]
pub unsafe fn depthwise_w_f16(
    input: *const f16,
    output: *mut f16,
    taps: &[f16],
    offsets: &[isize],
    bias: f16,
    len: usize,
    in_stride: isize,
) {
    unsafe {
        match in_stride {
            1 => strided::<1>(input, output, taps, offsets, bias, len),
            2 => strided::<2>(input, output, taps, offsets, bias, len),
            _ => scalar_w(input, output, taps, offsets, bias, 0, len, in_stride),
        }
    }
}

/// Eight consecutive output points' inputs, `S` elements apart, from `p`. Reads `8 * S`
/// elements, `S - 1` more than it keeps.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "fp16")]
#[inline]
unsafe fn load8<const S: usize>(p: *const f16) -> std::arch::aarch64::uint16x8_t {
    unsafe {
        use std::arch::aarch64::*;
        let p = p as *const u16;
        match S {
            1 => vld1q_u16(p),
            _ => vld2q_u16(p).0,
        }
    }
}

/// Taps outer and four vectors inner, so a long run keeps four independent accumulation chains
/// and pays each tap's broadcast once per 32 points.
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "fp16")]
unsafe fn strided<const S: usize>(
    input: *const f16,
    output: *mut f16,
    taps: &[f16],
    offsets: &[isize],
    bias: f16,
    len: usize,
) {
    unsafe {
        use std::arch::aarch64::*;
        // load8 reads past its last point at stride 2, into the next output point's inputs, so
        // that point has to be in the run too.
        let spare = (S > 1) as usize;
        let biasv = vdupq_n_u16(bias.to_bits());
        let mut i = 0usize;
        while i + 32 + spare <= len {
            let mut acc = [biasv; 4];
            for (&tap, &offset) in taps.iter().zip(offsets) {
                let k = vdupq_n_u16(tap.to_bits());
                let p = input.offset(offset).add(i * S);
                for (v, a) in acc.iter_mut().enumerate() {
                    *a = fma(*a, load8::<S>(p.add(8 * v * S)), k);
                }
            }
            for (v, a) in acc.iter().enumerate() {
                vst1q_u16(output.add(i + 8 * v) as *mut u16, *a);
            }
            i += 32;
        }
        while i + spare < len && len >= 8 + spare {
            let at = i.min(len - 8 - spare);
            let mut acc = biasv;
            for (&tap, &offset) in taps.iter().zip(offsets) {
                let p = input.offset(offset).add(at * S);
                acc = fma(acc, load8::<S>(p), vdupq_n_u16(tap.to_bits()));
            }
            vst1q_u16(output.add(at) as *mut u16, acc);
            i = at + 8;
        }
        scalar_w(input, output, taps, offsets, bias, i, len, S as isize);
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "fp16")]
#[allow(clippy::too_many_arguments)]
unsafe fn scalar_w(
    input: *const f16,
    output: *mut f16,
    taps: &[f16],
    offsets: &[isize],
    bias: f16,
    from: usize,
    len: usize,
    in_stride: isize,
) {
    unsafe {
        for i in from..len {
            let mut sum = bias;
            for (&tap, &offset) in taps.iter().zip(offsets) {
                sum = fma_scalar(sum, tap, *input.offset(offset + i as isize * in_stride));
            }
            *output.add(i) = sum;
        }
    }
}

bail_stub!(aarch64; pub unsafe fn depthwise_w_f16(
    *const f16, *mut f16, &[f16], &[isize], f16, usize, isize
));

/// [`crate::routines::DepthwiseCF16`]: 32 then 8 channels per step, the remaining channels
/// scalar.
///
/// # Safety
/// As [`crate::routines::DepthwiseCF32`].
#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "fp16")]
#[allow(clippy::too_many_arguments)]
pub unsafe fn depthwise_c_f16(
    input: *const f16,
    output: *mut f16,
    kernel: *const f16,
    tap_indices: &[usize],
    offsets: &[isize],
    bias: *const f16,
    channels: usize,
    k_stride: usize,
    len: usize,
    in_stride: isize,
    out_stride: isize,
) {
    unsafe {
        use std::arch::aarch64::*;
        let ld = |p: *const f16| vld1q_u16(p as *const u16);
        for i in 0..len {
            let iptr = input.offset(i as isize * in_stride);
            let optr = output.offset(i as isize * out_stride);
            let mut c = 0usize;
            while c + 32 <= channels {
                let mut acc = [
                    ld(bias.add(c)),
                    ld(bias.add(c + 8)),
                    ld(bias.add(c + 16)),
                    ld(bias.add(c + 24)),
                ];
                for (&kix, &off) in tap_indices.iter().zip(offsets) {
                    let kt = kernel.add(kix * k_stride + c);
                    let xt = iptr.offset(off).add(c);
                    for (v, a) in acc.iter_mut().enumerate() {
                        *a = fma(*a, ld(xt.add(8 * v)), ld(kt.add(8 * v)));
                    }
                }
                for (v, a) in acc.iter().enumerate() {
                    vst1q_u16(optr.add(c + 8 * v) as *mut u16, *a);
                }
                c += 32;
            }
            while c + 8 <= channels {
                let mut acc = ld(bias.add(c));
                for (&kix, &off) in tap_indices.iter().zip(offsets) {
                    acc = fma(acc, ld(iptr.offset(off).add(c)), ld(kernel.add(kix * k_stride + c)));
                }
                vst1q_u16(optr.add(c) as *mut u16, acc);
                c += 8;
            }
            while c < channels {
                let mut sum = *bias.add(c);
                for (&kix, &off) in tap_indices.iter().zip(offsets) {
                    sum =
                        fma_scalar(sum, *kernel.add(kix * k_stride + c), *iptr.offset(off).add(c));
                }
                *optr.add(c) = sum;
                c += 1;
            }
        }
    }
}

bail_stub!(aarch64; pub unsafe fn depthwise_c_f16(
    *const f16, *mut f16, *const f16, &[usize], &[isize], *const f16, usize, usize, usize, isize,
    isize
));

submit_routine!(aarch64; DepthwiseWF16, DepthwiseW, "arm64fp16_depthwise_w_f16", depthwise_w_f16,
    isa(Aarch64Fp16));
submit_routine!(aarch64; DepthwiseCF16, DepthwiseC, "arm64fp16_depthwise_c_f16", depthwise_c_f16,
    isa(Aarch64Fp16));

#[cfg(all(test, target_arch = "aarch64"))]
mod tests {
    use super::*;

    fn h(x: f32) -> f16 {
        f16::from_f32(x)
    }

    /// The f32 sum of the f16 terms, with the bound an f16 accumulation of them stays within.
    fn reference(terms: impl Iterator<Item = f32>, bias: f32) -> (f32, f32) {
        let (mut sum, mut abs) = (bias, bias.abs());
        for t in terms {
            sum += t;
            abs += t.abs();
        }
        (sum, abs * 4e-3 + 1e-3)
    }

    fn compare_w(taps: usize, len: usize, in_stride: isize) {
        if !crate::arm64::has_fp16() {
            return;
        }
        let k: Vec<f16> = (0..taps).map(|t| h((t as f32 * 0.7).sin())).collect();
        let offsets: Vec<isize> = (0..taps as isize).map(|t| t - taps as isize / 2).collect();
        let center = taps;
        let input: Vec<f16> = (0..center + (len - 1) * in_stride as usize + center)
            .map(|i| h((i as f32 * 0.11).cos()))
            .collect();
        let mut got = vec![f16::ZERO; len];
        unsafe {
            depthwise_w_f16(
                input.as_ptr().add(center),
                got.as_mut_ptr(),
                &k,
                &offsets,
                h(0.25),
                len,
                in_stride,
            )
        };
        for (i, g) in got.iter().enumerate() {
            let (want, tol) = reference(
                k.iter().zip(&offsets).map(|(t, o)| {
                    t.to_f32()
                        * input[(center as isize + o + i as isize * in_stride) as usize].to_f32()
                }),
                0.25,
            );
            assert!(
                (g.to_f32() - want).abs() <= tol,
                "taps={taps} len={len} stride={in_stride} at {i}: got {g}, want {want}"
            );
        }
    }

    #[test]
    fn w_matches_reference() {
        for taps in [1usize, 2, 3, 4, 5, 9, 16, 25] {
            for in_stride in [1isize, 2, 3] {
                for len in [1usize, 7, 8, 9, 16, 17, 31, 32, 33, 40, 65, 113, 481] {
                    compare_w(taps, len, in_stride);
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
        if !crate::arm64::has_fp16() {
            return;
        }
        let kernel: Vec<f16> = (0..taps * channels).map(|i| h((i as f32 * 0.7).sin())).collect();
        let tap_indices: Vec<usize> = (0..taps).rev().collect();
        let offsets: Vec<isize> =
            (0..taps as isize).map(|t| (t - taps as isize / 2) * channels as isize).collect();
        let bias: Vec<f16> = (0..channels).map(|i| h(i as f32 * 0.05 - 0.1)).collect();
        let in_stride = channels as isize * s;
        let out_stride = channels as isize;
        let center = (taps / 2) * channels;
        let input: Vec<f16> =
            (0..center + (len - 1) * in_stride as usize + (taps - taps / 2) * channels)
                .map(|i| h((i as f32 * 0.11).cos()))
                .collect();
        let out_len = (len - 1) * out_stride as usize + channels;
        let mut got = vec![f16::NAN; out_len];
        unsafe {
            depthwise_c_f16(
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
        for i in 0..len {
            for c in 0..channels {
                let g = got[i * out_stride as usize + c];
                if !block.contains(&c) {
                    assert!(g.is_nan(), "wrote outside the block at {i},{c}");
                    continue;
                }
                let (want, tol) = reference(
                    tap_indices.iter().zip(&offsets).map(|(&kix, &off)| {
                        kernel[kix * channels + c].to_f32()
                            * input[(center as isize + i as isize * in_stride + off) as usize + c]
                                .to_f32()
                    }),
                    bias[c].to_f32(),
                );
                assert!(
                    (g.to_f32() - want).abs() <= tol,
                    "taps={taps} ch={channels} len={len} s={s} at {i},{c}: got {g}, want {want}"
                );
            }
        }
    }

    #[test]
    fn c_matches_reference() {
        for taps in [1usize, 2, 3, 4, 5, 9] {
            for channels in [1usize, 7, 8, 9, 16, 31, 32, 33, 40, 64, 80] {
                for len in [1usize, 5, 8] {
                    for s in [1isize, 2] {
                        compare_c(taps, channels, 0..channels, len, s);
                    }
                }
            }
        }
    }

    #[test]
    fn c_block_matches_reference() {
        for taps in [1usize, 9] {
            for block in [0..32, 32..64, 16..48, 5..27, 63..64] {
                compare_c(taps, 64, block, 3, 1);
            }
        }
    }
}
