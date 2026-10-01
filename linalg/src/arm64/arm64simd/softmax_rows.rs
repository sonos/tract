/// Softmax in place over each `row_len` row of `buf`, `row_len` a non-zero multiple of 4. Per
/// row: the row max, `exp(x - max)` with the same arithmetic as the generic `Softmax2` kernel's
/// NEON path, summed in its order, then a multiply by `1 / sum`; a row holding a NaN comes out
/// all NaN, as through the per-row kernels. Rows go in pairs so two exp chains overlap.
#[cfg(target_arch = "aarch64")]
pub fn softmax_rows_f32(buf: &mut [f32], row_len: usize) {
    use std::arch::aarch64::*;
    debug_assert!(row_len > 0 && row_len.is_multiple_of(4));

    #[inline(always)]
    unsafe fn exp_ps(x: float32x4_t) -> float32x4_t {
        unsafe {
            let kf = vrndnq_f32(vmulq_f32(x, vdupq_n_f32(1.442_695_04)));
            let mut rr = vfmsq_f32(x, kf, vdupq_n_f32(0.693_145_75));
            rr = vfmsq_f32(rr, kf, vdupq_n_f32(1.428_606_8e-6));
            let mut q = vdupq_n_f32(8.297653546e-03);
            q = vfmaq_f32(vdupq_n_f32(4.191538191e-02), q, rr);
            q = vfmaq_f32(vdupq_n_f32(1.666757475e-01), q, rr);
            q = vfmaq_f32(vdupq_n_f32(4.999889485e-01), q, rr);
            q = vfmaq_f32(vdupq_n_f32(9.999996920e-01), q, rr);
            q = vfmaq_f32(vdupq_n_f32(1.000000072e+00), q, rr);
            let k = vcvtq_s32_f32(kf);
            let biased = vmaxq_s32(
                vminq_s32(vaddq_s32(k, vdupq_n_s32(127)), vdupq_n_s32(254)),
                vdupq_n_s32(1),
            );
            let scale = vreinterpretq_f32_s32(vshlq_n_s32(biased, 23));
            let out = vorrq_u32(vcltq_f32(x, vdupq_n_f32(-103.0)), vcgtq_f32(x, vdupq_n_f32(0.0)));
            vbslq_f32(out, vdupq_n_f32(0.0), vmulq_f32(q, scale))
        }
    }

    let num_rows = buf.len() / row_len;
    let mut row_idx = 0;

    // Process pairs of rows with interleaving for latency hiding
    while row_idx + 1 < num_rows {
        unsafe {
            let p1 = buf.as_mut_ptr().add(row_idx * row_len);
            let p2 = buf.as_mut_ptr().add((row_idx + 1) * row_len);

            // Max pass: 2-vector unroll per row
            let mut vmax1a = vdupq_n_f32(f32::NEG_INFINITY);
            let mut vmax1b = vdupq_n_f32(f32::NEG_INFINITY);
            let mut vmax2a = vdupq_n_f32(f32::NEG_INFINITY);
            let mut vmax2b = vdupq_n_f32(f32::NEG_INFINITY);
            let mut i = 0;
            while i + 8 <= row_len {
                vmax1a = vmaxq_f32(vmax1a, vld1q_f32(p1.add(i)));
                vmax1b = vmaxq_f32(vmax1b, vld1q_f32(p1.add(i + 4)));
                vmax2a = vmaxq_f32(vmax2a, vld1q_f32(p2.add(i)));
                vmax2b = vmaxq_f32(vmax2b, vld1q_f32(p2.add(i + 4)));
                i += 8;
            }
            while i < row_len {
                vmax1a = vmaxq_f32(vmax1a, vld1q_f32(p1.add(i)));
                vmax2a = vmaxq_f32(vmax2a, vld1q_f32(p2.add(i)));
                i += 4;
            }
            let m1 = vdupq_n_f32(vmaxvq_f32(vmaxq_f32(vmax1a, vmax1b)));
            let m2 = vdupq_n_f32(vmaxvq_f32(vmaxq_f32(vmax2a, vmax2b)));

            // Exp + sum pass: interleaved between rows
            let mut vsum1 = vdupq_n_f32(0.0);
            let mut vsum2 = vdupq_n_f32(0.0);
            i = 0;
            while i < row_len {
                let y1 = exp_ps(vsubq_f32(vld1q_f32(p1.add(i)), m1));
                let y2 = exp_ps(vsubq_f32(vld1q_f32(p2.add(i)), m2));
                vst1q_f32(p1.add(i), y1);
                vst1q_f32(p2.add(i), y2);
                vsum1 = vaddq_f32(vsum1, y1);
                vsum2 = vaddq_f32(vsum2, y2);
                i += 4;
            }
            let r1 = vdupq_n_f32(1.0 / vaddvq_f32(vsum1));
            let r2 = vdupq_n_f32(1.0 / vaddvq_f32(vsum2));

            // Scale pass: 2-vector unroll per row
            i = 0;
            while i + 8 <= row_len {
                vst1q_f32(p1.add(i), vmulq_f32(vld1q_f32(p1.add(i)), r1));
                vst1q_f32(p1.add(i + 4), vmulq_f32(vld1q_f32(p1.add(i + 4)), r1));
                vst1q_f32(p2.add(i), vmulq_f32(vld1q_f32(p2.add(i)), r2));
                vst1q_f32(p2.add(i + 4), vmulq_f32(vld1q_f32(p2.add(i + 4)), r2));
                i += 8;
            }
            while i < row_len {
                vst1q_f32(p1.add(i), vmulq_f32(vld1q_f32(p1.add(i)), r1));
                vst1q_f32(p2.add(i), vmulq_f32(vld1q_f32(p2.add(i)), r2));
                i += 4;
            }
        }
        row_idx += 2;
    }

    // Handle remaining single row
    if row_idx < num_rows {
        unsafe {
            let p = buf.as_mut_ptr().add(row_idx * row_len);

            // Max pass: 2-vector unroll
            let mut vmax1 = vdupq_n_f32(f32::NEG_INFINITY);
            let mut vmax2 = vdupq_n_f32(f32::NEG_INFINITY);
            let mut i = 0;
            while i + 8 <= row_len {
                vmax1 = vmaxq_f32(vmax1, vld1q_f32(p.add(i)));
                vmax2 = vmaxq_f32(vmax2, vld1q_f32(p.add(i + 4)));
                i += 8;
            }
            while i < row_len {
                vmax1 = vmaxq_f32(vmax1, vld1q_f32(p.add(i)));
                i += 4;
            }
            let m = vdupq_n_f32(vmaxvq_f32(vmaxq_f32(vmax1, vmax2)));

            // Exp + sum pass
            let mut vsum = vdupq_n_f32(0.0);
            i = 0;
            while i < row_len {
                let y = exp_ps(vsubq_f32(vld1q_f32(p.add(i)), m));
                vst1q_f32(p.add(i), y);
                vsum = vaddq_f32(vsum, y);
                i += 4;
            }
            let r = vdupq_n_f32(1.0 / vaddvq_f32(vsum));

            // Scale pass: 2-vector unroll
            i = 0;
            while i + 8 <= row_len {
                vst1q_f32(p.add(i), vmulq_f32(vld1q_f32(p.add(i)), r));
                vst1q_f32(p.add(i + 4), vmulq_f32(vld1q_f32(p.add(i + 4)), r));
                i += 8;
            }
            while i < row_len {
                vst1q_f32(p.add(i), vmulq_f32(vld1q_f32(p.add(i)), r));
                i += 4;
            }
        }
    }
}

bail_stub!(aarch64; pub fn softmax_rows_f32(&mut [f32], usize));

submit_routine!(aarch64; SoftmaxRowsF32, SoftmaxRows, "arm64simd_softmax_rows_f32", softmax_rows_f32);

#[cfg(all(test, target_arch = "aarch64"))]
mod tests {
    use super::*;
    use crate::routines::Func;

    /// The per-row kernels the softmax op runs without this one.
    fn per_row(buf: &mut [f32], row_len: usize) {
        let max = Func::ReduceMax.reduce_f32().unwrap();
        let exp_sum = Func::Softmax2.map_reduce_f32().unwrap();
        let scale = Func::MulByScalar.ew_f32_param().unwrap();
        for row in buf.chunks_mut(row_len) {
            let m = max.run(row).unwrap();
            let s = exp_sum.run_with_params(row, m).unwrap();
            scale.run_with_params(row, s.recip()).unwrap();
        }
    }

    fn bits(v: &[f32]) -> Vec<u32> {
        v.iter().map(|x| x.to_bits()).collect()
    }

    #[test]
    fn matches_the_per_row_kernels_bit_for_bit() {
        for row_len in (4..=128).step_by(4) {
            for rows in [1usize, 2, 3, 5, 7, 8, 9, 15, 16, 17] {
                let base: Vec<f32> =
                    (0..rows * row_len).map(|i| ((i as f32) * 0.37).sin() * 6.0).collect();
                let mut want = base.clone();
                per_row(&mut want, row_len);
                let mut got = base.clone();
                softmax_rows_f32(&mut got, row_len);
                assert_eq!(bits(&got), bits(&want), "rows={rows} row_len={row_len}");
            }
        }
    }

    /// NaN, a fully masked row and an infinite logit come out as through the per-row kernels.
    #[test]
    fn edge_rows_match_the_per_row_kernels() {
        let row_len = 12;
        let mut base: Vec<f32> = (0..4 * row_len).map(|i| (i as f32 * 0.21).cos()).collect();
        base[3] = f32::NAN;
        base[row_len..2 * row_len].fill(f32::NEG_INFINITY);
        base[2 * row_len + 5] = f32::INFINITY;
        let mut want = base.clone();
        per_row(&mut want, row_len);
        let mut got = base.clone();
        softmax_rows_f32(&mut got, row_len);
        assert!(got[..row_len].iter().all(|x| x.is_nan()));
        for r in 1..4 {
            let (g, w) = (&got[r * row_len..][..row_len], &want[r * row_len..][..row_len]);
            assert!(
                g.iter()
                    .zip(w)
                    .all(|(g, w)| g.to_bits() == w.to_bits() || g.is_nan() && w.is_nan()),
                "row {r}: {g:?} vs {w:?}"
            );
        }
    }

    #[test]
    fn registered_as_this_hosts_row_softmax() {
        assert!(crate::routines::softmax_rows_f32().is_some());
    }
}
