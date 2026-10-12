/// Softmax in place over each `row_len` row of `buf`, `row_len` a non-zero multiple of 4. Per
/// row: the row max, `exp(x - max)` with the same arithmetic as the generic `Softmax2` kernel's
/// simd128 path, summed in its order, then a multiply by `1 / sum`; a row holding a NaN comes out
/// all NaN, as through the per-row kernels.
#[cfg(all(target_arch = "wasm32", target_feature = "simd128"))]
pub fn softmax_rows_f32(buf: &mut [f32], row_len: usize) {
    use std::arch::wasm32::*;
    debug_assert!(row_len > 0 && row_len.is_multiple_of(4));

    #[inline(never)]
    fn exp_ps(x: v128) -> v128 {
        let kf = f32x4_nearest(f32x4_mul(x, f32x4_splat(1.442_695_04)));
        let mut rr = f32x4_sub(x, f32x4_mul(kf, f32x4_splat(0.693_145_75)));
        rr = f32x4_sub(rr, f32x4_mul(kf, f32x4_splat(1.428_606_8e-6)));
        let mut q = f32x4_splat(8.297653546e-03);
        q = f32x4_add(f32x4_splat(4.191538191e-02), f32x4_mul(q, rr));
        q = f32x4_add(f32x4_splat(1.666757475e-01), f32x4_mul(q, rr));
        q = f32x4_add(f32x4_splat(4.999889485e-01), f32x4_mul(q, rr));
        q = f32x4_add(f32x4_splat(9.999996920e-01), f32x4_mul(q, rr));
        q = f32x4_add(f32x4_splat(1.000000072e+00), f32x4_mul(q, rr));
        let k = i32x4_trunc_sat_f32x4(kf);
        let biased =
            i32x4_max(i32x4_min(i32x4_add(k, i32x4_splat(127)), i32x4_splat(254)), i32x4_splat(1));
        let scale = i32x4_shl(biased, 23);
        let out = v128_or(f32x4_lt(x, f32x4_splat(-103.0)), f32x4_gt(x, f32x4_splat(0.0)));
        v128_bitselect(f32x4_splat(0.0), f32x4_mul(q, scale), out)
    }

    #[inline(never)]
    fn max_pass(p: *const f32, row_len: usize) -> f32 {
        use std::arch::wasm32::*;
        let mut acc = f32x4_splat(f32::NEG_INFINITY);

        let mut i = 0;
        while i + 8 <= row_len {
            unsafe {
                acc = f32x4_pmax(acc, v128_load(p.add(i) as *const v128));
                acc = f32x4_pmax(acc, v128_load(p.add(i + 4) as *const v128));
            }
            i += 8;
        }

        if i < row_len {
            unsafe {
                acc = f32x4_pmax(acc, v128_load(p.add(i) as *const v128));
            }
        }

        f32x4_extract_lane::<0>(acc)
            .max(f32x4_extract_lane::<1>(acc))
            .max(f32x4_extract_lane::<2>(acc))
            .max(f32x4_extract_lane::<3>(acc))
    }

    #[inline(never)]
    fn exp_sum_pass(p: *mut f32, row_len: usize, m: f32) -> f32 {
        use std::arch::wasm32::*;
        let vm = f32x4_splat(m);
        let mut vsum = f32x4_splat(0.0);

        let mut i = 0;
        while i + 8 <= row_len {
            unsafe {
                let y0 = exp_ps(f32x4_sub(v128_load(p.add(i) as *const v128), vm));
                let y1 = exp_ps(f32x4_sub(v128_load(p.add(i + 4) as *const v128), vm));
                v128_store(p.add(i) as *mut v128, y0);
                v128_store(p.add(i + 4) as *mut v128, y1);
                vsum = f32x4_add(vsum, y0);
                vsum = f32x4_add(vsum, y1);
            }
            i += 8;
        }

        if i < row_len {
            unsafe {
                let y = exp_ps(f32x4_sub(v128_load(p.add(i) as *const v128), vm));
                v128_store(p.add(i) as *mut v128, y);
                vsum = f32x4_add(vsum, y);
            }
        }

        f32x4_extract_lane::<0>(vsum)
            + f32x4_extract_lane::<1>(vsum)
            + f32x4_extract_lane::<2>(vsum)
            + f32x4_extract_lane::<3>(vsum)
    }

    #[inline(never)]
    fn scale_pass(p: *mut f32, row_len: usize, r: f32) {
        use std::arch::wasm32::*;
        let vr = f32x4_splat(r);

        let mut i = 0;
        while i + 8 <= row_len {
            unsafe {
                v128_store(
                    p.add(i) as *mut v128,
                    f32x4_mul(v128_load(p.add(i) as *const v128), vr),
                );
                v128_store(
                    p.add(i + 4) as *mut v128,
                    f32x4_mul(v128_load(p.add(i + 4) as *const v128), vr),
                );
            }
            i += 8;
        }

        if i < row_len {
            unsafe {
                v128_store(
                    p.add(i) as *mut v128,
                    f32x4_mul(v128_load(p.add(i) as *const v128), vr),
                );
            }
        }
    }

    let num_rows = buf.len() / row_len;
    if num_rows == 0 {
        return;
    }

    let base_ptr = buf.as_mut_ptr();

    // Rows go through in groups of eight, four, two and one, each pass over the whole group
    // before the next, with the calls spelled out: V8 compiles that markedly faster than a
    // loop over rows or over the group.
    let mut row_idx = 0;
    while row_idx + 8 <= num_rows {
        let p0 = unsafe { base_ptr.add(row_idx * row_len) };
        let p1 = unsafe { base_ptr.add((row_idx + 1) * row_len) };
        let p2 = unsafe { base_ptr.add((row_idx + 2) * row_len) };
        let p3 = unsafe { base_ptr.add((row_idx + 3) * row_len) };
        let p4 = unsafe { base_ptr.add((row_idx + 4) * row_len) };
        let p5 = unsafe { base_ptr.add((row_idx + 5) * row_len) };
        let p6 = unsafe { base_ptr.add((row_idx + 6) * row_len) };
        let p7 = unsafe { base_ptr.add((row_idx + 7) * row_len) };

        let m0 = max_pass(p0, row_len);
        let m1 = max_pass(p1, row_len);
        let m2 = max_pass(p2, row_len);
        let m3 = max_pass(p3, row_len);
        let m4 = max_pass(p4, row_len);
        let m5 = max_pass(p5, row_len);
        let m6 = max_pass(p6, row_len);
        let m7 = max_pass(p7, row_len);

        let sum0 = exp_sum_pass(p0, row_len, m0);
        let sum1 = exp_sum_pass(p1, row_len, m1);
        let sum2 = exp_sum_pass(p2, row_len, m2);
        let sum3 = exp_sum_pass(p3, row_len, m3);
        let sum4 = exp_sum_pass(p4, row_len, m4);
        let sum5 = exp_sum_pass(p5, row_len, m5);
        let sum6 = exp_sum_pass(p6, row_len, m6);
        let sum7 = exp_sum_pass(p7, row_len, m7);

        scale_pass(p0, row_len, sum0.recip());
        scale_pass(p1, row_len, sum1.recip());
        scale_pass(p2, row_len, sum2.recip());
        scale_pass(p3, row_len, sum3.recip());
        scale_pass(p4, row_len, sum4.recip());
        scale_pass(p5, row_len, sum5.recip());
        scale_pass(p6, row_len, sum6.recip());
        scale_pass(p7, row_len, sum7.recip());

        row_idx += 8;
    }

    while row_idx + 4 <= num_rows {
        let p0 = unsafe { base_ptr.add(row_idx * row_len) };
        let p1 = unsafe { base_ptr.add((row_idx + 1) * row_len) };
        let p2 = unsafe { base_ptr.add((row_idx + 2) * row_len) };
        let p3 = unsafe { base_ptr.add((row_idx + 3) * row_len) };

        let m0 = max_pass(p0, row_len);
        let m1 = max_pass(p1, row_len);
        let m2 = max_pass(p2, row_len);
        let m3 = max_pass(p3, row_len);

        let sum0 = exp_sum_pass(p0, row_len, m0);
        let sum1 = exp_sum_pass(p1, row_len, m1);
        let sum2 = exp_sum_pass(p2, row_len, m2);
        let sum3 = exp_sum_pass(p3, row_len, m3);

        scale_pass(p0, row_len, sum0.recip());
        scale_pass(p1, row_len, sum1.recip());
        scale_pass(p2, row_len, sum2.recip());
        scale_pass(p3, row_len, sum3.recip());

        row_idx += 4;
    }

    while row_idx + 2 <= num_rows {
        let p0 = unsafe { base_ptr.add(row_idx * row_len) };
        let p1 = unsafe { base_ptr.add((row_idx + 1) * row_len) };

        let m0 = max_pass(p0, row_len);
        let m1 = max_pass(p1, row_len);

        let sum0 = exp_sum_pass(p0, row_len, m0);
        let sum1 = exp_sum_pass(p1, row_len, m1);

        scale_pass(p0, row_len, sum0.recip());
        scale_pass(p1, row_len, sum1.recip());

        row_idx += 2;
    }

    if row_idx < num_rows {
        let p = unsafe { base_ptr.add(row_idx * row_len) };
        let m = max_pass(p, row_len);
        let sum = exp_sum_pass(p, row_len, m);
        scale_pass(p, row_len, sum.recip());
    }
}

bail_stub!(wasm32; pub fn softmax_rows_f32(&mut [f32], usize));

submit_routine!(wasm32; SoftmaxRowsF32, SoftmaxRows, "wasm_simd128_softmax_rows_f32", softmax_rows_f32);

#[cfg(all(test, target_arch = "wasm32", target_feature = "simd128"))]
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
