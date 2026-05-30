use tract_nnef::internal::*;
use tract_nnef::prelude::tract_itertools::Itertools;
use tract_nnef::tract_ndarray::{Array2, Array4, ArrayView2, ArrayView4, ArrayViewMut2, Ix4, s};

/// Tract operator wrapper.
#[derive(Clone, Debug, PartialEq)]
pub struct FlashSdpaOp {
    pub causal: bool,
    pub scale: Option<f32>,
}
impl Eq for FlashSdpaOp {}

impl Op for FlashSdpaOp {
    fn name(&self) -> StaticName {
        "FlashSDPA".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("causal={}, scale={:?}", self.causal, self.scale)])
    }

    op_as_typed_op!();
}

impl TypedOp for FlashSdpaOp {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let (q, k, v, opt_m) = match inputs.len() {
            3 => (inputs[0], inputs[1], inputs[2], None),
            4 => (inputs[0], inputs[1], inputs[2], Some(inputs[3])),
            _ => bail!("FlashSDPA expects 3 or 4 inputs (Q,K,V, optional mask), got {inputs:?}"),
        };

        // dtype checks
        ensure!(q.datum_type.is_float(), "Q must be floating point");
        ensure!(k.datum_type.is_float(), "K must be floating point");
        ensure!(v.datum_type.is_float(), "V must be floating point");
        if let Some(m) = opt_m {
            ensure!(m.datum_type.is_float(), "Mask must be floating point");
        }

        // rank checks
        ensure!(
            q.rank() == k.rank()
                && q.rank() == v.rank()
                && opt_m.as_ref().is_none_or(|m| m.rank() == q.rank()),
            "Q, K, V, mask must have the same rank, found {}, {}, {}, {:?} resp.",
            q.rank(),
            k.rank(),
            v.rank(),
            opt_m.as_ref().map(|m| m.rank().to_string())
        );
        ensure!(
            q.rank() == 3 || q.rank() == 4,
            "Inputs must be of rank 3 or 4, found {}",
            q.rank()
        );

        let one = 1.to_dim();

        let ((bq, hq, _tq, dq), (bk, hkv, tk, dk), (bv, hkv2, tk2, dv)) = if q.rank() == 4 {
            (
                (&q.shape[0], &q.shape[1], &q.shape[2], &q.shape[3]),
                (&k.shape[0], &k.shape[1], &k.shape[2], &k.shape[3]),
                (&v.shape[0], &v.shape[1], &v.shape[2], &v.shape[3]),
            )
        } else {
            (
                (&q.shape[0], &one, &q.shape[1], &q.shape[2]),
                (&k.shape[0], &one, &k.shape[1], &k.shape[2]),
                (&v.shape[0], &one, &v.shape[1], &v.shape[2]),
            )
        };

        ensure!(bq == bk && bq == bv, "Batch dims must match for Q/K/V");
        ensure!(hkv == hkv2, "K/V head counts must match");
        ensure!(tk == tk2, "K/V lengths must match");
        ensure!(dq == dk && dq == dv, "Head dims (D) must match across Q/K/V");

        // If heads are fully known, check GQA divisibility (best-effort).
        if let (Ok(hq), Ok(hkv)) = (hq.to_usize(), hkv.to_usize()) {
            ensure!(hq % hkv == 0, "num_q_heads must be a multiple of num_kv_heads for GQA");
        }

        // Output has same shape and dtype as Q.
        Ok(tvec!(q.without_value()))
    }

    as_op!();
}

impl EvalOp for FlashSdpaOp {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        ensure!(inputs.len() == 3 || inputs.len() == 4);
        let [q, k, v] = &inputs[0..3] else {
            bail!("Expects 3 or 4 inptus (Q, K, V, optional mask)")
        };

        let input_dt = q.datum_type();

        let (q, k, v) = (q.cast_to::<f32>()?, k.cast_to::<f32>()?, v.cast_to::<f32>()?);
        let mut q = q.to_plain_array_view::<f32>()?;
        let mut k = k.to_plain_array_view::<f32>()?;
        let mut v = v.to_plain_array_view::<f32>()?;

        let is_3d_case = q.ndim() == 3;

        if is_3d_case {
            q.insert_axis_inplace(tract_ndarray::Axis(1));
            k.insert_axis_inplace(tract_ndarray::Axis(1));
            v.insert_axis_inplace(tract_ndarray::Axis(1));
        }

        let (q, k, v) = (
            q.into_dimensionality::<Ix4>()?,
            k.into_dimensionality::<Ix4>()?,
            v.into_dimensionality::<Ix4>()?,
        );

        let (batch_size, num_q_heads, query_len, head_dim) =
            (q.shape()[0], q.shape()[1], q.shape()[2], q.shape()[3]);
        let (bk, num_kv_heads, kv_len, hd_k) =
            (k.shape()[0], k.shape()[1], k.shape()[2], k.shape()[3]);

        ensure!(batch_size == bk, "Batch mismatch between Q and K");
        ensure!(head_dim == hd_k, "Head dim mismatch between Q and K");
        ensure!(num_kv_heads == v.shape()[1], "K/V head mismatch");
        ensure!(kv_len == v.shape()[2] && head_dim == v.shape()[3], "K/V shape mismatch");

        // Views as fixed 4-D
        let q4 = q.to_shape((batch_size, num_q_heads, query_len, head_dim))?;
        let k4 = k.to_shape((batch_size, num_kv_heads, kv_len, head_dim))?;
        let v4 = v.to_shape((batch_size, num_kv_heads, kv_len, head_dim))?;

        let m = if let Some(m) = inputs.get(3) {
            Some(
                m.cast_to::<f32>()?.into_owned().into_plain_array::<f32>()?.into_shape_with_order(
                    (m.shape()[0], if m.rank() == 3 { 1 } else { m.shape()[1] }, query_len, kv_len),
                )?,
            )
        } else {
            None
        };

        let mut out = self
            .flash_attention_gqa(q4.view(), k4.view(), v4.view(), m.as_ref().map(|m| m.view()))
            .into_dyn();
        if is_3d_case {
            out.index_axis_inplace(tract_ndarray::Axis(1), 0);
        }

        Ok(tvec!(out.into_tensor().cast_to_dt(input_dt)?.into_owned().into_tvalue()))
    }
}

impl FlashSdpaOp {
    /// Flash Attention forward with Grouped-Query Attention (GQA).
    ///
    /// Shapes:
    ///   Q: (batch_size, num_q_heads, query_len, head_dim)
    ///   K: (batch_size, num_kv_heads, kv_len,   head_dim)
    ///   V: (batch_size, num_kv_heads, kv_len,   head_dim)
    /// Returns O: (batch_size, num_q_heads, query_len, head_dim)
    ///
    /// GQA mapping: each query head qh maps to a key/value head index kh = qh / group_size
    /// where group_size = num_q_heads / num_kv_heads (must divide exactly).
    pub fn flash_attention_gqa(
        &self,
        q: ArrayView4<f32>,
        k: ArrayView4<f32>,
        v: ArrayView4<f32>,
        mask: Option<ArrayView4<f32>>,
    ) -> Array4<f32> {
        self.flash_attention_gqa_impl(q, k, v, mask, false)
    }

    fn flash_attention_gqa_impl(
        &self,
        q: ArrayView4<f32>,
        k: ArrayView4<f32>,
        v: ArrayView4<f32>,
        mask: Option<ArrayView4<f32>>,
        pv_strided: bool,
    ) -> Array4<f32> {
        let (batch_size, num_q_heads, q_len, head_dim) = q.dim();
        let (_, num_kv_heads, _, _) = k.dim();
        let group_size = num_q_heads / num_kv_heads;

        let mut out = Array4::<f32>::zeros((batch_size, num_q_heads, q_len, head_dim));

        // One independent task per (batch, q-head). Heads share only read-only Q/K/V
        // and write disjoint output slices, so this is embarrassingly parallel.
        let tasks: Vec<(usize, usize)> =
            (0..batch_size).flat_map(|b| (0..num_q_heads).map(move |qh| (b, qh))).collect();
        let compute = |&(b, qh): &(usize, usize)| {
            self.attend_one_head(b, qh, qh / group_size, q, k, v, mask.as_ref(), pv_strided)
        };

        let results: Vec<Array2<f32>> = if Self::should_thread(tasks.len()) {
            #[cfg(not(target_family = "wasm"))]
            {
                use rayon::prelude::*;
                tasks.par_iter().map(compute).collect()
            }
            #[cfg(target_family = "wasm")]
            {
                tasks.iter().map(compute).collect()
            }
        } else {
            tasks.iter().map(compute).collect()
        };

        for (&(b, qh), head_out) in tasks.iter().zip(results) {
            out.slice_mut(s!(b, qh, .., ..)).assign(&head_out);
        }
        out
    }

    /// Whether to spread the per-head tasks across cores (rayon's global pool).
    /// Off for a single task, on wasm (no thread spawn), or when `TRACT_FLASH_SDPA_ST` is set.
    fn should_thread(num_tasks: usize) -> bool {
        num_tasks > 1
            && cfg!(not(target_family = "wasm"))
            && std::env::var_os("TRACT_FLASH_SDPA_ST").is_none()
    }

    /// Flash-attention for a single (batch, q-head) pair. Returns O of shape
    /// `[q_len, head_dim]`. Pure function of the read-only Q/K/V/mask views, so it
    /// is safe to call concurrently for distinct heads.
    #[allow(clippy::too_many_arguments)]
    fn attend_one_head(
        &self,
        b: usize,
        qh: usize,
        kvh: usize,
        q: ArrayView4<f32>,
        k: ArrayView4<f32>,
        v: ArrayView4<f32>,
        mask: Option<&ArrayView4<f32>>,
        pv_strided: bool,
    ) -> Array2<f32> {
        let (_, _, q_len, head_dim) = q.dim();
        let kv_len = k.dim().2;
        let scale = self.scale.unwrap_or((head_dim as f32).recip().sqrt());

        let block_kv_len = 32;
        let block_q_len = 32;

        let mb = b.min(mask.map(|m| m.shape()[0] - 1).unwrap_or(0));
        let mh = qh.min(mask.map(|m| m.shape()[1] - 1).unwrap_or(0));

        let mut out = Array2::<f32>::zeros((q_len, head_dim));
        let mut l = vec![0f32; q_len];
        let mut m = vec![f32::NEG_INFINITY; q_len];
        for kbix in 0..kv_len.div_ceil(block_kv_len) {
            for qbix in 0..q_len.div_ceil(block_q_len) {
                let kv_range = (kbix * block_kv_len)..((kbix + 1) * block_kv_len).min(kv_len);
                let q_range = (qbix * block_q_len)..((qbix + 1) * block_q_len).min(q_len);
                if let Some(mask) = mask {
                    if mask
                        .slice(s!(mb, mh, q_range.clone(), kv_range.clone()))
                        .iter()
                        .all(|x| *x < -65503.0)
                    {
                        continue;
                    }
                }
                let m = &mut m[q_range.clone()];
                let l = &mut l[q_range.clone()];
                let qblock: ArrayView2<f32> = q.slice(s!(b, qh, q_range.clone(), ..));
                let kblock: ArrayView2<f32> = k.slice(s!(b, kvh, kv_range.clone(), ..));
                let vblock: ArrayView2<f32> = v.slice(s!(b, kvh, kv_range.clone(), ..));
                let mut oblock: ArrayViewMut2<f32> = out.slice_mut(s!(q_range.clone(), ..));
                // Sij <- QiKTj
                let mut s = qblock.dot(&kblock.t()) * scale;
                if let Some(mask) = mask {
                    s += &mask.slice(s!(mb, mh, q_range.clone(), kv_range.clone()));
                } else if self.causal {
                    let mask =
                        Array2::from_elem((q_range.len(), kv_range.len()), f32::NEG_INFINITY);
                    let mask = mask.triu(
                        q_range.start as isize - kv_range.start as isize - q_len as isize
                            + kv_len as isize
                            + 1,
                    );
                    s += &mask;
                };
                let tile_m: Vec<f32> = s
                    .rows()
                    .into_iter()
                    .map(|row| row.iter().copied().map(float_ord::FloatOrd).max().unwrap().0)
                    .collect_vec();
                for (row_ix, max) in tile_m.iter().enumerate() {
                    if max.is_finite() {
                        s.row_mut(row_ix).iter_mut().for_each(|x| *x -= max);
                    }
                }
                // Sij <- exp(Sij * scale - max_of_row)
                s.mapv_inplace(f32::exp);
                let tile_l = s.sum_axis(tract_ndarray::Axis(1)).insert_axis(tract_ndarray::Axis(1));
                // m_new = max(maxes, row_maxs)
                let m_new = (0..q_range.len()).map(|i| m[i].max(tile_m[i])).collect_vec();
                // l_new = exp(m[i] - m_new[i]) * l[i] + exp(tile_m[i] - m_new[i]) * tile_l[i]
                let l_new = (0..q_range.len())
                    .map(|i| {
                        (m[i] - m_new[i]).exp() * l[i]
                            + (tile_m[i] - m_new[i]).exp() * tile_l[(i, 0)]
                    })
                    .collect_vec();
                // P·V as one contiguous tile GEMM (s: [q,kv] · vblock: [kv,d]) instead of
                // `head_dim` strided column dots; then rescale each row with the online-softmax
                // factors. `pv_strided` keeps the old per-column path for A/B benching only.
                let sv_tile = if pv_strided { None } else { Some(s.dot(&vblock)) };
                for i in 0..q_range.len() {
                    let r_l_new = l_new[i].recip();
                    let mul_o = ((m[i] - m_new[i]).exp()) * l[i] * r_l_new;
                    let mul_sv = ((tile_m[i] - m_new[i]).exp()) * r_l_new;
                    if let Some(sv_tile) = &sv_tile {
                        for j in 0..head_dim {
                            oblock[(i, j)] = oblock[(i, j)] * mul_o + sv_tile[(i, j)] * mul_sv;
                        }
                    } else {
                        for j in 0..head_dim {
                            let sv = s.row(i).dot(&vblock.column(j));
                            oblock[(i, j)] = oblock[(i, j)] * mul_o + sv * mul_sv;
                        }
                    }
                }
                l.copy_from_slice(&l_new);
                m.copy_from_slice(&m_new);
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tract_nnef::tract_ndarray::Array4;

    fn rng(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
            })
            .collect()
    }

    #[test]
    fn flash_sdpa_pv_matches_naive() -> TractResult<()> {
        let (b, h, sq, sk, d) = (1usize, 2, 16, 24, 8); // non-square, multi-head
        let scale = 1.0f32 / (d as f32).sqrt();
        let qv = rng(b * h * sq * d, 1);
        let kv = rng(b * h * sk * d, 2);
        let vv = rng(b * h * sk * d, 3);
        let q = Array4::from_shape_vec((b, h, sq, d), qv.clone())?;
        let k = Array4::from_shape_vec((b, h, sk, d), kv.clone())?;
        let v = Array4::from_shape_vec((b, h, sk, d), vv.clone())?;
        let op = FlashSdpaOp { causal: false, scale: None };
        let got = op.flash_attention_gqa(q.view(), k.view(), v.view(), None);
        let gv: Vec<f32> = got.iter().copied().collect();
        let mut want = vec![0f32; b * h * sq * d];
        for bh in 0..b * h {
            for i in 0..sq {
                let mut sc = vec![0f32; sk];
                for j in 0..sk {
                    let mut a = 0f32;
                    for dd in 0..d {
                        a += qv[(bh * sq + i) * d + dd] * kv[(bh * sk + j) * d + dd];
                    }
                    sc[j] = a * scale;
                }
                let m = sc.iter().copied().fold(f32::MIN, f32::max);
                let mut sum = 0f32;
                for x in sc.iter_mut() {
                    *x = (*x - m).exp();
                    sum += *x;
                }
                for x in sc.iter_mut() {
                    *x /= sum;
                }
                for e in 0..d {
                    let mut a = 0f32;
                    for j in 0..sk {
                        a += sc[j] * vv[(bh * sk + j) * d + e];
                    }
                    want[(bh * sq + i) * d + e] = a;
                }
            }
        }
        let max_abs = gv.iter().zip(want.iter()).map(|(&g, &w)| (g - w).abs()).fold(0f32, f32::max);
        println!("flash_sdpa PV-gemm max_abs={max_abs:.6}");
        ensure!(max_abs < 1e-5, "flash_sdpa PV mismatch: {max_abs}");
        Ok(())
    }

    // A/B the P·V step (tile GEMM vs strided per-column dot) across the microbench
    // shape AND the real Llama-3.2-1B PP512 prefill shape (causal, GQA 32q/8kv).
    //   cargo test -p tract-transformers --release bench_flash_sdpa_pv -- --ignored --nocapture
    #[test]
    #[ignore]
    fn bench_flash_sdpa_pv() -> TractResult<()> {
        use std::time::Instant;
        // (label, hq, hkv, sq, sk, d, causal)
        let shapes = [
            (
                "microbench    non-causal Hq8  Hkv8 S512 D64",
                8usize,
                8usize,
                512usize,
                512usize,
                64usize,
                false,
            ),
            ("llama-3.2-1B  causal     Hq32 Hkv8 S512 D64", 32, 8, 512, 512, 64, true),
        ];
        for (label, hq, hkv, sq, sk, d, causal) in shapes {
            let q = Array4::from_shape_vec((1, hq, sq, d), rng(hq * sq * d, 1))?;
            let k = Array4::from_shape_vec((1, hkv, sk, d), rng(hkv * sk * d, 2))?;
            let v = Array4::from_shape_vec((1, hkv, sk, d), rng(hkv * sk * d, 3))?;
            let op = FlashSdpaOp { causal, scale: None };
            let bench = |strided: bool| -> f64 {
                for _ in 0..2 {
                    std::hint::black_box(op.flash_attention_gqa_impl(
                        q.view(),
                        k.view(),
                        v.view(),
                        None,
                        strided,
                    ));
                }
                let mut best = f64::MAX;
                for _ in 0..8 {
                    let t = Instant::now();
                    for _ in 0..3 {
                        std::hint::black_box(op.flash_attention_gqa_impl(
                            q.view(),
                            k.view(),
                            v.view(),
                            None,
                            strided,
                        ));
                    }
                    best = best.min(t.elapsed().as_secs_f64() / 3.0);
                }
                best * 1e3
            };
            let strided = bench(true);
            let gemm = bench(false);
            println!(
                "{label}: strided {strided:7.3} ms -> gemm {gemm:7.3} ms  ({:.2}x)  [per-layer; x16 layers = {:.1} -> {:.1} ms total SDPA]",
                strided / gemm,
                strided * 16.0,
                gemm * 16.0,
            );
        }
        Ok(())
    }
}
