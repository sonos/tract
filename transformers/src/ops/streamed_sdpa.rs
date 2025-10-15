use tract_nnef::internal::*;
use tract_nnef::tract_ndarray::{s, Array4, ArrayView1, ArrayView2, ArrayView4, Ix4};

/// Tract operator wrapper.
#[derive(Clone, Debug, PartialEq)]
pub struct StreamedSdpaOp {
    pub causal: bool,
    pub block_k: usize,
    pub scale: Option<f32>,
}

impl Op for StreamedSdpaOp {
    fn name(&self) -> StaticName {
        "StreamedSDPA".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!(
            "causal={}, block_k={}, scale={:?}",
            self.causal, self.block_k, self.scale
        )])
    }

    fn same_as(&self, other: &dyn Op) -> bool {
        other.downcast_ref::<Self>().is_some_and(|o| o == self)
    }

    op_as_typed_op!();
}

impl TypedOp for StreamedSdpaOp {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        let (q, k, v, opt_m) = match inputs.len() {
            3 => (inputs[0], inputs[1], inputs[2], None),
            4 => (inputs[0], inputs[1], inputs[2], Some(inputs[3])),
            _ => bail!("StreamSDPA expects 3 or 4 inputs (Q,K,V, optional mask), got {inputs:?}"),
        };

        // dtype checks
        ensure!(q.datum_type.is_float(), "Q must be floating point");
        ensure!(k.datum_type.is_float(), "K must be floating point");
        ensure!(v.datum_type.is_float(), "V must be floating point");
        if let Some(m) = opt_m {
            ensure!(m.datum_type.is_float(), "M must be floating point");
        }

        // rank checks
        ensure!(
            q.rank() == k.rank() && q.rank() == v.rank(),
            "Q, K and V must have the same rank, found {}, {}, {} resp.",
            q.rank(),
            k.rank(),
            v.rank()
        );
        ensure!(
            q.rank() == 3 || q.rank() == 4,
            "Inputs must be of rank 3 or 4, found {}",
            q.rank()
        );

        if let Some(m) = opt_m {
            ensure!(m.rank() == 2, "Mask must be of rank 2, found mask {:?}", m,);
        }

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

impl EvalOp for StreamedSdpaOp {
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
        let mut q = q.to_array_view::<f32>()?;
        let mut k = k.to_array_view::<f32>()?;
        let mut v = v.to_array_view::<f32>()?;

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
                m.cast_to::<f32>()?
                    .into_owned()
                    .into_array::<f32>()?
                    .into_shape_with_order((query_len, kv_len))?,
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

impl StreamedSdpaOp {
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
        mask: Option<ArrayView2<f32>>,
    ) -> Array4<f32> {
        // Explicit dimensions
        let (batch_size, num_q_heads, query_len, head_dim) = q.dim();
        let (_, num_kv_heads, kv_len, _) = k.dim();
        let scale = self.scale.unwrap_or((head_dim as f32).recip().sqrt());
        let block_k = self.block_k.max(1);

        let mut out = Array4::<f32>::zeros((batch_size, num_q_heads, query_len, head_dim));
        let group_size = num_q_heads / num_kv_heads;

        for b in 0..batch_size {
            for kh in 0..num_kv_heads {
                let k_bh = k.slice(s![b, kh, .., ..]); // (kv_len, head_dim)
                let v_bh = v.slice(s![b, kh, .., ..]); // (kv_len, head_dim)
                for inner_qh in 0..group_size {
                    let qh_ix = kh * group_size + inner_qh;
                    let qh = q.slice(s![b, qh_ix, .., ..]);

                    for t_q in 0..query_len {
                        let q_vec = qh.slice(s![t_q, ..]);

                        // Streaming softmax state
                        let mut m = f32::NEG_INFINITY; // running max
                        let mut l = 0.0f32; // running sum of exp(scores - m)
                        let mut acc = vec![0.0f32; head_dim];

                        // Process in KV tiles
                        let mut kb = 0usize;
                        while kb < kv_len {
                            let kend = (kb + block_k).min(kv_len);

                            let mut block_max = f32::NEG_INFINITY;
                            let mut scores: Vec<f32> = Vec::with_capacity(kend - kb);
                            for i_k in kb..kend {
                                let mask = if let Some(mask) = mask {
                                    mask[(t_q, i_k)]
                                } else if self.causal && i_k > t_q {
                                    f32::NEG_INFINITY
                                } else {
                                    0.0f32
                                };
                                let s = (dot1d(q_vec.view(), k_bh.row(i_k).view()) * scale) + mask;
                                if s > block_max {
                                    block_max = s;
                                }
                                scores.push(s);
                            }

                            if !block_max.is_finite() {
                                kb = kend;
                                continue;
                            }

                            let new_m = if m > block_max { m } else { block_max };
                            let alpha = if m.is_finite() { (m - new_m).exp() } else { 0.0 };

                            if alpha != 1.0 {
                                acc.iter_mut().for_each(|a| *a *= alpha);
                                l *= alpha;
                            }

                            let mut block_l = 0.0f32;
                            for (off, i_k) in (kb..kend).enumerate() {
                                let s = scores[off];
                                if !s.is_finite() {
                                    continue;
                                }
                                let w = (s - new_m).exp();
                                block_l += w;
                                let v_row = v_bh.row(i_k);
                                for d_idx in 0..head_dim {
                                    acc[d_idx] += w * v_row[d_idx];
                                }
                            }

                            l += block_l;
                            m = new_m;
                            kb = kend;
                        }

                        let inv_l = if l > 0.0 { 1.0 / l } else { 0.0 };
                        for d_idx in 0..head_dim {
                            out[[b, qh_ix, t_q, d_idx]] = acc[d_idx] * inv_l;
                        }
                    }
                }
            }
        }

        out
    }
}

#[inline]
fn dot1d(a: ArrayView1<f32>, b: ArrayView1<f32>) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut s = 0.0f32;
    for i in 0..a.len() {
        s += a[i] * b[i];
    }
    s
}
