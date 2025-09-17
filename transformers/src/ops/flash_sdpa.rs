use tract_nnef::internal::*;
use tract_nnef::tract_ndarray::{s, Array4, ArrayView1, ArrayView4, Ix4};

/// Tract operator wrapper.
#[derive(Clone, Debug, PartialEq)]
pub struct FlashAttnGqaOp {
    pub causal: bool,
    pub block_k: usize,
    pub scale: Option<f32>,
}

impl Op for FlashAttnGqaOp {
    fn name(&self) -> StaticName {
        "FlashAttnGqa".into()
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

impl TypedOp for FlashAttnGqaOp {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(inputs.len() == 3, "FlashAttnGqa expects 3 inputs (Q,K,V)");
        let q = inputs[0];
        let k = inputs[1];
        let v = inputs[2];

        // dtype checks
        ensure!(q.datum_type == f32::datum_type(), "Q must be f32");
        ensure!(k.datum_type == f32::datum_type(), "K must be f32");
        ensure!(v.datum_type == f32::datum_type(), "V must be f32");

        // rank checks
        ensure!(q.rank() == 4, "Q must be rank-4 (B, Hq, Tq, D)");
        ensure!(k.rank() == 4, "K must be rank-4 (B, Hkv, Tk, D)");
        ensure!(v.rank() == 4, "V must be rank-4 (B, Hkv, Tk, D)");

        // Basic shape relations if statically known (best-effort).
        let (bq, hq, _tq, dq) =
            (q.shape[0].clone(), q.shape[1].clone(), q.shape[2].clone(), q.shape[3].clone());
        let (bk, hkv, tk, dk) =
            (k.shape[0].clone(), k.shape[1].clone(), k.shape[2].clone(), k.shape[3].clone());
        let (bv, hkv2, tk2, dv) =
            (v.shape[0].clone(), v.shape[1].clone(), v.shape[2].clone(), v.shape[3].clone());

        ensure!(bq == bk && bq == bv, "Batch dims must match for Q/K/V");
        ensure!(hkv == hkv2, "K/V head counts must match");
        ensure!(tk == tk2, "K/V lengths must match");
        ensure!(dq == dk && dq == dv, "Head dims (D) must match across Q/K/V");

        // If heads are fully known, check GQA divisibility (best-effort).
        if let (Ok(hq), Ok(hkv)) = (hq.to_usize(), hkv.to_usize()) {
            ensure!(hq % hkv == 0, "num_q_heads must be a multiple of num_kv_heads for GQA");
        }

        // Output has same shape and dtype as Q.
        Ok(tvec!(TypedFact::dt_shape(f32::datum_type(), q.shape.clone())))
    }

    as_op!();
}

impl EvalOp for FlashAttnGqaOp {
    fn is_stateless(&self) -> bool {
        true
    }

    fn eval(&self, inputs: TVec<TValue>) -> TractResult<TVec<TValue>> {
        let (q_t, k_t, v_t) = args_3!(inputs);

        let q = q_t.to_array_view::<f32>()?.into_dimensionality::<Ix4>()?;
        let k = k_t.to_array_view::<f32>()?.into_dimensionality::<Ix4>()?;
        let v = v_t.to_array_view::<f32>()?.into_dimensionality::<Ix4>()?;

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

        // Run flash attention
        let out = self.flash_attention_gqa(q4.view(), k4.view(), v4.view());

        Ok(tvec!(out.into_tvalue()))
    }
}

impl FlashAttnGqaOp {
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
    ) -> Array4<f32> {
        // Explicit dimensions
        let (batch_size, num_q_heads, query_len, head_dim) = q.dim();
        let (_, num_kv_heads, kv_len, _) = k.dim();
        let group_size = num_q_heads / num_kv_heads;
        let scale = self.scale.unwrap_or((head_dim as f32).recip().sqrt());
        let block_k = self.block_k.max(1);

        let mut out = Array4::<f32>::zeros((batch_size, num_q_heads, query_len, head_dim));

        for b in 0..batch_size {
            for qh in 0..num_q_heads {
                let kh = qh / group_size;
                let k_bh = k.slice(s![b, kh, .., ..]); // (kv_len, head_dim)
                let v_bh = v.slice(s![b, kh, .., ..]); // (kv_len, head_dim)
                for t_q in 0..query_len {
                    let q_vec = q.slice(s![b, qh, t_q, ..]);

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
                            if self.causal && query_len == kv_len && i_k > t_q {
                                scores.push(f32::NEG_INFINITY);
                            } else {
                                let s = dot1d(q_vec.view(), k_bh.row(i_k).view()) * scale;
                                if s > block_max {
                                    block_max = s;
                                }
                                scores.push(s);
                            }
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
                        out[[b, qh, t_q, d_idx]] = acc[d_idx] * inv_l;
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
