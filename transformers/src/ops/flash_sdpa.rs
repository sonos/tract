use tract_nnef::internal::*;
use tract_nnef::prelude::tract_itertools::Itertools;
use tract_nnef::tract_ndarray::{s, Array2, Array4, ArrayView2, ArrayView4, ArrayViewMut2, Ix4};

/// Tract operator wrapper.
#[derive(Clone, Debug, PartialEq)]
pub struct FlashSdpaOp {
    pub causal: bool,
    pub scale: Option<f32>,
}

impl Op for FlashSdpaOp {
    fn name(&self) -> StaticName {
        "FlashSDPA".into()
    }

    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("causal={}, scale={:?}", self.causal, self.scale)])
    }

    fn same_as(&self, other: &dyn Op) -> bool {
        other.downcast_ref::<Self>().is_some_and(|o| o == self)
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
        mask: Option<ArrayView2<f32>>,
    ) -> Array4<f32> {
        // Explicit dimensions
        let (batch_size, num_q_heads, q_len, head_dim) = q.dim();
        let (_, num_kv_heads, kv_len, _) = k.dim();
        let scale = self.scale.unwrap_or((head_dim as f32).recip().sqrt());
        let group_size = num_q_heads / num_kv_heads;

        let block_kv_len = 32;
        let block_q_len = 32;

        let mut out = Array4::<f32>::zeros((batch_size, num_q_heads, q_len, head_dim));

        for b in 0..batch_size {
            for kvh in 0..num_kv_heads {
                for g in 0..group_size {
                    let qh = kvh * group_size + g;
                    let mut l = vec![0f32; q_len];
                    let mut m = vec![f32::NEG_INFINITY; q_len];
                    for kbix in 0..kv_len.div_ceil(block_kv_len) {
                        for qbix in 0..q_len.div_ceil(block_q_len) {
                            let kv_range =
                                (kbix * block_kv_len)..((kbix + 1) * block_kv_len).min(kv_len);
                            let q_range =
                                (qbix * block_q_len)..((qbix + 1) * block_q_len).min(q_len);
                            if let Some(mask) = &mask {
                                if mask
                                    .slice(s!(q_range.clone(), kv_range.clone()))
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
                            let mut oblock: ArrayViewMut2<f32> =
                                out.slice_mut(s!(b, qh, q_range.clone(), ..));
                            // Sij <- QiKTj
                            let mut s = qblock.dot(&kblock.t()) * scale;
                            if let Some(mask) = &mask {
                                s += &mask.slice(s!(q_range.clone(), kv_range.clone()));
                            } else if self.causal {
                                let mask = Array2::from_elem(
                                    (q_range.len(), kv_range.len()),
                                    f32::NEG_INFINITY,
                                );
                                let mask = mask.triu(
                                    q_range.start as isize
                                        - kv_range.start as isize
                                        - q_len as isize
                                        + kv_len as isize
                                        + 1,
                                );
                                s += &mask;
                            };
                            let tile_m: Vec<f32> = s
                                .rows()
                                .into_iter()
                                .map(|row| {
                                    row.iter().copied().map(float_ord::FloatOrd).max().unwrap().0
                                })
                                .collect_vec();
                            for (row_ix, max) in tile_m.iter().enumerate() {
                                if max.is_finite() {
                                    s.row_mut(row_ix).iter_mut().for_each(|x| *x -= max);
                                }
                            }
                            // Sij <- exp(Sij * scale - max_of_row)
                            s.mapv_inplace(f32::exp);
                            let tile_l = s
                                .sum_axis(tract_ndarray::Axis(1))
                                .insert_axis(tract_ndarray::Axis(1));
                            // m_new = max(maxes, row_maxs)
                            let m_new =
                                (0..q_range.len()).map(|i| m[i].max(tile_m[i])).collect_vec();
                            // l_new = exp(m[i] - m_new[i]) * l[i] - exp(tile_m[i] - m_new[i]) * tile_l[i]
                            let l_new = (0..q_range.len())
                                .map(|i| {
                                    (m[i] - m_new[i]).exp() * l[i]
                                        + (tile_m[i] - m_new[i]).exp() * tile_l[(i, 0)]
                                })
                                .collect_vec();
                            for i in 0..q_range.len() {
                                let r_l_new = l_new[i].recip();
                                let mul_o = ((m[i] - m_new[i]).exp()) * l[i] * r_l_new;
                                let mul_sv = ((tile_m[i] - m_new[i]).exp()) * r_l_new;
                                for j in 0..head_dim {
                                    let sv = s.row(i).dot(&vblock.column(j));
                                    oblock[(i, j)] = oblock[(i, j)] * mul_o + sv * mul_sv;
                                }
                            }
                            l.copy_from_slice(&l_new);
                            m.copy_from_slice(&m_new);
                        }
                    }
                }
            }
        }
        out
    }
}
