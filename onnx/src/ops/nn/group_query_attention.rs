use crate::model::ParsingContext;
use crate::pb::NodeProto;
use tract_core::ops::change_axes::AxisOp;
use tract_hir::internal::*;
use tract_transformers::ops::sdpa::Sdpa;

// com.microsoft GroupQueryAttention (prefill only).
//   inputs:  query(0), key(1), value(2), past_key(3), past_value(4), seqlens_k(5), total_seq(6)
//   outputs: output(0), present_key(1), present_value(2)
// Scoped to the prefill case (no past KV cache): query/key/value are [B, S, heads*head_size],
// attention is causal with q_seq == kv_seq, and present_key/value are the reshaped K/V.
// Sliding-window attention (local_window_size) is applied as a banded causal mask.
// Decode-step KV cache and internal rotary (do_rotary) are still rejected.
pub fn group_query_attention(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let num_heads: usize = node.get_attr("num_heads")?;
    let kv_num_heads: usize = node.get_attr("kv_num_heads")?;
    let scale = node.get_attr_opt::<f32>("scale")?;
    ensure!(
        node.get_attr_opt::<i64>("do_rotary")?.unwrap_or(0) == 0,
        "GroupQueryAttention: internal rotary (do_rotary) is unsupported; apply RotaryEmbedding separately"
    );
    // Sliding-window attention: a query attends only to the last `window` keys (in
    // addition to causal). <0 / absent means no window (full causal). Applied as a band.
    let window = node.get_attr_opt::<i64>("local_window_size")?.unwrap_or(0).max(0) as usize;
    ensure!(
        node.get_attr_opt::<f32>("softcap")?.unwrap_or(0.0) == 0.0,
        "GroupQueryAttention: softcap is unsupported"
    );
    let have_past = (node.input.len() > 3 && !node.input[3].is_empty())
        || (node.input.len() > 4 && !node.input[4].is_empty());
    ensure!(
        !have_past,
        "GroupQueryAttention: past KV cache (decode step) is unsupported; only prefill is handled"
    );
    Ok((expand(GroupQueryAttention { num_heads, kv_num_heads, scale, window }), vec![]))
}

#[derive(Debug, Clone)]
struct GroupQueryAttention {
    num_heads: usize,
    kv_num_heads: usize,
    scale: Option<f32>,
    /// Sliding-window size; 0 = full causal (no window).
    window: usize,
}

/// Additive attention mask for causal + optional sliding window: entry `(i, j)` is `0`
/// when query `i` may attend to key `j` (`j <= i`, and `i - j < window` when `window > 0`),
/// else `-inf`. `window == 0` is plain causal.
fn windowed_causal_mask(qs: usize, ks: usize, window: usize) -> tract_ndarray::Array2<f32> {
    tract_ndarray::Array2::<f32>::from_shape_fn((qs, ks), |(i, j)| {
        if j <= i && (window == 0 || i - j < window) { 0.0f32 } else { f32::NEG_INFINITY }
    })
}

// [B, S, heads*head_size] -> [B, heads, S, head_size]
fn to_4d(
    model: &mut TypedModel,
    prefix: &str,
    x: OutletId,
    total: TDim,
    heads: usize,
) -> TractResult<OutletId> {
    let head_dim = total.clone() / heads;
    let reshaped = model.wire_node(
        format!("{prefix}.reshape"),
        AxisOp::Reshape(2, tvec![total], tvec![heads.to_dim(), head_dim]),
        &[x],
    )?[0];
    Ok(model.wire_node(format!("{prefix}.transpose"), AxisOp::Move(2, 1), &[reshaped])?[0])
}

impl Expansion for GroupQueryAttention {
    fn name(&self) -> StaticName {
        "GroupQueryAttention".into()
    }

    fn nboutputs(&self) -> TractResult<usize> {
        Ok(3)
    }

    fn rules<'r, 'p: 'r, 's: 'r>(
        &'s self,
        s: &mut Solver<'r>,
        inputs: &'p [TensorProxy],
        outputs: &'p [TensorProxy],
    ) -> InferenceResult {
        check_output_arity(outputs, 3)?;
        s.equals(&inputs[0].datum_type, &outputs[0].datum_type)?;
        s.equals(&inputs[0].shape, &outputs[0].shape)?;
        s.equals(&inputs[0].datum_type, &outputs[1].datum_type)?;
        s.equals(&inputs[0].datum_type, &outputs[2].datum_type)?;
        // present_key / present_value = key/value reshaped to [B, kv_num_heads, S, head_dim].
        let kvh = self.kv_num_heads;
        s.given(&inputs[1].shape, move |s, ks| {
            s.equals(
                &outputs[1].shape,
                tvec![ks[0].clone(), kvh.to_dim(), ks[1].clone(), ks[2].clone() / kvh],
            )
        })?;
        s.given(&inputs[2].shape, move |s, vs| {
            s.equals(
                &outputs[2].shape,
                tvec![vs[0].clone(), kvh.to_dim(), vs[1].clone(), vs[2].clone() / kvh],
            )
        })?;
        Ok(())
    }

    fn wire(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
    ) -> TractResult<TVec<OutletId>> {
        let q_fact = model.outlet_fact(inputs[0])?.clone();
        let dt = q_fact.datum_type;
        ensure!(q_fact.rank() == 3, "GroupQueryAttention: expected 3D query [B, S, hidden]");
        let q_hidden = q_fact.shape[2].clone();
        let k_hidden = model.outlet_fact(inputs[1])?.shape[2].clone();
        let v_hidden = model.outlet_fact(inputs[2])?.shape[2].clone();

        let q4 = to_4d(model, &format!("{prefix}.q"), inputs[0], q_hidden.clone(), self.num_heads)?;
        let k4 = to_4d(model, &format!("{prefix}.k"), inputs[1], k_hidden, self.kv_num_heads)?;
        let v4 = to_4d(model, &format!("{prefix}.v"), inputs[2], v_hidden, self.kv_num_heads)?;

        // Causal (+ optional sliding-window) mask: materialise an explicit additive band
        // for concrete shapes — query i attends to keys j with `j <= i` (causal) and, if a
        // window is set, `i - j < window` (the last `window` keys). Fall back to Sdpa's own
        // is_causal for symbolic shapes (a static window can't be built there). Sdpa handles
        // GQA head grouping (kv heads < q heads).
        let q_seq = model.outlet_fact(q4)?.shape[2].to_usize().ok();
        let kv_seq = model.outlet_fact(k4)?.shape[2].to_usize().ok();
        let window = self.window;
        let (mask, is_causal) = if let (Some(qs), Some(ks)) = (q_seq, kv_seq) {
            let arr = windowed_causal_mask(qs, ks, window);
            let mask_tensor: Tensor = arr.into();
            let mut m = model.add_const(format!("{prefix}.causal_mask"), mask_tensor)?;
            for i in 0..2 {
                m = model.wire_node(
                    format!("{prefix}.mask_unsqueeze_{i}"),
                    AxisOp::Add(0),
                    &[m],
                )?[0];
            }
            (Some(m), false)
        } else {
            ensure!(
                window == 0,
                "GroupQueryAttention: sliding window (local_window_size) requires static \
                 sequence lengths to materialise the banded mask"
            );
            (None, true)
        };
        let mut sdpa_inputs = tvec![q4, k4, v4];
        if let Some(m) = mask {
            sdpa_inputs.push(m);
        }
        let sdpa = Sdpa {
            scale: self.scale.map(tensor0),
            datum_type: dt,
            acc_datum_type: DatumType::F32,
            is_causal,
        };
        let y4 = model.wire_node(format!("{prefix}.sdpa"), sdpa, &sdpa_inputs)?[0];

        // [B, num_heads, S, head_dim] -> [B, S, num_heads, head_dim] -> [B, S, hidden]
        let y_t = model.wire_node(format!("{prefix}.y_transpose"), AxisOp::Move(1, 2), &[y4])?[0];
        let yf = model.outlet_fact(y4)?.clone();
        let (heads_dim, head_dim) = (yf.shape[1].clone(), yf.shape[3].clone());
        let y = model.wire_node(
            format!("{prefix}.y_reshape"),
            AxisOp::Reshape(
                2,
                tvec![heads_dim.clone(), head_dim.clone()],
                tvec![heads_dim * head_dim],
            ),
            &[y_t],
        )?[0];

        Ok(tvec!(y, k4, v4))
    }
}

#[cfg(test)]
mod tests {
    use super::windowed_causal_mask;

    #[test]
    fn band_mask_causal_and_window() {
        // window 3 on a 5x5: query i attends to key j iff causal (j<=i) AND i-j<3
        let m = windowed_causal_mask(5, 5, 3);
        for i in 0..5 {
            for j in 0..5 {
                let want_open = j <= i && i - j < 3;
                assert_eq!(m[(i, j)] == 0.0, want_open, "window=3 at (i={i}, j={j})");
            }
        }
        // window 0 == plain causal
        let c = windowed_causal_mask(4, 4, 0);
        for i in 0..4 {
            for j in 0..4 {
                assert_eq!(c[(i, j)] == 0.0, j <= i, "causal at (i={i}, j={j})");
            }
        }
    }
}
