use crate::model::ParsingContext;
use crate::pb::NodeProto;
use tract_core::ops::array::{Gather, Slice, TypedConcat};
use tract_core::ops::cast::cast;
use tract_core::ops::change_axes::AxisOp;
use tract_core::ops::math::add;
use tract_hir::internal::*;
use tract_hir::ops::logic::wire_with_rank_broadcast;
use tract_transformers::ops::apply_rope::ApplyRope;
use tract_transformers::ops::sdpa::{Sdpa, SdpaMaskMode, wire_attention_mask};

/// com.microsoft GroupQueryAttention.
///
/// Query, key and value arrive as `[B, S, heads * head_size]`; the KV cache is BNSH
/// (`[B, kv_num_heads, seq, head_size]`). Past key/value are concatenated onto the new
/// step and returned as `present_key` / `present_value`, so both prefill and decode go
/// through the same wiring. Attention is causal by default, aligned to the *end* of the
/// key sequence: query `i` sits at absolute position `kv_len - q_len + i`.
///
/// `do_rotary` applies RoPE to Q and K before they are cached, reading `cos_cache` /
/// `sin_cache` of shape `[max_seq, head_size / 2]` at positions `0..S` for a first prompt
/// and at `seqlens_k` (`total_sequence_length - 1`) for a generated token.
///
/// Only those two step kinds are handled. A multi-token step against a non-empty cache is
/// the subsequent-prompt case, where past and present alias one buffer that the op fills
/// from index 0 instead of appending to; it is rejected rather than approximated by a
/// concatenation. Also rejected: packed QKV, interleaved rotary, partial rotary, softcap,
/// smooth softmax, head sink, quantized KV cache, fused Q/K norm, windowed cache buffers,
/// and the `position_ids` input (whose meaning changes with the step kind).
pub fn group_query_attention(
    _ctx: &ParsingContext,
    node: &NodeProto,
) -> TractResult<(Box<dyn InferenceOp>, Vec<String>)> {
    let num_heads: usize = node.get_attr("num_heads")?;
    let kv_num_heads: usize = node.get_attr("kv_num_heads")?;
    let scale = node.get_attr_opt::<f32>("scale")?;
    let causal = node.get_attr_opt::<i64>("causal")?.unwrap_or(1) != 0;
    let do_rotary = node.get_attr_opt::<i64>("do_rotary")?.unwrap_or(0) != 0;
    ensure!(
        !do_rotary || node.get_attr_opt::<i64>("rotary_interleaved")?.unwrap_or(0) == 0,
        "GroupQueryAttention: rotary_interleaved=1 is unsupported"
    );
    // <0 / absent means no window (full causal); the mask is banded when set.
    let window = node.get_attr_opt::<i64>("local_window_size")?.unwrap_or(0).max(0) as usize;
    ensure!(
        node.get_attr_opt::<f32>("softcap")?.unwrap_or(0.0) == 0.0,
        "GroupQueryAttention: softcap is unsupported"
    );
    ensure!(
        node.get_attr_opt::<i64>("smooth_softmax")?.unwrap_or(0) == 0,
        "GroupQueryAttention: smooth_softmax is unsupported"
    );
    ensure!(
        node.get_attr_opt::<i64>("qk_output")?.unwrap_or(0) == 0,
        "GroupQueryAttention: qk_output is unsupported"
    );
    ensure!(
        node.get_attr_opt::<i64>("sliding_window_cache")?.unwrap_or(0) == 0,
        "GroupQueryAttention: sliding_window_cache is unsupported"
    );
    for quant in ["k_quant_type", "v_quant_type"] {
        let mode = node.get_attr_opt::<String>(quant)?.unwrap_or_else(|| "NONE".to_string());
        ensure!(
            mode == "NONE",
            "GroupQueryAttention: {quant}={mode} (quantized KV) is unsupported"
        );
    }

    let mut opt = crate::model::optional_inputs(node);
    let _query = opt.next().unwrap();
    let key = opt.next().unwrap();
    let value = opt.next().unwrap();
    ensure!(
        key.is_some() && value.is_some(),
        "GroupQueryAttention: packed QKV (absent key/value) is unsupported"
    );
    let past_key = opt.next().unwrap();
    let past_value = opt.next().unwrap();
    let seqlens_k = opt.next().unwrap();
    let _total_sequence_length = opt.next().unwrap();
    let cos_cache = opt.next().unwrap();
    let sin_cache = opt.next().unwrap();
    let position_ids = opt.next().unwrap();
    let attention_bias = opt.next().unwrap();
    for (name, input) in [
        ("position_ids", position_ids),
        ("head_sink", opt.next().unwrap()),
        ("k_scale", opt.next().unwrap()),
        ("v_scale", opt.next().unwrap()),
        ("q_norm_weight", opt.next().unwrap()),
        ("k_norm_weight", opt.next().unwrap()),
    ] {
        ensure!(input.is_none(), "GroupQueryAttention: the {name} input is unsupported");
    }

    let past = match (past_key, past_value) {
        (Some(k), Some(v)) => Some((k, v)),
        (None, None) => None,
        _ => bail!("GroupQueryAttention: past_key and past_value must both be set or both absent"),
    };
    let rotary = match (do_rotary, cos_cache, sin_cache) {
        (true, Some(c), Some(s)) => Some((c, s)),
        (true, _, _) => {
            bail!("GroupQueryAttention: do_rotary=1 requires the cos_cache and sin_cache inputs")
        }
        (false, _, _) => None,
    };

    Ok((
        expand(GroupQueryAttention {
            num_heads,
            kv_num_heads,
            scale,
            window,
            causal,
            past,
            seqlens_k,
            rotary,
            attention_bias,
        }),
        vec![],
    ))
}

#[derive(Debug, Clone)]
struct GroupQueryAttention {
    num_heads: usize,
    kv_num_heads: usize,
    scale: Option<f32>,
    /// Sliding-window size; 0 = full causal (no window).
    window: usize,
    causal: bool,
    past: Option<(usize, usize)>,
    seqlens_k: Option<usize>,
    rotary: Option<(usize, usize)>,
    attention_bias: Option<usize>,
}

/// Additive attention mask for causal + optional sliding window over a key sequence that
/// may already hold `ks - qs` cached tokens: query `i` sits at absolute position
/// `(ks - qs) + i` and attends key `j` when `j <= i` (causal) and, with a window set,
/// `i - j < window`. `window == 0` is plain causal.
fn windowed_causal_mask(qs: usize, ks: usize, window: usize) -> tract_ndarray::Array2<f32> {
    let past = ks - qs;
    tract_ndarray::Array2::<f32>::from_shape_fn((qs, ks), |(i, j)| {
        let i = i + past;
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

impl GroupQueryAttention {
    /// Reads `cos_cache` / `sin_cache` at this step's positions and widens both halves to
    /// the full head size, giving the `[B or 1, 1, S, head_size]` operands `ApplyRope`
    /// multiplies against a BNSH tensor.
    fn wire_rope_tables(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
        q_seq: &TDim,
        head_dim: &TDim,
        first_prompt: bool,
    ) -> TractResult<(OutletId, OutletId)> {
        let (cos, sin) = self.rotary.unwrap();
        let half = model.outlet_fact(inputs[cos])?.shape[1].clone();
        ensure!(
            half.clone() * 2 == *head_dim,
            "GroupQueryAttention: partial rotary is unsupported; cos_cache is {half} wide \
             for a head size of {head_dim}"
        );

        let mut prep = |tag: &str, cache: OutletId| -> TractResult<OutletId> {
            let gathered = if first_prompt {
                // Positions are 0..S, so slicing the cache is the gather.
                let s = model.wire_node(
                    format!("{prefix}.{tag}_slice"),
                    Slice::new(0, 0.to_dim(), q_seq.clone()),
                    &[cache],
                )?[0];
                let b = model.wire_node(format!("{prefix}.{tag}_batch"), AxisOp::Add(0), &[s])?[0];
                model.wire_node(format!("{prefix}.{tag}_heads"), AxisOp::Add(0), &[b])?[0]
            } else {
                // Token generation: the new token sits at seqlens_k, which the op defines as
                // total_sequence_length - 1.
                let seqlens = self.seqlens_k.context(
                    "GroupQueryAttention: do_rotary with a past KV cache needs the seqlens_k input",
                )?;
                let positions = model.wire_node(
                    format!("{prefix}.{tag}_positions"),
                    cast(DatumType::I64),
                    &[inputs[seqlens]],
                )?[0];
                let g = model.wire_node(
                    format!("{prefix}.{tag}_gather"),
                    Gather::new(0),
                    &[cache, positions],
                )?[0];
                // seqlens_k is [B] or [B, 1]; pad whichever it is up to [B, 1, 1, half].
                let mut g = g;
                while model.outlet_fact(g)?.rank() < 4 {
                    let rank = model.outlet_fact(g)?.rank();
                    g = model.wire_node(
                        format!("{prefix}.{tag}_pad_{rank}"),
                        AxisOp::Add(rank - 1),
                        &[g],
                    )?[0];
                }
                g
            };
            Ok(model.wire_node(
                format!("{prefix}.{tag}_halves"),
                TypedConcat::new(3),
                &[gathered, gathered],
            )?[0])
        };
        Ok((prep("cos", inputs[cos])?, prep("sin", inputs[sin])?))
    }

    /// Causal / sliding-window band, plus the `attention_bias` input when present. Returns
    /// `None` when `Sdpa` can build the mask itself from `is_causal`.
    ///
    /// The band masks with the datum type's minimum rather than `-inf`, which the indicator
    /// times fill-value construction would turn into `NaN` on the *kept* entries. A query
    /// row whose every visible key is also masked by the bias therefore leaves both
    /// sentinels indistinguishable and spreads weight over them; such a row carries no
    /// meaningful output in any implementation.
    fn wire_mask(
        &self,
        prefix: &str,
        model: &mut TypedModel,
        inputs: &[OutletId],
        dt: DatumType,
        q_len: &TDim,
        kv_len: &TDim,
    ) -> TractResult<(Option<OutletId>, bool)> {
        let bias = self.attention_bias.map(|i| inputs[i]);
        if !self.causal {
            ensure!(
                self.window == 0,
                "GroupQueryAttention: local_window_size requires causal attention"
            );
            return Ok((bias, false));
        }
        if self.window == 0 && bias.is_none() {
            return Ok((None, true));
        }

        let band = if self.window > 0 {
            let (Some(qs), Some(ks)) = (q_len.to_usize().ok(), kv_len.to_usize().ok()) else {
                bail!(
                    "GroupQueryAttention: sliding window (local_window_size) requires static \
                     sequence lengths to materialise the banded mask"
                )
            };
            let mask: Tensor = windowed_causal_mask(qs, ks, self.window).into();
            let mut m = model
                .add_const(format!("{prefix}.causal_mask"), mask.cast_to_dt(dt)?.into_owned())?;
            for i in 0..2 {
                m = model.wire_node(
                    format!("{prefix}.mask_unsqueeze_{i}"),
                    AxisOp::Add(0),
                    &[m],
                )?[0];
            }
            m
        } else {
            wire_attention_mask(
                model,
                &format!("{prefix}.causal"),
                dt,
                SdpaMaskMode::Causal,
                4,
                q_len,
                kv_len,
            )?
        };

        let Some(bias) = bias else { return Ok((Some(band), false)) };
        let bias_len = model.outlet_fact(bias)?.shape.last().unwrap().clone();
        ensure!(
            bias_len == *kv_len,
            "GroupQueryAttention: attention_bias covers {bias_len} keys but the cache holds \
             {kv_len}; pin the sequence symbols so the two agree"
        );
        Ok((
            Some(
                wire_with_rank_broadcast(format!("{prefix}.mask"), model, add(), &[band, bias])?[0],
            ),
            false,
        ))
    }
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
        // present_key / present_value are the cache concatenated with this step's key/value,
        // in BNSH: [B, kv_num_heads, past_seq + S, head_dim].
        let kvh = self.kv_num_heads;
        for (kv, present, past) in
            [(1usize, 1usize, self.past.map(|p| p.0)), (2, 2, self.past.map(|p| p.1))]
        {
            if let Some(past) = past {
                s.given_2(&inputs[kv].shape, &inputs[past].shape, move |s, ks, ps| {
                    s.equals(
                        &outputs[present].shape,
                        tvec![
                            ks[0].clone(),
                            kvh.to_dim(),
                            ps[2].clone() + ks[1].clone(),
                            ks[2].clone() / kvh
                        ],
                    )
                })?;
            } else {
                s.given(&inputs[kv].shape, move |s, ks| {
                    s.equals(
                        &outputs[present].shape,
                        tvec![ks[0].clone(), kvh.to_dim(), ks[1].clone(), ks[2].clone() / kvh],
                    )
                })?;
            }
        }
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

        let mut q4 =
            to_4d(model, &format!("{prefix}.q"), inputs[0], q_hidden.clone(), self.num_heads)?;
        let mut k4 = to_4d(model, &format!("{prefix}.k"), inputs[1], k_hidden, self.kv_num_heads)?;
        let v4 = to_4d(model, &format!("{prefix}.v"), inputs[2], v_hidden, self.kv_num_heads)?;

        // The op splits into a first prompt (empty cache, positions 0..S) and token
        // generation (one new token appended to the cache). A multi-token step against a
        // non-empty cache is ORT's subsequent-prompt case, where past/present are a shared
        // buffer written from index 0 rather than a prefix to concatenate onto.
        let past_len = self
            .past
            .map(|(pk, _)| model.outlet_fact(inputs[pk]).map(|f| f.shape[2].clone()))
            .transpose()?
            .unwrap_or_else(|| 0.to_dim());
        let q_seq = model.outlet_fact(q4)?.shape[2].clone();
        let first_prompt = past_len == 0.to_dim();
        ensure!(
            first_prompt || q_seq.to_usize().ok() == Some(1),
            "GroupQueryAttention: a {q_seq}-token step against a {past_len}-long KV cache is \
             the subsequent-prompt case, which is unsupported; prefill with an empty cache or \
             decode one token at a time"
        );

        if self.rotary.is_some() {
            ensure!(
                ApplyRope::is_supported_dt(dt),
                "GroupQueryAttention: do_rotary is limited to f32 and f16, got {dt:?}"
            );
            let head_dim = model.outlet_fact(q4)?.shape[3].clone();
            let (cos, sin) =
                self.wire_rope_tables(prefix, model, inputs, &q_seq, &head_dim, first_prompt)?;
            q4 = model.wire_node(format!("{prefix}.q_rope"), ApplyRope, &[q4, cos, sin])?[0];
            k4 = model.wire_node(format!("{prefix}.k_rope"), ApplyRope, &[k4, cos, sin])?[0];
        }

        // The cache holds K after rotation, so the concat has to follow ApplyRope.
        let (k_all, v_all) = if let Some((pk, pv)) = self.past {
            let past_len = model.outlet_fact(inputs[pk])?.shape[2].clone();
            ensure!(
                model.outlet_fact(inputs[pv])?.shape[2] == past_len,
                "GroupQueryAttention: past_key and past_value disagree on the cached length"
            );
            let k_cat = model.wire_node(
                format!("{prefix}.k_cat"),
                TypedConcat::new(2),
                &[inputs[pk], k4],
            )?[0];
            let v_cat = model.wire_node(
                format!("{prefix}.v_cat"),
                TypedConcat::new(2),
                &[inputs[pv], v4],
            )?[0];
            (k_cat, v_cat)
        } else {
            (k4, v4)
        };

        let q_len = q_seq;
        let kv_len = model.outlet_fact(k_all)?.shape[2].clone();
        let (mask, is_causal) = self.wire_mask(prefix, model, inputs, dt, &q_len, &kv_len)?;

        let mut sdpa_inputs = tvec![q4, k_all, v_all];
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

        Ok(tvec!(y, k_all, v_all))
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

    #[test]
    fn band_mask_offsets_by_the_cached_length() {
        // 2 new queries against a 5-long cache: they sit at absolute positions 3 and 4.
        let m = windowed_causal_mask(2, 5, 0);
        for j in 0..5 {
            assert_eq!(m[(0, j)] == 0.0, j <= 3, "causal decode at (i=0, j={j})");
            assert_eq!(m[(1, j)] == 0.0, j <= 4, "causal decode at (i=1, j={j})");
        }
        // a single new token attends the whole cache
        let d = windowed_causal_mask(1, 4, 0);
        for j in 0..4 {
            assert!(d[(0, j)] == 0.0, "single-token decode at (i=0, j={j})");
        }
        // ... but only the last `window` keys when a window is set
        let w = windowed_causal_mask(1, 6, 2);
        for j in 0..6 {
            assert_eq!(w[(0, j)] == 0.0, j >= 4, "windowed decode at (i=0, j={j})");
        }
    }
}
