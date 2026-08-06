//! Fused GPT-OSS attention with an in-place KV cache.
//!
//! The GPT-OSS export never matches the generic Sdpa/DynKeyValueCache
//! detection: its attention is raw ops with SINKS in the softmax denominator,
//! and the KV cache is a plain in-graph `concat([in_cache, new], axis=2)`
//! whose output is both a model output and the attention input. That concat
//! copies the whole past per layer per token: O(T^2) over a decode, the
//! dominant long-context decode cost.
//!
//! `GptOssSdpa` replaces the per-layer subgraph
//! `{concat K/V, GQA expand, QK*scale, +mask, concat sinks, softmax, drop
//! sink, @V}` with one stateful op that owns geometrically-grown K/V capacity
//! buffers and appends only the new rows each step. The model KEEPS its cache
//! I/O signature: `out_cache_*` dims still carry the symbolic P+S the rest of
//! the graph resolves masks/positions/reshapes from. On the CPU path the
//! emitted cache is a contiguous copy (no worse than the concat it replaces);
//! device paths can emit zero-copy views of the capacity buffer, which is
//! where the O(T) win lands.
//!
//! Cache-input handling per step: if the incoming cache length equals the
//! state's accumulated length, this is a continuation and the input tensor is
//! ignored (no copy); anything else (fresh session, truncation, retry,
//! teacher forcing) rebuilds the state from the provided cache so external
//! semantics stay exact.

use tract_nnef::internal::*;
use tract_nnef::tract_core::ops::{FrozenOpState, OpStateFreeze};
use tract_nnef::tract_ndarray::{Array2, ArrayView2, Ix1, Ix4, s};

use crate::ops::inplace_kv_cache::InPlaceKvCache;

/// Sequence axis of `[B, H, S, D]`.
const SEQ_AXIS: usize = 2;

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct GptOssSdpa {
    /// Softmax scale as f32 bits (f32 lacks Eq/Hash).
    pub scale_bits: u32,
    /// Sliding-attention window in keys (0 = full attention). Extracted by
    /// the fuse rule from the mask-building subgraph; runtimes may clamp
    /// their attention reads to the last `window + S - 1` keys, since the
    /// mask sends everything older to -inf anyway. The mask input remains
    /// the semantic source of truth, so 0 is always safe.
    pub window: u32,
}

impl GptOssSdpa {
    pub fn scale(&self) -> f32 {
        f32::from_bits(self.scale_bits)
    }
}

impl Op for GptOssSdpa {
    fn name(&self) -> StaticName {
        "GptOssSdpa".into()
    }
    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("scale={}", self.scale())])
    }
    op_as_typed_op!();
}

impl EvalOp for GptOssSdpa {
    fn is_stateless(&self) -> bool {
        false
    }
    fn state(
        &self,
        _session: &TurnState,
        _node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(Some(Box::new(GptOssSdpaState {
            scale: self.scale(),
            k: InPlaceKvCache::new(SEQ_AXIS),
            v: InPlaceKvCache::new(SEQ_AXIS),
        })))
    }
}

impl TypedOp for GptOssSdpa {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(inputs.len() == 7, "GptOssSdpa expects [q, k_new, v_new, k_cache, v_cache, mask, sinks]");
        let (q, k_new, _v_new, k_cache, v_cache, _mask, sinks) =
            (inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6]);
        ensure!(q.rank() == 4 && k_new.rank() == 4 && k_cache.rank() == 4);
        let _ = sinks;
        // Attention output has Q's shape/dtype; caches grow to P+S along seq.
        let total = k_cache.shape[SEQ_AXIS].clone() + k_new.shape[SEQ_AXIS].clone();
        let mut k_out = k_cache.without_value();
        k_out.shape.set(SEQ_AXIS, total.clone());
        let mut v_out = v_cache.without_value();
        v_out.shape.set(SEQ_AXIS, total);
        Ok(tvec!(q.without_value(), k_out, v_out))
    }
    as_op!();
}

#[derive(Clone, Debug)]
pub struct GptOssSdpaState {
    scale: f32,
    k: InPlaceKvCache,
    v: InPlaceKvCache,
}

impl OpState for GptOssSdpaState {
    fn eval(
        &mut self,
        _state: &mut TurnState,
        _op: &dyn Op,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        ensure!(inputs.len() == 7);
        let (q_in, k_new, v_new, k_cache, v_cache, mask, sinks) = (
            &inputs[0], &inputs[1], &inputs[2], &inputs[3], &inputs[4], &inputs[5], &inputs[6],
        );
        let out_dt = q_in.datum_type();
        let past = k_cache.shape()[SEQ_AXIS];

        // Continuation fast-path vs rebuild (fresh run / truncation / retry).
        if past != self.k.len() {
            self.k = InPlaceKvCache::new(SEQ_AXIS);
            self.v = InPlaceKvCache::new(SEQ_AXIS);
            if past > 0 {
                self.k.push(&to_f32(k_cache)?)?;
                self.v.push(&to_f32(v_cache)?)?;
            }
        }
        self.k.push(&to_f32(k_new)?)?;
        self.v.push(&to_f32(v_new)?)?;

        if std::env::var_os("TRACT_DEBUG_GPT_OSS_SDPA").is_some() {
            let stats = |t: &TValue, tag: &str| -> TractResult<()> {
                let h = t.cast_to::<f32>()?.into_owned();
                let v = h.try_as_plain()?.as_slice::<f32>()?;
                let mx = v.iter().cloned().fold(f32::MIN, f32::max);
                let mn = v.iter().cloned().fold(f32::MAX, f32::min);
                let sum: f32 = v.iter().sum();
                eprintln!(
                    "gptoss-cpu-dbg {tag}: shape={:?} min={mn:.4} max={mx:.4} mean={:.6}",
                    t.shape(),
                    sum / v.len() as f32
                );
                Ok(())
            };
            stats(q_in, "q")?;
            stats(k_new, "k_new")?;
            stats(mask, "mask")?;
        }
        let q = to_f32(q_in)?;
        let q = q.to_plain_array_view::<f32>()?.into_dimensionality::<Ix4>()?;
        let mask_t = to_f32(mask)?;
        let mask_rank = mask_t.rank();
        ensure!(mask_rank >= 2);
        ensure!(
            mask_t.shape()[..mask_rank - 2].iter().all(|&d| d == 1),
            "mask leading dims must be broadcast (1), got {:?}",
            mask_t.shape()
        );
        let mask_2d: TVec<usize> = mask_t.shape()[mask_rank - 2..].into();
        let mask_t = mask_t.into_shape(&mask_2d)?;
        let mask = mask_t
            .to_plain_array_view::<f32>()?
            .into_dimensionality::<tract_nnef::tract_ndarray::Ix2>()?;
        let mut sinks_t = to_f32(sinks)?;
        let sinks_len = sinks_t.len();
        sinks_t = sinks_t.into_shape(&[sinks_len])?;
        let sinks = sinks_t.to_plain_array_view::<f32>()?.into_dimensionality::<Ix1>()?;
        let k = self.k.valid_view::<f32>()?.into_dimensionality::<Ix4>()?;
        let v = self.v.valid_view::<f32>()?.into_dimensionality::<Ix4>()?;

        let (b_sz, hq, s_len, d) = q.dim();
        let hkv = k.dim().1;
        ensure!(hq % hkv == 0, "q heads {hq} not a multiple of kv heads {hkv}");
        let group = hq / hkv;
        let kv_len = k.dim().2;
        ensure!(kv_len == past + k_new.shape()[SEQ_AXIS]);
        ensure!(mask.dim().1 == kv_len, "mask keys {} != cache len {kv_len}", mask.dim().1);
        ensure!(sinks.len() == hq);

        let mut out = Tensor::zero::<f32>(&[b_sz, hq, s_len, d])?;
        {
            let mut ov = out.to_plain_array_view_mut::<f32>()?.into_dimensionality::<Ix4>()?;
            for b in 0..b_sz {
                for h in 0..hq {
                    let o = attend_one_head_with_sink(
                        q.slice(s!(b, h, .., ..)),
                        k.slice(s!(b, h / group, .., ..)),
                        v.slice(s!(b, h / group, .., ..)),
                        mask,
                        sinks[h],
                        self.scale,
                    );
                    ov.slice_mut(s!(b, h, .., ..)).assign(&o);
                }
            }
        }

        // CPU path: emit contiguous copies (device paths emit views instead).
        let k_out = self.k.valid_contiguous()?.cast_to_dt(out_dt)?.into_owned();
        let v_out = self.v.valid_contiguous()?.cast_to_dt(out_dt)?.into_owned();
        Ok(tvec!(
            out.cast_to_dt(out_dt)?.into_owned().into_tvalue(),
            k_out.into_tvalue(),
            v_out.into_tvalue(),
        ))
    }
}

#[derive(Clone, Debug)]
pub struct FrozenGptOssSdpaState(GptOssSdpaState);

impl OpStateFreeze for GptOssSdpaState {
    fn freeze(&self) -> Box<dyn FrozenOpState> {
        Box::new(FrozenGptOssSdpaState(self.clone()))
    }
}

impl FrozenOpState for FrozenGptOssSdpaState {
    fn unfreeze(&self) -> Box<dyn OpState> {
        Box::new(self.0.clone())
    }
}

/// Two-pass softmax attention for one (batch, q-head), with the GPT-OSS sink:
/// an extra logit participating in normalization with no value row. Matches
/// `softmax(concat([q.k*scale + mask, sink]))[..-1] @ v` exactly.
fn attend_one_head_with_sink(
    q: ArrayView2<f32>,
    k: ArrayView2<f32>,
    v: ArrayView2<f32>,
    mask: ArrayView2<f32>,
    sink: f32,
    scale: f32,
) -> Array2<f32> {
    let mut logits = q.dot(&k.t());
    logits *= scale;
    logits += &mask;
    let (s_len, d) = (q.dim().0, q.dim().1);
    let mut out = Array2::<f32>::zeros((s_len, d));
    for i in 0..s_len {
        let row = logits.slice(s!(i, ..));
        let m = row.iter().copied().fold(sink, f32::max);
        let mut den = (sink - m).exp();
        let mut weights = row.to_owned();
        for w in weights.iter_mut() {
            *w = (*w - m).exp();
            den += *w;
        }
        weights /= den;
        out.slice_mut(s!(i, ..)).assign(&weights.dot(&v));
    }
    out
}

fn to_f32(t: &TValue) -> TractResult<Tensor> {
    Ok(t.cast_to::<f32>()?.into_owned())
}

pub fn register(registry: &mut Registry) {
    registry.register_dumper(dump);
    registry.register_primitive(
        "tract_transformers_gpt_oss_sdpa",
        &parameters(),
        &[
            ("output", TypeName::Scalar.tensor()),
            ("k_cache_out", TypeName::Scalar.tensor()),
            ("v_cache_out", TypeName::Scalar.tensor()),
        ],
        load,
    );
}

fn parameters() -> Vec<Parameter> {
    vec![
        TypeName::Scalar.tensor().named("q"),
        TypeName::Scalar.tensor().named("k_new"),
        TypeName::Scalar.tensor().named("v_new"),
        TypeName::Scalar.tensor().named("k_cache"),
        TypeName::Scalar.tensor().named("v_cache"),
        TypeName::Scalar.tensor().named("mask"),
        TypeName::Scalar.tensor().named("sinks"),
        TypeName::Scalar.named("scale"),
        TypeName::Integer.named("window").default(0),
    ]
}

fn dump(ast: &mut IntoAst, node: &TypedNode, op: &GptOssSdpa) -> TractResult<Option<Arc<RValue>>> {
    let inputs: Vec<Arc<RValue>> =
        node.inputs.iter().map(|i| ast.mapping[i].clone()).collect();
    Ok(Some(invocation(
        "tract_transformers_gpt_oss_sdpa",
        &inputs,
        &[("scale", numeric(op.scale())), ("window", numeric(op.window))],
    )))
}

fn load(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<Value> {
    let q = invocation.named_arg_as(builder, "q")?;
    let k_new = invocation.named_arg_as(builder, "k_new")?;
    let v_new = invocation.named_arg_as(builder, "v_new")?;
    let k_cache = invocation.named_arg_as(builder, "k_cache")?;
    let v_cache = invocation.named_arg_as(builder, "v_cache")?;
    let mask = invocation.named_arg_as(builder, "mask")?;
    let sinks = invocation.named_arg_as(builder, "sinks")?;
    let scale: f32 = invocation.named_arg_as(builder, "scale")?;
    let window: i64 = invocation.named_arg_as(builder, "window")?;
    builder.wire(
        GptOssSdpa { scale_bits: scale.to_bits(), window: window as u32 },
        &[q, k_new, v_new, k_cache, v_cache, mask, sinks],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use tract_nnef::tract_ndarray::Array4;

    fn rng_tensor(shape: &[usize], seed: &mut u64) -> Tensor {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n)
            .map(|_| {
                *seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((*seed >> 33) as f32 / (1u64 << 31) as f32) - 1.0
            })
            .collect();
        Tensor::from_shape(shape, &data).unwrap()
    }

    /// Reference: literally the exported subgraph math.
    /// softmax(concat([QK*scale + mask, sinks]))[..,:-1] @ V, with GQA expand.
    fn reference(
        q: &Tensor,
        k_all: &Tensor,
        v_all: &Tensor,
        mask: &Tensor,
        sinks: &[f32],
        scale: f32,
    ) -> Tensor {
        let q = q.to_plain_array_view::<f32>().unwrap().into_dimensionality::<Ix4>().unwrap();
        let k = k_all.to_plain_array_view::<f32>().unwrap().into_dimensionality::<Ix4>().unwrap();
        let v = v_all.to_plain_array_view::<f32>().unwrap().into_dimensionality::<Ix4>().unwrap();
        let mask = mask.to_plain_array_view::<f32>().unwrap().into_dimensionality::<Ix4>().unwrap();
        let (b_sz, hq, s_len, d) = q.dim();
        let group = hq / k.dim().1;
        let kv_len = k.dim().2;
        let mut out = Array4::<f32>::zeros((b_sz, hq, s_len, d));
        for b in 0..b_sz {
            for h in 0..hq {
                for i in 0..s_len {
                    let mut logits: Vec<f32> = (0..kv_len)
                        .map(|j| {
                            let mut acc = 0f32;
                            for x in 0..d {
                                acc += q[(b, h, i, x)] * k[(b, h / group, j, x)];
                            }
                            acc * scale + mask[(0, 0, i, j)]
                        })
                        .collect();
                    logits.push(sinks[h]);
                    let m = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let exps: Vec<f32> = logits.iter().map(|l| (l - m).exp()).collect();
                    let den: f32 = exps.iter().sum();
                    for x in 0..d {
                        let mut acc = 0f32;
                        for j in 0..kv_len {
                            acc += exps[j] / den * v[(b, h / group, j, x)];
                        }
                        out[(b, h, i, x)] = acc;
                    }
                }
            }
        }
        out.into_tensor()
    }

    fn causal_mask(s_len: usize, kv_len: usize) -> Tensor {
        let past = kv_len - s_len;
        let mut m = Array4::<f32>::zeros((1, 1, s_len, kv_len));
        for i in 0..s_len {
            for j in 0..kv_len {
                if j > past + i {
                    m[(0, 0, i, j)] = f32::MIN;
                }
            }
        }
        m.into_tensor()
    }

    fn run_state(
        state: &mut GptOssSdpaState,
        op: &GptOssSdpa,
        q: &Tensor,
        k_new: &Tensor,
        v_new: &Tensor,
        k_cache: &Tensor,
        v_cache: &Tensor,
        mask: &Tensor,
        sinks: &Tensor,
    ) -> TVec<TValue> {
        let mut session = TurnState::default();
        state
            .eval(
                &mut session,
                op,
                tvec!(
                    q.clone().into_tvalue(),
                    k_new.clone().into_tvalue(),
                    v_new.clone().into_tvalue(),
                    k_cache.clone().into_tvalue(),
                    v_cache.clone().into_tvalue(),
                    mask.clone().into_tvalue(),
                    sinks.clone().into_tvalue(),
                ),
            )
            .unwrap()
    }

    #[test]
    fn matches_reference_over_prefill_and_decode() -> TractResult<()> {
        let (hq, hkv, d) = (4, 2, 8);
        let scale = (d as f32).sqrt().recip();
        let op = GptOssSdpa { scale_bits: scale.to_bits(), window: 0 };
        let mut state = GptOssSdpaState {
            scale,
            k: InPlaceKvCache::new(SEQ_AXIS),
            v: InPlaceKvCache::new(SEQ_AXIS),
        };
        let mut seed = 42u64;
        let sinks_t = rng_tensor(&[hq], &mut seed);
        let sinks: Vec<f32> = sinks_t.try_as_plain()?.as_slice::<f32>()?.to_vec();

        // Accumulated "external" cache, grown the reference way.
        let mut k_all = Tensor::zero::<f32>(&[1, hkv, 0, d])?;
        let mut v_all = Tensor::zero::<f32>(&[1, hkv, 0, d])?;

        for step_len in [5usize, 1, 1, 3, 1] {
            let past = k_all.shape()[SEQ_AXIS];
            let q = rng_tensor(&[1, hq, step_len, d], &mut seed);
            let k_new = rng_tensor(&[1, hkv, step_len, d], &mut seed);
            let v_new = rng_tensor(&[1, hkv, step_len, d], &mut seed);
            let mask = causal_mask(step_len, past + step_len);

            let outputs =
                run_state(&mut state, &op, &q, &k_new, &v_new, &k_all, &v_all, &mask, &sinks_t);

            // Reference over the full concatenated cache.
            let k_ref = Tensor::stack_tensors(SEQ_AXIS, &[&k_all, &k_new])?;
            let v_ref = Tensor::stack_tensors(SEQ_AXIS, &[&v_all, &v_new])?;
            let want = reference(&q, &k_ref, &v_ref, &mask, &sinks, scale);
            outputs[0]
                .clone()
                .into_tensor()
                .close_enough(&want, Approximation::Approximate)?;
            outputs[1].clone().into_tensor().close_enough(&k_ref, Approximation::Exact)?;
            outputs[2].clone().into_tensor().close_enough(&v_ref, Approximation::Exact)?;
            k_all = k_ref;
            v_all = v_ref;
        }
        Ok(())
    }

    #[test]
    fn rebuilds_state_on_cache_mismatch() -> TractResult<()> {
        let (hq, hkv, d) = (2, 1, 4);
        let scale = 0.5f32;
        let op = GptOssSdpa { scale_bits: scale.to_bits(), window: 0 };
        let mut state = GptOssSdpaState {
            scale,
            k: InPlaceKvCache::new(SEQ_AXIS),
            v: InPlaceKvCache::new(SEQ_AXIS),
        };
        let mut seed = 7u64;
        let sinks_t = rng_tensor(&[hq], &mut seed);
        let sinks: Vec<f32> = sinks_t.try_as_plain()?.as_slice::<f32>()?.to_vec();

        // Prime the state with a 4-token pass.
        let q0 = rng_tensor(&[1, hq, 4, d], &mut seed);
        let k0 = rng_tensor(&[1, hkv, 4, d], &mut seed);
        let v0 = rng_tensor(&[1, hkv, 4, d], &mut seed);
        let empty_k = Tensor::zero::<f32>(&[1, hkv, 0, d])?;
        let empty_v = Tensor::zero::<f32>(&[1, hkv, 0, d])?;
        run_state(&mut state, &op, &q0, &k0, &v0, &empty_k, &empty_v, &causal_mask(4, 4), &sinks_t);

        // Now pretend the caller truncated to 2 tokens (retry): the provided
        // cache disagrees with state len 4 and must win.
        let k_trunc = k0.slice(SEQ_AXIS, 0, 2)?;
        let v_trunc = v0.slice(SEQ_AXIS, 0, 2)?;
        let q1 = rng_tensor(&[1, hq, 1, d], &mut seed);
        let k1 = rng_tensor(&[1, hkv, 1, d], &mut seed);
        let v1 = rng_tensor(&[1, hkv, 1, d], &mut seed);
        let mask = causal_mask(1, 3);
        let outputs =
            run_state(&mut state, &op, &q1, &k1, &v1, &k_trunc, &v_trunc, &mask, &sinks_t);

        let k_ref = Tensor::stack_tensors(SEQ_AXIS, &[&k_trunc, &k1])?;
        let v_ref = Tensor::stack_tensors(SEQ_AXIS, &[&v_trunc, &v1])?;
        let want = reference(&q1, &k_ref, &v_ref, &mask, &sinks, scale);
        outputs[0].clone().into_tensor().close_enough(&want, Approximation::Approximate)?;
        outputs[1].clone().into_tensor().close_enough(&k_ref, Approximation::Exact)?;
        Ok(())
    }
}

// ===================================================================================
// Detection: fuse the exported GPT-OSS attention subgraph on the decluttered
// graph. Anchor = the sinks concat `Concat([qk+mask, broadcast(sinks)], last)`.
// ===================================================================================

use tract_nnef::tract_core::ops::array::{MultiBroadcastTo, TypedConcat};
use tract_nnef::tract_core::ops::binary::TypedBinOp;
use tract_nnef::tract_core::ops::cast::Cast;
use tract_nnef::tract_core::ops::einsum::EinSum;
use tract_nnef::tract_core::ops::konst::Const;
use tract_nnef::tract_core::ops::nn::{Reduce, Reducer, Softmax};
use tract_nnef::tract_core::ops::source::TypedSource;
use tract_nnef::tract_core::transform::ModelTransform;

#[derive(Debug)]
pub struct GptOssInPlaceSdpaTransform;

impl ModelTransform for GptOssInPlaceSdpaTransform {
    fn name(&self) -> StaticName {
        "fuse_gpt_oss_sdpa".into()
    }
    fn transform(&self, model: &mut TypedModel) -> TractResult<()> {
        Rewriter::default()
            .with_rule_for("fuse-gpt-oss-sdpa", fuse_gpt_oss_sdpa_rule)
            .rewrite(&(), model)?;
        model.compact()
    }
}

fn prev<'m>(model: &'m TypedModel, node: &TypedNode, slot: usize) -> &'m TypedNode {
    model.node(node.inputs[slot].node)
}

fn is_bin(node: &TypedNode, name: &str) -> bool {
    node.op_as::<TypedBinOp>().is_some_and(|b| b.0.name() == name)
}

fn single_consumer<'m>(model: &'m TypedModel, node: &TypedNode) -> Option<&'m TypedNode> {
    let succ = &model.outlet_successors(node.id.into());
    if succ.len() == 1 { Some(model.node(succ[0].node)) } else { None }
}

/// Walk one QK einsum branch back to (q_half outlet, K concat node, slice start).
fn qk_branch(
    model: &TypedModel,
    reshape_out: &TypedNode,
) -> Option<(OutletId, usize, i64)> {
    // Reshape(folded_output) <- EinSum <- [Reshape(q_half), Slice(bcast(AddAxis(KConcat)))]
    if !reshape_out.op_is::<AxisOp>() {
        return None;
    }
    let einsum = prev(model, reshape_out, 0);
    einsum.op_as::<EinSum>()?;
    let q_reshape = prev(model, einsum, 0);
    if !q_reshape.op_is::<AxisOp>() {
        return None;
    }
    let q_half = q_reshape.inputs[0];
    let k_slice = prev(model, einsum, 1);
    let slice = k_slice.op_as::<tract_nnef::tract_core::ops::array::Slice>()?;
    let start = slice.start.to_i64().ok()?;
    let bcast = prev(model, k_slice, 0);
    bcast.op_as::<MultiBroadcastTo>()?;
    let addaxis = prev(model, bcast, 0);
    if !addaxis.op_is::<AxisOp>() {
        return None;
    }
    let kconcat = prev(model, addaxis, 0);
    kconcat.op_as::<TypedConcat>()?;
    Some((q_half, kconcat.id, start))
}

/// Walk the mask-building subgraph upstream of `mask_outlet` looking for the
/// sliding-window size. The exported causal mask is built from ranges and
/// 0/1/huge-negative constants only; the sliding variant additionally embeds
/// the window as a small integer constant in its band comparison. Returns 0
/// (full attention) when no such constant exists, which is always safe: the
/// mask stays the semantic source of truth.
fn extract_sliding_window(model: &TypedModel, mask_outlet: OutletId) -> u32 {
    let debug = std::env::var_os("TRACT_DEBUG_GPT_OSS_WINDOW").is_some();
    let mut seen = std::collections::HashSet::new();
    let mut stack = vec![mask_outlet.node];
    let mut candidates: Vec<u32> = vec![];
    while let Some(id) = stack.pop() {
        if !seen.insert(id) || seen.len() > 256 {
            continue;
        }
        let n = model.node(id);
        if debug {
            let k = n
                .op_as::<Const>()
                .map(|k| format!(" = {:?}", k.val()))
                .unwrap_or_default();
            eprintln!("mask-subgraph[{}]: {} {}{k}", mask_outlet.node, n.name, n.op.name());
        }
        if let Some(k) = n.op_as::<Const>() {
            let v = k.val();
            if v.len() == 1
                && (v.datum_type().is_integer() || v.datum_type() == DatumType::TDim)
            {
                if let Ok(x) = v.cast_to::<i64>() {
                    if let Ok(x) = x.try_as_plain() {
                        if let Ok(x) = x.as_slice::<i64>() {
                            let w = x[0].unsigned_abs();
                            if (2..=1_000_000).contains(&w) {
                                candidates.push(w as u32);
                            }
                        }
                    }
                }
            }
        }
        for input in &n.inputs {
            stack.push(input.node);
        }
    }
    match candidates.as_slice() {
        [w] => *w,
        [] => 0,
        many => {
            // Ambiguous: refuse to clamp rather than risk wrong semantics.
            eprintln!(
                "gpt-oss fuse: mask subgraph has several window-like constants {many:?}; not clamping"
            );
            0
        }
    }
}

pub fn fuse_gpt_oss_sdpa_rule(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    node_name: &str,
    concat: &TypedConcat,
) -> TractResult<Option<TypedModelPatch>> {
    // ---- anchor: Concat([logits, sinks_broadcast], axis = last) ----
    if node.inputs.len() != 2 {
        return Ok(None);
    }
    let rank = node.outputs[0].fact.rank();
    if concat.axis != rank - 1 {
        return Ok(None);
    }
    let sinks_bcast = prev(model, node, 1);
    if sinks_bcast.op_as::<MultiBroadcastTo>().is_none() {
        return Ok(None);
    }
    let sinks_const = prev(model, sinks_bcast, 0);
    let Some(sinks_k) = sinks_const.op_as::<Const>() else { return Ok(None) };
    let sinks_tensor = sinks_k.val().clone();

    let mask_add = prev(model, node, 0);
    if !is_bin(mask_add, "Add") {
        return Ok(None);
    }
    let mask_outlet = mask_add.inputs[1];
    let cast_f32 = prev(model, mask_add, 0);
    if cast_f32.op_as::<Cast>().is_none() {
        return Ok(None);
    }
    let scale_mul = prev(model, cast_f32, 0);
    if !is_bin(scale_mul, "Mul") {
        return Ok(None);
    }
    let Some(scale_k) = prev(model, scale_mul, 1).op_as::<Const>() else { return Ok(None) };
    let scale = scale_k.val().cast_to::<f32>()?.try_as_plain()?.as_slice::<f32>()?[0];

    let qk_add = prev(model, scale_mul, 0);
    if !is_bin(qk_add, "Add") {
        return Ok(None);
    }
    let Some((q_a, kc_a, start_a)) = qk_branch(model, prev(model, qk_add, 0)) else {
        return Ok(None);
    };
    let Some((q_b, kc_b, _)) = qk_branch(model, prev(model, qk_add, 1)) else {
        return Ok(None);
    };
    if kc_a != kc_b {
        return Ok(None);
    }
    let (q_first, q_second) = if start_a == 0 { (q_a, q_b) } else { (q_b, q_a) };
    let kconcat = model.node(kc_a);
    if prev(model, kconcat, 0).op_as::<TypedSource>().is_none() {
        return Ok(None);
    }
    let k_cache = kconcat.inputs[0];
    let k_new = kconcat.inputs[1];

    // ---- downstream: max/sub/softmax/slice/cast/reshape/AV-einsum ----
    // node (concat) has two consumers: Reduce<Max> and Sub.
    let succ = model.outlet_successors(node.id.into());
    if succ.len() != 2 {
        return Ok(None);
    }
    let (mx, sub) = {
        let a = model.node(succ[0].node);
        let b = model.node(succ[1].node);
        if a.op_as::<Reduce>().is_some_and(|r| r.reducer == Reducer::Max) {
            (a, b)
        } else {
            (b, a)
        }
    };
    if mx.op_as::<Reduce>().is_none() || !is_bin(sub, "Sub") {
        return Ok(None);
    }
    let Some(softmax) = single_consumer(model, sub) else { return Ok(None) };
    if softmax.op_as::<Softmax>().is_none() {
        return Ok(None);
    }
    let Some(drop_sink) = single_consumer(model, softmax) else { return Ok(None) };
    if drop_sink.op_as::<tract_nnef::tract_core::ops::array::Slice>().is_none() {
        return Ok(None);
    }
    let Some(cast_f16) = single_consumer(model, drop_sink) else { return Ok(None) };
    if cast_f16.op_as::<Cast>().is_none() {
        return Ok(None);
    }
    let Some(probs_reshape) = single_consumer(model, cast_f16) else { return Ok(None) };
    if !probs_reshape.op_is::<AxisOp>() {
        return Ok(None);
    }
    let Some(av) = single_consumer(model, probs_reshape) else { return Ok(None) };
    if av.op_as::<EinSum>().is_none() {
        return Ok(None);
    }
    let vconcat = prev(model, av, 1);
    if vconcat.op_as::<TypedConcat>().is_none()
        || prev(model, vconcat, 0).op_as::<TypedSource>().is_none()
    {
        return Ok(None);
    }
    let v_cache = vconcat.inputs[0];
    let v_new = vconcat.inputs[1];

    // Cache concats must be model outputs (their slots get the op's views).
    let k_out_outlet = OutletId::new(kconcat.id, 0);
    let v_out_outlet = OutletId::new(vconcat.id, 0);
    if !model.outputs.contains(&k_out_outlet) || !model.outputs.contains(&v_out_outlet) {
        return Ok(None);
    }

    // ---- build the patch ----
    let mut patch = TypedModelPatch::new(format!("fuse-gpt-oss-sdpa @ {node_name}"));
    let q1 = patch.tap_model(model, q_first)?;
    let q2 = patch.tap_model(model, q_second)?;
    let k_new_t = patch.tap_model(model, k_new)?;
    let v_new_t = patch.tap_model(model, v_new)?;
    let k_cache_t = patch.tap_model(model, k_cache)?;
    let v_cache_t = patch.tap_model(model, v_cache)?;
    let mask_t = patch.tap_model(model, mask_outlet)?;
    let window = extract_sliding_window(model, mask_outlet);
    let sinks_t = patch.add_const(format!("{node_name}.sinks"), sinks_tensor)?;

    let q_rank = model.outlet_fact(q_first)?.rank();
    let q = patch.wire_node(
        format!("{node_name}.q"),
        TypedConcat { axis: q_rank - 1 },
        &[q1, q2],
    )?[0];

    let fused = patch.wire_node(
        format!("{node_name}.gpt_oss_sdpa"),
        GptOssSdpa { scale_bits: scale.to_bits(), window },
        &[q, k_new_t, v_new_t, k_cache_t, v_cache_t, mask_t, sinks_t],
    )?;

    // Attention out [1, Hq, S, D] -> the AV einsum's [G, Hkv, S, D] shape.
    let av_fact = model.outlet_fact(OutletId::new(av.id, 0))?;
    let g = av_fact.shape[0].to_usize()?;
    let hkv = av_fact.shape[1].to_usize()?;
    let attn = patch.wire_node(
        format!("{node_name}.attn_reshape"),
        AxisOp::Reshape(
            0,
            tvec![1.to_dim(), (g * hkv).to_dim()],
            tvec![g.to_dim(), hkv.to_dim()],
        ),
        &[fused[0]],
    )?[0];

    patch.shunt_outside(model, OutletId::new(av.id, 0), attn)?;
    patch.shunt_outside(model, k_out_outlet, fused[1])?;
    patch.shunt_outside(model, v_out_outlet, fused[2])?;
    Ok(Some(patch))
}
