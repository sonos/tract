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
        ensure!(sinks.rank() == 1, "sinks must be rank 1 [num_q_heads]");
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

        let q = to_f32(q_in)?;
        let q = q.to_plain_array_view::<f32>()?.into_dimensionality::<Ix4>()?;
        let mask_t = to_f32(mask)?;
        let mask = mask_t.to_plain_array_view::<f32>()?.into_dimensionality::<Ix4>()?;
        let sinks_t = to_f32(sinks)?;
        let sinks = sinks_t.to_plain_array_view::<f32>()?.into_dimensionality::<Ix1>()?;
        let k = self.k.valid_view::<f32>()?.into_dimensionality::<Ix4>()?;
        let v = self.v.valid_view::<f32>()?.into_dimensionality::<Ix4>()?;

        let (b_sz, hq, s_len, d) = q.dim();
        let hkv = k.dim().1;
        ensure!(hq % hkv == 0, "q heads {hq} not a multiple of kv heads {hkv}");
        let group = hq / hkv;
        let kv_len = k.dim().2;
        ensure!(kv_len == past + k_new.shape()[SEQ_AXIS]);
        ensure!(mask.dim().3 == kv_len, "mask keys {} != cache len {kv_len}", mask.dim().3);
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
                        mask.slice(s!(0, 0, .., ..)),
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
    ]
}

fn dump(ast: &mut IntoAst, node: &TypedNode, op: &GptOssSdpa) -> TractResult<Option<Arc<RValue>>> {
    let inputs: Vec<Arc<RValue>> =
        node.inputs.iter().map(|i| ast.mapping[i].clone()).collect();
    Ok(Some(invocation(
        "tract_transformers_gpt_oss_sdpa",
        &inputs,
        &[("scale", numeric(op.scale()))],
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
    builder.wire(
        GptOssSdpa { scale_bits: scale.to_bits() },
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
        let op = GptOssSdpa { scale_bits: scale.to_bits() };
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
        let op = GptOssSdpa { scale_bits: scale.to_bits() };
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
