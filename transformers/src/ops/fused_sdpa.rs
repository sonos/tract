//! Fused attention with an in-place KV cache, generic across transformer
//! exports (GPT-OSS, Qwen3.5 full-attention layers, ...).
//!
//! These exports never match the generic Sdpa/DynKeyValueCache detection:
//! attention is raw ops (optionally with SINKS in the softmax denominator),
//! and the KV cache is a plain in-graph `concat([in_cache, new], axis=2)`
//! whose output is both a model output and the attention input. That concat
//! copies the whole past per layer per token: O(T^2) over a decode, the
//! dominant long-context decode cost.
//!
//! `FusedSdpa` replaces the per-layer subgraph
//! `{concat K/V, GQA expand, QK*scale, +mask, [concat sinks], softmax,
//! [drop sink], @V}` with one stateful op that owns geometrically-grown K/V
//! capacity buffers and appends only the new rows each step. The model KEEPS
//! its cache I/O signature: `out_cache_*` dims still carry the symbolic P+S
//! the rest of the graph resolves masks/positions/reshapes from. On the CPU
//! path the emitted cache is a contiguous copy (no worse than the concat it
//! replaces); device paths can emit zero-copy views of the capacity buffer,
//! which is where the O(T) win lands.
//!
//! Everything geometry-related is parametric: head_dim, GQA ratio (Hq/Hkv),
//! sequence lengths. Attention sinks are optional (`has_sinks`), the sliding
//! window is optional (`window == 0` means full attention).
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
pub struct FusedSdpa {
    /// Softmax scale as f32 bits (f32 lacks Eq/Hash).
    pub scale_bits: u32,
    /// Sliding-attention window in keys (0 = full attention). Extracted by
    /// the fuse rule from the mask-building subgraph; runtimes may clamp
    /// their attention reads to the last `window + S - 1` keys, since the
    /// mask sends everything older to -inf anyway. The mask input remains
    /// the semantic source of truth, so 0 is always safe.
    pub window: u32,
    /// GPT-OSS-style attention sinks: one extra logit per q head in the
    /// softmax denominator, no value row. When set, the op takes a trailing
    /// `sinks` input of `Hq` f32 values.
    pub has_sinks: bool,
}

impl FusedSdpa {
    pub fn scale(&self) -> f32 {
        f32::from_bits(self.scale_bits)
    }
    pub fn input_count(&self) -> usize {
        6 + self.has_sinks as usize
    }
}

impl Op for FusedSdpa {
    fn name(&self) -> StaticName {
        "FusedSdpa".into()
    }
    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!(
            "scale={} window={} sinks={}",
            self.scale(),
            self.window,
            self.has_sinks
        )])
    }
    op_as_typed_op!();
}

impl EvalOp for FusedSdpa {
    fn is_stateless(&self) -> bool {
        false
    }
    fn state(
        &self,
        _session: &TurnState,
        _node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(Some(Box::new(FusedSdpaState {
            scale: self.scale(),
            has_sinks: self.has_sinks,
            k: InPlaceKvCache::new(SEQ_AXIS),
            v: InPlaceKvCache::new(SEQ_AXIS),
        })))
    }
}

impl TypedOp for FusedSdpa {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(
            inputs.len() == self.input_count(),
            "FusedSdpa expects [q, k_new, v_new, k_cache, v_cache, mask{}], got {} inputs",
            if self.has_sinks { ", sinks" } else { "" },
            inputs.len()
        );
        let (q, k_new, _v_new, k_cache, v_cache) =
            (inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]);
        ensure!(q.rank() == 4 && k_new.rank() == 4 && k_cache.rank() == 4);
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
pub struct FusedSdpaState {
    scale: f32,
    has_sinks: bool,
    k: InPlaceKvCache,
    v: InPlaceKvCache,
}

impl OpState for FusedSdpaState {
    fn eval(
        &mut self,
        _state: &mut TurnState,
        _op: &dyn Op,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        ensure!(inputs.len() == 6 + self.has_sinks as usize);
        let (q_in, k_new, v_new, k_cache, v_cache, mask) = (
            &inputs[0], &inputs[1], &inputs[2], &inputs[3], &inputs[4], &inputs[5],
        );
        let sinks_in = self.has_sinks.then(|| &inputs[6]);
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
                    "fused-sdpa-cpu-dbg {tag}: shape={:?} min={mn:.4} max={mx:.4} mean={:.6}",
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
        let sinks_t = match sinks_in {
            Some(sinks) => {
                let mut t = to_f32(sinks)?;
                let n = t.len();
                t = t.into_shape(&[n])?;
                Some(t)
            }
            None => None,
        };
        let sinks = match sinks_t.as_ref() {
            Some(t) => Some(t.to_plain_array_view::<f32>()?.into_dimensionality::<Ix1>()?),
            None => None,
        };
        let k = self.k.valid_view::<f32>()?.into_dimensionality::<Ix4>()?;
        let v = self.v.valid_view::<f32>()?.into_dimensionality::<Ix4>()?;

        let (b_sz, hq, s_len, d) = q.dim();
        let hkv = k.dim().1;
        ensure!(hq % hkv == 0, "q heads {hq} not a multiple of kv heads {hkv}");
        let group = hq / hkv;
        let kv_len = k.dim().2;
        ensure!(kv_len == past + k_new.shape()[SEQ_AXIS]);
        ensure!(mask.dim().1 == kv_len, "mask keys {} != cache len {kv_len}", mask.dim().1);
        if let Some(sinks) = &sinks {
            ensure!(sinks.len() == hq);
        }

        let mut out = Tensor::zero::<f32>(&[b_sz, hq, s_len, d])?;
        {
            let mut ov = out.to_plain_array_view_mut::<f32>()?.into_dimensionality::<Ix4>()?;
            for b in 0..b_sz {
                for h in 0..hq {
                    // Without sinks the -inf sink contributes exp(-inf) = 0
                    // to the denominator: exactly the plain masked softmax.
                    let sink = sinks.as_ref().map_or(f32::NEG_INFINITY, |s| s[h]);
                    let o = attend_one_head_with_sink(
                        q.slice(s!(b, h, .., ..)),
                        k.slice(s!(b, h / group, .., ..)),
                        v.slice(s!(b, h / group, .., ..)),
                        mask,
                        sink,
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
pub struct FrozenFusedSdpaState(FusedSdpaState);

impl OpStateFreeze for FusedSdpaState {
    fn freeze(&self) -> Box<dyn FrozenOpState> {
        Box::new(FrozenFusedSdpaState(self.clone()))
    }
}

impl FrozenOpState for FrozenFusedSdpaState {
    fn unfreeze(&self) -> Box<dyn OpState> {
        Box::new(self.0.clone())
    }
}

/// Two-pass softmax attention for one (batch, q-head), with an optional
/// GPT-OSS-style sink: an extra logit participating in normalization with no
/// value row. Matches `softmax(concat([q.k*scale + mask, sink]))[..-1] @ v`
/// exactly; `sink = -inf` degenerates to the plain masked softmax.
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
    // Sinks variant keeps the historical primitive name so existing GPT-OSS
    // dumps keep loading.
    registry.register_primitive(
        "tract_transformers_gpt_oss_sdpa",
        &parameters(true),
        &outputs(),
        load_with_sinks,
    );
    registry.register_primitive(
        "tract_transformers_fused_sdpa",
        &parameters(false),
        &outputs(),
        load_without_sinks,
    );
}

fn outputs() -> Vec<(&'static str, tract_nnef::ast::TypeSpec)> {
    vec![
        ("output", TypeName::Scalar.tensor()),
        ("k_cache_out", TypeName::Scalar.tensor()),
        ("v_cache_out", TypeName::Scalar.tensor()),
    ]
}

fn parameters(with_sinks: bool) -> Vec<Parameter> {
    let mut params = vec![
        TypeName::Scalar.tensor().named("q"),
        TypeName::Scalar.tensor().named("k_new"),
        TypeName::Scalar.tensor().named("v_new"),
        TypeName::Scalar.tensor().named("k_cache"),
        TypeName::Scalar.tensor().named("v_cache"),
        TypeName::Scalar.tensor().named("mask"),
    ];
    if with_sinks {
        params.push(TypeName::Scalar.tensor().named("sinks"));
    }
    params.push(TypeName::Scalar.named("scale"));
    params.push(TypeName::Integer.named("window").default(0));
    params
}

fn dump(ast: &mut IntoAst, node: &TypedNode, op: &FusedSdpa) -> TractResult<Option<Arc<RValue>>> {
    let inputs: Vec<Arc<RValue>> =
        node.inputs.iter().map(|i| ast.mapping[i].clone()).collect();
    let name = if op.has_sinks {
        "tract_transformers_gpt_oss_sdpa"
    } else {
        "tract_transformers_fused_sdpa"
    };
    Ok(Some(invocation(
        name,
        &inputs,
        &[("scale", numeric(op.scale())), ("window", numeric(op.window))],
    )))
}

fn load_common(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
    has_sinks: bool,
) -> TractResult<Value> {
    let mut inputs: TVec<OutletId> = tvec!(
        invocation.named_arg_as(builder, "q")?,
        invocation.named_arg_as(builder, "k_new")?,
        invocation.named_arg_as(builder, "v_new")?,
        invocation.named_arg_as(builder, "k_cache")?,
        invocation.named_arg_as(builder, "v_cache")?,
        invocation.named_arg_as(builder, "mask")?,
    );
    if has_sinks {
        inputs.push(invocation.named_arg_as(builder, "sinks")?);
    }
    let scale: f32 = invocation.named_arg_as(builder, "scale")?;
    let window: i64 = invocation.named_arg_as(builder, "window")?;
    builder.wire(
        FusedSdpa { scale_bits: scale.to_bits(), window: window as u32, has_sinks },
        &inputs,
    )
}

fn load_with_sinks(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<Value> {
    load_common(builder, invocation, true)
}

fn load_without_sinks(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<Value> {
    load_common(builder, invocation, false)
}

// ===================================================================================
// Detection: fuse the exported attention subgraph on the decluttered graph.
// Anchor = the attention Softmax. Variants of the same skeleton:
//
//   GPT-OSS:  Add(qk branches) -> Mul(scale) -> Cast(f32) -> Add(mask)
//             -> Concat(sinks) -> [Reduce<Max>, Sub] -> Softmax -> Slice(drop
//             sink) -> Cast(f16) -> Reshape([Hkv,G,S,T]) -> EinSum(@V)
//   Qwen3.5:  Add(qk branches) -> Mul(scale) -> Cast(f32) -> Add(mask)
//             -> Softmax -> Cast(f16) -> Reshape([Hkv,G,S,T]) -> EinSum(@V)
//   Granite:  EinSum(f32 casts inside) -> Mul(scale) -> Add(mask) -> Softmax
//             -> Cast(f16) -> Reshape([Hkv,G,S,T]) -> EinSum(@V)
//   OLMoE:    EinSum -> Mul(scale) -> Cast(f32) -> Add(mask) -> Softmax
//             -> Cast(f16) -> EinSum(@V)  (MHA: no GQA probs fold, group = 1)
//
// A qk branch is [Reshape](EinSum([Cast][Reshape](q_part), k_side)); the outer
// reshape, the q-side f32 cast and the q-side layout reshape are each optional.
// k_side is `Slice(expand(KConcat))` covering one head_dim range (the rope
// split), or directly `expand(KConcat)` (single-branch models, start = 0; the
// expand chain may be empty on MHA exports). Rope-split exports sum two such
// branches. q parts may be laid out [1,Hq,S,d] or [1,S,Hq,d]. The f32 cast
// between Mul(scale) and Add(mask) exists only when the QK einsum ran in f16.
// The AV einsum output is [Hkv,G,S,D], [1,Hkv,G,S,D], or, on MHA exports
// without the probs fold, [Hq,S,D] or [S,Hq,D]. K/V concats are
// `concat([in_cache, new], axis=2)` and must be model outputs (their slots get
// the op's views).
// ===================================================================================

use tract_nnef::tract_core::ops::array::{MultiBroadcastTo, Slice, TypedConcat};
use tract_nnef::tract_core::ops::binary::TypedBinOp;
use tract_nnef::tract_core::ops::cast::Cast;
use tract_nnef::tract_core::ops::math::Mul;
use tract_nnef::tract_core::ops::einsum::EinSum;
use tract_nnef::tract_core::ops::konst::Const;
use tract_nnef::tract_core::ops::nn::{Reduce, Reducer, Softmax};
use tract_nnef::tract_core::ops::source::TypedSource;
use tract_nnef::tract_core::transform::ModelTransform;

#[derive(Debug)]
pub struct FusedSdpaTransform;

impl ModelTransform for FusedSdpaTransform {
    fn name(&self) -> StaticName {
        "fuse_sdpa_inplace_kv".into()
    }
    fn transform(&self, model: &mut TypedModel) -> TractResult<()> {
        Rewriter::default()
            .with_rule_for("fuse-sdpa-inplace-kv", fuse_sdpa_rule)
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

/// Walk a GQA-expansion chain (AddAxis/Reshape/MultiBroadcastTo/Cast, in any
/// order) from `outlet` up to the KV cache concat. Returns the concat node id.
fn expand_chain_to_concat(model: &TypedModel, outlet: OutletId) -> Option<usize> {
    let mut id = outlet.node;
    for _ in 0..8 {
        let n = model.node(id);
        if n.op_as::<TypedConcat>().is_some() {
            return Some(id);
        }
        if n.op_is::<AxisOp>()
            || n.op_as::<MultiBroadcastTo>().is_some()
            || n.op_as::<Cast>().is_some()
        {
            id = n.inputs[0].node;
        } else {
            return None;
        }
    }
    None
}

/// Walk one QK einsum branch back to (q_part outlet, K concat node, head_dim
/// slice start). The branch is `[Reshape](EinSum([Cast][Reshape](q_part),
/// k_side))` where k_side is `Slice(expand(KConcat))` (rope-split models) or
/// directly `expand(KConcat)` (single-branch models, start = 0). The outer
/// reshape, the q-side f32 cast (einsum-in-f32 exports) and the q-side layout
/// reshape are each optional; the q-part shape checks in the caller guard the
/// tolerant walk.
fn qk_branch(
    model: &TypedModel,
    branch_root: &TypedNode,
) -> Option<(OutletId, usize, i64)> {
    let einsum =
        if branch_root.op_is::<AxisOp>() { prev(model, branch_root, 0) } else { branch_root };
    einsum.op_as::<EinSum>()?;
    if einsum.inputs.len() != 2 {
        return None;
    }
    let mut q_part = einsum.inputs[0];
    if model.node(q_part.node).op_as::<Cast>().is_some() {
        q_part = model.node(q_part.node).inputs[0];
    }
    if model.node(q_part.node).op_is::<AxisOp>() {
        q_part = model.node(q_part.node).inputs[0];
    }
    let k_side = prev(model, einsum, 1);
    let (start, k_expand_root) = if let Some(slice) = k_side.op_as::<Slice>() {
        let start = slice.start.to_i64().ok()?;
        (start, k_side.inputs[0])
    } else {
        (0, OutletId::new(k_side.id, 0))
    };
    let kconcat = expand_chain_to_concat(model, k_expand_root)?;
    Some((q_part, kconcat, start))
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
                "fused-sdpa fuse: mask subgraph has several window-like constants {many:?}; not clamping"
            );
            0
        }
    }
}

pub fn fuse_sdpa_rule(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    node_name: &str,
    _softmax: &Softmax,
) -> TractResult<Option<TypedModelPatch>> {
    // ---- variant detection at the softmax input ----
    // GPT-OSS: Softmax(Sub(Concat([logits, sinks]), Reduce<Max>(same Concat)))
    // Plain:   Softmax(Add(logits, mask))
    let sm_in = prev(model, node, 0);
    let (mask_add, sinks_tensor) = if is_bin(sm_in, "Sub") {
        let cat = prev(model, sm_in, 0);
        let mx = prev(model, sm_in, 1);
        let Some(concat) = cat.op_as::<TypedConcat>() else { return Ok(None) };
        if cat.inputs.len() != 2 || concat.axis != cat.outputs[0].fact.rank() - 1 {
            return Ok(None);
        }
        if !mx.op_as::<Reduce>().is_some_and(|r| r.reducer == Reducer::Max)
            || mx.inputs[0] != sm_in.inputs[0]
        {
            return Ok(None);
        }
        let sinks_bcast = prev(model, cat, 1);
        if sinks_bcast.op_as::<MultiBroadcastTo>().is_none() {
            return Ok(None);
        }
        let Some(sinks_k) = prev(model, sinks_bcast, 0).op_as::<Const>() else {
            return Ok(None);
        };
        (prev(model, cat, 0), Some(sinks_k.val().clone()))
    } else if is_bin(sm_in, "Add") {
        (sm_in, None)
    } else {
        return Ok(None);
    };
    let has_sinks = sinks_tensor.is_some();

    // ---- shared upstream: Add(mask) <- [Cast(f32)] <- Mul(scale) <- QK ----
    // The cast exists when the QK einsum ran in f16 (GPT-OSS, Qwen3.5, OLMoE);
    // einsum-in-f32 exports (Granite) go Mul -> Add directly.
    if !is_bin(mask_add, "Add") {
        return Ok(None);
    }
    let mask_outlet = mask_add.inputs[1];
    let mut scale_mul = prev(model, mask_add, 0);
    if scale_mul.op_as::<Cast>().is_some() {
        scale_mul = prev(model, scale_mul, 0);
    }
    if !is_bin(scale_mul, "Mul") {
        return Ok(None);
    }
    // Scalar scale constant on either side of the Mul (exports are not
    // consistent about binop operand order).
    let Some(scale_slot) = (0..2).find(|&s| {
        prev(model, scale_mul, s).op_as::<Const>().is_some_and(|k| k.val().len() == 1)
    }) else {
        return Ok(None);
    };
    let scale_k = prev(model, scale_mul, scale_slot).op_as::<Const>().unwrap();
    let scale = scale_k.val().cast_to::<f32>()?.try_as_plain()?.as_slice::<f32>()?[0];

    // ---- QK branches: rope-split exports sum two dim-range einsums ----
    let qk_root = prev(model, scale_mul, 1 - scale_slot);
    // Exports that upcast q/k to f32 for the QK matmul (Granite: q max ~73,
    // k max ~214, dots way past f16 max 65504) rely on f32 logits; device
    // runtimes compute scores in f16 and NaN out on them. Detected here so
    // the patch can fold the softmax scale into q up front: post-scale
    // logits are softmax-sized, so f16 scores are safe again. f16-QK
    // exports are left untouched (bit-faithful to their own graph).
    let qk_f32 = model.outlet_fact(scale_mul.inputs[1 - scale_slot])?.datum_type
        == DatumType::F32;
    let mut branches: Vec<(OutletId, usize, i64)> = vec![];
    if is_bin(qk_root, "Add") {
        for slot in 0..2 {
            let Some(b) = qk_branch(model, prev(model, qk_root, slot)) else {
                return Ok(None);
            };
            branches.push(b);
        }
    } else if let Some(b) = qk_branch(model, qk_root) {
        branches.push(b);
    } else {
        return Ok(None);
    }
    if !branches.iter().all(|(_, kc, _)| *kc == branches[0].1) {
        return Ok(None);
    }
    branches.sort_by_key(|(_, _, start)| *start);
    let kconcat = model.node(branches[0].1);
    if kconcat.inputs.len() != 2
        || kconcat.op_as::<TypedConcat>().is_none_or(|c| c.axis != SEQ_AXIS)
        || prev(model, kconcat, 0).op_as::<TypedSource>().is_none()
    {
        return Ok(None);
    }
    let k_cache = kconcat.inputs[0];
    let k_new = kconcat.inputs[1];
    let k_fact = model.outlet_fact(OutletId::new(kconcat.id, 0))?;
    if k_fact.rank() != 4 {
        return Ok(None);
    }
    let Ok(hkv) = k_fact.shape[1].to_usize() else { return Ok(None) };
    let Ok(d) = k_fact.shape[3].to_usize() else { return Ok(None) };

    // ---- downstream: [drop sink slice], cast, probs reshape, AV einsum ----
    let mut cursor = node;
    if has_sinks {
        let Some(drop_sink) = single_consumer(model, cursor) else { return Ok(None) };
        if drop_sink.op_as::<Slice>().is_none() {
            return Ok(None);
        }
        cursor = drop_sink;
    }
    let Some(cast_f16) = single_consumer(model, cursor) else { return Ok(None) };
    if cast_f16.op_as::<Cast>().is_none() {
        return Ok(None);
    }
    let Some(after_cast) = single_consumer(model, cast_f16) else { return Ok(None) };
    let (group, av) = if after_cast.op_is::<AxisOp>() {
        // GQA: probs fold to [Hkv, G, S, T], the head split pins hkv/group.
        let probs_fact = &after_cast.outputs[0].fact;
        if probs_fact.rank() != 4 {
            return Ok(None);
        }
        let (Ok(p_hkv), Ok(group)) =
            (probs_fact.shape[0].to_usize(), probs_fact.shape[1].to_usize())
        else {
            return Ok(None);
        };
        if p_hkv != hkv {
            return Ok(None);
        }
        let Some(av) = single_consumer(model, after_cast) else { return Ok(None) };
        (group, av)
    } else {
        // MHA exports skip the fold: probs stay [Hq, S, T], group = 1.
        let probs_fact = &cast_f16.outputs[0].fact;
        if probs_fact.rank() != 3 || probs_fact.shape[0].to_usize().ok() != Some(hkv) {
            return Ok(None);
        }
        (1, after_cast)
    };
    let hq = hkv * group;
    if av.op_as::<EinSum>().is_none() || av.inputs.len() != 2 {
        return Ok(None);
    }
    let Some(vconcat_id) = expand_chain_to_concat(model, av.inputs[1]) else {
        return Ok(None);
    };
    let vconcat = model.node(vconcat_id);
    if vconcat.inputs.len() != 2
        || vconcat.op_as::<TypedConcat>().is_none_or(|c| c.axis != SEQ_AXIS)
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

    // ---- q parts: normalize each to [1, Hq, S, d_part] and check coverage ----
    // Parts appear either head-major [1,Hq,S,d] or seq-major [1,S,Hq,d]; S is
    // symbolic at fuse time so matching Hq against a concrete dim is safe.
    enum Layout {
        HeadMajor,
        SeqMajor,
    }
    let mut parts: Vec<(OutletId, Layout, usize)> = vec![];
    let mut covered = 0usize;
    for (q_part, _, start) in &branches {
        let f = model.outlet_fact(*q_part)?;
        if f.rank() != 4 || f.shape[0].to_usize().ok() != Some(1) {
            return Ok(None);
        }
        let Ok(d_part) = f.shape[3].to_usize() else { return Ok(None) };
        if *start != covered as i64 {
            return Ok(None);
        }
        let layout = if f.shape[1].to_usize().ok() == Some(hq) && f.shape[2].to_usize().is_err()
        {
            Layout::HeadMajor
        } else if f.shape[2].to_usize().ok() == Some(hq) {
            Layout::SeqMajor
        } else if f.shape[1].to_usize().ok() == Some(hq) {
            Layout::HeadMajor
        } else {
            return Ok(None);
        };
        covered += d_part;
        parts.push((*q_part, layout, d_part));
    }
    if covered != d {
        return Ok(None);
    }

    // AV output must be a known re-layout of the op's [1,Hq,S,D]: the GQA
    // foldings [Hkv,G,S,D] / [1,Hkv,G,S,D], or the MHA rank-3 forms [Hq,S,D]
    // (head-major) / [S,Hq,D] (seq-major). S is symbolic at fuse time, so
    // matching concrete head dims disambiguates the rank-3 layouts.
    #[derive(Clone, Copy, PartialEq)]
    enum AvLayout {
        Fold4,
        Fold5,
        HeadMajor3,
        SeqMajor3,
    }
    let av_fact = &av.outputs[0].fact;
    let av_layout = match av_fact.rank() {
        3 if group == 1
            && av_fact.shape[0].to_usize().ok() == Some(hq)
            && av_fact.shape[2].to_usize().ok() == Some(d) =>
        {
            Some(AvLayout::HeadMajor3)
        }
        3 if group == 1
            && av_fact.shape[1].to_usize().ok() == Some(hq)
            && av_fact.shape[2].to_usize().ok() == Some(d) =>
        {
            Some(AvLayout::SeqMajor3)
        }
        4 if av_fact.shape[0].to_usize().ok() == Some(hkv)
            && av_fact.shape[1].to_usize().ok() == Some(group)
            && av_fact.shape[3].to_usize().ok() == Some(d) =>
        {
            Some(AvLayout::Fold4)
        }
        5 if av_fact.shape[0].to_usize().ok() == Some(1)
            && av_fact.shape[1].to_usize().ok() == Some(hkv)
            && av_fact.shape[2].to_usize().ok() == Some(group)
            && av_fact.shape[4].to_usize().ok() == Some(d) =>
        {
            Some(AvLayout::Fold5)
        }
        _ => None,
    };
    let Some(av_layout) = av_layout else { return Ok(None) };

    // ---- build the patch ----
    let mut patch = TypedModelPatch::new(format!("fuse-sdpa-inplace-kv @ {node_name}"));
    let mut q_parts: Vec<OutletId> = vec![];
    for (i, (outlet, layout, _)) in parts.iter().enumerate() {
        let tapped = patch.tap_model(model, *outlet)?;
        let normalized = match layout {
            Layout::HeadMajor => tapped,
            Layout::SeqMajor => {
                patch.wire_node(
                    format!("{node_name}.q_part{i}_hsd"),
                    AxisOp::Move(2, 1),
                    &[tapped],
                )?[0]
            }
        };
        q_parts.push(normalized);
    }
    let q = if q_parts.len() == 1 {
        q_parts[0]
    } else {
        patch.wire_node(format!("{node_name}.q"), TypedConcat { axis: 3 }, &q_parts)?[0]
    };
    // f32-QK exports: fold the softmax scale into q (see `qk_f32` above) so
    // f16 device scores stay in range; the op then runs with scale 1. In q's
    // dtype the fold is exact for power-of-two scales (Granite: 2^-6); any
    // residual rounding is at f16 epsilon, far below the f16 QK rounding the
    // other exports already live with.
    let (q, op_scale) = if qk_f32 {
        let q_dt = patch.outlet_fact(q)?.datum_type;
        let scale_t =
            tensor0(scale).cast_to_dt(q_dt)?.into_owned().broadcast_into_rank(4)?;
        let scale_c =
            patch.add_const(format!("{node_name}.q_prescale"), scale_t.into_arc_tensor())?;
        let scaled = patch.wire_node(
            format!("{node_name}.q_prescaled"),
            TypedBinOp(Box::new(Mul), None),
            &[q, scale_c],
        )?[0];
        (scaled, 1f32)
    } else {
        (q, scale)
    };

    let k_new_t = patch.tap_model(model, k_new)?;
    let v_new_t = patch.tap_model(model, v_new)?;
    let k_cache_t = patch.tap_model(model, k_cache)?;
    let v_cache_t = patch.tap_model(model, v_cache)?;
    let mask_t = patch.tap_model(model, mask_outlet)?;
    let window = extract_sliding_window(model, mask_outlet);
    let mut op_inputs = tvec!(q, k_new_t, v_new_t, k_cache_t, v_cache_t, mask_t);
    if let Some(sinks) = sinks_tensor {
        op_inputs.push(patch.add_const(format!("{node_name}.sinks"), sinks)?);
    }

    let fused = patch.wire_node(
        format!("{node_name}.fused_sdpa"),
        FusedSdpa { scale_bits: op_scale.to_bits(), window, has_sinks },
        &op_inputs,
    )?;

    // Attention out [1, Hq, S, D] -> the AV einsum's output layout.
    let attn = match av_layout {
        AvLayout::Fold5 => patch.wire_node(
            format!("{node_name}.attn_reshape"),
            AxisOp::Reshape(1, tvec![hq.to_dim()], tvec![hkv.to_dim(), group.to_dim()]),
            &[fused[0]],
        )?[0],
        AvLayout::Fold4 => patch.wire_node(
            format!("{node_name}.attn_reshape"),
            AxisOp::Reshape(
                0,
                tvec![1.to_dim(), hq.to_dim()],
                tvec![hkv.to_dim(), group.to_dim()],
            ),
            &[fused[0]],
        )?[0],
        AvLayout::HeadMajor3 | AvLayout::SeqMajor3 => {
            let squeezed = patch.wire_node(
                format!("{node_name}.attn_rm_batch"),
                AxisOp::Rm(0),
                &[fused[0]],
            )?[0];
            if av_layout == AvLayout::SeqMajor3 {
                patch.wire_node(
                    format!("{node_name}.attn_seq_major"),
                    AxisOp::Move(0, 1),
                    &[squeezed],
                )?[0]
            } else {
                squeezed
            }
        }
    };

    patch.shunt_outside(model, OutletId::new(av.id, 0), attn)?;
    patch.shunt_outside(model, k_out_outlet, fused[1])?;
    patch.shunt_outside(model, v_out_outlet, fused[2])?;
    Ok(Some(patch))
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
    /// softmax(concat([QK*scale + mask, sinks?]))[..,:kv] @ V, with GQA expand.
    fn reference(
        q: &Tensor,
        k_all: &Tensor,
        v_all: &Tensor,
        mask: &Tensor,
        sinks: Option<&[f32]>,
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
                    if let Some(sinks) = sinks {
                        logits.push(sinks[h]);
                    }
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

    #[allow(clippy::too_many_arguments)]
    fn run_state(
        state: &mut FusedSdpaState,
        op: &FusedSdpa,
        q: &Tensor,
        k_new: &Tensor,
        v_new: &Tensor,
        k_cache: &Tensor,
        v_cache: &Tensor,
        mask: &Tensor,
        sinks: Option<&Tensor>,
    ) -> TVec<TValue> {
        let mut session = TurnState::default();
        let mut inputs = tvec!(
            q.clone().into_tvalue(),
            k_new.clone().into_tvalue(),
            v_new.clone().into_tvalue(),
            k_cache.clone().into_tvalue(),
            v_cache.clone().into_tvalue(),
            mask.clone().into_tvalue(),
        );
        if let Some(sinks) = sinks {
            inputs.push(sinks.clone().into_tvalue());
        }
        state.eval(&mut session, op, inputs).unwrap()
    }

    fn state_for(op: &FusedSdpa) -> FusedSdpaState {
        FusedSdpaState {
            scale: op.scale(),
            has_sinks: op.has_sinks,
            k: InPlaceKvCache::new(SEQ_AXIS),
            v: InPlaceKvCache::new(SEQ_AXIS),
        }
    }

    fn run_reference_check(hq: usize, hkv: usize, d: usize, with_sinks: bool) -> TractResult<()> {
        let scale = (d as f32).sqrt().recip();
        let op = FusedSdpa { scale_bits: scale.to_bits(), window: 0, has_sinks: with_sinks };
        let mut state = state_for(&op);
        let mut seed = 42u64;
        let sinks_t = with_sinks.then(|| rng_tensor(&[hq], &mut seed));
        let sinks: Option<Vec<f32>> = sinks_t
            .as_ref()
            .map(|t| t.try_as_plain().unwrap().as_slice::<f32>().unwrap().to_vec());

        // Accumulated "external" cache, grown the reference way.
        let mut k_all = Tensor::zero::<f32>(&[1, hkv, 0, d])?;
        let mut v_all = Tensor::zero::<f32>(&[1, hkv, 0, d])?;

        for step_len in [5usize, 1, 1, 3, 1] {
            let past = k_all.shape()[SEQ_AXIS];
            let q = rng_tensor(&[1, hq, step_len, d], &mut seed);
            let k_new = rng_tensor(&[1, hkv, step_len, d], &mut seed);
            let v_new = rng_tensor(&[1, hkv, step_len, d], &mut seed);
            let mask = causal_mask(step_len, past + step_len);

            let outputs = run_state(
                &mut state,
                &op,
                &q,
                &k_new,
                &v_new,
                &k_all,
                &v_all,
                &mask,
                sinks_t.as_ref(),
            );

            // Reference over the full concatenated cache.
            let k_ref = Tensor::stack_tensors(SEQ_AXIS, &[&k_all, &k_new])?;
            let v_ref = Tensor::stack_tensors(SEQ_AXIS, &[&v_all, &v_new])?;
            let want = reference(&q, &k_ref, &v_ref, &mask, sinks.as_deref(), scale);
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
    fn matches_reference_over_prefill_and_decode_with_sinks() -> TractResult<()> {
        run_reference_check(4, 2, 8, true)
    }

    #[test]
    fn matches_reference_no_sinks() -> TractResult<()> {
        run_reference_check(4, 2, 8, false)
    }

    /// Qwen3.5-35B full-attention geometry: head_dim 256, GQA 16q/2kv, no
    /// sinks, no window.
    #[test]
    fn matches_reference_qwen35_geometry() -> TractResult<()> {
        run_reference_check(16, 2, 256, false)
    }

    /// Granite 3.0 MoE geometry: head_dim 64, GQA 16q/8kv, no sinks.
    #[test]
    fn matches_reference_granite_geometry() -> TractResult<()> {
        run_reference_check(16, 8, 64, false)
    }

    /// OLMoE geometry: MHA 16q/16kv (group 1), head_dim 128, no sinks.
    #[test]
    fn matches_reference_olmoe_geometry() -> TractResult<()> {
        run_reference_check(16, 16, 128, false)
    }

    #[test]
    fn rebuilds_state_on_cache_mismatch() -> TractResult<()> {
        let (hq, hkv, d) = (2, 1, 4);
        let scale = 0.5f32;
        let op = FusedSdpa { scale_bits: scale.to_bits(), window: 0, has_sinks: true };
        let mut state = state_for(&op);
        let mut seed = 7u64;
        let sinks_t = rng_tensor(&[hq], &mut seed);
        let sinks: Vec<f32> = sinks_t.try_as_plain()?.as_slice::<f32>()?.to_vec();

        // Prime the state with a 4-token pass.
        let q0 = rng_tensor(&[1, hq, 4, d], &mut seed);
        let k0 = rng_tensor(&[1, hkv, 4, d], &mut seed);
        let v0 = rng_tensor(&[1, hkv, 4, d], &mut seed);
        let empty_k = Tensor::zero::<f32>(&[1, hkv, 0, d])?;
        let empty_v = Tensor::zero::<f32>(&[1, hkv, 0, d])?;
        run_state(
            &mut state,
            &op,
            &q0,
            &k0,
            &v0,
            &empty_k,
            &empty_v,
            &causal_mask(4, 4),
            Some(&sinks_t),
        );

        // Now pretend the caller truncated to 2 tokens (retry): the provided
        // cache disagrees with state len 4 and must win.
        let k_trunc = k0.slice(SEQ_AXIS, 0, 2)?;
        let v_trunc = v0.slice(SEQ_AXIS, 0, 2)?;
        let q1 = rng_tensor(&[1, hq, 1, d], &mut seed);
        let k1 = rng_tensor(&[1, hkv, 1, d], &mut seed);
        let v1 = rng_tensor(&[1, hkv, 1, d], &mut seed);
        let mask = causal_mask(1, 3);
        let outputs = run_state(
            &mut state,
            &op,
            &q1,
            &k1,
            &v1,
            &k_trunc,
            &v_trunc,
            &mask,
            Some(&sinks_t),
        );

        let k_ref = Tensor::stack_tensors(SEQ_AXIS, &[&k_trunc, &k1])?;
        let v_ref = Tensor::stack_tensors(SEQ_AXIS, &[&v_trunc, &v1])?;
        let want = reference(&q1, &k_ref, &v_ref, &mask, Some(&sinks), scale);
        outputs[0].clone().into_tensor().close_enough(&want, Approximation::Approximate)?;
        outputs[1].clone().into_tensor().close_enough(&k_ref, Approximation::Exact)?;
        Ok(())
    }
}
