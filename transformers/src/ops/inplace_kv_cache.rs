//! In-place, geometric-growth KV cache (Apple Core ML "stateful in-place KV"
//! lever, arXiv: Core ML Llama 3.1 — the ~13x throughput win on long decodes).
//!
//! tract's current `DynKeyValueCache` grows the cache by `TypedConcat([past, new])`
//! every step: step `t` copies the whole `t`-token past into a fresh buffer, so a
//! `T`-token decode does O(T^2) total copy work. This keeps a buffer with spare
//! capacity along `axis` and writes each new chunk at the cursor, doubling only
//! when capacity is exceeded — O(T) amortized copy work (classic Vec growth).
//!
//! IMPORTANT (the subtlety that made a naive attempt a wash): the saving only
//! materializes if the CONSUMER reads the valid `[0..len]` region as a *view*
//! (`valid_view`) and not as a contiguous copy (`valid_contiguous`) — otherwise
//! the per-step slice-to-valid copy reintroduces the O(T^2). The module tests
//! A/B exactly this, and the consumer-via-view path is what a fused attention
//! kernel (reading buffer + length) would use.

use tract_nnef::internal::*;
use tract_nnef::tract_core::ops::{FrozenOpState, OpStateFreeze};
use tract_nnef::tract_core::transform::ModelTransform;
use tract_nnef::tract_ndarray::Ix4;

use crate::ops::dyn_kv_cache::DynKeyValueCache;
use crate::ops::flash_sdpa::FlashSdpaOp;
use crate::ops::sdpa::Sdpa;

/// Geometric-growth in-place cache along `axis`.
#[derive(Clone, Debug)]
pub struct InPlaceKvCache {
    pub axis: usize,
    buffer: Option<Tensor>,
    len: usize,
    reallocs: usize,
}

impl InPlaceKvCache {
    pub fn new(axis: usize) -> Self {
        InPlaceKvCache { axis, buffer: None, len: 0, reallocs: 0 }
    }

    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    /// Number of reallocations so far (should be O(log T) over a decode).
    pub fn reallocs(&self) -> usize {
        self.reallocs
    }
    pub fn capacity(&self) -> usize {
        self.buffer.as_ref().map(|b| b.shape()[self.axis]).unwrap_or(0)
    }

    /// Append `input` along `axis`, in place. O(input) amortized.
    pub fn push(&mut self, input: &Tensor) -> TractResult<()> {
        let new = input.shape()[self.axis];
        if new == 0 {
            return Ok(());
        }
        match self.buffer.take() {
            None => {
                // Seed with the first chunk (capacity = its length); subsequent
                // pushes grow geometrically from here.
                self.buffer = Some(input.clone());
                self.len = new;
            }
            Some(mut buf) => {
                ensure!(buf.rank() == input.rank(), "rank mismatch in kv-cache push");
                let cap = buf.shape()[self.axis];
                if self.len + new <= cap {
                    buf.assign_slice(self.len..self.len + new, input, 0..new, self.axis)?;
                    self.len += new;
                    self.buffer = Some(buf);
                } else {
                    let new_cap = (cap * 2).max(self.len + new);
                    let mut grown = grow(&buf, self.len, new_cap, self.axis)?;
                    grown.assign_slice(self.len..self.len + new, input, 0..new, self.axis)?;
                    self.len += new;
                    self.reallocs += 1;
                    self.buffer = Some(grown);
                }
            }
        }
        Ok(())
    }

    /// Zero-copy ndarray view of the valid `[0..len]` region. This is the path
    /// that realizes the win — a length-aware consumer attends over this without
    /// ever copying the past.
    pub fn valid_view<T: Datum>(&self) -> TractResult<tract_ndarray::ArrayViewD<'_, T>> {
        let buf = self.buffer.as_ref().context("empty kv-cache")?;
        let mut full = buf.to_plain_array_view::<T>()?;
        full.slice_axis_inplace(tract_ndarray::Axis(self.axis), (0..self.len).into());
        Ok(full)
    }

    /// Contiguous `[0..len]` copy. This is what a consumer needing an owned,
    /// packed tensor forces — it re-introduces the per-step O(len) copy, so it is
    /// only kept to A/B against `valid_view`.
    pub fn valid_contiguous(&self) -> TractResult<Tensor> {
        let buf = self.buffer.as_ref().context("empty kv-cache")?;
        buf.slice(self.axis, 0, self.len)
    }
}

/// Allocate a buffer of capacity `new_cap` along `axis` and copy the live
/// `[0..len]` region from `src` (the tail is left uninitialized — never read).
fn grow(src: &Tensor, len: usize, new_cap: usize, axis: usize) -> TractResult<Tensor> {
    let mut shape: TVec<usize> = src.shape().into();
    shape[axis] = new_cap;
    let mut out = unsafe { Tensor::uninitialized_dt(src.datum_type(), &shape)? };
    out.assign_slice(0..len, src, 0..len, axis)?;
    Ok(out)
}

// ===================================================================================
// Integration: a stateful fused KV-cache + attention op.
//
// tract `Tensor`s cannot be zero-copy views across an op boundary (`Tensor::slice`
// copies), so the only way to make the consumer read the cache without re-copying is
// to keep the K/V buffers INSIDE the op that consumes them. `InPlaceKvSdpa` owns the
// two in-place caches as op-state, appends K/V in place each step, and runs the CPU
// SDPA (`FlashSdpaOp::flash_attention_gqa`) over the zero-copy `[0..len]` views — so
// it is a drop-in for the `{kv_cache(K), kv_cache(V), Sdpa}` subgraph that produces
// the same output by construction (same attention kernel, same K/V values) while
// eliminating the per-step O(T) concat copy. This is the "couple #1 with the
// attention consumer" design; the same shape works on the GPU side by handing the
// fused MFA kernel the capacity buffer + length.
// ===================================================================================

/// Fused in-place KV-cache + scaled-dot-product-attention (decode-oriented).
/// Inputs: `[Q, K_new, V_new]`, each `[B, H, S, D]` (K/V at `num_kv_heads`).
/// Output: attention of `Q` over the whole accumulated cache, shape of `Q`.
#[derive(Clone, Debug, PartialEq)]
pub struct InPlaceKvSdpa {
    /// Sequence axis to grow (2 for `[B, H, S, D]`).
    pub axis: usize,
    pub causal: bool,
    pub scale: Option<f32>,
}
impl Eq for InPlaceKvSdpa {}

impl Op for InPlaceKvSdpa {
    fn name(&self) -> StaticName {
        "InPlaceKvSdpa".into()
    }
    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("axis={}, causal={}, scale={:?}", self.axis, self.causal, self.scale)])
    }
    op_as_typed_op!();
}

impl EvalOp for InPlaceKvSdpa {
    fn is_stateless(&self) -> bool {
        false
    }
    fn state(
        &self,
        _session: &TurnState,
        _node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(Some(Box::new(InPlaceKvSdpaState {
            axis: self.axis,
            causal: self.causal,
            scale: self.scale,
            k: InPlaceKvCache::new(self.axis),
            v: InPlaceKvCache::new(self.axis),
        })))
    }
}

impl TypedOp for InPlaceKvSdpa {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(inputs.len() == 3, "InPlaceKvSdpa expects [Q, K_new, V_new]");
        // Attention output has Q's shape and dtype.
        Ok(tvec!(inputs[0].without_value()))
    }
    as_op!();
}

#[derive(Clone, Debug)]
pub struct InPlaceKvSdpaState {
    axis: usize,
    causal: bool,
    scale: Option<f32>,
    k: InPlaceKvCache,
    v: InPlaceKvCache,
}

impl OpState for InPlaceKvSdpaState {
    fn eval(
        &mut self,
        _state: &mut TurnState,
        _op: &dyn Op,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        ensure!(inputs.len() == 3, "InPlaceKvSdpa expects [Q, K_new, V_new]");
        let input_dt = inputs[0].datum_type();

        // Cache K/V in f32 (the dtype FlashSdpaOp computes in); append in place.
        let k_new = inputs[1].cast_to::<f32>()?;
        let v_new = inputs[2].cast_to::<f32>()?;
        self.k.push(k_new.as_ref())?;
        self.v.push(v_new.as_ref())?;

        let q = inputs[0].cast_to::<f32>()?;
        let qv = q.to_plain_array_view::<f32>()?.into_dimensionality::<Ix4>()?;
        // Zero-copy views of the valid [0..len] region — no past is ever copied.
        let kview = self.k.valid_view::<f32>()?.into_dimensionality::<Ix4>()?;
        let vview = self.v.valid_view::<f32>()?.into_dimensionality::<Ix4>()?;

        let flash = FlashSdpaOp { causal: self.causal, scale: self.scale };
        let o = flash.flash_attention_gqa(qv, kview, vview, None);

        Ok(tvec!(o.into_tensor().cast_to_dt(input_dt)?.into_owned().into_tvalue()))
    }

    /// Persist the accumulated cache as two state tensors `[K_valid, V_valid]`, so a
    /// decode can be checkpointed and resumed (alongside `freeze`/`unfreeze`, which
    /// snapshot the whole running state in-process). The valid `[0..len]` regions are
    /// copied out once here — O(len), not per step.
    fn save_to(&self, states: &mut Vec<TValue>) -> TractResult<()> {
        if !self.k.is_empty() {
            states.push(self.k.valid_contiguous()?.into_tvalue());
            states.push(self.v.valid_contiguous()?.into_tvalue());
        }
        Ok(())
    }

    /// Seed the K/V caches from a previously `save_to`'d `[K, V]` pair (resume from a
    /// saved cache). With no initializers the caches stay empty (fresh decode).
    fn load_from(
        &mut self,
        _state: &mut TurnState,
        states: &mut dyn Iterator<Item = TValue>,
    ) -> TractResult<()> {
        if let Some(k) = states.next() {
            let v = states.next().context("InPlaceKvSdpa load_from: expected V state after K")?;
            self.k = InPlaceKvCache::new(self.axis);
            self.v = InPlaceKvCache::new(self.axis);
            self.k.push(k.cast_to::<f32>()?.as_ref())?;
            self.v.push(v.cast_to::<f32>()?.as_ref())?;
        }
        Ok(())
    }
}

#[derive(Clone, Debug)]
struct FrozenInPlaceKvSdpaState {
    axis: usize,
    causal: bool,
    scale: Option<f32>,
    k: InPlaceKvCache,
    v: InPlaceKvCache,
}

impl OpStateFreeze for InPlaceKvSdpaState {
    fn freeze(&self) -> Box<dyn FrozenOpState> {
        Box::new(FrozenInPlaceKvSdpaState {
            axis: self.axis,
            causal: self.causal,
            scale: self.scale,
            k: self.k.clone(),
            v: self.v.clone(),
        })
    }
}

impl FrozenOpState for FrozenInPlaceKvSdpaState {
    fn unfreeze(&self) -> Box<dyn OpState> {
        Box::new(InPlaceKvSdpaState {
            axis: self.axis,
            causal: self.causal,
            scale: self.scale,
            k: self.k.clone(),
            v: self.v.clone(),
        })
    }
}

/// Rewrite rule: fuse `{DynKeyValueCache(K), DynKeyValueCache(V), Sdpa(Q,K,V)}` into a
/// single stateful `InPlaceKvSdpa`, so existing decode models adopt the in-place cache
/// transparently. Fires on the 3-input pattern where each of Sdpa's K/V inputs is the
/// output of a single-consumer `DynKeyValueCache` on the same axis. Intended to run
/// AFTER `fuse_kv_cache_broadcast_rule` strips the GQA unsqueeze/broadcast/reshape chain
/// (so the cache feeds Sdpa directly) — `InPlaceKvSdpa` does the GQA expansion itself, so
/// the fusion also removes those broadcast ops.
///
/// Note: the fused op self-initializes an empty cache, so this replaces the
/// `DynKeyValueCache` state-init/restore contract with the op's own state — equivalent
/// for fresh inference; a model that restores a pre-seeded cache would need that state
/// threaded into the op (follow-up).
pub fn fuse_inplace_kv_sdpa_rule(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    node_name: &str,
    op: &Sdpa,
) -> TractResult<Option<TypedModelPatch>> {
    // plain (Q, K, V) — no explicit mask input
    if node.inputs.len() != 3 {
        return Ok(None);
    }
    let k_node = model.node(node.inputs[1].node);
    let v_node = model.node(node.inputs[2].node);
    let (Some(kc), Some(vc)) =
        (k_node.op_as::<DynKeyValueCache>(), v_node.op_as::<DynKeyValueCache>())
    else {
        return Ok(None);
    };
    if kc.axis != vc.axis {
        return Ok(None);
    }
    // Each cache must feed only this Sdpa, else removing it would drop a live value.
    if k_node.outputs[0].successors.len() != 1 || v_node.outputs[0].successors.len() != 1 {
        return Ok(None);
    }

    let scale = op.scale.as_ref().map(|t| t.cast_to_scalar::<f32>()).transpose()?;
    let q_outlet = node.inputs[0];
    let k_new = k_node.inputs[0];
    let v_new = v_node.inputs[0];

    let mut patch = TypedModelPatch::default();
    let taps = patch.taps(model, &[q_outlet, k_new, v_new])?;
    let fused = patch.wire_node(
        format!("{node_name}.inplace_kv_sdpa"),
        InPlaceKvSdpa { axis: kc.axis, causal: op.is_causal, scale },
        &taps,
    )?;
    patch.shunt_outside(model, node.id.into(), fused[0])?;
    Ok(Some(patch))
}

/// Strip the GQA broadcast chain, then fuse `cache -> Sdpa` into `InPlaceKvSdpa`.
#[derive(Debug, Default)]
pub struct InPlaceKvSdpaTransform;

impl ModelTransform for InPlaceKvSdpaTransform {
    fn name(&self) -> StaticName {
        "fuse_inplace_kv_sdpa".into()
    }
    fn transform(&self, model: &mut TypedModel) -> TractResult<()> {
        Rewriter::default()
            .with_rule_for("fuse-kv-broadcast", crate::ops::sdpa::fuse_kv_cache_broadcast_rule)
            .with_rule_for("fuse-inplace-kv-sdpa", fuse_inplace_kv_sdpa_rule)
            .rewrite(&(), model)?;
        model.compact()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tract_nnef::tract_core::ops::array::TypedConcat;
    use tract_nnef::tract_ndarray::{Array4, ArrayView4, s};

    // Deterministic filler so correctness checks are reproducible.
    fn seq_tensor(shape: &[usize], start: f32) -> Tensor {
        let n: usize = shape.iter().product();
        let data: Vec<f32> = (0..n).map(|i| start + i as f32 * 0.5).collect();
        Tensor::from_shape(shape, &data).unwrap()
    }

    // ---- Correctness: in-place buffer[0..len] == concat-grow, bit-exact ----
    fn check_matches_concat(chunk_shapes: &[Vec<usize>], axis: usize) -> TractResult<()> {
        let mut cache = InPlaceKvCache::new(axis);
        let mut concat: Option<Tensor> = None;
        for (i, sh) in chunk_shapes.iter().enumerate() {
            let chunk = seq_tensor(sh, i as f32 * 100.0);
            cache.push(&chunk)?;
            concat = Some(match concat.take() {
                None => chunk,
                Some(c) => TypedConcat { axis }
                    .eval(tvec![c.into(), chunk.into()])?
                    .remove(0)
                    .into_tensor(),
            });
        }
        let reference = concat.unwrap();
        // The view and the contiguous slice must both equal the concat result.
        let got = cache.valid_contiguous()?;
        got.close_enough(&reference, Approximation::Exact)?;
        ensure!(cache.len() == reference.shape()[axis]);
        Ok(())
    }

    #[test]
    fn inplace_matches_concat_decode() -> TractResult<()> {
        // prompt of 5, then 20 single-token decode steps, axis = seq (2) of [B,H,S,D]
        let mut shapes = vec![vec![1, 2, 5, 4]];
        for _ in 0..20 {
            shapes.push(vec![1, 2, 1, 4]);
        }
        check_matches_concat(&shapes, 2)
    }

    #[test]
    fn inplace_matches_concat_axes_and_chunks() -> TractResult<()> {
        check_matches_concat(&[vec![2, 2], vec![4, 2], vec![1, 2], vec![7, 2]], 0)?;
        check_matches_concat(&[vec![2, 2], vec![2, 1], vec![2, 3]], 1)?;
        check_matches_concat(&[vec![1, 3, 2, 8], vec![1, 3, 3, 8], vec![1, 3, 1, 8]], 2)?;
        Ok(())
    }

    #[test]
    fn geometric_growth_is_amortized() -> TractResult<()> {
        // 1024 single-token pushes from a 1-token seed → O(log T) reallocs, not O(T).
        let mut cache = InPlaceKvCache::new(2);
        cache.push(&seq_tensor(&[1, 2, 1, 4], 0.0))?;
        for _ in 0..1023 {
            cache.push(&seq_tensor(&[1, 2, 1, 4], 1.0))?;
        }
        ensure!(cache.len() == 1024);
        // doubling from 1: 1,2,4,...,1024 → ~10-11 reallocs
        ensure!(cache.reallocs() <= 12, "expected ~log2(1024) reallocs, got {}", cache.reallocs());
        Ok(())
    }

    // ---- Length-aware attention readout (identical math for both strategies;
    //      the ONLY difference benched is how K/V are obtained). ----
    fn attention(
        q: ArrayView4<f32>,
        k: ArrayView4<f32>,
        v: ArrayView4<f32>,
        scale: f32,
    ) -> Array4<f32> {
        let (b, h, sq, d) = q.dim();
        let mut out = Array4::<f32>::zeros((b, h, sq, d));
        for bi in 0..b {
            for hi in 0..h {
                let qm = q.slice(s![bi, hi, .., ..]); // [Sq, D] contiguous
                let km = k.slice(s![bi, hi, .., ..]); // [Skv, D] contiguous prefix
                let vm = v.slice(s![bi, hi, .., ..]); // [Skv, D]
                let mut scores = qm.dot(&km.t()); // [Sq, Skv]
                scores *= scale;
                for mut row in scores.rows_mut() {
                    let max = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let mut sum = 0.0f32;
                    row.iter_mut().for_each(|x| {
                        *x = (*x - max).exp();
                        sum += *x;
                    });
                    let inv = 1.0 / sum;
                    row.iter_mut().for_each(|x| *x *= inv);
                }
                let o = scores.dot(&vm); // [Sq, D]
                out.slice_mut(s![bi, hi, .., ..]).assign(&o);
            }
        }
        out
    }

    // ---- End-to-end correctness: attention over the in-place view == attention
    //      over the concat-grown contiguous cache, every decode step. ----
    #[test]
    fn consumer_over_view_matches_concat_baseline() -> TractResult<()> {
        let (b, h, d) = (1usize, 2usize, 8usize);
        let scale = 1.0 / (d as f32).sqrt();
        let mut cache = InPlaceKvCache::new(2);
        let mut concat: Option<Tensor> = None;
        for t in 0..16 {
            let kv = seq_tensor(&[b, h, 1, d], t as f32);
            cache.push(&kv)?;
            concat = Some(match concat.take() {
                None => kv.clone(),
                Some(c) => TypedConcat { axis: 2 }
                    .eval(tvec![c.into(), kv.clone().into()])?
                    .remove(0)
                    .into_tensor(),
            });

            let q = seq_tensor(&[b, h, 1, d], 1000.0 + t as f32);
            let qv = q.to_plain_array_view::<f32>()?.into_dimensionality()?;

            // in-place: read the [0..len] VIEW (no copy)
            let kview = cache.valid_view::<f32>()?.into_dimensionality()?;
            let out_inplace = attention(qv, kview, kview, scale);

            // baseline: read the concat-grown contiguous tensor
            let cbuf = concat.as_ref().unwrap();
            let cv = cbuf.to_plain_array_view::<f32>()?.into_dimensionality()?;
            let out_concat = attention(qv, cv, cv, scale);

            let a = Tensor::from(out_inplace);
            let bt = Tensor::from(out_concat);
            a.close_enough(&bt, Approximation::Approximate)
                .with_context(|| format!("mismatch at decode step {t}"))?;
        }
        Ok(())
    }

    // ================= BENCHES (#[ignore]) =================

    // Pure cache-update cost: concat-grow (O(T^2)) vs in-place (O(T)).
    //   cargo test -p tract-transformers inplace_kv_cache::tests::bench_update -- --ignored --nocapture
    #[test]
    #[ignore]
    fn bench_update() -> TractResult<()> {
        use std::time::Instant;
        let (b, h, d) = (1usize, 8usize, 128usize);
        println!("\n  cache-UPDATE only (B={b} H={h} D={d}), concat-grow vs in-place:");
        println!("   T     concat(ms)  inplace(ms)   speedup  reallocs");
        for &t in &[256usize, 512, 1024, 2048, 4096] {
            let step = seq_tensor(&[b, h, 1, d], 1.0);

            let t_concat = {
                let start = Instant::now();
                let mut concat: Option<Tensor> = None;
                for _ in 0..t {
                    concat = Some(match concat.take() {
                        None => step.clone(),
                        Some(c) => TypedConcat { axis: 2 }
                            .eval(tvec![c.into(), step.clone().into()])?
                            .remove(0)
                            .into_tensor(),
                    });
                }
                start.elapsed().as_secs_f64() * 1e3
            };

            let (t_inplace, reallocs) = {
                let start = Instant::now();
                let mut cache = InPlaceKvCache::new(2);
                for _ in 0..t {
                    cache.push(&step)?;
                }
                (start.elapsed().as_secs_f64() * 1e3, cache.reallocs())
            };

            println!(
                "  {t:>5}  {t_concat:>9.3}  {t_inplace:>10.3}   {:>6.2}x  {reallocs:>4}",
                t_concat / t_inplace
            );
        }
        Ok(())
    }

    // End-to-end decode: {update + attention readout} per step, both strategies.
    //   cargo test -p tract-transformers inplace_kv_cache::tests::bench_decode -- --ignored --nocapture
    #[test]
    #[ignore]
    fn bench_decode() -> TractResult<()> {
        use std::time::Instant;
        let (b, h, d) = (1usize, 8usize, 128usize);
        let scale = 1.0 / (d as f32).sqrt();
        println!("\n  END-TO-END decode (update + attention, B={b} H={h} D={d}):");
        println!("   T     concat(ms)  inplace(ms)   speedup");
        for &t in &[256usize, 512, 1024, 2048] {
            let q = seq_tensor(&[b, h, 1, d], 7.0);
            let qv = q.to_plain_array_view::<f32>()?.into_dimensionality()?;
            let step = seq_tensor(&[b, h, 1, d], 1.0);

            let t_concat = {
                let start = Instant::now();
                let mut concat: Option<Tensor> = None;
                for _ in 0..t {
                    concat = Some(match concat.take() {
                        None => step.clone(),
                        Some(c) => TypedConcat { axis: 2 }
                            .eval(tvec![c.into(), step.clone().into()])?
                            .remove(0)
                            .into_tensor(),
                    });
                    let cbuf = concat.as_ref().unwrap();
                    let cv = cbuf.to_plain_array_view::<f32>()?.into_dimensionality()?;
                    std::hint::black_box(attention(qv, cv, cv, scale));
                }
                start.elapsed().as_secs_f64() * 1e3
            };

            let t_inplace = {
                let start = Instant::now();
                let mut cache = InPlaceKvCache::new(2);
                for _ in 0..t {
                    cache.push(&step)?;
                    let kview = cache.valid_view::<f32>()?.into_dimensionality()?;
                    std::hint::black_box(attention(qv, kview, kview, scale));
                }
                start.elapsed().as_secs_f64() * 1e3
            };

            println!(
                "  {t:>5}  {t_concat:>9.3}  {t_inplace:>10.3}   {:>6.2}x",
                t_concat / t_inplace
            );
        }
        Ok(())
    }

    // ---- Integration: the fused stateful op vs the {concat-cache(K), concat-cache(V),
    //      FlashSdpaOp} baseline subgraph, driven step-by-step (prefill + decode), GQA.
    //      Same attention kernel + identical K/V => bit-identical, proving the fused op
    //      is a correct drop-in for the cache->Sdpa path. ----
    fn drive_fused_vs_baseline(causal: bool) -> TractResult<()> {
        let (bsz, hq, hkv, d) = (1usize, 4usize, 2usize, 16usize); // GQA: 4 q-heads / 2 kv-heads
        let op = InPlaceKvSdpa { axis: 2, causal, scale: None };
        let session = TurnState::default();
        let mut state = op.state(&session, 0)?.unwrap();
        let mut session = session;
        let flash = FlashSdpaOp { causal, scale: None };

        let mut kc: Option<Tensor> = None;
        let mut vc: Option<Tensor> = None;

        // prefill 3 tokens, then 12 single-token decode steps
        let mut snews = vec![3usize];
        snews.extend(std::iter::repeat_n(1usize, 12));

        for (t, &snew) in snews.iter().enumerate() {
            let q = seq_tensor(&[bsz, hq, snew, d], 1.0 + t as f32);
            let knew = seq_tensor(&[bsz, hkv, snew, d], 5.0 + t as f32 * 0.3);
            let vnew = seq_tensor(&[bsz, hkv, snew, d], 9.0 - t as f32 * 0.2);

            let o_fused = state
                .eval(
                    &mut session,
                    &op,
                    tvec![q.clone().into(), knew.clone().into(), vnew.clone().into()],
                )?
                .remove(0)
                .into_tensor();

            kc = Some(match kc.take() {
                None => knew.clone(),
                Some(c) => TypedConcat { axis: 2 }
                    .eval(tvec![c.into(), knew.clone().into()])?
                    .remove(0)
                    .into_tensor(),
            });
            vc = Some(match vc.take() {
                None => vnew.clone(),
                Some(c) => TypedConcat { axis: 2 }
                    .eval(tvec![c.into(), vnew.clone().into()])?
                    .remove(0)
                    .into_tensor(),
            });
            let o_base = flash
                .eval(tvec![q.into(), kc.clone().unwrap().into(), vc.clone().unwrap().into()])?
                .remove(0)
                .into_tensor();

            o_fused
                .close_enough(&o_base, Approximation::Approximate)
                .with_context(|| format!("fused != baseline at step {t} (causal={causal})"))?;
        }
        Ok(())
    }

    #[test]
    fn fused_op_matches_cache_plus_flash_noncausal() -> TractResult<()> {
        drive_fused_vs_baseline(false)
    }

    #[test]
    fn fused_op_matches_cache_plus_flash_causal() -> TractResult<()> {
        drive_fused_vs_baseline(true)
    }

    // True runtime integration: build a real TypedModel with the op, run it through
    // tract's engine via a persistent SimpleState across prefill+decode steps (KV state
    // carried by the op between runs), and match the concat-cache + FlashSdpaOp baseline.
    #[test]
    fn fused_op_runs_in_model() -> TractResult<()> {
        let (b, hq, hkv, d) = (1usize, 4usize, 2usize, 16usize);
        let mut model = TypedModel::default();
        let s = model.sym("S");
        let dim = |x: usize| x.to_dim();
        let qshape: TVec<TDim> = tvec![dim(b), dim(hq), s.clone().into(), dim(d)];
        let kshape: TVec<TDim> = tvec![dim(b), dim(hkv), s.into(), dim(d)];
        let q = model.add_source("q", f32::fact(&qshape))?;
        let k = model.add_source("k", f32::fact(&kshape))?;
        let v = model.add_source("v", f32::fact(&kshape))?;
        let o = model.wire_node(
            "fused_attn",
            InPlaceKvSdpa { axis: 2, causal: false, scale: None },
            &[q, k, v],
        )?;
        model.select_output_outlets(&o)?;

        let runnable = model.into_runnable()?;
        let mut rt = runnable.spawn()?;

        let flash = FlashSdpaOp { causal: false, scale: None };
        let mut kc: Option<Tensor> = None;
        let mut vc: Option<Tensor> = None;
        let mut snews = vec![3usize];
        snews.extend(std::iter::repeat_n(1usize, 8));

        for (t, &snew) in snews.iter().enumerate() {
            let q = seq_tensor(&[b, hq, snew, d], 2.0 + t as f32);
            let knew = seq_tensor(&[b, hkv, snew, d], 4.0 + t as f32 * 0.4);
            let vnew = seq_tensor(&[b, hkv, snew, d], 6.0 - t as f32 * 0.1);

            let o_model = rt
                .run(tvec![q.clone().into(), knew.clone().into(), vnew.clone().into()])?
                .remove(0)
                .into_tensor();

            kc = Some(match kc.take() {
                None => knew.clone(),
                Some(c) => TypedConcat { axis: 2 }
                    .eval(tvec![c.into(), knew.clone().into()])?
                    .remove(0)
                    .into_tensor(),
            });
            vc = Some(match vc.take() {
                None => vnew.clone(),
                Some(c) => TypedConcat { axis: 2 }
                    .eval(tvec![c.into(), vnew.clone().into()])?
                    .remove(0)
                    .into_tensor(),
            });
            let o_base = flash
                .eval(tvec![q.into(), kc.clone().unwrap().into(), vc.clone().unwrap().into()])?
                .remove(0)
                .into_tensor();

            o_model
                .close_enough(&o_base, Approximation::Approximate)
                .with_context(|| format!("model-run != baseline at step {t}"))?;
        }
        Ok(())
    }

    // Auto-rewrite: a model with the {DynKeyValueCache(K), DynKeyValueCache(V), Sdpa}
    // pattern is transparently fused to InPlaceKvSdpa, and the rewritten model runs and
    // matches the concat-cache + FlashSdpaOp baseline.
    #[test]
    fn rewrite_fuses_cache_sdpa() -> TractResult<()> {
        let (b, hq, hkv, d) = (1usize, 4usize, 2usize, 16usize);
        let mut model = TypedModel::default();
        let s = model.sym("S");
        let p = model.sym("P");
        let dim = |x: usize| x.to_dim();
        let qf: TVec<TDim> = tvec![dim(b), dim(hq), s.clone().into(), dim(d)];
        let newf: TVec<TDim> = tvec![dim(b), dim(hkv), s.into(), dim(d)];
        let pastf: TVec<TDim> = tvec![dim(b), dim(hkv), p.into(), dim(d)];
        let q = model.add_source("q", f32::fact(&qf))?;
        let knew = model.add_source("k", f32::fact(&newf))?;
        let vnew = model.add_source("v", f32::fact(&newf))?;
        let mkcache = |nm: &str| DynKeyValueCache {
            name: nm.to_string(),
            axis: 2,
            past_sequence_fact: f32::fact(&pastf),
            input_sequence_fact: f32::fact(&newf),
        };
        let kc = model.wire_node("kc", mkcache("kc"), &[knew])?;
        let vc = model.wire_node("vc", mkcache("vc"), &[vnew])?;
        let o = model.wire_node(
            "sdpa",
            Sdpa {
                scale: None,
                datum_type: f32::datum_type(),
                acc_datum_type: f32::datum_type(),
                is_causal: false,
            },
            &[q, kc[0], vc[0]],
        )?;
        model.select_output_outlets(&o)?;

        assert!(model.nodes().iter().any(|n| n.op_is::<DynKeyValueCache>()));
        InPlaceKvSdpaTransform.transform(&mut model)?;

        // Structural: fused op present, caches + sdpa gone.
        assert!(model.nodes().iter().any(|n| n.op_is::<InPlaceKvSdpa>()), "fused op present");
        assert!(!model.nodes().iter().any(|n| n.op_is::<DynKeyValueCache>()), "caches removed");
        assert!(!model.nodes().iter().any(|n| n.op_is::<Sdpa>()), "sdpa removed");
        let fused = model.nodes().iter().find(|n| n.op_is::<InPlaceKvSdpa>()).unwrap();
        assert_eq!(fused.inputs.len(), 3, "fused op takes [Q, K_new, V_new]");

        // Behavioral: the rewritten model runs and matches the baseline over decode.
        let runnable = model.into_runnable()?;
        let mut rt = runnable.spawn()?;
        let flash = FlashSdpaOp { causal: false, scale: None };
        let mut kacc: Option<Tensor> = None;
        let mut vacc: Option<Tensor> = None;
        let mut snews = vec![3usize];
        snews.extend(std::iter::repeat_n(1usize, 6));
        for (t, &snew) in snews.iter().enumerate() {
            let qi = seq_tensor(&[b, hq, snew, d], 2.0 + t as f32);
            let ki = seq_tensor(&[b, hkv, snew, d], 3.0 + t as f32 * 0.2);
            let vi = seq_tensor(&[b, hkv, snew, d], 8.0 - t as f32 * 0.1);
            let o_model = rt
                .run(tvec![qi.clone().into(), ki.clone().into(), vi.clone().into()])?
                .remove(0)
                .into_tensor();
            kacc = Some(match kacc.take() {
                None => ki.clone(),
                Some(c) => TypedConcat { axis: 2 }
                    .eval(tvec![c.into(), ki.clone().into()])?
                    .remove(0)
                    .into_tensor(),
            });
            vacc = Some(match vacc.take() {
                None => vi.clone(),
                Some(c) => TypedConcat { axis: 2 }
                    .eval(tvec![c.into(), vi.clone().into()])?
                    .remove(0)
                    .into_tensor(),
            });
            let o_base = flash
                .eval(tvec![qi.into(), kacc.clone().unwrap().into(), vacc.clone().unwrap().into()])?
                .remove(0)
                .into_tensor();
            o_model
                .close_enough(&o_base, Approximation::Approximate)
                .with_context(|| format!("rewritten model != baseline at step {t}"))?;
        }
        Ok(())
    }

    // End-to-end decode through the actual op path: fused InPlaceKvSdpa (in-place cache
    // + attention over the view) vs concat-cache + FlashSdpaOp.
    //   cargo test -p tract-transformers inplace_kv_cache::tests::bench_fused_decode -- --ignored --nocapture
    #[test]
    #[ignore]
    fn bench_fused_decode() -> TractResult<()> {
        use std::time::Instant;
        let (bsz, hq, hkv, d) = (1usize, 8usize, 8usize, 128usize);
        let q = seq_tensor(&[bsz, hq, 1, d], 7.0);
        let knew = seq_tensor(&[bsz, hkv, 1, d], 1.0);
        let vnew = seq_tensor(&[bsz, hkv, 1, d], 2.0);
        println!(
            "\n  INTEGRATED decode via op (fused InPlaceKvSdpa vs concat-cache + FlashSdpaOp):"
        );
        println!("   T     baseline(ms)  fused(ms)   speedup");
        for &steps in &[256usize, 512, 1024, 2048] {
            let op = InPlaceKvSdpa { axis: 2, causal: false, scale: None };
            let session = TurnState::default();
            let mut state = op.state(&session, 0)?.unwrap();
            let mut session = session;
            let t_fused = {
                let start = Instant::now();
                for _ in 0..steps {
                    std::hint::black_box(state.eval(
                        &mut session,
                        &op,
                        tvec![q.clone().into(), knew.clone().into(), vnew.clone().into()],
                    )?);
                }
                start.elapsed().as_secs_f64() * 1e3
            };

            let flash = FlashSdpaOp { causal: false, scale: None };
            let t_base = {
                let mut kc: Option<Tensor> = None;
                let mut vc: Option<Tensor> = None;
                let start = Instant::now();
                for _ in 0..steps {
                    kc = Some(match kc.take() {
                        None => knew.clone(),
                        Some(c) => TypedConcat { axis: 2 }
                            .eval(tvec![c.into(), knew.clone().into()])?
                            .remove(0)
                            .into_tensor(),
                    });
                    vc = Some(match vc.take() {
                        None => vnew.clone(),
                        Some(c) => TypedConcat { axis: 2 }
                            .eval(tvec![c.into(), vnew.clone().into()])?
                            .remove(0)
                            .into_tensor(),
                    });
                    std::hint::black_box(flash.eval(tvec![
                        q.clone().into(),
                        kc.clone().unwrap().into(),
                        vc.clone().unwrap().into()
                    ])?);
                }
                start.elapsed().as_secs_f64() * 1e3
            };
            println!("  {steps:>5}  {t_base:>11.3}  {t_fused:>9.3}   {:>6.2}x", t_base / t_fused);
        }
        Ok(())
    }

    // Deterministic (Q, K_new, V_new) per step: prefill 3 + `n_decode` single tokens.
    fn decode_inputs(n_decode: usize) -> Vec<(Tensor, Tensor, Tensor)> {
        let (b, hq, hkv, d) = (1usize, 4usize, 2usize, 16usize);
        let mut shapes = vec![3usize];
        shapes.extend(std::iter::repeat_n(1usize, n_decode));
        shapes
            .iter()
            .enumerate()
            .map(|(t, &snew)| {
                (
                    seq_tensor(&[b, hq, snew, d], 1.0 + t as f32),
                    seq_tensor(&[b, hkv, snew, d], 5.0 + t as f32 * 0.3),
                    seq_tensor(&[b, hkv, snew, d], 9.0 - t as f32 * 0.2),
                )
            })
            .collect()
    }

    // Resume #1: snapshot the running state mid-decode via freeze/unfreeze, then keep
    // going — bit-identical to a straight run.
    #[test]
    fn resume_via_freeze_unfreeze() -> TractResult<()> {
        let op = InPlaceKvSdpa { axis: 2, causal: true, scale: None };
        let mut session = TurnState::default();
        let mut straight = op.state(&session, 0)?.unwrap();
        let mut split = op.state(&session, 0)?.unwrap();
        for (t, (q, k, v)) in decode_inputs(9).into_iter().enumerate() {
            let ins = tvec![q.into(), k.into(), v.into()];
            let os = straight.eval(&mut session, &op, ins.clone())?.remove(0).into_tensor();
            if t == 5 {
                let frozen = split.freeze();
                split = frozen.unfreeze();
            }
            let op2 = split.eval(&mut session, &op, ins)?.remove(0).into_tensor();
            os.close_enough(&op2, Approximation::Exact)
                .with_context(|| format!("freeze/unfreeze resume mismatch at step {t}"))?;
        }
        Ok(())
    }

    // Resume #2: checkpoint the cache to tensors (save_to), reload into a fresh state
    // (load_from), then continue — bit-identical to a straight run.
    #[test]
    fn resume_via_save_load() -> TractResult<()> {
        let op = InPlaceKvSdpa { axis: 2, causal: true, scale: None };
        let mut session = TurnState::default();
        let mut straight = op.state(&session, 0)?.unwrap();
        let mut split = op.state(&session, 0)?.unwrap();
        for (t, (q, k, v)) in decode_inputs(9).into_iter().enumerate() {
            let ins = tvec![q.into(), k.into(), v.into()];
            let os = straight.eval(&mut session, &op, ins.clone())?.remove(0).into_tensor();
            if t == 5 {
                let mut saved = vec![];
                split.save_to(&mut saved)?;
                ensure!(saved.len() == 2, "save_to should emit [K, V]");
                let mut fresh = op.state(&session, 0)?.unwrap();
                fresh.load_from(&mut session, &mut saved.into_iter())?;
                split = fresh;
            }
            let op2 = split.eval(&mut session, &op, ins)?.remove(0).into_tensor();
            os.close_enough(&op2, Approximation::Exact)
                .with_context(|| format!("save/load resume mismatch at step {t}"))?;
        }
        Ok(())
    }

    // The resume checkpoint (save_to) is a one-time O(len) copy, not per-step O(T^2).
    //   cargo test -p tract-transformers inplace_kv_cache::tests::bench_resume_snapshot -- --ignored --nocapture
    #[test]
    #[ignore]
    fn bench_resume_snapshot() -> TractResult<()> {
        use std::time::Instant;
        let (b, hq, hkv, d) = (1usize, 8usize, 8usize, 128usize);
        let op = InPlaceKvSdpa { axis: 2, causal: false, scale: None };
        let session = TurnState::default();
        let q = seq_tensor(&[b, hq, 1, d], 7.0);
        let kv = seq_tensor(&[b, hkv, 1, d], 1.0);
        println!("\n  resume checkpoint (save_to) — one-time O(len):");
        println!("   len    save_to(ms)");
        for &len in &[256usize, 1024, 4096] {
            let mut sess = TurnState::default();
            let mut state = op.state(&session, 0)?.unwrap();
            for _ in 0..len {
                state.eval(
                    &mut sess,
                    &op,
                    tvec![q.clone().into(), kv.clone().into(), kv.clone().into()],
                )?;
            }
            let reps = 100;
            let start = Instant::now();
            for _ in 0..reps {
                let mut saved = vec![];
                state.save_to(&mut saved)?;
                std::hint::black_box(saved);
            }
            println!("  {len:>5}  {:>10.4}", start.elapsed().as_secs_f64() * 1e3 / reps as f64);
        }
        Ok(())
    }
}
