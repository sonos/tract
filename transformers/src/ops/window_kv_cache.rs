//! Bounded (sliding-window) KV cache — a fixed-capacity ring buffer for models
//! trained with sliding-window attention (Mistral, Gemma-style local/global, …).
//!
//! Instead of growing the cache with the sequence (O(T) memory + O(T) per-step
//! attention), keep a fixed `window` slots and overwrite the oldest on append. So a
//! decode of arbitrary length runs at **constant** memory and per-step cost — and it's
//! **lossless**, because the model was trained to attend only within the window.
//!
//! Key trick that makes the ring buffer cheap: **decode attention is order-invariant
//! over keys** — `O = Σ_j softmax_j · V_j` is unchanged if you permute (K,V) together
//! (the set of scores, hence the softmax weights, is identical). So we never have to
//! un-rotate the buffer: the consumer attends over the W physical slots in whatever
//! order they sit, and the result equals attending over the ordered last-W (up to
//! floating-point summation order). Validated in the tests.
//!
//! Companion to the in-place cache (#2321): this is "the in-place cache with a cap and
//! wraparound." For prefill (multi-query) the window also needs a banded causal mask;
//! that lives on the attention op, not here.

use tract_nnef::internal::*;
use tract_nnef::tract_core::ops::{FrozenOpState, OpStateFreeze};
use tract_nnef::tract_core::transform::ModelTransform;
use tract_nnef::tract_ndarray::Ix4;

use crate::ops::dyn_kv_cache::DynKeyValueCache;
use crate::ops::flash_sdpa::FlashSdpaOp;
use crate::ops::sdpa::Sdpa;

/// NNEF (de)serialization for the fused `WindowKvSdpa` op.
pub fn register(registry: &mut Registry) {
    registry.register_dumper(ser_window_kv_sdpa);
    registry.register_primitive(
        "tract_transformers_window_kv_sdpa",
        &[
            TypeName::Scalar.tensor().named("q"),
            TypeName::Scalar.tensor().named("k"),
            TypeName::Scalar.tensor().named("v"),
            TypeName::Integer.named("axis"),
            TypeName::Integer.named("window"),
            TypeName::Scalar.named("scale"),
        ],
        &[("output", TypeName::Scalar.tensor())],
        de_window_kv_sdpa,
    );
}

fn ser_window_kv_sdpa(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &WindowKvSdpa,
) -> TractResult<Option<Arc<RValue>>> {
    let q = ast.mapping[&node.inputs[0]].clone();
    let k = ast.mapping[&node.inputs[1]].clone();
    let v = ast.mapping[&node.inputs[2]].clone();
    let mut attrs = vec![("axis", numeric(op.axis)), ("window", numeric(op.window))];
    if let Some(scale) = op.scale {
        attrs.push(("scale", numeric(scale)));
    }
    Ok(Some(invocation("tract_transformers_window_kv_sdpa", &[q, k, v], &attrs)))
}

fn de_window_kv_sdpa(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<Value> {
    let q = invocation.named_arg_as(builder, "q")?;
    let k = invocation.named_arg_as(builder, "k")?;
    let v = invocation.named_arg_as(builder, "v")?;
    let axis: usize = invocation.named_arg_as(builder, "axis")?;
    let window: usize = invocation.named_arg_as(builder, "window")?;
    let scale: Option<f32> = invocation.get_named_arg_as(builder, "scale")?;
    builder.wire(WindowKvSdpa { axis, window, scale }, &[q, k, v])
}

/// Fixed-capacity sliding-window KV cache (ring buffer) along `axis`.
#[derive(Clone, Debug)]
pub struct WindowKvCache {
    pub axis: usize,
    pub window: usize,
    buf: Option<Tensor>, // capacity `window` along `axis`, allocated on first push
    len: usize,          // valid slots, ≤ window
    cursor: usize,       // next write position, in 0..window
}

impl WindowKvCache {
    pub fn new(axis: usize, window: usize) -> Self {
        assert!(window > 0, "window must be > 0");
        WindowKvCache { axis, window, buf: None, len: 0, cursor: 0 }
    }

    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    /// Always the window capacity once allocated — memory is bounded regardless of T.
    pub fn capacity(&self) -> usize {
        self.buf.as_ref().map(|b| b.shape()[self.axis]).unwrap_or(0)
    }

    fn ensure_buf(&mut self, like: &Tensor) -> TractResult<()> {
        if self.buf.is_none() {
            let mut shape: TVec<usize> = like.shape().into();
            shape[self.axis] = self.window;
            self.buf = Some(unsafe { Tensor::uninitialized_dt(like.datum_type(), &shape)? });
        }
        Ok(())
    }

    /// Append `input` along `axis`, overwriting the oldest slots once full. O(min(new, window)).
    pub fn push(&mut self, input: &Tensor) -> TractResult<()> {
        let new = input.shape()[self.axis];
        if new == 0 {
            return Ok(());
        }
        self.ensure_buf(input)?;
        let w = self.window;

        if new >= w {
            // Only the last `w` of the input survive; lay them out [0..w], cursor resets.
            let buf = self.buf.as_mut().unwrap();
            buf.assign_slice(0..w, input, (new - w)..new, self.axis)?;
            self.cursor = 0;
            self.len = w;
            return Ok(());
        }

        // new < w: write at the cursor, wrapping around the end.
        let end = self.cursor + new;
        let buf = self.buf.as_mut().unwrap();
        if end <= w {
            buf.assign_slice(self.cursor..end, input, 0..new, self.axis)?;
            self.cursor = if end == w { 0 } else { end };
        } else {
            let first = w - self.cursor;
            buf.assign_slice(self.cursor..w, input, 0..first, self.axis)?;
            buf.assign_slice(0..(new - first), input, first..new, self.axis)?;
            self.cursor = new - first;
        }
        self.len = (self.len + new).min(w);
        Ok(())
    }

    /// Zero-copy view of the `len` valid slots. Once full this is the whole buffer in
    /// *physical* (rotated) order — correct for decode attention by order-invariance.
    pub fn valid_view<T: Datum>(&self) -> TractResult<tract_ndarray::ArrayViewD<'_, T>> {
        let buf = self.buf.as_ref().context("empty window cache")?;
        let mut v = buf.to_plain_array_view::<T>()?;
        v.slice_axis_inplace(tract_ndarray::Axis(self.axis), (0..self.len).into());
        Ok(v)
    }
}

/// Fused sliding-window KV-cache + attention (decode). Owns K/V ring buffers of size
/// `window`; each step appends `K_new`/`V_new` and attends `Q` over the (≤window) cache.
/// The bounded cache *is* the sliding window — attending over it equals windowed
/// attention — so decode runs at constant memory + per-step cost, losslessly. Inputs
/// `[Q, K_new, V_new]`, each `[B, H, S, D]`; output has Q's shape.
#[derive(Clone, Debug, PartialEq)]
pub struct WindowKvSdpa {
    pub axis: usize,
    pub window: usize,
    pub scale: Option<f32>,
}
impl Eq for WindowKvSdpa {}

impl Op for WindowKvSdpa {
    fn name(&self) -> StaticName {
        "WindowKvSdpa".into()
    }
    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("axis={}, window={}, scale={:?}", self.axis, self.window, self.scale)])
    }
    op_as_typed_op!();
}

impl EvalOp for WindowKvSdpa {
    fn is_stateless(&self) -> bool {
        false
    }
    fn state(
        &self,
        _session: &TurnState,
        _node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(Some(Box::new(WindowKvSdpaState {
            window: self.window,
            scale: self.scale,
            k: WindowKvCache::new(self.axis, self.window),
            v: WindowKvCache::new(self.axis, self.window),
        })))
    }
}

impl TypedOp for WindowKvSdpa {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(inputs.len() == 3, "WindowKvSdpa expects [Q, K_new, V_new]");
        Ok(tvec!(inputs[0].without_value()))
    }
    as_op!();
}

#[derive(Clone, Debug)]
pub struct WindowKvSdpaState {
    window: usize,
    scale: Option<f32>,
    k: WindowKvCache,
    v: WindowKvCache,
}

impl OpState for WindowKvSdpaState {
    fn eval(
        &mut self,
        _state: &mut TurnState,
        _op: &dyn Op,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        ensure!(inputs.len() == 3, "WindowKvSdpa expects [Q, K_new, V_new]");
        let input_dt = inputs[0].datum_type();
        let k_new = inputs[1].cast_to::<f32>()?;
        let v_new = inputs[2].cast_to::<f32>()?;
        self.k.push(k_new.as_ref())?;
        self.v.push(v_new.as_ref())?;

        let q = inputs[0].cast_to::<f32>()?;
        let qv = q.to_plain_array_view::<f32>()?.into_dimensionality::<Ix4>()?;
        let kview = self.k.valid_view::<f32>()?.into_dimensionality::<Ix4>()?;
        let vview = self.v.valid_view::<f32>()?.into_dimensionality::<Ix4>()?;

        // The ring buffer already bounds which keys are visible (the last `window`), so
        // attention is "attend all" — every cached key is within the current query's window.
        let flash = FlashSdpaOp { causal: false, scale: self.scale };
        let o = flash.flash_attention_gqa(qv, kview, vview, None);
        Ok(tvec!(o.into_tensor().cast_to_dt(input_dt)?.into_owned().into_tvalue()))
    }
}

#[derive(Clone, Debug)]
struct FrozenWindowKvSdpaState {
    window: usize,
    scale: Option<f32>,
    k: WindowKvCache,
    v: WindowKvCache,
}
impl OpStateFreeze for WindowKvSdpaState {
    fn freeze(&self) -> Box<dyn FrozenOpState> {
        Box::new(FrozenWindowKvSdpaState {
            window: self.window,
            scale: self.scale,
            k: self.k.clone(),
            v: self.v.clone(),
        })
    }
}
impl FrozenOpState for FrozenWindowKvSdpaState {
    fn unfreeze(&self) -> Box<dyn OpState> {
        Box::new(WindowKvSdpaState {
            window: self.window,
            scale: self.scale,
            k: self.k.clone(),
            v: self.v.clone(),
        })
    }
}

/// Rewrite rule: fuse `{DynKeyValueCache(K), DynKeyValueCache(V), Sdpa(Q,K,V)}` into a
/// `WindowKvSdpa` with the window supplied via the Rewriter context — so an imported
/// decode model uses a bounded sliding-window cache. The window comes from the model
/// (the GQA `local_window_size` / config), passed to `WindowKvSdpaTransform`.
pub fn fuse_window_kv_sdpa_rule(
    window: &usize,
    model: &TypedModel,
    node: &TypedNode,
    node_name: &str,
    op: &Sdpa,
) -> TractResult<Option<TypedModelPatch>> {
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
        format!("{node_name}.window_kv_sdpa"),
        WindowKvSdpa { axis: kc.axis, window: *window, scale },
        &taps,
    )?;
    patch.shunt_outside(model, node.id.into(), fused[0])?;
    Ok(Some(patch))
}

/// Strip the GQA broadcast chain, then fuse `cache -> Sdpa` into `WindowKvSdpa` with
/// `window` — making an imported decode model use the bounded sliding-window cache.
#[derive(Debug, Clone)]
pub struct WindowKvSdpaTransform {
    pub window: usize,
}

impl ModelTransform for WindowKvSdpaTransform {
    fn name(&self) -> StaticName {
        "fuse_window_kv_sdpa".into()
    }
    fn transform(&self, model: &mut TypedModel) -> TractResult<()> {
        Rewriter::default()
            .with_rule_for("fuse-kv-broadcast", crate::ops::sdpa::fuse_kv_cache_broadcast_rule)
            .rewrite(&(), model)?;
        Rewriter::default()
            .with_rule_for("fuse-window-kv-sdpa", fuse_window_kv_sdpa_rule)
            .rewrite(&self.window, model)?;
        model.compact()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tract_nnef::tract_ndarray::{Array4, ArrayView4, s};

    fn tok(shape: &[usize], v: f32) -> Tensor {
        let n: usize = shape.iter().product();
        Tensor::from_shape(shape, &vec![v; n]).unwrap()
    }

    // ---- ring-buffer mechanics: holds exactly the last `window` items (as a set) ----
    #[test]
    fn window_holds_last_w_as_a_set() -> TractResult<()> {
        let w = 4;
        let mut c = WindowKvCache::new(2, w); // [B,H,S,D], seq axis
        let mut full: Vec<f32> = vec![];
        for t in 0..10 {
            c.push(&tok(&[1, 1, 1, 1], t as f32))?;
            full.push(t as f32);
            assert!(c.len() <= w, "len bounded by window");
            // valid_view set == last min(t+1,w) tokens
            let view = c.valid_view::<f32>()?;
            let mut got: Vec<f32> = view.iter().copied().collect();
            got.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let mut want: Vec<f32> = full.iter().rev().take(w).copied().collect();
            want.sort_by(|a, b| a.partial_cmp(b).unwrap());
            assert_eq!(got, want, "step {t}: window must hold the last {w} as a set");
        }
        assert_eq!(c.capacity(), w, "memory stays bounded at the window");
        Ok(())
    }

    #[test]
    fn prefill_chunk_larger_than_window_keeps_last_w() -> TractResult<()> {
        let w = 3;
        let mut c = WindowKvCache::new(2, w);
        // one chunk of 7 tokens with distinct values, axis=2
        let chunk = Tensor::from_shape(&[1, 1, 7, 1], &[0f32, 1., 2., 3., 4., 5., 6.])?;
        c.push(&chunk)?;
        let mut got: Vec<f32> = c.valid_view::<f32>()?.iter().copied().collect();
        got.sort_by(|a, b| a.partial_cmp(b).unwrap());
        assert_eq!(got, vec![4.0, 5.0, 6.0], "keeps the last 3");
        Ok(())
    }

    // ---- the correctness property: decode attention over the (rotated) window ==
    //      attention over the ordered last-W slice, by order-invariance ----
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
                let qm = q.slice(s![bi, hi, .., ..]);
                let km = k.slice(s![bi, hi, .., ..]);
                let vm = v.slice(s![bi, hi, .., ..]);
                let mut sc = qm.dot(&km.t());
                sc *= scale;
                for mut row in sc.rows_mut() {
                    let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                    let mut s = 0.0;
                    row.iter_mut().for_each(|x| {
                        *x = (*x - m).exp();
                        s += *x;
                    });
                    row.iter_mut().for_each(|x| *x /= s);
                }
                out.slice_mut(s![bi, hi, .., ..]).assign(&sc.dot(&vm));
            }
        }
        out
    }

    #[test]
    fn windowed_attention_matches_last_w_full() -> TractResult<()> {
        let (h, d, w) = (2usize, 8usize, 6usize);
        let scale = 1.0 / (d as f32).sqrt();
        // deterministic varied K/V per token so order actually differs after wrap
        let seq = |s: usize, base: f32| -> Tensor {
            let data: Vec<f32> = (0..h * s * d).map(|i| base + (i as f32 * 0.013).sin()).collect();
            Tensor::from_shape(&[1, h, s, d], &data).unwrap()
        };
        let mut kc = WindowKvCache::new(2, w);
        let mut vc = WindowKvCache::new(2, w);
        let mut kfull: Option<Tensor> = None;
        let mut vfull: Option<Tensor> = None;
        use tract_nnef::tract_core::ops::array::TypedConcat;
        for t in 0..20 {
            let knew = seq(1, 1.0 + t as f32 * 0.1);
            let vnew = seq(1, 5.0 - t as f32 * 0.07);
            kc.push(&knew)?;
            vc.push(&vnew)?;
            let grow = |acc: Option<Tensor>, x: Tensor| -> TractResult<Tensor> {
                Ok(match acc {
                    None => x,
                    Some(a) => TypedConcat { axis: 2 }
                        .eval(tvec![a.into(), x.into()])?
                        .remove(0)
                        .into_tensor(),
                })
            };
            kfull = Some(grow(kfull.take(), knew)?);
            vfull = Some(grow(vfull.take(), vnew)?);

            let q = seq(1, 9.0 + t as f32 * 0.05);
            let qv = q.to_plain_array_view::<f32>()?.into_dimensionality()?;

            // windowed (rotated physical order)
            let o_win = attention(
                qv,
                kc.valid_view::<f32>()?.into_dimensionality()?,
                vc.valid_view::<f32>()?.into_dimensionality()?,
                scale,
            );
            // reference: ordered last-W of the full cache
            let len = kc.len();
            let kf = kfull.as_ref().unwrap();
            let s = kf.shape()[2];
            let kslice = kf.slice(2, s - len, s)?;
            let vslice = vfull.as_ref().unwrap().slice(2, s - len, s)?;
            let o_ref = attention(
                qv,
                kslice.to_plain_array_view::<f32>()?.into_dimensionality()?,
                vslice.to_plain_array_view::<f32>()?.into_dimensionality()?,
                scale,
            );
            let a = Tensor::from(o_win);
            let b = Tensor::from(o_ref);
            a.close_enough(&b, Approximation::Approximate)
                .with_context(|| format!("windowed != last-W at step {t}"))?;
        }
        Ok(())
    }

    // The fused decode op, run through tract's engine over a long sequence with a small
    // window, equals full attention over the last-W each step — i.e. correct sliding-window
    // decode, with the cache bounded to `window` regardless of how long we decode.
    #[test]
    fn window_sdpa_decode_matches_last_w_in_model() -> TractResult<()> {
        use tract_nnef::tract_core::ops::array::TypedConcat;
        let (b, h, d, w) = (1usize, 2usize, 16usize, 5usize);
        let scale = 1.0 / (d as f32).sqrt();
        let mut model = TypedModel::default();
        let s = model.sym("S");
        let dim = |x: usize| x.to_dim();
        let f: TVec<TDim> = tvec![dim(b), dim(h), s.into(), dim(d)];
        let q = model.add_source("q", f32::fact(&f))?;
        let k = model.add_source("k", f32::fact(&f))?;
        let v = model.add_source("v", f32::fact(&f))?;
        let o =
            model.wire_node("win", WindowKvSdpa { axis: 2, window: w, scale: None }, &[q, k, v])?;
        model.select_output_outlets(&o)?;
        let mut rt = model.into_runnable()?.spawn()?;

        let mk = |base: f32| -> Tensor {
            let data: Vec<f32> = (0..b * h * d).map(|i| base + (i as f32 * 0.013).sin()).collect();
            Tensor::from_shape(&[b, h, 1, d], &data).unwrap()
        };
        let grow = |acc: Option<Tensor>, x: Tensor| -> TractResult<Tensor> {
            Ok(match acc {
                None => x,
                Some(a) => {
                    TypedConcat { axis: 2 }.eval(tvec![a.into(), x.into()])?.remove(0).into_tensor()
                }
            })
        };
        let (mut kf, mut vf): (Option<Tensor>, Option<Tensor>) = (None, None);
        for t in 0..15 {
            let qi = mk(9.0 + t as f32 * 0.1);
            let ki = mk(1.0 + t as f32 * 0.07);
            let vi = mk(5.0 - t as f32 * 0.05);
            let o_model = rt
                .run(tvec![qi.clone().into(), ki.clone().into(), vi.clone().into()])?
                .remove(0)
                .into_tensor();
            kf = Some(grow(kf.take(), ki)?);
            vf = Some(grow(vf.take(), vi)?);
            let fk = kf.as_ref().unwrap();
            let sk = fk.shape()[2];
            let len = sk.min(w);
            let kslice = fk.slice(2, sk - len, sk)?;
            let vslice = vf.as_ref().unwrap().slice(2, sk - len, sk)?;
            let qv = qi.to_plain_array_view::<f32>()?.into_dimensionality()?;
            let o_ref = attention(
                qv,
                kslice.to_plain_array_view::<f32>()?.into_dimensionality()?,
                vslice.to_plain_array_view::<f32>()?.into_dimensionality()?,
                scale,
            );
            o_model
                .close_enough(&Tensor::from(o_ref), Approximation::Approximate)
                .with_context(|| format!("window decode != last-{w} at step {t}"))?;
        }
        Ok(())
    }

    // Auto-wiring: a {DynKeyValueCache(K), DynKeyValueCache(V), Sdpa} decode subgraph is
    // fused to WindowKvSdpa{window} by the transform, and the rewritten model then does
    // correct windowed decode (vs the un-windowed full attention it had before).
    #[test]
    fn transform_fuses_cache_sdpa_to_windowed_decode() -> TractResult<()> {
        use crate::ops::dyn_kv_cache::DynKeyValueCache;
        use crate::ops::sdpa::Sdpa;
        use tract_nnef::tract_core::ops::array::TypedConcat;
        let (b, h, d, w) = (1usize, 2usize, 16usize, 5usize);
        let scale = 1.0 / (d as f32).sqrt();
        let mut model = TypedModel::default();
        let s = model.sym("S");
        let p = model.sym("P");
        let dim = |x: usize| x.to_dim();
        let newf: TVec<TDim> = tvec![dim(b), dim(h), s.clone().into(), dim(d)];
        let qf: TVec<TDim> = tvec![dim(b), dim(h), s.into(), dim(d)];
        let pastf: TVec<TDim> = tvec![dim(b), dim(h), p.into(), dim(d)];
        let q = model.add_source("q", f32::fact(&qf))?;
        let knew = model.add_source("k", f32::fact(&newf))?;
        let vnew = model.add_source("v", f32::fact(&newf))?;
        let mkc = |nm: &str| DynKeyValueCache {
            name: nm.to_string(),
            axis: 2,
            past_sequence_fact: f32::fact(&pastf),
            input_sequence_fact: f32::fact(&newf),
        };
        let kc = model.wire_node("kc", mkc("kc"), &[knew])?;
        let vc = model.wire_node("vc", mkc("vc"), &[vnew])?;
        let o = model.wire_node(
            "sdpa",
            Sdpa {
                scale: None,
                datum_type: f32::datum_type(),
                acc_datum_type: f32::datum_type(),
                is_causal: true,
            },
            &[q, kc[0], vc[0]],
        )?;
        model.select_output_outlets(&o)?;

        WindowKvSdpaTransform { window: w }.transform(&mut model)?;

        assert!(model.nodes().iter().any(|n| n.op_is::<WindowKvSdpa>()), "fused to WindowKvSdpa");
        assert!(!model.nodes().iter().any(|n| n.op_is::<DynKeyValueCache>()), "caches removed");
        assert!(!model.nodes().iter().any(|n| n.op_is::<Sdpa>()), "sdpa removed");

        let mut rt = model.into_runnable()?.spawn()?;
        let mk = |base: f32| -> Tensor {
            let data: Vec<f32> = (0..b * h * d).map(|i| base + (i as f32 * 0.013).sin()).collect();
            Tensor::from_shape(&[b, h, 1, d], &data).unwrap()
        };
        let grow = |acc: Option<Tensor>, x: Tensor| -> TractResult<Tensor> {
            Ok(match acc {
                None => x,
                Some(a) => {
                    TypedConcat { axis: 2 }.eval(tvec![a.into(), x.into()])?.remove(0).into_tensor()
                }
            })
        };
        let (mut kf, mut vf): (Option<Tensor>, Option<Tensor>) = (None, None);
        for t in 0..15 {
            let qi = mk(9.0 + t as f32 * 0.1);
            let ki = mk(1.0 + t as f32 * 0.07);
            let vi = mk(5.0 - t as f32 * 0.05);
            let o_model = rt
                .run(tvec![qi.clone().into(), ki.clone().into(), vi.clone().into()])?
                .remove(0)
                .into_tensor();
            kf = Some(grow(kf.take(), ki)?);
            vf = Some(grow(vf.take(), vi)?);
            let fk = kf.as_ref().unwrap();
            let sk = fk.shape()[2];
            let len = sk.min(w);
            let kslice = fk.slice(2, sk - len, sk)?;
            let vslice = vf.as_ref().unwrap().slice(2, sk - len, sk)?;
            let qv = qi.to_plain_array_view::<f32>()?.into_dimensionality()?;
            let o_ref = attention(
                qv,
                kslice.to_plain_array_view::<f32>()?.into_dimensionality()?,
                vslice.to_plain_array_view::<f32>()?.into_dimensionality()?,
                scale,
            );
            o_model
                .close_enough(&Tensor::from(o_ref), Approximation::Approximate)
                .with_context(|| format!("rewritten windowed decode != last-{w} at step {t}"))?;
        }
        Ok(())
    }

    // NNEF ser/de round-trip: WindowKvSdpa survives write_to_tar -> model_for_read.
    #[test]
    fn window_kv_sdpa_nnef_round_trip() -> TractResult<()> {
        use crate::WithTractTransformers;
        let (b, h, d) = (1usize, 2usize, 16usize);
        let mut model = TypedModel::default();
        let s = model.sym("S");
        let dim = |x: usize| x.to_dim();
        let f: TVec<TDim> = tvec![dim(b), dim(h), s.into(), dim(d)];
        let q = model.add_source("q", f32::fact(&f))?;
        let k = model.add_source("k", f32::fact(&f))?;
        let v = model.add_source("v", f32::fact(&f))?;
        let o = model.wire_node(
            "win",
            WindowKvSdpa { axis: 2, window: 4096, scale: Some(0.125) },
            &[q, k, v],
        )?;
        model.select_output_outlets(&o)?;

        let nnef = tract_nnef::nnef().with_tract_transformers();
        let mut buffer = vec![];
        nnef.write_to_tar(&model, &mut buffer)?;
        let reloaded = nnef.model_for_read(&mut &*buffer)?;

        let n = reloaded
            .nodes()
            .iter()
            .find(|n| n.op_is::<WindowKvSdpa>())
            .context("WindowKvSdpa survived the round-trip")?;
        let op = n.op_as::<WindowKvSdpa>().unwrap();
        assert_eq!(op.axis, 2);
        assert_eq!(op.window, 4096);
        assert_eq!(op.scale, Some(0.125));
        Ok(())
    }
}
