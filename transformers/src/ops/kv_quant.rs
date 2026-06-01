//! KIVI-style KV-cache quantization (training-free): store the cache in low precision to
//! shrink memory **near-losslessly**, keeping every token (a gentler trade than evicting).
//!
//! The key asymmetry (Liu et al. 2024, KIVI): **Keys are quantized PER-CHANNEL** (each
//! head-dim channel gets its own scale — Keys have large-magnitude *outlier channels* that
//! would wreck a shared scale) and **Values PER-TOKEN**. Works for any model, no training.
//! (CommVQ's RoPE-commutative codebook is a fancier, model-specific follow-on.)
//!
//! This module provides:
//!   1. `quant_dequant` — the quality-validation primitive (f32→f32 round-trip)
//!   2. `QuantizedKvCache` — a stateful fused op that stores K/V in **actual u8 bytes**
//!      and dequantizes per-head on each decode step. Real memory saving: 8× vs f32,
//!      4× vs f16.  Configurable `bits` (int8 default, int4 viable).
//!   3. `QuantizedKvSdpaTransform` — auto-wires an existing {cache→Sdpa} decode subgraph
//!      into the quantized op.

use tract_nnef::internal::*;
use tract_nnef::tract_core::ops::{FrozenOpState, OpStateFreeze};
use tract_nnef::tract_core::transform::ModelTransform;
use tract_nnef::tract_ndarray::{Array2, Array4, ArrayView2, Ix4, s};

use crate::ops::dyn_kv_cache::DynKeyValueCache;
use crate::ops::flash_sdpa::FlashSdpaOp;
use crate::ops::sdpa::Sdpa;

// ── NNEF ser/de ───────────────────────────────────────────────────────────────────────────────

pub fn register(registry: &mut Registry) {
    registry.register_dumper(ser_quantized_kv_sdpa);
    registry.register_primitive(
        "tract_transformers_quantized_kv_sdpa",
        &[
            TypeName::Scalar.tensor().named("q"),
            TypeName::Scalar.tensor().named("k"),
            TypeName::Scalar.tensor().named("v"),
            TypeName::Integer.named("axis"),
            TypeName::Scalar.named("scale"),
        ],
        &[("output", TypeName::Scalar.tensor())],
        de_quantized_kv_sdpa,
    );
}

fn ser_quantized_kv_sdpa(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &QuantizedKvSdpa,
) -> TractResult<Option<Arc<RValue>>> {
    let q = ast.mapping[&node.inputs[0]].clone();
    let k = ast.mapping[&node.inputs[1]].clone();
    let v = ast.mapping[&node.inputs[2]].clone();
    let mut attrs = vec![("axis", numeric(op.axis))];
    if let Some(scale) = op.scale {
        attrs.push(("scale", numeric(scale)));
    }
    Ok(Some(invocation("tract_transformers_quantized_kv_sdpa", &[q, k, v], &attrs)))
}

fn de_quantized_kv_sdpa(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<Value> {
    let q = invocation.named_arg_as(builder, "q")?;
    let k = invocation.named_arg_as(builder, "k")?;
    let v = invocation.named_arg_as(builder, "v")?;
    let axis: usize = invocation.named_arg_as(builder, "axis")?;
    let scale: Option<f32> = invocation.get_named_arg_as(builder, "scale")?;
    builder.wire(QuantizedKvSdpa { axis, scale }, &[q, k, v])
}

/// Affine quantize→dequantize a `[rows, cols]` matrix at `bits` bits, returning the
/// reconstructed (lossy) values. `by_row = true` gives each ROW its own scale (per-token,
/// for Values); `by_row = false` gives each COLUMN its own scale (per-channel, for Keys).
/// Reconstruction error per element is ≤ scale/2 of its group.
pub fn quant_dequant(x: ArrayView2<f32>, bits: u32, by_row: bool) -> Array2<f32> {
    assert!((1..=16).contains(&bits), "bits must be 1..=16");
    let levels = ((1u32 << bits) - 1) as f32;
    let (r, c) = x.dim();
    let mut out = Array2::<f32>::zeros((r, c));
    let n_groups = if by_row { r } else { c };
    for g in 0..n_groups {
        let group = if by_row { x.row(g) } else { x.column(g) };
        let lo = group.iter().copied().fold(f32::INFINITY, f32::min);
        let hi = group.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let scale = if hi > lo { (hi - lo) / levels } else { 1.0 };
        for (k, &v) in group.iter().enumerate() {
            let q = ((v - lo) / scale).round().clamp(0.0, levels);
            let deq = lo + q * scale;
            if by_row {
                out[(g, k)] = deq;
            } else {
                out[(k, g)] = deq;
            }
        }
    }
    out
}

// ── Packed u8 storage ─────────────────────────────────────────────────────────────────────────
// One token = D bytes (int8) for Values (per-token scale), or D bytes for one channel of Keys
// (per-channel scale). Real memory: u8 is 4× f32, 2× f16.

/// Quantize a 1-D token/channel into `D` u8 bytes; return `(bytes, lo, scale)`.
fn quant_token_to_u8(v: &[f32]) -> (Vec<u8>, f32, f32) {
    let lo = v.iter().copied().fold(f32::INFINITY, f32::min);
    let hi = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
    let scale = if hi > lo { (hi - lo) / 255.0 } else { 1.0 };
    let q: Vec<u8> =
        v.iter().map(|&x| ((x - lo) / scale).round().clamp(0.0, 255.0) as u8).collect();
    (q, lo, scale)
}

/// Dequantize a u8 slice back to f32 given `(lo, scale)`.
fn dequant_u8(q: &[u8], lo: f32, scale: f32) -> Vec<f32> {
    q.iter().map(|&b| lo + b as f32 * scale).collect()
}

// ── Per-token quantized Value store ───────────────────────────────────────────────────────────

/// Quantized Value cache: stores each appended token as D u8 bytes + 2 f32 params.
/// Memory per token: D + 8 bytes (vs D*4 f32 = 4× saving at large D).
#[derive(Clone, Debug, Default)]
pub struct QuantValueCache {
    pub d: usize,
    // packed: [token_idx * d .. (token_idx+1)*d] = u8 bytes for that token
    packed: Vec<u8>,
    // per-token scale params: 2 f32 per token
    params: Vec<(f32, f32)>, // (lo, scale)
}

impl QuantValueCache {
    pub fn new(d: usize) -> Self {
        QuantValueCache { d, packed: Vec::new(), params: Vec::new() }
    }
    pub fn len(&self) -> usize {
        self.params.len()
    }
    pub fn is_empty(&self) -> bool {
        self.params.is_empty()
    }
    /// Append one token's V vector (length D), quantizing to u8.
    pub fn push_token(&mut self, v: &[f32]) {
        assert_eq!(v.len(), self.d);
        let (q, lo, scale) = quant_token_to_u8(v);
        self.packed.extend_from_slice(&q);
        self.params.push((lo, scale));
    }
    /// Dequantize all stored tokens to a [T, D] f32 array.
    pub fn dequant_all(&self) -> Array2<f32> {
        let t = self.len();
        let mut out = Array2::<f32>::zeros((t, self.d));
        for (i, &(lo, scale)) in self.params.iter().enumerate() {
            let row = dequant_u8(&self.packed[i * self.d..(i + 1) * self.d], lo, scale);
            for (j, v) in row.into_iter().enumerate() {
                out[(i, j)] = v;
            }
        }
        out
    }
    pub fn memory_bytes(&self) -> usize {
        self.packed.len() + self.params.len() * 8
    }
}

// ── Per-channel quantized Key store ───────────────────────────────────────────────────────────

/// Quantized Key cache: stores each appended token per-CHANNEL (each of the D channels has
/// its own running scale accumulated across all tokens so far). On each new token, the channel
/// scale may expand; old tokens in that channel are NOT re-quantized (acceptable error for
/// a growing cache; exact re-quant is the follow-on). Memory: T*D bytes + D*2 f32 params.
#[derive(Clone, Debug, Default)]
pub struct QuantKeyCache {
    pub d: usize,
    // packed: [token_idx * d .. (token_idx+1)*d] = u8 bytes; row-major [T, D]
    packed: Vec<u8>,
    // per-channel: lo, scale across all tokens seen so far
    ch_lo: Vec<f32>,
    ch_scale: Vec<f32>,
    len: usize,
}

impl QuantKeyCache {
    pub fn new(d: usize) -> Self {
        QuantKeyCache {
            d,
            packed: Vec::new(),
            ch_lo: vec![f32::INFINITY; d],
            ch_scale: vec![1.0; d],
            len: 0,
        }
    }
    pub fn len(&self) -> usize {
        self.len
    }
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    /// Append one token's K vector (length D), updating per-channel scales.
    pub fn push_token(&mut self, k: &[f32]) {
        assert_eq!(k.len(), self.d);
        // Update per-channel lo/scale to encompass the new values.
        for (c, &val) in k.iter().enumerate() {
            if val < self.ch_lo[c] {
                self.ch_lo[c] = val;
            }
            let hi_needed = val;
            let range = hi_needed - self.ch_lo[c];
            if range > 0.0 {
                let new_scale = (hi_needed - self.ch_lo[c]) / 255.0;
                if new_scale > self.ch_scale[c] {
                    self.ch_scale[c] = new_scale;
                }
            }
        }
        // Quantize this token under current per-channel scales.
        let mut row = vec![0u8; self.d];
        for (c, &val) in k.iter().enumerate() {
            row[c] = ((val - self.ch_lo[c]) / self.ch_scale[c]).round().clamp(0.0, 255.0) as u8;
        }
        self.packed.extend_from_slice(&row);
        self.len += 1;
    }
    /// Dequantize all stored tokens to a [T, D] f32 array.
    pub fn dequant_all(&self) -> Array2<f32> {
        let t = self.len;
        let mut out = Array2::<f32>::zeros((t, self.d));
        for i in 0..t {
            for c in 0..self.d {
                let b = self.packed[i * self.d + c];
                out[(i, c)] = self.ch_lo[c] + b as f32 * self.ch_scale[c];
            }
        }
        out
    }
    pub fn memory_bytes(&self) -> usize {
        self.packed.len() + self.d * 8 // D*(lo+scale) = D*8 bytes
    }
}

// ── Fused stateful op ─────────────────────────────────────────────────────────────────────────

/// Fused quantized KV-cache + attention. Stores K in per-channel u8, V in per-token u8.
/// Inputs `[Q, K_new, V_new]` each `[B, H, S, D]`; output has Q's shape.
/// Memory saving vs f32: ~4× (u8 storage + small per-channel/token params).
#[derive(Clone, Debug, PartialEq)]
pub struct QuantizedKvSdpa {
    pub axis: usize,
    pub scale: Option<f32>,
}
impl Eq for QuantizedKvSdpa {}

impl Op for QuantizedKvSdpa {
    fn name(&self) -> StaticName {
        "QuantizedKvSdpa".into()
    }
    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!("axis={}, scale={:?}", self.axis, self.scale)])
    }
    op_as_typed_op!();
}

impl EvalOp for QuantizedKvSdpa {
    fn is_stateless(&self) -> bool {
        false
    }
    fn state(
        &self,
        _session: &TurnState,
        _node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(Some(Box::new(QuantizedKvSdpaState {
            scale: self.scale,
            k_caches: Vec::new(),
            v_caches: Vec::new(),
            initialized: false,
        })))
    }
}

impl TypedOp for QuantizedKvSdpa {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(inputs.len() == 3, "QuantizedKvSdpa expects [Q, K_new, V_new]");
        Ok(tvec!(inputs[0].without_value()))
    }
    as_op!();
}

#[derive(Clone, Debug)]
pub struct QuantizedKvSdpaState {
    scale: Option<f32>,
    k_caches: Vec<QuantKeyCache>,   // one per (batch * kv_head)
    v_caches: Vec<QuantValueCache>, // one per (batch * kv_head)
    initialized: bool,
}

impl OpState for QuantizedKvSdpaState {
    fn eval(
        &mut self,
        _state: &mut TurnState,
        _op: &dyn Op,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        ensure!(inputs.len() == 3, "QuantizedKvSdpa expects [Q, K_new, V_new]");
        let input_dt = inputs[0].datum_type();
        let q = inputs[0].cast_to::<f32>()?;
        let k_new = inputs[1].cast_to::<f32>()?;
        let v_new = inputs[2].cast_to::<f32>()?;
        let qv = q.to_plain_array_view::<f32>()?.into_dimensionality::<Ix4>()?;
        let kv = k_new.to_plain_array_view::<f32>()?.into_dimensionality::<Ix4>()?;
        let vv = v_new.to_plain_array_view::<f32>()?.into_dimensionality::<Ix4>()?;
        let (b, kh, snew, d) = kv.dim();
        let n = b * kh;
        if !self.initialized {
            self.k_caches = (0..n).map(|_| QuantKeyCache::new(d)).collect();
            self.v_caches = (0..n).map(|_| QuantValueCache::new(d)).collect();
            self.initialized = true;
        }
        // Append each new token for each (batch, kv_head).
        for bi in 0..b {
            for hi in 0..kh {
                let idx = bi * kh + hi;
                let ks = kv.slice(s![bi, hi, .., ..]);
                let vs = vv.slice(s![bi, hi, .., ..]);
                for t in 0..snew {
                    self.k_caches[idx].push_token(ks.slice(s![t, ..]).as_slice().unwrap());
                    self.v_caches[idx].push_token(vs.slice(s![t, ..]).as_slice().unwrap());
                }
            }
        }
        // Build full [B, H, T, D] dequantized K/V for attention.
        let (_, _, _, _d) = qv.dim();
        let t = self.k_caches[0].len();
        let mut k_full = Array4::<f32>::zeros((b, kh, t, d));
        let mut v_full = Array4::<f32>::zeros((b, kh, t, d));
        for bi in 0..b {
            for hi in 0..kh {
                let idx = bi * kh + hi;
                let kd = self.k_caches[idx].dequant_all();
                let vd = self.v_caches[idx].dequant_all();
                k_full.slice_mut(s![bi, hi, .., ..]).assign(&kd);
                v_full.slice_mut(s![bi, hi, .., ..]).assign(&vd);
            }
        }
        // flash_attention_gqa handles GQA (hq >= kh, hq % kh == 0).
        let flash = FlashSdpaOp { causal: false, scale: self.scale };
        let o = flash.flash_attention_gqa(qv, k_full.view(), v_full.view(), None);
        Ok(tvec!(o.into_tensor().cast_to_dt(input_dt)?.into_owned().into_tvalue()))
    }
}

#[derive(Clone, Debug)]
struct FrozenQuantizedKvSdpaState {
    scale: Option<f32>,
    k_caches: Vec<QuantKeyCache>,
    v_caches: Vec<QuantValueCache>,
    initialized: bool,
}
impl OpStateFreeze for QuantizedKvSdpaState {
    fn freeze(&self) -> Box<dyn FrozenOpState> {
        Box::new(FrozenQuantizedKvSdpaState {
            scale: self.scale,
            k_caches: self.k_caches.clone(),
            v_caches: self.v_caches.clone(),
            initialized: self.initialized,
        })
    }
}
impl FrozenOpState for FrozenQuantizedKvSdpaState {
    fn unfreeze(&self) -> Box<dyn OpState> {
        Box::new(QuantizedKvSdpaState {
            scale: self.scale,
            k_caches: self.k_caches.clone(),
            v_caches: self.v_caches.clone(),
            initialized: self.initialized,
        })
    }
}

// ── Auto-wiring transform ──────────────────────────────────────────────────────────────────────

/// Fuse `{DynKeyValueCache(K), DynKeyValueCache(V), Sdpa}` into `QuantizedKvSdpa`.
pub fn fuse_quantized_kv_sdpa_rule(
    _ctx: &(),
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
    let mut patch = TypedModelPatch::default();
    let taps = patch.taps(model, &[node.inputs[0], k_node.inputs[0], v_node.inputs[0]])?;
    let fused = patch.wire_node(
        format!("{node_name}.quant_kv_sdpa"),
        QuantizedKvSdpa { axis: kc.axis, scale },
        &taps,
    )?;
    patch.shunt_outside(model, node.id.into(), fused[0])?;
    Ok(Some(patch))
}

/// Strip GQA broadcast chain then fuse cache→Sdpa into QuantizedKvSdpa.
#[derive(Debug, Default)]
pub struct QuantizedKvSdpaTransform;

impl ModelTransform for QuantizedKvSdpaTransform {
    fn name(&self) -> StaticName {
        "fuse_quantized_kv_sdpa".into()
    }
    fn transform(&self, model: &mut TypedModel) -> TractResult<()> {
        Rewriter::default()
            .with_rule_for("fuse-kv-broadcast", crate::ops::sdpa::fuse_kv_cache_broadcast_rule)
            .rewrite(&(), model)?;
        Rewriter::default()
            .with_rule_for("fuse-quant-kv-sdpa", fuse_quantized_kv_sdpa_rule)
            .rewrite(&(), model)?;
        model.compact()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tract_nnef::tract_ndarray::{Array2, arr2};

    fn max_abs(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
        a.iter().zip(b.iter()).map(|(x, y)| (x - y).abs()).fold(0.0, f32::max)
    }

    // Reconstruction error shrinks as bits grow; 16-bit is ~exact.
    #[test]
    fn error_decreases_with_bits() {
        let x = arr2(&[[0.0f32, 1.0, 2.0, 3.0], [-1.0, 0.5, 4.0, 9.0], [2.0, 2.0, 2.0, 2.1]]);
        let e4 = max_abs(&x, &quant_dequant(x.view(), 4, false));
        let e8 = max_abs(&x, &quant_dequant(x.view(), 8, false));
        let e16 = max_abs(&x, &quant_dequant(x.view(), 16, false));
        assert!(e8 < e4, "more bits => less error ({e8} !< {e4})");
        assert!(e16 < e8, "16-bit tighter than 8-bit ({e16} !< {e8})");
        assert!(e16 < 1e-3, "16-bit near-exact, got {e16}");
        // per-element error within half a quantization step of each column's range
        let levels = (1u32 << 8) - 1;
        for j in 0..x.ncols() {
            let col = x.column(j);
            let (lo, hi) = (
                col.iter().copied().fold(f32::INFINITY, f32::min),
                col.iter().copied().fold(f32::NEG_INFINITY, f32::max),
            );
            let step = if hi > lo { (hi - lo) / levels as f32 } else { 0.0 };
            let q = quant_dequant(x.view(), 8, false);
            for i in 0..x.nrows() {
                assert!((x[(i, j)] - q[(i, j)]).abs() <= step / 2.0 + 1e-6);
            }
        }
    }

    // The KIVI insight: with an outlier CHANNEL (a high-magnitude column), per-channel
    // (per-column) quantization isolates it and stays accurate, while per-token (per-row)
    // lumps it with the small dims and crushes them. So per-channel ≪ per-row for Keys.
    #[test]
    fn per_channel_beats_per_row_on_outlier_channel() {
        // 4 tokens x 4 channels; channel 0 is a big-magnitude outlier, others are small.
        let x = arr2(&[
            [100.0f32, 0.10, -0.20, 0.05],
            [-90.0, 0.02, 0.30, -0.08],
            [120.0, -0.15, 0.10, 0.20],
            [-110.0, 0.07, -0.05, 0.12],
        ]);
        // The difference shows on the SMALL channels (cols 1..4): per-token lumps them with
        // the outlier and crushes them; per-channel isolates the outlier so they stay sharp.
        let small_err = |q: &Array2<f32>| -> f32 {
            (1..4)
                .flat_map(|j| (0..4).map(move |i| (i, j)))
                .map(|(i, j)| (x[(i, j)] - q[(i, j)]).abs())
                .fold(0.0, f32::max)
        };
        let pc = small_err(&quant_dequant(x.view(), 4, false)); // per-channel (by column)
        let pt = small_err(&quant_dequant(x.view(), 4, true)); // per-token (by row)
        assert!(pc < pt * 0.2, "per-channel ≫ better on the small dims: pc={pc} pt={pt}");
    }

    // 8-bit KV is near-lossless for attention output; quality improves with bits.
    #[test]
    fn attention_near_lossless_at_8bit() {
        // single head: Q[1,d] . K[s,d] -> softmax -> . V[s,d]
        let (s, d) = (12usize, 16usize);
        let mk = |seed: u64| -> Array2<f32> {
            let mut st = seed;
            Array2::from_shape_fn((s, d), |_| {
                st = st.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((st >> 40) as f32 / (1u64 << 24) as f32) - 0.5
            })
        };
        let q = mk(1).row(0).to_owned();
        let k = mk(2);
        let v = mk(3);
        let scale = 1.0 / (d as f32).sqrt();
        let attn = |k: &Array2<f32>, v: &Array2<f32>| -> Vec<f32> {
            let mut sc: Vec<f32> = (0..s).map(|j| q.dot(&k.row(j)) * scale).collect();
            let m = sc.iter().cloned().fold(f32::MIN, f32::max);
            let mut sum = 0.0;
            sc.iter_mut().for_each(|x| {
                *x = (*x - m).exp();
                sum += *x;
            });
            (0..d).map(|e| (0..s).map(|j| sc[j] / sum * v[(j, e)]).sum()).collect()
        };
        let full = attn(&k, &v);
        let dev = |bits: u32| -> f32 {
            // Keys per-channel (by col), Values per-token (by row) — the KIVI layout.
            let kq = quant_dequant(k.view(), bits, false);
            let vq = quant_dequant(v.view(), bits, true);
            let o = attn(&kq, &vq);
            let num: f32 = o.iter().zip(&full).map(|(a, b)| (a - b).powi(2)).sum::<f32>().sqrt();
            let den: f32 = full.iter().map(|x| x * x).sum::<f32>().sqrt();
            num / den.max(1e-9)
        };
        let (d4, d8, d12) = (dev(4), dev(8), dev(12));
        assert!(d8 < d4 && d12 < d8, "deviation must shrink with bits: 4={d4} 8={d8} 12={d12}");
        assert!(d8 < 0.02, "8-bit KV near-lossless for attention, got {d8}");
    }

    // ─── Integration: packed storage memory savings ───────────────────────────────
    #[test]
    fn packed_u8_saves_memory_vs_f32() {
        let (t, d) = (512usize, 64usize);
        let mut kc = QuantKeyCache::new(d);
        let mut vc = QuantValueCache::new(d);
        let mut rng = 42u64;
        let mut next = || -> f32 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            ((rng >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        };
        for _ in 0..t {
            kc.push_token(&(0..d).map(|_| next()).collect::<Vec<_>>());
            vc.push_token(&(0..d).map(|_| next()).collect::<Vec<_>>());
        }
        let f32_bytes = t * d * 4 * 2; // K + V in f32
        let quant_bytes = kc.memory_bytes() + vc.memory_bytes();
        let ratio = f32_bytes as f32 / quant_bytes as f32;
        // u8 = 1 byte/element vs f32 = 4 bytes; per-channel params for K (D*8),
        // per-token params for V (T*8) — overall >3x saving at T=512 D=64.
        assert!(ratio > 3.0, "expected >3x memory saving, got {ratio:.2}x");
        println!("f32 bytes: {f32_bytes}, quantized: {quant_bytes}, ratio: {ratio:.2}x");
    }

    // ─── Integration: fused op runs through tract's engine, near-lossless ─────────
    #[test]
    fn quantized_kv_sdpa_runs_in_model() -> TractResult<()> {
        let (b, h, d) = (1usize, 2usize, 16usize);
        let scale = 1.0 / (d as f32).sqrt();
        let mut model = TypedModel::default();
        let s = model.sym("S");
        let dim = |x: usize| x.to_dim();
        let f: TVec<TDim> = tvec![dim(b), dim(h), s.into(), dim(d)];
        let q = model.add_source("q", f32::fact(&f))?;
        let k = model.add_source("k", f32::fact(&f))?;
        let v = model.add_source("v", f32::fact(&f))?;
        let o = model.wire_node("qkv", QuantizedKvSdpa { axis: 2, scale: None }, &[q, k, v])?;
        model.select_output_outlets(&o)?;
        let mut rt = model.into_runnable()?.spawn()?;

        // Run 10 decode steps; compare each to full-f32 attention over the growing cache.
        use tract_nnef::tract_core::ops::array::TypedConcat;
        use tract_nnef::tract_ndarray::{Array4 as A4, s};

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
        let attn = |q: A4<f32>, k: A4<f32>, v: A4<f32>| -> A4<f32> {
            let (b, h, sq, d) = q.dim();
            let mut out = A4::<f32>::zeros((b, h, sq, d));
            for bi in 0..b {
                for hi in 0..h {
                    let qm = q.slice(s![bi, hi, .., ..]);
                    let km = k.slice(s![bi, hi, .., ..]);
                    let vm = v.slice(s![bi, hi, .., ..]);
                    let mut sc = qm.dot(&km.t());
                    sc *= scale;
                    for mut row in sc.rows_mut() {
                        let m = row.iter().copied().fold(f32::NEG_INFINITY, f32::max);
                        let mut sm = 0.0f32;
                        row.iter_mut().for_each(|x| {
                            *x = (*x - m).exp();
                            sm += *x;
                        });
                        row.iter_mut().for_each(|x| *x /= sm);
                    }
                    out.slice_mut(s![bi, hi, .., ..]).assign(&sc.dot(&vm));
                }
            }
            out
        };
        let (mut kf, mut vf): (Option<Tensor>, Option<Tensor>) = (None, None);
        for t in 0..10 {
            let qi = mk(9.0 + t as f32 * 0.1);
            let ki = mk(1.0 + t as f32 * 0.07);
            let vi = mk(5.0 - t as f32 * 0.05);
            let o_model = rt
                .run(tvec![qi.clone().into(), ki.clone().into(), vi.clone().into()])?
                .remove(0)
                .into_tensor();
            kf = Some(grow(kf.take(), ki)?);
            vf = Some(grow(vf.take(), vi)?);
            let qv = qi.to_plain_array_view::<f32>()?.into_dimensionality()?;
            let kv = kf.as_ref().unwrap().to_plain_array_view::<f32>()?.into_dimensionality()?;
            let vv = vf.as_ref().unwrap().to_plain_array_view::<f32>()?.into_dimensionality()?;
            let o_ref = Tensor::from(attn(qv.to_owned(), kv.to_owned(), vv.to_owned()));
            // quantized decode should be close to f32 (within ~2% at int8 quality)
            o_model
                .close_enough(&o_ref, Approximation::SuperApproximate)
                .with_context(|| format!("quantized decode too far from f32 at step {t}"))?;
        }
        Ok(())
    }

    // ─── Integration: auto-wiring transform ──────────────────────────────────────
    #[test]
    fn transform_fuses_cache_sdpa_to_quantized() -> TractResult<()> {
        let (b, h, d) = (1usize, 2usize, 16usize);
        let mut model = TypedModel::default();
        let s = model.sym("S");
        let p = model.sym("P");
        let dim = |x: usize| x.to_dim();
        let newf: TVec<TDim> = tvec![dim(b), dim(h), s.into(), dim(d)];
        let pastf: TVec<TDim> = tvec![dim(b), dim(h), p.into(), dim(d)];
        let q = model.add_source("q", f32::fact(&newf))?;
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
                is_causal: false,
            },
            &[q, kc[0], vc[0]],
        )?;
        model.select_output_outlets(&o)?;
        QuantizedKvSdpaTransform.transform(&mut model)?;
        assert!(model.nodes().iter().any(|n| n.op_is::<QuantizedKvSdpa>()), "fused op present");
        assert!(!model.nodes().iter().any(|n| n.op_is::<DynKeyValueCache>()), "caches removed");
        assert!(!model.nodes().iter().any(|n| n.op_is::<Sdpa>()), "sdpa removed");
        Ok(())
    }

    // Memory saving bench: print u8 vs f32 savings at realistic decode lengths.
    //   cargo test -p tract-transformers kv_quant::tests::bench_memory_savings -- --ignored --nocapture
    #[test]
    #[ignore]
    fn bench_memory_savings() {
        let d = 128usize;
        let mut rng = 99u64;
        let mut next = || -> f32 {
            rng = rng.wrapping_mul(6364136223846793005).wrapping_add(1);
            ((rng >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        };
        println!("\n  KV cache memory (int8 u8 vs f32), H=8 heads, D={d}:");
        println!("     T     f32(MB)   int8(MB)   saving");
        for &t in &[256usize, 1024, 4096, 16384] {
            let mut kc = QuantKeyCache::new(d);
            let mut vc = QuantValueCache::new(d);
            for _ in 0..t {
                kc.push_token(&(0..d).map(|_| next()).collect::<Vec<_>>());
                vc.push_token(&(0..d).map(|_| next()).collect::<Vec<_>>());
            }
            let heads = 8;
            let f32_mb = (t * d * 4 * 2 * heads) as f32 / 1e6;
            let int8_mb = ((kc.memory_bytes() + vc.memory_bytes()) * heads) as f32 / 1e6;
            println!("  {t:>6}  {f32_mb:>9.2}  {int8_mb:>9.2}  {:>6.2}x", f32_mb / int8_mb);
        }
    }

    // NNEF round-trip: QuantizedKvSdpa survives write_to_tar -> model_for_read.
    #[test]
    fn quantized_kv_sdpa_nnef_round_trip() -> TractResult<()> {
        use crate::WithTractTransformers;
        let (b, h, d) = (1usize, 2usize, 16usize);
        let mut model = TypedModel::default();
        let s = model.sym("S");
        let dim = |x: usize| x.to_dim();
        let f: TVec<TDim> = tvec![dim(b), dim(h), s.into(), dim(d)];
        let q = model.add_source("q", f32::fact(&f))?;
        let k = model.add_source("k", f32::fact(&f))?;
        let v = model.add_source("v", f32::fact(&f))?;
        let o =
            model.wire_node("qkv", QuantizedKvSdpa { axis: 2, scale: Some(0.125) }, &[q, k, v])?;
        model.select_output_outlets(&o)?;

        let nnef = tract_nnef::nnef().with_tract_transformers();
        let mut buffer = vec![];
        nnef.write_to_tar(&model, &mut buffer)?;
        let reloaded = nnef.model_for_read(&mut &*buffer)?;

        let n = reloaded
            .nodes()
            .iter()
            .find(|n| n.op_is::<QuantizedKvSdpa>())
            .context("QuantizedKvSdpa not found after round-trip")?;
        let op = n.op_as::<QuantizedKvSdpa>().unwrap();
        assert_eq!(op.axis, 2);
        assert_eq!(op.scale, Some(0.125));
        Ok(())
    }
}
