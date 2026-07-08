//! Storage-only quantized KV cache: a drop-in replacement for `DynKeyValueCache` that keeps the
//! resident cache in int8/int4 and dequantizes back to the input dtype on read. It stays a
//! cache-shaped op, so it still resolves the past-length symbol via `init_tensor_fact` /
//! `resolve_symbols`; downstream position/RoPE/mask/attention are untouched.
//!
//! KIVI layout (Liu et al. 2024): Keys are quantized **per channel** (each head-dim channel has
//! its own scale — Keys have large-magnitude outlier channels a shared scale would crush), Values
//! **per token**. To stay consistent in a streaming cache, Keys are quantized in fixed-size blocks
//! whose per-channel scales are frozen once a block is full; the current partial block is kept in
//! f32 until it fills.

use std::str::FromStr;

use tract_nnef::internal::*;
use tract_nnef::prelude::tract_itertools::Itertools;
use tract_nnef::ser::{datum_type, tdims};
use tract_nnef::tract_core::ops::array::MultiBroadcastTo;
use tract_nnef::tract_core::ops::cast::Cast;
use tract_nnef::tract_core::ops::change_axes::AxisOp;
use tract_nnef::tract_core::ops::{FrozenOpState, OpStateFreeze};
use tract_nnef::tract_core::transform::ModelTransform;
use tract_nnef::tract_ndarray::{Array2, Ix4, s};

use crate::ops::apply_rope::ApplyRope;
use crate::ops::dyn_kv_cache::DynKeyValueCache;
use crate::ops::sdpa::Sdpa;

pub fn register(registry: &mut Registry) {
    registry.register_dumper(ser_quant_dyn_kv_cache);
    registry.register_primitive(
        "tract_transformers_quantized_dyn_kv_cache",
        &[
            TypeName::Scalar.tensor().named("input"),
            TypeName::String.named("name"),
            TypeName::Integer.named("axis"),
            TypeName::Integer.named("bits"),
            TypeName::Integer.named("per_channel"),
            TypeName::String.named("datum_type"),
            TypeName::Integer.array().named("past_sequence_shape"),
            TypeName::Integer.array().named("input_sequence_shape"),
        ],
        &[("output", TypeName::Scalar.tensor())],
        de_quant_dyn_kv_cache,
    );
}

// ── affine int8 / int4 packing ──────────────────────────────────────────────────────────────────

/// Max code value at `bits` bits (`levels - 1`). `bits` is 4 or 8.
#[inline]
fn max_code(bits: u32) -> f32 {
    ((1u32 << bits) - 1) as f32
}

/// Bytes needed to store `d` codes at `bits` bits/code, packed.
#[inline]
fn row_bytes(d: usize, bits: u32) -> usize {
    (d * bits as usize).div_ceil(8)
}

/// Pack `codes` (each `<= max_code(bits)`) at `bits` bits/code, appending to `out`.
/// bits=8: one code per byte. bits=4: two codes per byte (even index → low nibble).
fn pack_codes(codes: &[u8], bits: u32, out: &mut Vec<u8>) {
    match bits {
        8 => out.extend_from_slice(codes),
        4 => {
            for pair in codes.chunks(2) {
                let lo = pair[0] & 0x0F;
                let hi = pair.get(1).map(|c| c & 0x0F).unwrap_or(0);
                out.push(lo | (hi << 4));
            }
        }
        _ => unreachable!("bits must be 4 or 8"),
    }
}

/// Read the `c`-th code from a packed row `bytes` at `bits` bits/code.
#[inline]
fn unpack_code(bytes: &[u8], c: usize, bits: u32) -> u8 {
    match bits {
        8 => bytes[c],
        4 => {
            let b = bytes[c >> 1];
            if c & 1 == 0 { b & 0x0F } else { b >> 4 }
        }
        _ => unreachable!("bits must be 4 or 8"),
    }
}

// ── per-token Value store ───────────────────────────────────────────────────────────────────────

/// Values quantized per token: each token gets one `(lo, scale)` pair over its `D` channels.
#[derive(Clone, Debug, Default)]
struct PerTokenStore {
    d: usize,
    bits: u32,
    row_bytes: usize,
    packed: Vec<u8>,
    params: Vec<(f32, f32)>,
}

impl PerTokenStore {
    fn with_bits(d: usize, bits: u32) -> Self {
        PerTokenStore { d, bits, row_bytes: row_bytes(d, bits), ..Default::default() }
    }
    fn len(&self) -> usize {
        self.params.len()
    }
    fn push_token(&mut self, v: &[f32]) {
        let lv = max_code(self.bits);
        let lo = v.iter().copied().fold(f32::INFINITY, f32::min);
        let hi = v.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let scale = if hi > lo { (hi - lo) / lv } else { 1.0 };
        let codes: Vec<u8> =
            v.iter().map(|&x| ((x - lo) / scale).round().clamp(0.0, lv) as u8).collect();
        pack_codes(&codes, self.bits, &mut self.packed);
        self.params.push((lo, scale));
    }
    fn dequant_all(&self) -> Array2<f32> {
        let (d, rb, bits) = (self.d, self.row_bytes, self.bits);
        let mut out = Array2::<f32>::zeros((self.len(), d));
        for (t, &(lo, scale)) in self.params.iter().enumerate() {
            let src = &self.packed[t * rb..t * rb + rb];
            for c in 0..d {
                out[(t, c)] = lo + unpack_code(src, c, bits) as f32 * scale;
            }
        }
        out
    }
}

// ── block-wise per-channel Key store ────────────────────────────────────────────────────────────

const KEY_BLOCK: usize = 32;

/// Keys quantized per channel in blocks of `KEY_BLOCK` tokens. Each finalized block stores `D`
/// `lo`/`scale` values computed over that block and frozen, so every code dequantizes against the
/// exact scale it was encoded with. The current partial block is held in f32 until it fills.
#[derive(Clone, Debug, Default)]
struct BlockChannelStore {
    d: usize,
    bits: u32,
    row_bytes: usize,
    packed: Vec<u8>,       // finalized blocks, [n_finalized * KEY_BLOCK, row_bytes]
    block_lo: Vec<f32>,    // per finalized block: D lo values (block * D + c)
    block_scale: Vec<f32>, // per finalized block: D scale values
    residual: Vec<f32>,    // current partial block, row-major [n_res, D]
    n_res: usize,
    n_finalized: usize,
}

impl BlockChannelStore {
    fn with_bits(d: usize, bits: u32) -> Self {
        BlockChannelStore { d, bits, row_bytes: row_bytes(d, bits), ..Default::default() }
    }
    fn len(&self) -> usize {
        self.n_finalized * KEY_BLOCK + self.n_res
    }
    fn push_token(&mut self, k: &[f32]) {
        self.residual.extend_from_slice(k);
        self.n_res += 1;
        if self.n_res == KEY_BLOCK {
            self.finalize_block();
        }
    }
    fn finalize_block(&mut self) {
        let (d, bits) = (self.d, self.bits);
        let lv = max_code(bits);
        let mut lo = vec![f32::INFINITY; d];
        let mut hi = vec![f32::NEG_INFINITY; d];
        for t in 0..self.n_res {
            for c in 0..d {
                let v = self.residual[t * d + c];
                lo[c] = lo[c].min(v);
                hi[c] = hi[c].max(v);
            }
        }
        let scale: Vec<f32> =
            (0..d).map(|c| if hi[c] > lo[c] { (hi[c] - lo[c]) / lv } else { 1.0 }).collect();
        let mut codes = vec![0u8; d];
        for t in 0..self.n_res {
            for c in 0..d {
                codes[c] =
                    ((self.residual[t * d + c] - lo[c]) / scale[c]).round().clamp(0.0, lv) as u8;
            }
            pack_codes(&codes, bits, &mut self.packed);
        }
        self.block_lo.extend_from_slice(&lo);
        self.block_scale.extend_from_slice(&scale);
        self.residual.clear();
        self.n_res = 0;
        self.n_finalized += 1;
    }
    fn dequant_all(&self) -> Array2<f32> {
        let (d, rb, bits) = (self.d, self.row_bytes, self.bits);
        let mut out = Array2::<f32>::zeros((self.len(), d));
        for bi in 0..self.n_finalized {
            let lo = &self.block_lo[bi * d..bi * d + d];
            let sc = &self.block_scale[bi * d..bi * d + d];
            for ti in 0..KEY_BLOCK {
                let g = bi * KEY_BLOCK + ti;
                let src = &self.packed[g * rb..g * rb + rb];
                for c in 0..d {
                    out[(g, c)] = lo[c] + unpack_code(src, c, bits) as f32 * sc[c];
                }
            }
        }
        let base = self.n_finalized * KEY_BLOCK;
        for ti in 0..self.n_res {
            for c in 0..d {
                out[(base + ti, c)] = self.residual[ti * d + c];
            }
        }
        out
    }
}

/// Per-head quantized store, dispatched on the KIVI layout (Keys per-channel, Values per-token).
#[derive(Clone, Debug)]
enum HeadCache {
    Key(BlockChannelStore),
    Value(PerTokenStore),
}

impl HeadCache {
    fn new(per_channel: bool, d: usize, bits: u32) -> Self {
        if per_channel {
            HeadCache::Key(BlockChannelStore::with_bits(d, bits))
        } else {
            HeadCache::Value(PerTokenStore::with_bits(d, bits))
        }
    }
    fn push(&mut self, row: &[f32]) {
        match self {
            HeadCache::Key(c) => c.push_token(row),
            HeadCache::Value(c) => c.push_token(row),
        }
    }
    fn dequant_all(&self) -> Array2<f32> {
        match self {
            HeadCache::Key(c) => c.dequant_all(),
            HeadCache::Value(c) => c.dequant_all(),
        }
    }
}

// ── op ──────────────────────────────────────────────────────────────────────────────────────────

/// Quantized replacement for `DynKeyValueCache`. `per_channel` selects the KIVI layout
/// (Keys: true, Values: false); `bits` is 4 or 8. The sequence axis must be `rank-2` with the head
/// dim last (the standard `[B, H, S, D]` decode-cache layout).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct QuantizedDynKeyValueCache {
    pub name: String,
    pub axis: usize,
    pub bits: u32,
    pub per_channel: bool,
    pub past_sequence_fact: TypedFact,
    pub input_sequence_fact: TypedFact,
}

impl Op for QuantizedDynKeyValueCache {
    fn name(&self) -> StaticName {
        "QuantizedDynKeyValueCache".to_string().into()
    }
    fn info(&self) -> TractResult<Vec<String>> {
        Ok(vec![format!(
            "bits={}, per_channel={}, axis={}",
            self.bits, self.per_channel, self.axis
        )])
    }
    op_as_typed_op!();
}

impl EvalOp for QuantizedDynKeyValueCache {
    fn is_stateless(&self) -> bool {
        false
    }
    fn state(
        &self,
        _session: &TurnState,
        _node_id: usize,
    ) -> TractResult<Option<Box<dyn OpState>>> {
        Ok(Some(Box::new(QuantizedDynKvCacheState {
            name: self.name.clone(),
            axis: self.axis,
            bits: self.bits,
            per_channel: self.per_channel,
            past_sequence_fact: self.past_sequence_fact.clone(),
            caches: Vec::new(),
            lead_shape: tvec!(),
            d: 0,
            len: 0,
        })))
    }
}

impl TypedOp for QuantizedDynKeyValueCache {
    fn output_facts(&self, inputs: &[&TypedFact]) -> TractResult<TVec<TypedFact>> {
        ensure!(inputs.len() == 1);
        let mut fact = inputs[0].without_value();
        fact.shape.set(
            self.axis,
            self.past_sequence_fact.shape.dims()[self.axis].clone()
                + self.input_sequence_fact.shape.dims()[self.axis].clone(),
        );
        Ok(tvec!(fact))
    }

    fn cost(&self, _inputs: &[&TypedFact]) -> TractResult<TVec<(Cost, TDim)>> {
        let values_per_token = self
            .past_sequence_fact
            .shape
            .iter()
            .enumerate()
            .filter(|(axis, _)| *axis != self.axis)
            .map(|(_, d)| d)
            .product::<TDim>();
        // Resident bytes per token: `bits/8` per value, plus small per-token (Values) or
        // amortized per-block (Keys) scale params — reported as a fraction of the value count.
        let resident_bytes = values_per_token.clone() * self.bits.max(1) as i64 / 8;
        Ok(tvec!(
            (Cost::Custom(false, "KVCacheValuesPerToken".to_string()), values_per_token),
            (Cost::Custom(false, "KVCacheQuantBytesPerToken".to_string()), resident_bytes),
        ))
    }

    as_op!();
}

#[derive(Clone, Debug)]
struct QuantizedDynKvCacheState {
    name: String,
    axis: usize,
    bits: u32,
    per_channel: bool,
    past_sequence_fact: TypedFact,
    caches: Vec<HeadCache>,
    lead_shape: TVec<usize>, // dims before the seq axis (e.g. [batch, heads])
    d: usize,                // head dim (last axis)
    len: usize,              // accumulated sequence length
}

impl QuantizedDynKvCacheState {
    /// Bind the single unresolved symbol in `past_sequence_fact` (the past length) to `len`.
    fn bind_past(&self, state: &mut TurnState, len: usize) -> TractResult<()> {
        let unresolved = self
            .past_sequence_fact
            .shape
            .iter()
            .filter_map(|symb| match symb {
                TDim::Sym(s) if state.resolved_symbols.get(s).is_none() => Some(s),
                _ => None,
            })
            .collect_vec();
        if unresolved.is_empty() {
            return Ok(());
        }
        ensure!(unresolved.len() == 1);
        let sym = unresolved[0];
        state.resolved_symbols.set(sym, len as i64);
        if state.scenario.is_none() {
            state.scenario = sym.scope().unwrap().guess_scenario(&state.resolved_symbols)?;
        }
        Ok(())
    }

    /// Append every new token of `f32in` (`[B, H, S, D]`, seq axis = 2) to the packed caches.
    fn ingest(&mut self, f32in: &Tensor) -> TractResult<()> {
        let view = f32in.to_plain_array_view::<f32>()?;
        ensure!(view.ndim() == 4, "quantized KV cache supports rank-4 [B, H, S, D] caches");
        ensure!(self.axis == 2, "seq axis must be 2 ([B, H, S, D])");
        let view = view.into_dimensionality::<Ix4>()?;
        let (b, h, s, d) = view.dim();
        if self.caches.is_empty() {
            self.d = d;
            self.lead_shape = tvec!(b, h);
            self.caches =
                (0..b * h).map(|_| HeadCache::new(self.per_channel, d, self.bits)).collect();
        }
        ensure!(
            d == self.d && self.lead_shape.as_slice() == [b, h],
            "cache shape changed between steps"
        );
        for bi in 0..b {
            for hi in 0..h {
                let cache = &mut self.caches[bi * h + hi];
                for t in 0..s {
                    cache.push(view.slice(s![bi, hi, t, ..]).as_slice().unwrap());
                }
            }
        }
        self.len += s;
        Ok(())
    }

    /// Reconstruct the full `[B, H, T, D]` f32 cache from the packed per-head stores.
    fn dequantized(&self) -> TractResult<Tensor> {
        let (t, d) = (self.len, self.d);
        let leading: usize = self.lead_shape.iter().product();
        let mut data = vec![0f32; leading * t * d];
        for (idx, cache) in self.caches.iter().enumerate() {
            let deq = cache.dequant_all();
            let base = idx * t * d;
            for ti in 0..t {
                for di in 0..d {
                    data[base + ti * d + di] = deq[(ti, di)];
                }
            }
        }
        let mut shape: Vec<usize> = self.lead_shape.to_vec();
        shape.push(t);
        shape.push(d);
        Tensor::from_shape(&shape, &data)
    }
}

impl OpState for QuantizedDynKvCacheState {
    // Declaring the past-sequence fact as this op's state makes the past-length symbol resolvable
    // (like `DynKeyValueCache`), so downstream shapes that depend on it plan and bind correctly.
    fn init_tensor_fact(&self) -> Option<(String, TypedFact)> {
        Some((self.name.clone(), self.past_sequence_fact.clone()))
    }
    fn has_init_tensor_fact(&self) -> bool {
        true
    }

    fn load_from(
        &mut self,
        state: &mut TurnState,
        states: &mut dyn Iterator<Item = TValue>,
    ) -> TractResult<()> {
        let init = states.next().context("Not enough state initializers")?;
        self.bind_past(state, init.shape()[self.axis])?;
        let f32init = init.cast_to::<f32>()?;
        self.ingest(&f32init)?;
        Ok(())
    }

    fn save_to(&self, states: &mut Vec<TValue>) -> TractResult<()> {
        states.push(self.dequantized()?.into_tvalue());
        Ok(())
    }

    fn resolve_symbols(&mut self, state: &mut TurnState) -> TractResult<()> {
        self.bind_past(state, self.len)
    }

    fn eval(
        &mut self,
        _state: &mut TurnState,
        _op: &dyn Op,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        let input = args_1!(inputs);
        let input_dt = input.datum_type();
        let f32in = input.cast_to::<f32>()?;
        self.ingest(&f32in)?;
        Ok(tvec!(self.dequantized()?.cast_to_dt(input_dt)?.into_owned().into_tvalue()))
    }
}

#[derive(Clone, Debug)]
struct FrozenQuantizedDynKvCacheState {
    name: String,
    axis: usize,
    bits: u32,
    per_channel: bool,
    past_sequence_fact: TypedFact,
    caches: Vec<HeadCache>,
    lead_shape: TVec<usize>,
    d: usize,
    len: usize,
}

impl OpStateFreeze for QuantizedDynKvCacheState {
    fn freeze(&self) -> Box<dyn FrozenOpState> {
        Box::new(FrozenQuantizedDynKvCacheState {
            name: self.name.clone(),
            axis: self.axis,
            bits: self.bits,
            per_channel: self.per_channel,
            past_sequence_fact: self.past_sequence_fact.clone(),
            caches: self.caches.clone(),
            lead_shape: self.lead_shape.clone(),
            d: self.d,
            len: self.len,
        })
    }
}

impl FrozenOpState for FrozenQuantizedDynKvCacheState {
    fn unfreeze(&self) -> Box<dyn OpState> {
        Box::new(QuantizedDynKvCacheState {
            name: self.name.clone(),
            axis: self.axis,
            bits: self.bits,
            per_channel: self.per_channel,
            past_sequence_fact: self.past_sequence_fact.clone(),
            caches: self.caches.clone(),
            lead_shape: self.lead_shape.clone(),
            d: self.d,
            len: self.len,
        })
    }
}

// ── NNEF ser/de ─────────────────────────────────────────────────────────────────────────────────

fn ser_quant_dyn_kv_cache(
    ast: &mut IntoAst,
    node: &TypedNode,
    op: &QuantizedDynKeyValueCache,
) -> TractResult<Option<Arc<RValue>>> {
    let input = ast.mapping[&node.inputs[0]].clone();
    Ok(Some(invocation(
        "tract_transformers_quantized_dyn_kv_cache",
        &[input],
        &[
            ("name", string(&op.name)),
            ("axis", numeric(op.axis)),
            ("bits", numeric(op.bits)),
            ("per_channel", numeric(op.per_channel as i64)),
            ("datum_type", datum_type(op.past_sequence_fact.datum_type)),
            ("past_sequence_shape", tdims(op.past_sequence_fact.shape.dims())),
            ("input_sequence_shape", tdims(op.input_sequence_fact.shape.dims())),
        ],
    )))
}

fn de_quant_dyn_kv_cache(
    builder: &mut ModelBuilder,
    invocation: &ResolvedInvocation,
) -> TractResult<Value> {
    let input = invocation.named_arg_as(builder, "input")?;
    let name: String = invocation.named_arg_as(builder, "name")?;
    let axis: usize = invocation.named_arg_as(builder, "axis")?;
    let bits: i64 = invocation.named_arg_as(builder, "bits")?;
    let per_channel: i64 = invocation.named_arg_as(builder, "per_channel")?;
    let dt = DatumType::from_str(&invocation.named_arg_as::<String>(builder, "datum_type")?)?;
    let past_sequence_shape: TVec<TDim> = builder
        .allowing_new_symbols(|builder| invocation.named_arg_as(builder, "past_sequence_shape"))?;
    let input_sequence_shape: TVec<TDim> = builder
        .allowing_new_symbols(|builder| invocation.named_arg_as(builder, "input_sequence_shape"))?;
    builder.wire(
        QuantizedDynKeyValueCache {
            name,
            axis,
            bits: bits as u32,
            per_channel: per_channel != 0,
            past_sequence_fact: dt.fact(&*past_sequence_shape),
            input_sequence_fact: dt.fact(&*input_sequence_shape),
        },
        &[input],
    )
}

// ── transform ───────────────────────────────────────────────────────────────────────────────────

/// Walk an Sdpa K/V input back through cache-read plumbing (`MultiBroadcastTo` / `AxisOp` / `Cast`
/// / on-read `ApplyRope`) to the `DynKeyValueCache` node; return its id. Every hop must be
/// single-consumer so the cache can be replaced safely.
fn walk_to_cache_node(model: &TypedModel, start: OutletId) -> Option<usize> {
    let mut outlet = start;
    loop {
        let n = model.node(outlet.node);
        if n.outputs[outlet.slot].successors.len() != 1 {
            return None;
        }
        if n.op_is::<DynKeyValueCache>() {
            return Some(n.id);
        } else if n.op_is::<ApplyRope>()
            || n.op_is::<MultiBroadcastTo>()
            || n.op_is::<AxisOp>()
            || n.op_is::<Cast>()
        {
            outlet = n.inputs[0];
        } else {
            return None;
        }
    }
}

fn replace_with_quant_cache(
    patch: &mut TypedModelPatch,
    model: &TypedModel,
    cache_node_id: usize,
    per_channel: bool,
    bits: u32,
) -> TractResult<()> {
    let cnode = model.node(cache_node_id);
    let dkv = cnode.op_as::<DynKeyValueCache>().context("expected DynKeyValueCache")?;
    let tap = patch.taps(model, &[cnode.inputs[0]])?;
    let new = patch.wire_node(
        format!("{}.quant", cnode.name),
        QuantizedDynKeyValueCache {
            name: dkv.name.clone(),
            axis: dkv.axis,
            bits,
            per_channel,
            past_sequence_fact: dkv.past_sequence_fact.clone(),
            input_sequence_fact: dkv.input_sequence_fact.clone(),
        },
        &tap,
    )?;
    patch.shunt_outside(model, cache_node_id.into(), new[0])?;
    Ok(())
}

/// For each `Sdpa`, replace its two upstream KV caches with quantized ones: the K cache (input 1)
/// per-channel and the V cache (input 2) per-token (KIVI). Attention/RoPE/mask are left untouched.
fn quantize_kv_storage_rule(
    ctx: &u32,
    model: &TypedModel,
    node: &TypedNode,
    _node_name: &str,
    _op: &Sdpa,
) -> TractResult<Option<TypedModelPatch>> {
    if node.inputs.len() != 3 && node.inputs.len() != 4 {
        return Ok(None);
    }
    let (Some(k_cache), Some(v_cache)) =
        (walk_to_cache_node(model, node.inputs[1]), walk_to_cache_node(model, node.inputs[2]))
    else {
        return Ok(None);
    };
    if k_cache == v_cache {
        return Ok(None);
    }
    let mut patch = TypedModelPatch::default();
    replace_with_quant_cache(&mut patch, model, k_cache, true, *ctx)?;
    replace_with_quant_cache(&mut patch, model, v_cache, false, *ctx)?;
    Ok(Some(patch))
}

/// Replace the KV caches feeding each attention with storage-quantized ones (int8/int4). Keeps a
/// cache-shaped op so the past-length symbol still resolves; attention math is unchanged. `bits`
/// selects int8 (default) or int4.
#[derive(Debug, Clone, Copy)]
pub struct QuantizeKvStorageTransform {
    pub bits: u32,
}

impl Default for QuantizeKvStorageTransform {
    fn default() -> Self {
        QuantizeKvStorageTransform { bits: 8 }
    }
}

impl ModelTransform for QuantizeKvStorageTransform {
    fn name(&self) -> StaticName {
        "quantize_kv_storage".into()
    }
    fn transform(&self, model: &mut TypedModel) -> TractResult<()> {
        ensure!(self.bits == 4 || self.bits == 8, "KV quantization bits must be 4 or 8");
        // Pre-passes so the rule sees a single ApplyRope and an internalized cache (idempotent if
        // an earlier stage already ran them).
        crate::rewriter::ApplyRopeTransform.transform(model)?;
        crate::rewriter::KeyValueCacheTransform.transform(model)?;
        Rewriter::default()
            .with_rule_for("quantize-kv-storage", quantize_kv_storage_rule)
            .rewrite(&self.bits, model)?;
        model.compact()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tract_nnef::tract_core::ops::array::TypedConcat;

    fn model_with_cache(bits: u32, per_channel: bool) -> TractResult<TypedModel> {
        let mut model = TypedModel::default();
        let s = model.sym("S");
        let p = model.sym("P");
        let (b, h, d) = (1usize, 2usize, 8usize);
        let inf: TVec<TDim> = tvec![b.to_dim(), h.to_dim(), s.into(), d.to_dim()];
        let pf: TVec<TDim> = tvec![b.to_dim(), h.to_dim(), p.into(), d.to_dim()];
        let input = model.add_source("input", f32::fact(&inf))?;
        let op = QuantizedDynKeyValueCache {
            name: "kv0".to_string(),
            axis: 2,
            bits,
            per_channel,
            past_sequence_fact: f32::fact(&pf),
            input_sequence_fact: f32::fact(&inf),
        };
        let out = model.wire_node("kv", op, &[input])?;
        model.select_output_outlets(&out)?;
        Ok(model)
    }

    // The quantized cache accumulates like a plain concat and dequantizes near-losslessly at int8.
    #[test]
    fn quant_cache_accumulates_near_lossless_int8() -> TractResult<()> {
        for per_channel in [false, true] {
            let mut rt = model_with_cache(8, per_channel)?.into_runnable()?.spawn()?;
            let (b, h, d) = (1usize, 2usize, 8usize);
            let mut st = 3u64;
            let mut nf = || {
                st = st.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((st >> 40) as f32 / (1u64 << 24) as f32) - 0.5
            };
            let mut acc: Option<Tensor> = None;
            for _ in 0..40 {
                let step = Tensor::from_shape(
                    &[b, h, 1, d],
                    &(0..b * h * d).map(|_| nf()).collect::<Vec<f32>>(),
                )?;
                let out = rt.run(tvec![step.clone().into()])?.remove(0).into_tensor();
                acc = Some(match acc.take() {
                    None => step,
                    Some(a) => TypedConcat { axis: 2 }
                        .eval(tvec![a.into(), step.into()])?
                        .remove(0)
                        .into_tensor(),
                });
                acc.as_ref().unwrap().close_enough(&out, Approximation::SuperApproximate)?;
            }
        }
        Ok(())
    }

    // Block-wise per-channel Keys stay consistent even when a channel's range grows late — the
    // case a running per-channel scale would mis-dequantize.
    #[test]
    fn block_per_channel_key_consistent_over_growing_range() {
        let d = 16usize;
        let mut kc = BlockChannelStore::with_bits(d, 8);
        let mut rows = Vec::new();
        for t in 0..80usize {
            let row: Vec<f32> = (0..d)
                .map(|c| {
                    if c == 0 { t as f32 * 2.0 } else { (((t * 7 + c) as f32) * 0.013).sin() * 0.1 }
                })
                .collect();
            kc.push_token(&row);
            rows.push(row);
        }
        assert_eq!(kc.len(), 80);
        let deq = kc.dequant_all();
        let mut maxerr = 0f32;
        for t in 0..80 {
            for c in 0..d {
                maxerr = maxerr.max((deq[(t, c)] - rows[t][c]).abs());
            }
        }
        assert!(maxerr < 1.0, "block-consistent int8 dequant, got {maxerr}");
    }

    #[test]
    fn quant_cache_nnef_round_trip() -> TractResult<()> {
        use crate::WithTractTransformers;
        let model = model_with_cache(4, true)?;
        let nnef = tract_nnef::nnef().with_tract_transformers();
        let mut buffer = vec![];
        nnef.write_to_tar(&model, &mut buffer)?;
        let reloaded = nnef.model_for_read(&mut &*buffer)?;
        let n = reloaded
            .nodes()
            .iter()
            .find_map(|n| n.op_as::<QuantizedDynKeyValueCache>())
            .context("QuantizedDynKeyValueCache missing after round-trip")?;
        assert_eq!(n.bits, 4);
        assert!(n.per_channel);
        assert_eq!(n.axis, 2);
        Ok(())
    }
}
