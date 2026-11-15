//! Routed MoE kernels: route selection, combine, the per-route GEMV over Q4_0, Q2_0_T, f16 and
//! f32 expert weights, and the postamble. i64 ids live as (low, high) u32 word pairs; the kernels
//! read and write the low word.

use tract_core::internal::*;
use tract_core::tract_linalg::block_quant::BlockQuantFact;
use tract_gpu::ops::routing::FusePairPolicy;
use tract_gpu::ops::routing::GpuRouteSelect;
use tract_gpu::ops::routing::GpuRoutedCombine;
use tract_gpu::ops::routing::GpuRoutedLinear;
use tract_gpu::ops::routing::RoutedFormat;
use tract_gpu::ops::routing::RoutedLinearKernels;
use tract_gpu::ops::routing::RoutedStrategy;
use tract_gpu::ops::routing::RoutedTuning;
use tract_gpu::ops::routing::RoutingKernels;
use tract_gpu::sync::DeviceSyncKind;
use tract_gpu::sync::sync_inputs_if_required;
use tract_gpu::tensor::DeviceTensor;
use tract_transformers::ops::moe_ffn::MoeFfn;
use tract_transformers::ops::moe_ffn::RouteSelect;
use tract_transformers::ops::moe_ffn::RoutedCombine;
use tract_transformers::ops::moe_ffn::RoutedLinear;

use crate::kernels::matmul_bq::bq_fact;
use crate::kernels::shaders::EntryPoint;
use crate::kernels::shaders::GRID_LIMIT;
use crate::kernels::shaders::LayoutKind;
use crate::kernels::shaders::ShaderDtype;
use crate::kernels::shaders::WORKGROUP;
use crate::kernels::shaders::pack_u32s;
use crate::kernels::shaders::preamble;
use crate::kernels::tiled_gemm::MATMUL_TILE;
use crate::kernels::tiled_gemm::TILED_THREADS;
use crate::kernels::tiled_gemm::tiled_gemm_core;
use crate::kernels::tiled_gemm::tiled_gemm_tiles;
use crate::register_wgpu_op;
use crate::transform::Mapping;
use crate::utils::element_offset;
use crate::utils::get_wgpu_buffer;
use crate::with_wgpu_queue;

/// Output rows of one routed GEMV workgroup.
const MV_ROWS: u32 = 4;

/// Routes of one expert a tiled workgroup takes: the rows of its output tile.
const SORT_CHUNK: usize = 32;

/// Most experts the route sort bins.
const SORT_MAX_EXPERTS: usize = 256;

/// Word offset of a tensor in its buffer.
fn word_offset(t: &DeviceTensor) -> TractResult<u32> {
    let bytes = t.buffer_offset::<usize>();
    ensure!(bytes.is_multiple_of(4), "routed kernels read whole words");
    u32::try_from(bytes / 4).context("buffer offset does not fit a WGSL u32")
}

fn elem_offset(t: &DeviceTensor) -> TractResult<u32> {
    u32::try_from(element_offset(t, 0)).context("buffer offset does not fit a WGSL u32")
}

fn u32_of(v: usize) -> TractResult<u32> {
    u32::try_from(v).context("routed size does not fit a WGSL u32")
}

const ROUTE_SELECT: &str = r#"
struct Params {
    off_scores: u32,
    off_bias: u32,
    off_tokens: u32,
    off_experts: u32,
    off_weights: u32,
    tokens: u32,
    experts: u32,
    k: u32,
    gate: u32,
    has_bias: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> scores: array<f32>;
@group(0) @binding(1) var<storage, read> bias: array<f32>;
@group(0) @binding(2) var<storage, read_write> token_ids: array<u32>;
@group(0) @binding(3) var<storage, read_write> expert_ids: array<u32>;
@group(0) @binding(4) var<storage, read_write> weights: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

const NONE: u32 = 0xFFFFFFFFu;
var<workgroup> red_f: array<f32, 64>;
var<workgroup> red_key: array<u32, 64>;
var<workgroup> red_expert: array<u32, 64>;

// Order-preserving image of a score: a larger key is a larger score, -0.0 maps with 0.0.
fn route_key(v: f32) -> u32 {
    var bits = bitcast<u32>(v);
    if (v == 0.0) { bits = 0u; }
    return select(bits | 0x80000000u, ~bits, (bits & 0x80000000u) != 0u);
}

fn route_score(key: u32) -> f32 {
    return bitcast<f32>(select(~key, key ^ 0x80000000u, (key & 0x80000000u) != 0u));
}

fn reduce_max(t: u32, v: f32) -> f32 {
    red_f[t] = v;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride = stride >> 1u) {
        if (t < stride) { red_f[t] = max(red_f[t], red_f[t + stride]); }
        workgroupBarrier();
    }
    let r = red_f[0];
    workgroupBarrier();
    return r;
}

fn reduce_sum(t: u32, v: f32) -> f32 {
    red_f[t] = v;
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride = stride >> 1u) {
        if (t < stride) { red_f[t] = red_f[t] + red_f[t + stride]; }
        workgroupBarrier();
    }
    let r = red_f[0];
    workgroupBarrier();
    return r;
}

// One workgroup per token; expert e lives in thread e % 64, slot e / 64. Each pick takes the
// largest key, then the lowest expert among equal keys.
@compute @workgroup_size(64)
fn route_select(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) t: u32,
) {
    let token = wid.y * 65535u + wid.x;
    if (token >= params.tokens) { return; }
    var s: array<f32, 4>;
    var valid = 0u;
    var local_max = bitcast<f32>(0xff800000u);
    for (var r = 0u; r < 4u; r++) {
        let e = r * 64u + t;
        s[r] = bitcast<f32>(0xff800000u);
        if (e < params.experts) {
            valid = valid | (1u << r);
            var v = scores[params.off_scores + token * params.experts + e];
            if (params.has_bias != 0u) { v = v + bias[params.off_bias + e]; }
            s[r] = v;
            local_max = max(local_max, v);
        }
    }
    let max_all = reduce_max(t, local_max);
    var local_sum = 0.0;
    for (var r = 0u; r < 4u; r++) {
        if ((valid & (1u << r)) != 0u) { local_sum = local_sum + exp(s[r] - max_all); }
    }
    let denom_all = reduce_sum(t, local_sum);

    var top = 0.0;
    var denom_topk = 0.0;
    for (var slot = 0u; slot < params.k; slot++) {
        var best_key = 0u;
        var best_expert = NONE;
        for (var r = 0u; r < 4u; r++) {
            if ((valid & (1u << r)) != 0u) {
                let key = route_key(s[r]);
                if (best_expert == NONE || key > best_key) {
                    best_key = key;
                    best_expert = r * 64u + t;
                }
            }
        }
        red_key[t] = best_key;
        red_expert[t] = best_expert;
        workgroupBarrier();
        for (var stride = 32u; stride > 0u; stride = stride >> 1u) {
            if (t < stride) {
                let ke = red_key[t + stride];
                let ee = red_expert[t + stride];
                let mine = red_expert[t];
                if (ee != NONE && (mine == NONE || ke > red_key[t] || (ke == red_key[t] && ee < mine))) {
                    red_key[t] = ke;
                    red_expert[t] = ee;
                }
            }
            workgroupBarrier();
        }
        let win_key = red_key[0];
        let win_expert = red_expert[0];
        workgroupBarrier();
        let win_score = route_score(win_key);
        if (slot == 0u) { top = win_score; }
        denom_topk = denom_topk + exp(win_score - top);
        if (win_expert % 64u == t) { valid = valid & ~(1u << (win_expert / 64u)); }
        if (t == 0u) {
            let route = token * params.k + slot;
            token_ids[2u * (params.off_tokens + route)] = token;
            token_ids[2u * (params.off_tokens + route) + 1u] = 0u;
            expert_ids[2u * (params.off_experts + route)] = win_expert;
            expert_ids[2u * (params.off_experts + route) + 1u] = 0u;
            var w = exp(win_score - top);
            if (params.gate == 1u) {
                w = exp(win_score - max_all) / denom_all;
            } else if (params.gate == 2u) {
                w = 1.0 / (1.0 + exp(-win_score));
            } else if (params.gate == 3u) {
                w = win_score;
            }
            weights[params.off_weights + route] = w;
        }
    }
    if (params.gate == 0u && t == 0u) {
        for (var slot = 0u; slot < params.k; slot++) {
            let ix = params.off_weights + token * params.k + slot;
            weights[ix] = weights[ix] / denom_topk;
        }
    }
}
"#;

fn select(
    scores: &DeviceTensor,
    bias: Option<&DeviceTensor>,
    k: usize,
    gate: u32,
    token_ids: &DeviceTensor,
    expert_ids: &DeviceTensor,
    weights: &DeviceTensor,
) -> TractResult<()> {
    let (tokens, experts) = (scores.shape()[0], scores.shape()[1]);
    ensure!(experts <= 256 && k <= 16, "Wgpu route select takes 256 experts and 16 routes");
    with_wgpu_queue(|q| {
        let bias_or_scores = bias.unwrap_or(scores);
        for t in [scores, bias_or_scores, token_ids, expert_ids, weights] {
            q.retain_tensor(t);
        }
        let layout = LayoutKind::TailWrites { n: 5, writes: 3 };
        let pipeline = q.context().chain_pipeline(
            "route_select",
            layout,
            EntryPoint::plain("route_select"),
            || ROUTE_SELECT.to_string(),
        )?;
        let bg = q.context().bind_group(
            layout,
            &[
                get_wgpu_buffer(scores),
                get_wgpu_buffer(bias_or_scores),
                get_wgpu_buffer(token_ids),
                get_wgpu_buffer(expert_ids),
                get_wgpu_buffer(weights),
            ],
            q.uniform(),
        )?;
        let params = pack_u32s(&[
            elem_offset(scores)?,
            elem_offset(bias_or_scores)?,
            elem_offset(token_ids)?,
            elem_offset(expert_ids)?,
            elem_offset(weights)?,
            u32_of(tokens)?,
            u32_of(experts)?,
            u32_of(k)?,
            gate,
            bias.is_some() as u32,
            0,
            0,
        ]);
        let dyn_off = q.alloc_uniform(&params)?;
        let grid = [
            tokens.min(GRID_LIMIT as usize) as u32,
            tokens.div_ceil(GRID_LIMIT as usize) as u32,
            1,
        ];
        q.dispatch_grid("route_select", &pipeline, &bg, dyn_off, grid)
    })
}

fn combine_module(dt: ShaderDtype) -> String {
    let t = dt.wgsl();
    format!(
        r#"{preamble}
struct Params {{
    off_values: u32,
    off_weights: u32,
    off_out: u32,
    tokens: u32,
    d: u32,
    k: u32,
    _pad0: u32,
    _pad1: u32,
}}

@group(0) @binding(0) var<storage, read> values: array<f32>;
@group(0) @binding(1) var<storage, read> weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> out: array<{t}>;
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size({WORKGROUP})
fn routed_combine_{t}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.y * {GRID_LIMIT}u * {WORKGROUP}u + gid.x;
    if (i >= params.tokens * params.d) {{ return; }}
    let token = i / params.d;
    let col = i - token * params.d;
    var acc = 0.0;
    for (var j = 0u; j < params.k; j++) {{
        let route = token * params.k + j;
        acc = acc + weights[params.off_weights + route] * values[params.off_values + route * params.d + col];
    }}
    out[params.off_out + i] = {t}(acc);
}}
"#,
        preamble = preamble(dt)
    )
}

fn combine(
    values: &DeviceTensor,
    weights: &DeviceTensor,
    k: usize,
    out: &DeviceTensor,
) -> TractResult<()> {
    let d = *out.shape().last().context("combine output has no feature axis")?;
    let tokens = out.len() / d.max(1);
    ensure!(values.shape() == [tokens * k, d] && weights.shape() == [tokens * k]);
    with_wgpu_queue(|q| {
        for t in [values, weights, out] {
            q.retain_tensor(t);
        }
        let dt = ShaderDtype::from_datum(out.datum_type())?;
        let layout = LayoutKind::TailWrites { n: 3, writes: 1 };
        let pipeline = q.context().chain_pipeline(
            ("routed_combine", dt),
            layout,
            EntryPoint::typed("routed_combine", dt),
            || combine_module(dt),
        )?;
        let bg = q.context().bind_group(
            layout,
            &[get_wgpu_buffer(values), get_wgpu_buffer(weights), get_wgpu_buffer(out)],
            q.uniform(),
        )?;
        let params = pack_u32s(&[
            elem_offset(values)?,
            elem_offset(weights)?,
            elem_offset(out)?,
            u32_of(tokens)?,
            u32_of(d)?,
            u32_of(k)?,
            0,
            0,
        ]);
        let dyn_off = q.alloc_uniform(&params)?;
        q.dispatch("routed_combine", &pipeline, &bg, dyn_off, (tokens * d) as u64)
    })
}

const WGPU_ROUTING_KERNELS: RoutingKernels = RoutingKernels { select, combine };

register_wgpu_op!(RouteSelect, |source, node, op| {
    Ok(GpuRouteSelect::from_tract_core(source, node, op, "Wgpu", WGPU_ROUTING_KERNELS)?
        .map(|op| Box::new(op) as _))
});

/// Units of work along one row of `k` weights: code words for the block formats, runs of four
/// weights for the plain ones.
fn units(format: RoutedFormat, k: usize) -> usize {
    match format {
        RoutedFormat::Q40 => k / 32 * 4,
        RoutedFormat::Q20T => k / 32 * 2,
        RoutedFormat::F16 | RoutedFormat::F32 => k / 4,
    }
}

/// wgpu's routed strategies: the sorted tiled matmul from 12 routes per expert for block-quantized
/// weights and 5 for plain ones, whose GEMV decodes slowest; no grouped GEMV.
const WGPU_ROUTED_TUNING: RoutedTuning = RoutedTuning {
    tiled_min_per_expert: |format| Some(if format.block_quant().is_some() { 12 } else { 5 }),
    tiled_chunk: SORT_CHUNK,
    grouped_min_per_expert: |_| None,
    grouped_chunk: 0,
    sorted_max_experts: SORT_MAX_EXPERTS,
    sorted_k_multiple: 1,
};

/// The format of `[E, N, K]` expert weights, if the kernels read them: plain weights need K a
/// multiple of 4.
fn format_of(dt: DatumType, block: Option<&BlockQuantFact>, k: usize) -> Option<RoutedFormat> {
    match block {
        Some(b) => RoutedFormat::of_block(&*b.format, k),
        None if k.is_multiple_of(4) => RoutedFormat::of_plain(dt),
        None => None,
    }
}

/// Expert weights the routed GEMV reads, `[E, N, K]`: Q4_0 or Q2_0_T, or plain f32 or f16 with
/// K a multiple of 4.
fn supports(fact: &TypedFact) -> bool {
    let Some(k) = fact.shape.as_concrete().filter(|s| s.len() == 3).map(|s| s[2]) else {
        return false;
    };
    let exotic = fact.exotic_fact();
    let block = exotic.and_then(|e| e.downcast_ref::<BlockQuantFact>());
    (exotic.is_none() || block.is_some()) && format_of(fact.datum_type, block, k).is_some()
}

/// The per-format body of the GEMV loop over `unit`: adds this unit's products to `acc[r]` for
/// the workgroup's rows. Block formats decode a code word and its block scale; plain ones take
/// four weights.
fn mv_unit_body(format: RoutedFormat) -> &'static str {
    match format {
        RoutedFormat::Q40 => {
            r#"
        let block = unit >> 2u;
        let w = unit & 3u;
        let xb = x_base + block * 32u + 4u * w;
        let xl = vec4<f32>(x[xb], x[xb + 1u], x[xb + 2u], x[xb + 3u]);
        let xh = vec4<f32>(x[xb + 16u], x[xb + 17u], x[xb + 18u], x[xb + 19u]);
        let sum_x = dot(xl + xh, vec4<f32>(1.0));
        for (var r = 0u; r < 4u; r++) {
            let bi = wr[r] * params.blocks_per_row + block;
            let scale = unpack2x16float(wbuf[params.scale_off + bi / params.group]).x;
            let word = wbuf[params.code_off + bi * 4u + w];
            let lo = vec4<f32>(vec4<u32>(word, word >> 8u, word >> 16u, word >> 24u) & vec4<u32>(15u));
            let hi = vec4<f32>(vec4<u32>(word >> 4u, word >> 12u, word >> 20u, word >> 28u) & vec4<u32>(15u));
            acc[r] = acc[r] + scale * (dot(lo, xl) + dot(hi, xh) - 8.0 * sum_x);
        }"#
        }
        RoutedFormat::Q20T => {
            r#"
        let block = unit >> 1u;
        let w = unit & 1u;
        let xb = x_base + block * 32u + 16u * w;
        var xs: array<vec4<f32>, 4>;
        var sum_x = 0.0;
        for (var j = 0u; j < 4u; j++) {
            xs[j] = vec4<f32>(x[xb + 4u * j], x[xb + 4u * j + 1u], x[xb + 4u * j + 2u], x[xb + 4u * j + 3u]);
            sum_x = sum_x + dot(xs[j], vec4<f32>(1.0));
        }
        for (var r = 0u; r < 4u; r++) {
            let bi = wr[r] * params.blocks_per_row + block;
            let scale = unpack2x16float(wbuf[params.scale_off + bi / params.group]).x;
            let word = wbuf[params.code_off + bi * 2u + w];
            var cx = 0.0;
            for (var j = 0u; j < 4u; j++) {
                let sh = 8u * j;
                let c = vec4<f32>(vec4<u32>(word >> sh, word >> (sh + 2u), word >> (sh + 4u), word >> (sh + 6u)) & vec4<u32>(3u));
                cx = cx + dot(c, xs[j]);
            }
            acc[r] = acc[r] + scale * (cx - sum_x);
        }"#
        }
        RoutedFormat::F16 => {
            r#"
        let xb = x_base + 4u * unit;
        let xv = vec4<f32>(x[xb], x[xb + 1u], x[xb + 2u], x[xb + 3u]);
        for (var r = 0u; r < 4u; r++) {
            let wo = params.code_off + (wr[r] * params.k + 4u * unit) / 2u;
            let wv = vec4<f32>(unpack2x16float(wbuf[wo]), unpack2x16float(wbuf[wo + 1u]));
            acc[r] = acc[r] + dot(wv, xv);
        }"#
        }
        RoutedFormat::F32 => {
            r#"
        let xb = x_base + 4u * unit;
        let xv = vec4<f32>(x[xb], x[xb + 1u], x[xb + 2u], x[xb + 3u]);
        for (var r = 0u; r < 4u; r++) {
            let wo = params.code_off + wr[r] * params.k + 4u * unit;
            let wv = vec4<f32>(bitcast<f32>(wbuf[wo]), bitcast<f32>(wbuf[wo + 1u]), bitcast<f32>(wbuf[wo + 2u]), bitcast<f32>(wbuf[wo + 3u]));
            acc[r] = acc[r] + dot(wv, xv);
        }"#
        }
    }
}

fn mv_module(format: RoutedFormat) -> String {
    format!(
        r#"
struct Params {{
    code_off: u32,
    scale_off: u32,
    group: u32,
    blocks_per_row: u32,
    k: u32,
    n: u32,
    routes: u32,
    units: u32,
    off_x: u32,
    off_tokens: u32,
    off_experts: u32,
    off_out: u32,
    has_rows: u32,
    _pad0: u32,
    _pad1: u32,
    _pad2: u32,
}}

@group(0) @binding(0) var<storage, read> wbuf: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> token_ids: array<u32>;
@group(0) @binding(3) var<storage, read> expert_ids: array<u32>;
@group(0) @binding(4) var<storage, read_write> out: array<f32>;
@group(0) @binding(5) var<uniform> params: Params;

var<workgroup> red: array<f32, 256>;

// One workgroup per four output rows of route wid.y, the threads striding over the row's units.
// The last row group of an expert clamps its rows to n - 1, storing only those below n.
@compute @workgroup_size(64)
fn routed_mv_{name}(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) t: u32,
) {{
    let route = wid.y;
    let row0 = wid.x * 4u;
    let expert = expert_ids[2u * (params.off_experts + route)];
    var in_row = route;
    if (params.has_rows != 0u) {{ in_row = token_ids[2u * (params.off_tokens + route)]; }}
    let rows = min(4u, params.n - row0);
    var wr: array<u32, 4>;
    for (var r = 0u; r < 4u; r++) {{ wr[r] = expert * params.n + min(row0 + r, params.n - 1u); }}
    let x_base = params.off_x + in_row * params.k;
    var acc = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
    for (var unit = t; unit < params.units; unit += 64u) {{{body}
    }}
    for (var r = 0u; r < 4u; r++) {{ red[r * 64u + t] = acc[r]; }}
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride = stride >> 1u) {{
        if (t < stride) {{
            for (var r = 0u; r < 4u; r++) {{ red[r * 64u + t] = red[r * 64u + t] + red[r * 64u + t + stride]; }}
        }}
        workgroupBarrier();
    }}
    if (t < rows) {{
        out[params.off_out + route * params.n + row0 + t] = red[t * 64u];
    }}
}}
"#,
        name = format.name(),
        body = mv_unit_body(format),
    )
}

const ROUTE_PLAN: &str = r#"
struct Params {
    off_experts: u32,
    off_sorted: u32,
    off_chunks: u32,
    routes: u32,
    experts: u32,
    max_chunks: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(0) var<storage, read> expert_ids: array<u32>;
@group(0) @binding(1) var<storage, read_write> sorted: array<u32>;
@group(0) @binding(2) var<storage, read_write> chunks: array<u32>;
@group(0) @binding(3) var<uniform> params: Params;

const CHUNK: u32 = 32u;
var<workgroup> cursor: array<atomic<u32>, 256>;

// One workgroup: `sorted` lists each expert's routes contiguously, `chunks` holds
// [expert, base, len] for every CHUNK routes of an expert, then expert 0xFFFFFFFF.
@compute @workgroup_size(256)
fn route_plan(@builtin(local_invocation_index) t: u32) {
    if (t < params.experts) { atomicStore(&cursor[t], 0u); }
    workgroupBarrier();
    for (var r = t; r < params.routes; r += 256u) {
        atomicAdd(&cursor[expert_ids[2u * (params.off_experts + r)]], 1u);
    }
    workgroupBarrier();
    if (t == 0u) {
        var run = 0u;
        var chunk = 0u;
        for (var e = 0u; e < params.experts; e++) {
            let count = atomicLoad(&cursor[e]);
            atomicStore(&cursor[e], run);
            for (var b = 0u; b < count; b += CHUNK) {
                let c = params.off_chunks + 3u * chunk;
                chunks[c] = e;
                chunks[c + 1u] = run + b;
                chunks[c + 2u] = min(CHUNK, count - b);
                chunk++;
            }
            run += count;
        }
        for (; chunk < params.max_chunks; chunk++) {
            chunks[params.off_chunks + 3u * chunk] = 0xFFFFFFFFu;
        }
    }
    workgroupBarrier();
    for (var r = t; r < params.routes; r += 256u) {
        let slot = atomicAdd(&cursor[expert_ids[2u * (params.off_experts + r)]], 1u);
        sorted[params.off_sorted + slot] = r;
    }
}
"#;

/// Reads weight `(k, c)` of expert `ctx.x` as f32, zero past the weights.
fn mm_ld_b(format: RoutedFormat) -> &'static str {
    match format {
        RoutedFormat::Q40 => {
            "    let bi = (ctx.x * params.n + c) * params.blocks_per_row + (k >> 5u);
    let scale = unpack2x16float(wbuf[params.scale_off + bi / params.group]).x;
    let inner = k & 31u;
    let word = wbuf[params.code_off + bi * 4u + ((inner & 15u) >> 2u)];
    let shift = 8u * (inner & 3u) + select(0u, 4u, inner >= 16u);
    return (f32((word >> shift) & 15u) - 8.0) * scale;"
        }
        RoutedFormat::Q20T => {
            "    let bi = (ctx.x * params.n + c) * params.blocks_per_row + (k >> 5u);
    let scale = unpack2x16float(wbuf[params.scale_off + bi / params.group]).x;
    let inner = k & 31u;
    let word = wbuf[params.code_off + bi * 2u + (inner >> 4u)];
    return (f32((word >> (2u * (inner & 15u))) & 3u) - 1.0) * scale;"
        }
        RoutedFormat::F16 => {
            "    let e = (ctx.x * params.n + c) * params.k + k;
    let pair = unpack2x16float(wbuf[params.code_off + e / 2u]);
    return select(pair.x, pair.y, (e & 1u) == 1u);"
        }
        RoutedFormat::F32 => {
            "    return bitcast<f32>(wbuf[params.code_off + (ctx.x * params.n + c) * params.k + k]);"
        }
    }
}

fn mm_module(format: RoutedFormat) -> String {
    format!(
        r#"
struct Params {{
    code_off: u32,
    scale_off: u32,
    group: u32,
    blocks_per_row: u32,
    k: u32,
    n: u32,
    off_x: u32,
    off_tokens: u32,
    off_sorted: u32,
    off_chunks: u32,
    off_out: u32,
    has_rows: u32,
}}

@group(0) @binding(0) var<storage, read> wbuf: array<u32>;
@group(0) @binding(1) var<storage, read> x: array<f32>;
@group(0) @binding(2) var<storage, read> token_ids: array<u32>;
@group(0) @binding(3) var<storage, read> sorted: array<u32>;
@group(0) @binding(4) var<storage, read> chunks: array<u32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;
@group(0) @binding(6) var<uniform> params: Params;

{tiles}
var<workgroup> chunk_expert: u32;

// Row r of the chunk at ctx.x, ctx.y rows long: the input row of route sorted[ctx.x + r].
fn ld_a(ctx: vec4<u32>, r: u32, k: u32, guard: bool) -> f32 {{
    if (r >= ctx.y || k >= params.k) {{ return 0.0; }}
    let route = sorted[params.off_sorted + ctx.x + r];
    var in_row = route;
    if (params.has_rows != 0u) {{ in_row = token_ids[2u * (params.off_tokens + route)]; }}
    return x[params.off_x + in_row * params.k + k];
}}

fn ld_b(ctx: vec4<u32>, k: u32, c: u32, guard: bool) -> f32 {{
    if (c >= params.n || k >= params.k) {{ return 0.0; }}
{ld_b}
}}

fn store_out(base: u32, len: u32, row: u32, col: u32, acc: f32) {{
    if (row >= len || col >= params.n) {{ return; }}
    out[params.off_out + sorted[params.off_sorted + base + row] * params.n + col] = acc;
}}

// One workgroup per chunk wid.y of one expert's routes and {tile} output columns from wid.x.
@compute @workgroup_size({threads})
fn routed_mm_{name}(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>) {{
    let chunk = params.off_chunks + 3u * wid.y;
    if (lid == 0u) {{ chunk_expert = chunks[chunk]; }}
    let expert = workgroupUniformLoad(&chunk_expert);
    if (expert == 0xFFFFFFFFu) {{ return; }}
    let base = chunks[chunk + 1u];
    let len = chunks[chunk + 2u];
    let actx = vec4<u32>(base, len, 0u, 0u);
    let bctx = vec4<u32>(expert, 0u, 0u, 0u);
    let row0 = 0u;
    let col0 = wid.x * {tile}u;
    let k_len = params.k;
    let interior = false;
{core}
}}
"#,
        tiles = tiled_gemm_tiles("f32", SORT_CHUNK),
        ld_b = mm_ld_b(format),
        tile = MATMUL_TILE,
        threads = TILED_THREADS,
        name = format.name(),
        core = tiled_gemm_core("f32", "true", "false", "base, len, ", "", SORT_CHUNK),
    )
}

/// Where a projection's weights sit in their buffer: code (or plain weight) word offset, scale
/// word offset, scale run and blocks per row.
fn weight_layout(
    format: RoutedFormat,
    weights: &DeviceTensor,
) -> TractResult<(u32, u32, u32, u32)> {
    let (n, k) = (weights.shape()[1], weights.shape()[2]);
    let base = match format {
        RoutedFormat::F16 => {
            let elems = element_offset(weights, 0);
            ensure!(elems.is_multiple_of(2), "f16 weights start on a word");
            u32_of(elems / 2)?
        }
        _ => word_offset(weights)?,
    };
    Ok(match bq_fact(weights) {
        Some(b) => {
            let group = b.group.max(1) as usize;
            let blocks = weights.shape()[0] * n * (k / 32);
            (base + u32_of(blocks.div_ceil(group))?, base, u32_of(group)?, u32_of(k / 32)?)
        }
        None => (base, 0, 1, 0),
    })
}

fn weights_format(weights: &DeviceTensor) -> TractResult<RoutedFormat> {
    let k = weights.shape()[2];
    match bq_fact(weights) {
        Some(b) => format_of(weights.datum_type(), Some(&b.inner), k),
        None => format_of(weights.datum_type(), None, k),
    }
    .context("no routed kernel for these weights")
}

/// The routes sorted by expert, then the tiled matmul over every chunk of [`SORT_CHUNK`].
fn project_sorted(
    format: RoutedFormat,
    input: &DeviceTensor,
    weights: &DeviceTensor,
    rows: Option<&DeviceTensor>,
    expert_ids: &DeviceTensor,
    out: &DeviceTensor,
) -> TractResult<()> {
    let (routes, experts) = (expert_ids.shape()[0], weights.shape()[0]);
    let (n, k) = (weights.shape()[1], weights.shape()[2]);
    let max_chunks = routes.div_ceil(SORT_CHUNK) + experts;
    ensure!(max_chunks <= GRID_LIMIT as usize, "too many routes for the sorted routed matmul");
    let sorted = DeviceTensor::uninitialized_dt(u32::datum_type(), &[routes])?;
    let chunks = DeviceTensor::uninitialized_dt(u32::datum_type(), &[3 * max_chunks])?;
    let (code_off, scale_off, group, blocks_per_row) = weight_layout(format, weights)?;
    with_wgpu_queue(|q| {
        let token_ids = rows.unwrap_or(expert_ids);
        for t in [input, weights, token_ids, expert_ids, &sorted, &chunks, out] {
            q.retain_tensor(t);
        }
        let layout = LayoutKind::TailWrites { n: 3, writes: 2 };
        let pipeline = q.context().chain_pipeline(
            "route_plan",
            layout,
            EntryPoint::plain("route_plan"),
            || ROUTE_PLAN.to_string(),
        )?;
        let bg = q.context().bind_group(
            layout,
            &[get_wgpu_buffer(expert_ids), get_wgpu_buffer(&sorted), get_wgpu_buffer(&chunks)],
            q.uniform(),
        )?;
        let params = pack_u32s(&[
            elem_offset(expert_ids)?,
            elem_offset(&sorted)?,
            elem_offset(&chunks)?,
            u32_of(routes)?,
            u32_of(experts)?,
            u32_of(max_chunks)?,
            0,
            0,
        ]);
        let dyn_off = q.alloc_uniform(&params)?;
        q.dispatch_grid("route_plan", &pipeline, &bg, dyn_off, [1, 1, 1])?;

        let layout = LayoutKind::TailWrites { n: 6, writes: 1 };
        let pipeline = q.context().chain_pipeline(
            ("routed_mm", format),
            layout,
            EntryPoint::plain(match format {
                RoutedFormat::Q40 => "routed_mm_q4_0",
                RoutedFormat::Q20T => "routed_mm_q2_0_t",
                RoutedFormat::F16 => "routed_mm_f16",
                RoutedFormat::F32 => "routed_mm_f32",
            }),
            || mm_module(format),
        )?;
        let bg = q.context().bind_group(
            layout,
            &[
                get_wgpu_buffer(weights),
                get_wgpu_buffer(input),
                get_wgpu_buffer(token_ids),
                get_wgpu_buffer(&sorted),
                get_wgpu_buffer(&chunks),
                get_wgpu_buffer(out),
            ],
            q.uniform(),
        )?;
        let params = pack_u32s(&[
            code_off,
            scale_off,
            group,
            blocks_per_row,
            u32_of(k)?,
            u32_of(n)?,
            elem_offset(input)?,
            elem_offset(token_ids)?,
            elem_offset(&sorted)?,
            elem_offset(&chunks)?,
            elem_offset(out)?,
            rows.is_some() as u32,
        ]);
        let dyn_off = q.alloc_uniform(&params)?;
        let grid = [u32_of(n.div_ceil(MATMUL_TILE))?, u32_of(max_chunks)?, 1];
        q.dispatch_grid("routed_mm", &pipeline, &bg, dyn_off, grid)
    })
}

fn project(
    input: &DeviceTensor,
    weights: &DeviceTensor,
    rows: Option<&DeviceTensor>,
    expert_ids: &DeviceTensor,
    out: &DeviceTensor,
) -> TractResult<()> {
    let (routes, experts) = (expert_ids.shape()[0], weights.shape()[0]);
    let (n, k) = (weights.shape()[1], weights.shape()[2]);
    ensure!(input.shape()[1] == k && out.shape() == [routes, n]);
    let format = weights_format(weights)?;
    if WGPU_ROUTED_TUNING.strategy(routes, experts, k, format) != RoutedStrategy::PerRoute {
        return project_sorted(format, input, weights, rows, expert_ids, out);
    }
    ensure!(routes <= GRID_LIMIT as usize, "Wgpu RoutedLinear takes up to {GRID_LIMIT} routes");
    let (code_off, scale_off, group, blocks_per_row) = weight_layout(format, weights)?;
    with_wgpu_queue(|q| {
        let token_ids = rows.unwrap_or(expert_ids);
        for t in [input, weights, token_ids, expert_ids, out] {
            q.retain_tensor(t);
        }
        let layout = LayoutKind::TailWrites { n: 5, writes: 1 };
        let pipeline = q.context().chain_pipeline(
            ("routed_mv", format),
            layout,
            EntryPoint::plain(match format {
                RoutedFormat::Q40 => "routed_mv_q4_0",
                RoutedFormat::Q20T => "routed_mv_q2_0_t",
                RoutedFormat::F16 => "routed_mv_f16",
                RoutedFormat::F32 => "routed_mv_f32",
            }),
            || mv_module(format),
        )?;
        let bg = q.context().bind_group(
            layout,
            &[
                get_wgpu_buffer(weights),
                get_wgpu_buffer(input),
                get_wgpu_buffer(token_ids),
                get_wgpu_buffer(expert_ids),
                get_wgpu_buffer(out),
            ],
            q.uniform(),
        )?;
        let params = pack_u32s(&[
            code_off,
            scale_off,
            group,
            blocks_per_row,
            u32_of(k)?,
            u32_of(n)?,
            u32_of(routes)?,
            u32_of(units(format, k))?,
            elem_offset(input)?,
            elem_offset(token_ids)?,
            elem_offset(expert_ids)?,
            elem_offset(out)?,
            rows.is_some() as u32,
            0,
            0,
            0,
        ]);
        let dyn_off = q.alloc_uniform(&params)?;
        let grid = [u32_of(n.div_ceil(MV_ROWS as usize))?, u32_of(routes)?, 1];
        q.dispatch_grid("routed_mv", &pipeline, &bg, dyn_off, grid)
    })
}

/// The activations: 0 none, 1 silu, 2 tanh gelu, 3 relu, and for a gate and up pair either
/// one of those of the gate times up, or clamped SwiGLU (4).
const ACTIVATIONS: &str = r#"
fn activate(v: f32, activation: u32) -> f32 {
    if (activation == 1u) { return v / (1.0 + exp(-v)); }
    // Tanh GELU as v * sigmoid(2c): tanh overflows to NaN on some backends.
    if (activation == 2u) { return v / (1.0 + exp(-1.5957691216 * (v + 0.044715 * v * v * v))); }
    if (activation == 3u) { return max(v, 0.0); }
    return v;
}

fn activate_pair(g: f32, u: f32, activation: u32, alpha: f32, limit: f32) -> f32 {
    if (activation == 4u) {
        let gc = min(g, limit);
        let uc = clamp(u, -limit, limit);
        return (uc + 1.0) * gc / (1.0 + exp(-alpha * gc));
    }
    return activate(g, activation) * u;
}
"#;

const POSTAMBLE: &str = r#"
struct Params {
    off_gate: u32,
    off_up: u32,
    off_bias: u32,
    off_bias_up: u32,
    off_experts: u32,
    off_out: u32,
    routes: u32,
    n: u32,
    activation: u32,
    flags: u32,
    alpha: f32,
    limit: f32,
}

@group(0) @binding(0) var<storage, read> gate: array<f32>;
@group(0) @binding(1) var<storage, read> up: array<f32>;
@group(0) @binding(2) var<storage, read> bias: array<f32>;
@group(0) @binding(3) var<storage, read> bias_up: array<f32>;
@group(0) @binding(4) var<storage, read> expert_ids: array<u32>;
@group(0) @binding(5) var<storage, read_write> out: array<f32>;
@group(0) @binding(6) var<uniform> params: Params;

// Bias at the route's expert, then the activation (0 none, 1 silu, 2 tanh gelu, 3 relu) times
// `up` for a pair, or clamped SwiGLU (4) of the pair. flags: 1 up, 2 bias, 4 bias_up.
@compute @workgroup_size(64)
fn routed_postamble(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = gid.y * 65535u * 64u + gid.x;
    if (i >= params.routes * params.n) { return; }
    let route = i / params.n;
    let col = i - route * params.n;
    let bias_ix = expert_ids[2u * (params.off_experts + route)] * params.n + col;
    var g = gate[params.off_gate + i];
    if ((params.flags & 2u) != 0u) { g = g + bias[params.off_bias + bias_ix]; }
    if ((params.flags & 1u) == 0u) {
        out[params.off_out + i] = activate(g, params.activation);
        return;
    }
    var u = up[params.off_up + i];
    if ((params.flags & 4u) != 0u) { u = u + bias_up[params.off_bias_up + bias_ix]; }
    out[params.off_out + i] = activate_pair(g, u, params.activation, params.alpha, params.limit);
}
"#;

#[allow(clippy::too_many_arguments)]
fn postamble(
    gate: &DeviceTensor,
    up: Option<&DeviceTensor>,
    bias: Option<&DeviceTensor>,
    bias_up: Option<&DeviceTensor>,
    expert_ids: &DeviceTensor,
    (activation, alpha, limit): (u32, f32, f32),
    out: &DeviceTensor,
) -> TractResult<()> {
    let (routes, n) = (gate.shape()[0], gate.shape()[1]);
    with_wgpu_queue(|q| {
        let inputs =
            [gate, up.unwrap_or(gate), bias.unwrap_or(gate), bias_up.unwrap_or(gate), expert_ids];
        for t in inputs.iter().chain([&out]) {
            q.retain_tensor(t);
        }
        let layout = LayoutKind::TailWrites { n: 6, writes: 1 };
        let pipeline = q.context().chain_pipeline(
            "routed_postamble",
            layout,
            EntryPoint::plain("routed_postamble"),
            || format!("{ACTIVATIONS}{POSTAMBLE}"),
        )?;
        let buffers: Vec<_> = inputs.iter().chain([&out]).map(|t| get_wgpu_buffer(t)).collect();
        let bg = q.context().bind_group(layout, &buffers, q.uniform())?;
        let flags =
            up.is_some() as u32 | (bias.is_some() as u32) << 1 | (bias_up.is_some() as u32) << 2;
        let mut params = pack_u32s(&[
            elem_offset(inputs[0])?,
            elem_offset(inputs[1])?,
            elem_offset(inputs[2])?,
            elem_offset(inputs[3])?,
            elem_offset(expert_ids)?,
            elem_offset(out)?,
            u32_of(routes)?,
            u32_of(n)?,
            activation,
            flags,
        ]);
        params.extend_from_slice(&alpha.to_le_bytes());
        params.extend_from_slice(&limit.to_le_bytes());
        let dyn_off = q.alloc_uniform(&params)?;
        q.dispatch("routed_postamble", &pipeline, &bg, dyn_off, (routes * n) as u64)
    })
}

/// The GEMV body for a pair: the unit's activations once, then the gate rows from `wbuf`
/// into `acc` and the up rows from `wup` into `acc_up`.
fn mv_pair_unit_body(format: RoutedFormat) -> String {
    let body = mv_unit_body(format);
    let split =
        body.find("        for (var r = 0u; r < 4u; r++) {").expect("one row loop per body");
    let (x_part, rows) = body.split_at(split);
    let up_rows = rows
        .replace("wbuf", "wup")
        .replace("acc[r]", "acc_up[r]")
        .replace("params.code_off", "params.code_off_up")
        .replace("params.scale_off", "params.scale_off_up")
        .replace("params.group", "params.group_up");
    format!("{x_part}{rows}\n{up_rows}")
}

fn mv_pair_module(format: RoutedFormat) -> String {
    format!(
        r#"
struct Params {{
    code_off: u32,
    scale_off: u32,
    group: u32,
    code_off_up: u32,
    scale_off_up: u32,
    group_up: u32,
    blocks_per_row: u32,
    k: u32,
    n: u32,
    units: u32,
    off_x: u32,
    off_tokens: u32,
    off_experts: u32,
    off_bias: u32,
    off_bias_up: u32,
    off_out: u32,
    has_rows: u32,
    has_bias: u32,
    activation: u32,
    _pad0: u32,
    alpha: f32,
    limit: f32,
    _pad1: u32,
    _pad2: u32,
}}

@group(0) @binding(0) var<storage, read> wbuf: array<u32>;
@group(0) @binding(1) var<storage, read> wup: array<u32>;
@group(0) @binding(2) var<storage, read> x: array<f32>;
@group(0) @binding(3) var<storage, read> token_ids: array<u32>;
@group(0) @binding(4) var<storage, read> expert_ids: array<u32>;
@group(0) @binding(5) var<storage, read> bias: array<f32>;
@group(0) @binding(6) var<storage, read> bias_up: array<f32>;
@group(0) @binding(7) var<storage, read_write> out: array<f32>;
@group(0) @binding(8) var<uniform> params: Params;
{activations}
var<workgroup> red: array<f32, 512>;

// routed_mv_{name} over both weights of a pair, then the bias at the route's expert and the
// pair activation.
@compute @workgroup_size(64)
fn routed_mv_pair_{name}(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_index) t: u32,
) {{
    let route = wid.y;
    let row0 = wid.x * 4u;
    let expert = expert_ids[2u * (params.off_experts + route)];
    var in_row = route;
    if (params.has_rows != 0u) {{ in_row = token_ids[2u * (params.off_tokens + route)]; }}
    let rows = min(4u, params.n - row0);
    var wr: array<u32, 4>;
    for (var r = 0u; r < 4u; r++) {{ wr[r] = expert * params.n + min(row0 + r, params.n - 1u); }}
    let x_base = params.off_x + in_row * params.k;
    var acc = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
    var acc_up = array<f32, 4>(0.0, 0.0, 0.0, 0.0);
    for (var unit = t; unit < params.units; unit += 64u) {{{body}
    }}
    for (var r = 0u; r < 4u; r++) {{
        red[r * 64u + t] = acc[r];
        red[256u + r * 64u + t] = acc_up[r];
    }}
    workgroupBarrier();
    for (var stride = 32u; stride > 0u; stride = stride >> 1u) {{
        if (t < stride) {{
            for (var r = 0u; r < 8u; r++) {{ red[r * 64u + t] = red[r * 64u + t] + red[r * 64u + t + stride]; }}
        }}
        workgroupBarrier();
    }}
    if (t < rows) {{
        var g = red[t * 64u];
        var u = red[256u + t * 64u];
        let col = row0 + t;
        if (params.has_bias != 0u) {{
            g = g + bias[params.off_bias + expert * params.n + col];
            u = u + bias_up[params.off_bias_up + expert * params.n + col];
        }}
        out[params.off_out + route * params.n + col] = activate_pair(g, u, params.activation, params.alpha, params.limit);
    }}
}}
"#,
        activations = ACTIVATIONS,
        name = format.name(),
        body = mv_pair_unit_body(format),
    )
}

/// Both projections of a pair and their postamble in one GEMV pass; false, having done nothing,
/// for weights of different formats or more routes than a grid takes.
#[allow(clippy::too_many_arguments)]
fn fused_pair(
    input: &DeviceTensor,
    weights: &DeviceTensor,
    weights_up: &DeviceTensor,
    biases: Option<(&DeviceTensor, &DeviceTensor)>,
    rows: Option<&DeviceTensor>,
    expert_ids: &DeviceTensor,
    (activation, alpha, limit): (u32, f32, f32),
    out: &DeviceTensor,
) -> TractResult<bool> {
    let (routes, n, k) = (expert_ids.shape()[0], weights.shape()[1], weights.shape()[2]);
    let format = weights_format(weights)?;
    if weights_up.shape() != weights.shape()
        || weights_format(weights_up)? != format
        || routes > GRID_LIMIT as usize
    {
        return Ok(false);
    }
    let (code_off, scale_off, group, blocks_per_row) = weight_layout(format, weights)?;
    let (code_off_up, scale_off_up, group_up, _) = weight_layout(format, weights_up)?;
    with_wgpu_queue(|q| {
        let token_ids = rows.unwrap_or(expert_ids);
        let (bias, bias_up) = biases.unwrap_or((input, input));
        for t in [input, weights, weights_up, token_ids, expert_ids, bias, bias_up, out] {
            q.retain_tensor(t);
        }
        let layout = LayoutKind::TailWrites { n: 8, writes: 1 };
        let pipeline = q.context().chain_pipeline(
            ("routed_mv_pair", format),
            layout,
            EntryPoint::plain(match format {
                RoutedFormat::Q40 => "routed_mv_pair_q4_0",
                RoutedFormat::Q20T => "routed_mv_pair_q2_0_t",
                RoutedFormat::F16 => "routed_mv_pair_f16",
                RoutedFormat::F32 => "routed_mv_pair_f32",
            }),
            || mv_pair_module(format),
        )?;
        let bg = q.context().bind_group(
            layout,
            &[
                get_wgpu_buffer(weights),
                get_wgpu_buffer(weights_up),
                get_wgpu_buffer(input),
                get_wgpu_buffer(token_ids),
                get_wgpu_buffer(expert_ids),
                get_wgpu_buffer(bias),
                get_wgpu_buffer(bias_up),
                get_wgpu_buffer(out),
            ],
            q.uniform(),
        )?;
        let mut params = pack_u32s(&[
            code_off,
            scale_off,
            group,
            code_off_up,
            scale_off_up,
            group_up,
            blocks_per_row,
            u32_of(k)?,
            u32_of(n)?,
            u32_of(units(format, k))?,
            elem_offset(input)?,
            elem_offset(token_ids)?,
            elem_offset(expert_ids)?,
            elem_offset(bias)?,
            elem_offset(bias_up)?,
            elem_offset(out)?,
            rows.is_some() as u32,
            biases.is_some() as u32,
            activation,
            0,
        ]);
        params.extend_from_slice(&alpha.to_le_bytes());
        params.extend_from_slice(&limit.to_le_bytes());
        params.extend_from_slice(&pack_u32s(&[0, 0]));
        let dyn_off = q.alloc_uniform(&params)?;
        let grid = [u32_of(n.div_ceil(MV_ROWS as usize))?, u32_of(routes)?, 1];
        q.dispatch_grid("routed_mv_pair", &pipeline, &bg, dyn_off, grid)
    })?;
    Ok(true)
}

const WGPU_ROUTED_LINEAR_KERNELS: RoutedLinearKernels = RoutedLinearKernels {
    supports,
    project,
    postamble,
    fused_pair: Some((FusePairPolicy::SingleRow, fused_pair)),
};

/// Translates a `RoutedLinear` [`GpuRoutedLinear`] takes, its i64 ids moved to the device with
/// the rest: the registry's dtype gate would refuse them.
pub fn translate_routed_linear(
    source: &TypedModel,
    node: &TypedNode,
    target: &mut TypedModel,
    mapping: &Mapping,
) -> TractResult<Option<TVec<OutletId>>> {
    rule_if_some!(op = node.op_as::<RoutedLinear>());
    rule_if_some!(
        op =
            GpuRoutedLinear::from_tract_core(source, node, op, "Wgpu", WGPU_ROUTED_LINEAR_KERNELS)?
    );
    let device_inputs = sync_inputs_if_required(target, node, mapping, DeviceSyncKind::ToDevice)?;
    target.wire_node(node.name.clone(), op, &device_inputs).map(Some)
}

/// Translates a `RoutedCombine` [`GpuRoutedCombine`] takes, its i64 token ids moved to the
/// device with the rest.
pub fn translate_routed_combine(
    source: &TypedModel,
    node: &TypedNode,
    target: &mut TypedModel,
    mapping: &Mapping,
) -> TractResult<Option<TVec<OutletId>>> {
    rule_if_some!(op = node.op_as::<RoutedCombine>());
    rule_if!(crate::utils::is_supported_fact(source.outlet_fact(node.inputs[0])?));
    rule_if_some!(
        op = GpuRoutedCombine::from_tract_core(source, node, op, "Wgpu", WGPU_ROUTING_KERNELS)?
    );
    let device_inputs = sync_inputs_if_required(target, node, mapping, DeviceSyncKind::ToDevice)?;
    target.wire_node(node.name.clone(), op, &device_inputs).map(Some)
}

/// [`tract_gpu::ops::routing::lower_moe_ffn`] for the weights the wgpu GEMV reads.
pub fn lower_moe_ffn(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    _name: &str,
    op: &MoeFfn,
) -> TractResult<Option<TypedModelPatch>> {
    tract_gpu::ops::routing::lower_moe_ffn(model, node, op, supports)
}
