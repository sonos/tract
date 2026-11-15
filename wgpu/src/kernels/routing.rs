//! Routed MoE kernels: route selection, combine, the per-route GEMV over Q4_0, Q2_0_T, f16 and
//! f32 expert weights, and the postamble. i64 ids live as (low, high) u32 word pairs; the kernels
//! read and write the low word.

use tract_core::internal::*;
use tract_core::tract_linalg::block_quant::BlockQuantFact;
use tract_core::tract_linalg::block_quant::Q2_0_T;
use tract_core::tract_linalg::block_quant::Q4_0;
use tract_gpu::ops::routing::GpuRouteSelect;
use tract_gpu::ops::routing::GpuRoutedCombine;
use tract_gpu::ops::routing::GpuRoutedLinear;
use tract_gpu::ops::routing::RoutedLinearKernels;
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
use crate::register_wgpu_op;
use crate::transform::Mapping;
use crate::utils::element_offset;
use crate::utils::get_wgpu_buffer;
use crate::with_wgpu_queue;

/// Output rows of one routed GEMV workgroup.
const MV_ROWS: u32 = 4;

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

/// An expert weight format the routed GEMV reads.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
enum Format {
    Q40,
    Q20T,
    F16,
    F32,
}

impl Format {
    fn name(self) -> &'static str {
        match self {
            Format::Q40 => "q4_0",
            Format::Q20T => "q2_0_t",
            Format::F16 => "f16",
            Format::F32 => "f32",
        }
    }

    /// Units of work along one row of `k` weights: code words for the block formats, runs of
    /// four weights for the plain ones.
    fn units(self, k: usize) -> usize {
        match self {
            Format::Q40 => k / 32 * 4,
            Format::Q20T => k / 32 * 2,
            Format::F16 | Format::F32 => k / 4,
        }
    }

    /// The format of `[E, N, K]` expert weights of this datum type and block format, if the GEMV
    /// reads them.
    fn of(dt: DatumType, block: Option<&BlockQuantFact>, k: usize) -> Option<Format> {
        match block {
            Some(b) if b.format.dyn_eq(&Q4_0) && k.is_multiple_of(32) => Some(Format::Q40),
            Some(b) if b.format.dyn_eq(&Q2_0_T) && k.is_multiple_of(32) => Some(Format::Q20T),
            Some(_) => None,
            None if !k.is_multiple_of(4) => None,
            None if dt == f16::datum_type() => Some(Format::F16),
            None if dt == f32::datum_type() => Some(Format::F32),
            None => None,
        }
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
    (exotic.is_none() || block.is_some()) && Format::of(fact.datum_type, block, k).is_some()
}

/// The per-format body of the GEMV loop over `unit`: adds this unit's products to `acc[r]` for
/// the workgroup's rows. Block formats decode a code word and its block scale; plain ones take
/// four weights.
fn mv_unit_body(format: Format) -> &'static str {
    match format {
        Format::Q40 => {
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
        Format::Q20T => {
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
        Format::F16 => {
            r#"
        let xb = x_base + 4u * unit;
        let xv = vec4<f32>(x[xb], x[xb + 1u], x[xb + 2u], x[xb + 3u]);
        for (var r = 0u; r < 4u; r++) {
            let wo = params.code_off + (wr[r] * params.k + 4u * unit) / 2u;
            let wv = vec4<f32>(unpack2x16float(wbuf[wo]), unpack2x16float(wbuf[wo + 1u]));
            acc[r] = acc[r] + dot(wv, xv);
        }"#
        }
        Format::F32 => {
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

fn mv_module(format: Format) -> String {
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

fn project(
    input: &DeviceTensor,
    weights: &DeviceTensor,
    rows: Option<&DeviceTensor>,
    expert_ids: &DeviceTensor,
    out: &DeviceTensor,
) -> TractResult<()> {
    let (routes, n, k) = (expert_ids.shape()[0], weights.shape()[1], weights.shape()[2]);
    ensure!(input.shape()[1] == k && out.shape() == [routes, n]);
    ensure!(routes <= GRID_LIMIT as usize, "Wgpu RoutedLinear takes up to {GRID_LIMIT} routes");
    let block = bq_fact(weights);
    let format = match &block {
        Some(b) => Format::of(weights.datum_type(), Some(&b.inner), k),
        None => Format::of(weights.datum_type(), None, k),
    }
    .context("no routed GEMV for these weights")?;
    let base = match format {
        Format::F16 => {
            let elems = element_offset(weights, 0);
            ensure!(elems.is_multiple_of(2), "f16 weights start on a word");
            u32_of(elems / 2)?
        }
        _ => word_offset(weights)?,
    };
    let (scale_off, code_off, group, blocks_per_row) = match &block {
        Some(b) => {
            let group = b.group.max(1) as usize;
            let blocks = weights.shape()[0] * n * (k / 32);
            (base, base + u32_of(blocks.div_ceil(group))?, u32_of(group)?, u32_of(k / 32)?)
        }
        None => (0, base, 1, 0),
    };
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
                Format::Q40 => "routed_mv_q4_0",
                Format::Q20T => "routed_mv_q2_0_t",
                Format::F16 => "routed_mv_f16",
                Format::F32 => "routed_mv_f32",
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
            u32_of(format.units(k))?,
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

fn activate(v: f32, activation: u32) -> f32 {
    if (activation == 1u) { return v / (1.0 + exp(-v)); }
    // Tanh GELU as v * sigmoid(2c): tanh overflows to NaN on some backends.
    if (activation == 2u) { return v / (1.0 + exp(-1.5957691216 * (v + 0.044715 * v * v * v))); }
    if (activation == 3u) { return max(v, 0.0); }
    return v;
}

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
    if (params.activation == 4u) {
        g = min(g, params.limit);
        u = clamp(u, -params.limit, params.limit);
        out[params.off_out + i] = (u + 1.0) * g / (1.0 + exp(-params.alpha * g));
    } else {
        out[params.off_out + i] = activate(g, params.activation) * u;
    }
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
            || POSTAMBLE.to_string(),
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

const WGPU_ROUTED_LINEAR_KERNELS: RoutedLinearKernels =
    RoutedLinearKernels { supports, project, postamble, fused_pair: None };

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
