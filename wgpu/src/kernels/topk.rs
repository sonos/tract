use tract_core::internal::*;
use tract_core::ops::array::Topk;
use tract_gpu::ops::topk::GpuTopK;
use tract_gpu::ops::topk::TOPK_CHUNK;
use tract_gpu::ops::topk::TopKKernels;
use tract_gpu::sync::DeviceSyncKind;
use tract_gpu::sync::sync_inputs_if_required;
use tract_gpu::tensor::DeviceTensor;

use crate::kernels::shaders::{
    EntryPoint, GRID_LIMIT, LayoutKind, ShaderDtype, WORKGROUP, pack_u32s, preamble,
};
use crate::transform::Mapping;
use crate::utils::{element_offset, get_wgpu_buffer};
use crate::with_wgpu_queue;

const THREADS: u32 = 256;

fn chunk_grid(rows: usize, len: usize) -> TractResult<(usize, [u32; 3])> {
    let chunks = len.div_ceil(TOPK_CHUNK);
    ensure!(rows * len <= u32::MAX as usize, "Wgpu TopK input too large for 32-bit indexing");
    ensure!(chunks <= GRID_LIMIT as usize && rows <= GRID_LIMIT as usize);
    Ok((chunks, [chunks as u32, rows as u32, 1]))
}

fn chunk_values(
    largest: bool,
    values: &DeviceTensor,
    rows: usize,
    n: usize,
    k: usize,
    keys: &DeviceTensor,
) -> TractResult<()> {
    let (chunks, grid) = chunk_grid(rows, n)?;
    with_wgpu_queue(|q| {
        q.retain_tensor(values);
        q.retain_tensor(keys);
        let dt = ShaderDtype::from_datum(values.datum_type())?;
        let layout = LayoutKind::Chain(2);
        let pipeline = q.context().chain_pipeline(
            ("topk_chunk_values", dt),
            layout,
            EntryPoint::typed("topk_chunk_values", dt),
            || chunk_module(Some(dt)),
        )?;
        let bg = q.context().bind_group(
            layout,
            &[get_wgpu_buffer(values), get_wgpu_buffer(keys)],
            q.uniform(),
        )?;
        let params = pack_u32s(&[
            element_offset(values, 0) as u32,
            element_offset(keys, 0) as u32,
            n as u32,
            k as u32,
            largest as u32,
            (chunks * k) as u32,
            0,
            0,
        ]);
        let dyn_off = q.alloc_uniform(&params)?;
        q.dispatch_grid("topk_chunk_values", &pipeline, &bg, dyn_off, grid)
    })
}

fn chunk_keys(
    keys: &DeviceTensor,
    rows: usize,
    len: usize,
    k: usize,
    out: &DeviceTensor,
) -> TractResult<()> {
    let (chunks, grid) = chunk_grid(rows, len)?;
    with_wgpu_queue(|q| {
        q.retain_tensor(keys);
        q.retain_tensor(out);
        let layout = LayoutKind::Chain(2);
        let pipeline = q.context().chain_pipeline(
            "topk_chunk_keys",
            layout,
            EntryPoint::plain("topk_chunk_keys"),
            || chunk_module(None),
        )?;
        let bg = q.context().bind_group(
            layout,
            &[get_wgpu_buffer(keys), get_wgpu_buffer(out)],
            q.uniform(),
        )?;
        let params = pack_u32s(&[
            element_offset(keys, 0) as u32,
            element_offset(out, 0) as u32,
            len as u32,
            k as u32,
            0,
            (chunks * k) as u32,
            0,
            0,
        ]);
        let dyn_off = q.alloc_uniform(&params)?;
        q.dispatch_grid("topk_chunk_keys", &pipeline, &bg, dyn_off, grid)
    })
}

fn finish(
    keys: &DeviceTensor,
    values: &DeviceTensor,
    rows: usize,
    n: usize,
    k: usize,
    out_values: &DeviceTensor,
    out_indices: &DeviceTensor,
) -> TractResult<()> {
    with_wgpu_queue(|q| {
        for t in [keys, values, out_values, out_indices] {
            q.retain_tensor(t);
        }
        let dt = ShaderDtype::from_datum(values.datum_type())?;
        let layout = LayoutKind::TailWrites { n: 4, writes: 2 };
        let pipeline = q.context().chain_pipeline(
            ("topk_finish", dt),
            layout,
            EntryPoint::typed("topk_finish", dt),
            || finish_module(dt),
        )?;
        let bg = q.context().bind_group(
            layout,
            &[
                get_wgpu_buffer(keys),
                get_wgpu_buffer(values),
                get_wgpu_buffer(out_values),
                get_wgpu_buffer(out_indices),
            ],
            q.uniform(),
        )?;
        let params = pack_u32s(&[
            element_offset(keys, 0) as u32,
            element_offset(values, 0) as u32,
            element_offset(out_values, 0) as u32,
            element_offset(out_indices, 0) as u32,
            n as u32,
            k as u32,
            (rows * k) as u32,
            0,
        ]);
        let dyn_off = q.alloc_uniform(&params)?;
        q.dispatch("topk_finish", &pipeline, &bg, dyn_off, (rows * k) as u64)
    })
}

pub const WGPU_TOPK_KERNELS: TopKKernels = TopKKernels { chunk_values, chunk_keys, finish };

/// Translates a `Topk` [`GpuTopK`] takes, its i64 `k` moved to the device with the values:
/// the registry's dtype gate would refuse it, and the op never reads it.
pub fn translate_topk(
    source: &TypedModel,
    node: &TypedNode,
    target: &mut TypedModel,
    mapping: &Mapping,
) -> TractResult<Option<TVec<OutletId>>> {
    rule_if_some!(op = node.op_as::<Topk>());
    rule_if!(crate::utils::is_supported_fact(source.outlet_fact(node.inputs[0])?));
    rule_if_some!(op = GpuTopK::from_tract_core(source, node, op, "Wgpu", WGPU_TOPK_KERNELS)?);
    let device_inputs = sync_inputs_if_required(target, node, mapping, DeviceSyncKind::ToDevice)?;
    target.wire_node(node.name.clone(), op, &device_inputs).map(Some)
}

const KEY_ORDER: &str = r#"
const CHUNK: u32 = 2048u;

struct Params {
    off_in: u32,
    off_out: u32,
    len: u32,
    k: u32,
    largest: u32,
    out_len: u32,
    _pad0: u32,
    _pad1: u32,
}

@group(0) @binding(2) var<uniform> params: Params;
var<workgroup> hi: array<u32, 2048>;
var<workgroup> lo: array<u32, 2048>;

fn order(v: f32, largest: bool) -> u32 {
    var u = bitcast<u32>(v);
    let magnitude = u & 0x7FFFFFFFu;
    if (magnitude > 0x7F800000u) { return 1u; }
    if (magnitude == 0u) {
        u = 0x80000000u;
    } else if ((u & 0x80000000u) != 0u) {
        u = ~u;
    } else {
        u = u | 0x80000000u;
    }
    if (largest) { return u; }
    return ~u;
}

fn sort_chunk(t: u32) {
    for (var size = 2u; size <= CHUNK; size = size << 1u) {
        for (var stride = size >> 1u; stride > 0u; stride = stride >> 1u) {
            workgroupBarrier();
            for (var p = t; p < CHUNK / 2u; p += THREADS) {
                let a = 2u * p - (p & (stride - 1u));
                let b = a + stride;
                let desc = (a & size) == 0u;
                let less = hi[a] < hi[b] || (hi[a] == hi[b] && lo[a] < lo[b]);
                if (less == desc) {
                    let h = hi[a];
                    let l = lo[a];
                    hi[a] = hi[b];
                    lo[a] = lo[b];
                    hi[b] = h;
                    lo[b] = l;
                }
            }
        }
    }
    workgroupBarrier();
}

fn store_chunk(t: u32, row: u32, chunk: u32) {
    for (var j = t; j < params.k; j += THREADS) {
        let o = params.off_out + row * params.out_len + chunk * params.k + j;
        dst[2u * o] = lo[j];
        dst[2u * o + 1u] = hi[j];
    }
}
"#;

/// Sorts one chunk of [`TOPK_CHUNK`] u64 keys per workgroup, held as (high, low) u32 words,
/// and writes its best `k` low word first. With a dtype it builds the keys from values:
/// the value's order-preserving bits high, `~index` low; without, it reads keys.
fn chunk_module(dt: Option<ShaderDtype>) -> String {
    let mut s = dt.map(preamble).unwrap_or_default();
    s.push_str(&format!("const THREADS: u32 = {THREADS}u;\n"));
    let src_t = dt.map(|dt| dt.wgsl()).unwrap_or("u32");
    s.push_str(&format!(
        "@group(0) @binding(0) var<storage, read> src: array<{src_t}>;\n\
         @group(0) @binding(1) var<storage, read_write> dst: array<u32>;\n"
    ));
    s.push_str(KEY_ORDER);
    let (name, load) = match dt {
        Some(dt) => (
            format!("topk_chunk_values_{}", dt.suffix()),
            "hi[i] = order(f32(src[params.off_in + row * params.len + ix]), params.largest != 0u);
            lo[i] = 0xFFFFFFFFu - ix;",
        ),
        None => (
            "topk_chunk_keys".to_string(),
            "let w = params.off_in + row * params.len + ix;
            lo[i] = src[2u * w];
            hi[i] = src[2u * w + 1u];",
        ),
    };
    s.push_str(&format!(
        r#"
@compute @workgroup_size({THREADS})
fn {name}(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    let chunk = wid.x;
    let row = wid.y;
    let t = lid.x;
    for (var i = t; i < CHUNK; i += THREADS) {{
        let ix = chunk * CHUNK + i;
        if (ix < params.len) {{
            {load}
        }} else {{
            hi[i] = 0u;
            lo[i] = 0u;
        }}
    }}
    sort_chunk(t);
    store_chunk(t, row, chunk);
}}
"#
    ));
    s
}

/// One thread per output element: the value and i64 index, as two u32 words, of the
/// entry each sorted key names.
fn finish_module(dt: ShaderDtype) -> String {
    let t = dt.wgsl();
    let suf = dt.suffix();
    let mut s = preamble(dt);
    s.push_str(&format!(
        r#"
struct Params {{
    off_keys: u32,
    off_values: u32,
    off_out_values: u32,
    off_out_indices: u32,
    n: u32,
    k: u32,
    len: u32,
    _pad: u32,
}}

@group(0) @binding(0) var<storage, read> keys: array<u32>;
@group(0) @binding(1) var<storage, read> values: array<{t}>;
@group(0) @binding(2) var<storage, read_write> out_values: array<{t}>;
@group(0) @binding(3) var<storage, read_write> out_indices: array<u32>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size({WORKGROUP})
fn topk_finish_{suf}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.y * {GRID_LIMIT}u * {WORKGROUP}u + gid.x;
    if (i >= params.len) {{ return; }}
    let row = i / params.k;
    let ix = 0xFFFFFFFFu - keys[2u * (params.off_keys + i)];
    out_values[params.off_out_values + i] = values[params.off_values + row * params.n + ix];
    out_indices[2u * (params.off_out_indices + i)] = ix;
    out_indices[2u * (params.off_out_indices + i) + 1u] = 0u;
}}
"#
    ));
    s
}
