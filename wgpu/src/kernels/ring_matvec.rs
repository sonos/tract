use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

use crate::kernels::shaders::{
    EntryPoint, GRID_LIMIT, LayoutKind, ShaderDtype, pack_u32s, preamble,
};
use crate::utils::{element_offset, get_wgpu_buffer};
use crate::with_wgpu_queue;

/// Rows of the activation the ring matvec kernel serves.
pub const RING_MATVEC_MAX_M: usize = 8;
/// Columns a workgroup covers.
const RING_MATVEC_COLS: usize = 32;
/// The most slices of `k` a workgroup splits its columns over.
const RING_MATVEC_SLICES: usize = 8;
/// Rows of the window a slice walks at the least, below which a workgroup takes
/// fewer slices.
const RING_MATVEC_ROWS_PER_SLICE: usize = 32;

/// The axis of a channel's window a ring turns on.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RingAxis {
    /// The contracted axis.
    K,
    /// The output columns.
    N,
}

/// `out[c, r, j] = sum_kk act[c, r, kk] * window(c, kk, j)` for `channels`
/// windows held in `ring`, element `(kk, j)` at `kk * k_stride + j * n_stride`
/// from the start `table` gives for channel `c`, along with how far its window
/// has turned along `axis` and where its `[m, k]` rows of `act` start. `out` is
/// `[channels, m, n]` and contiguous; `table` holds three `u32` per channel,
/// stored as the bits of an f32 tensor.
#[allow(clippy::too_many_arguments)]
pub fn wgpu_ring_matvec_dispatch(
    act: &DeviceTensor,
    ring: &DeviceTensor,
    table: &DeviceTensor,
    out: &DeviceTensor,
    (channels, m, k, n): (usize, usize, usize, usize),
    (k_stride, n_stride): (usize, usize),
    axis: RingAxis,
) -> TractResult<()> {
    let dt = ShaderDtype::from_datum(act.datum_type())?;
    ensure!(ring.datum_type() == act.datum_type() && out.datum_type() == act.datum_type());
    ensure!(
        m <= RING_MATVEC_MAX_M,
        "The ring matvec serves up to {RING_MATVEC_MAX_M} rows, not {m}"
    );
    ensure!(out.len() == channels * m * n);
    ensure!(
        act.strides() == natural_strides(act.shape()).as_slice(),
        "The ring matvec reads a contiguous activation"
    );
    ensure!(table.len() == 3 * channels && table.datum_type() == f32::datum_type());
    ensure!(channels <= GRID_LIMIT as usize, "{channels} channels exceed the grid");
    let slices = (k / RING_MATVEC_ROWS_PER_SLICE).clamp(1, RING_MATVEC_SLICES);
    with_wgpu_queue(|q| {
        for t in [act, ring, table, out] {
            q.retain_tensor(t);
        }
        let layout = LayoutKind::Chain(4);
        let pipeline = q.context().chain_pipeline(
            ("ring_matvec", dt, slices),
            layout,
            EntryPoint::typed("ring_matvec", dt),
            || ring_matvec_module(dt, slices),
        )?;
        let buffers = [
            get_wgpu_buffer(act),
            get_wgpu_buffer(ring),
            get_wgpu_buffer(table),
            get_wgpu_buffer(out),
        ];
        let bg = q.context().bind_group(layout, &buffers, q.uniform())?;
        let vals = [
            m as u32,
            k as u32,
            n as u32,
            (axis == RingAxis::K) as u32,
            k_stride as u32,
            n_stride as u32,
            element_offset(act, 0) as u32,
            element_offset(ring, 0) as u32,
            element_offset(table, 0) as u32,
            element_offset(out, 0) as u32,
            0,
            0,
        ];
        let dyn_off = q.alloc_uniform(&pack_u32s(&vals))?;
        let grid = [n.div_ceil(RING_MATVEC_COLS) as u32, channels as u32, 1];
        q.dispatch_grid("ring_matvec", &pipeline, &bg, dyn_off, grid)
    })
}

/// A workgroup covers 32 columns of one channel, its `slices` rows of lanes each
/// walking a slice of `k`, so a row of lanes reads a row of a `[k, n]` window at
/// consecutive addresses; the slices are summed in workgroup memory.
fn ring_matvec_module(dt: ShaderDtype, slices: usize) -> String {
    let t = dt.wgsl();
    let suf = dt.suffix();
    let cols = RING_MATVEC_COLS;
    let max_m = RING_MATVEC_MAX_M;
    let mut s = preamble(dt);
    s.push_str(&format!(
        r#"
struct Params {{
    m: u32,
    k: u32,
    n: u32,
    rotate_k: u32,
    k_stride: u32,
    n_stride: u32,
    off_act: u32,
    off_ring: u32,
    off_table: u32,
    off_out: u32,
    _p0: u32,
    _p1: u32,
}}

@group(0) @binding(0) var<storage, read> act: array<{t}>;
@group(0) @binding(1) var<storage, read> ring: array<{t}>;
@group(0) @binding(2) var<storage, read> table: array<u32>;
@group(0) @binding(3) var<storage, read_write> outp: array<{t}>;
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> partial: array<f32, {partial_len}>;

fn wrap(x: u32, extent: u32) -> u32 {{
    return select(x, x - extent, x >= extent);
}}

@compute @workgroup_size({cols}, {slices})
fn ring_matvec_{suf}(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{
    let c = wid.y;
    let lane = lid.x;
    let slice = lid.y;
    let j = wid.x * {cols}u + lane;
    let m = params.m;
    let k = params.k;
    let n = params.n;
    let turn = table[params.off_table + 3u * c];
    let window = params.off_ring + table[params.off_table + 3u * c + 1u];
    let a0 = params.off_act + table[params.off_table + 3u * c + 2u];
    var acc: array<f32, {max_m}>;
    for (var r = 0u; r < {max_m}u; r++) {{ acc[r] = 0.0; }}
    if (j < n) {{
        var col = j;
        if (params.rotate_k == 0u) {{
            col = wrap(col + turn, n);
        }}
        let column = window + col * params.n_stride;
        var row = wrap(slice + select(0u, turn, params.rotate_k != 0u), k);
        var kk = slice;
        loop {{
            if (kk + 3u * {slices}u >= k) {{ break; }}
            let r1 = wrap(row + {slices}u, k);
            let r2 = wrap(r1 + {slices}u, k);
            let r3 = wrap(r2 + {slices}u, k);
            let w0 = f32(ring[column + row * params.k_stride]);
            let w1 = f32(ring[column + r1 * params.k_stride]);
            let w2 = f32(ring[column + r2 * params.k_stride]);
            let w3 = f32(ring[column + r3 * params.k_stride]);
            for (var r = 0u; r < m; r++) {{
                let x = a0 + r * k + kk;
                acc[r] += f32(act[x]) * w0 + f32(act[x + {slices}u]) * w1
                    + f32(act[x + 2u * {slices}u]) * w2 + f32(act[x + 3u * {slices}u]) * w3;
            }}
            row = wrap(r3 + {slices}u, k);
            kk += 4u * {slices}u;
        }}
        loop {{
            if (kk >= k) {{ break; }}
            let wv = f32(ring[column + row * params.k_stride]);
            for (var r = 0u; r < m; r++) {{
                acc[r] += f32(act[a0 + r * k + kk]) * wv;
            }}
            row = wrap(row + {slices}u, k);
            kk += {slices}u;
        }}
    }}
    for (var r = 0u; r < m; r++) {{
        partial[(slice * {max_m}u + r) * {cols}u + lane] = acc[r];
    }}
    workgroupBarrier();
    if (slice == 0u && j < n) {{
        let o = params.off_out + c * m * n + j;
        for (var r = 0u; r < m; r++) {{
            var sum = 0.0;
            for (var sl = 0u; sl < {slices}u; sl++) {{
                sum += partial[(sl * {max_m}u + r) * {cols}u + lane];
            }}
            outp[o + r * n] = {t}(sum);
        }}
    }}
}}
"#,
        partial_len = slices * max_m * cols,
    ));
    s
}
