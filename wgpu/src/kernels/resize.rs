use anyhow::ensure;
use tract_core::internal::*;
use tract_core::ops::nn::resize::Resize as CoreResize;
use tract_gpu::ops::resize::GpuResize;
use tract_gpu::tensor::DeviceTensor;

use crate::kernels::shaders::{
    EntryPoint, GRID_LIMIT, LayoutKind, ModuleKey, ModuleKind, PipelineKey, ShaderDtype, WORKGROUP,
    keys_for, pack_u32s, preamble,
};
use crate::ops::resize2d::WgpuResize2d;
use crate::utils::{element_offset, get_wgpu_buffer};
use crate::with_wgpu_queue;

pub fn all_pipeline_keys(shader_f16: bool) -> Vec<PipelineKey> {
    keys_for(ModuleKind::Resize, &["resize_axis"], shader_f16)
}

/// Both trailing axes in one launch; `rows` and `cols` are (indices, weights,
/// window) plans as [`wgpu_resize_axis_dispatch`] takes them.
pub fn wgpu_resize_2d_dispatch(
    input: &DeviceTensor,
    rows: (&DeviceTensor, &DeviceTensor, usize),
    cols: (&DeviceTensor, &DeviceTensor, usize),
    output: &DeviceTensor,
) -> TractResult<()> {
    with_wgpu_queue(|q| {
        for t in [input, rows.0, rows.1, cols.0, cols.1, output] {
            q.retain_tensor(t);
        }
        let rank = input.rank();
        ensure!(rank >= 2 && output.rank() == rank);
        ensure!(
            rows.0.datum_type() == i32::datum_type() && cols.0.datum_type() == i32::datum_type()
        );
        ensure!(
            rows.1.datum_type() == f32::datum_type() && cols.1.datum_type() == f32::datum_type()
        );
        let dt = ShaderDtype::from_datum(input.datum_type())?;
        let layout = LayoutKind::Chain(6);
        let outer: usize = input.shape()[..rank - 2].iter().product::<usize>().max(1);
        let (h_in, w_in) = (input.shape()[rank - 2], input.shape()[rank - 1]);
        let (h_out, w_out) = (output.shape()[rank - 2], output.shape()[rank - 1]);
        let [wx, wy, wz] = RESIZE_2D_WG.map(|d| d as usize);
        let grid = [w_out.div_ceil(wx), h_out.div_ceil(wy), outer.div_ceil(wz)];
        let bilinear = rows.2 == 2 && cols.2 == 2 && grid.iter().all(|&g| g <= 65535);
        let pipeline = if bilinear {
            q.context().chain_pipeline(
                ("resize_2d_bilinear", dt),
                layout,
                EntryPoint::typed("resize_2d_bilinear", dt),
                || resize_2d_bilinear_module(dt),
            )?
        } else {
            q.context().chain_pipeline(
                ("resize_2d", dt),
                layout,
                EntryPoint::typed("resize_2d", dt),
                || resize_2d_module(dt),
            )?
        };
        let bg = q.context().bind_group(
            layout,
            &[
                get_wgpu_buffer(input),
                get_wgpu_buffer(rows.0),
                get_wgpu_buffer(rows.1),
                get_wgpu_buffer(cols.0),
                get_wgpu_buffer(cols.1),
                get_wgpu_buffer(output),
            ],
            q.uniform(),
        )?;
        let params = pack_u32s(&[
            element_offset(input, 0) as u32,
            element_offset(rows.0, 0) as u32,
            element_offset(rows.1, 0) as u32,
            element_offset(cols.0, 0) as u32,
            element_offset(cols.1, 0) as u32,
            element_offset(output, 0) as u32,
            outer as u32,
            h_in as u32,
            w_in as u32,
            h_out as u32,
            w_out as u32,
            rows.2 as u32,
            cols.2 as u32,
            0,
            0,
            0,
        ]);
        let dyn_off = q.alloc_uniform(&params)?;
        if bilinear {
            let grid = grid.map(|g| g as u32);
            return q.dispatch_grid("resize_2d_bilinear", &pipeline, &bg, dyn_off, grid);
        }
        q.dispatch("resize_2d", &pipeline, &bg, dyn_off, (outer * h_out * w_out) as u64)
    })
}

pub fn wgpu_resize_axis_dispatch(
    input: &DeviceTensor,
    axis: usize,
    indices: &DeviceTensor,
    weights: &DeviceTensor,
    window: usize,
    output: &DeviceTensor,
) -> TractResult<()> {
    with_wgpu_queue(|q| {
        q.retain_tensor(input);
        q.retain_tensor(indices);
        q.retain_tensor(weights);
        q.retain_tensor(output);
        ensure!(input.rank() > axis);
        ensure!(indices.datum_type() == i32::datum_type());
        ensure!(weights.datum_type() == f32::datum_type());
        let dt = ShaderDtype::from_datum(input.datum_type())?;
        let pipeline = q.context().pipeline(PipelineKey {
            module: ModuleKey { kind: ModuleKind::Resize, dtype: dt },
            entry: EntryPoint::typed("resize_axis", dt),
        })?;
        let len_in = input.shape()[axis];
        let len_out = output.shape()[axis];
        let inner: usize = input.shape()[axis + 1..].iter().product();
        let outer: usize = input.shape()[..axis].iter().product();
        let n = (outer.max(1) * len_out * inner.max(1)) as u64;
        let in_buf = get_wgpu_buffer(input);
        let idx_buf = get_wgpu_buffer(indices);
        let w_buf = get_wgpu_buffer(weights);
        let out_buf = get_wgpu_buffer(output);
        let bg = q.context().bind_group(
            LayoutKind::Resize,
            &[in_buf, idx_buf, w_buf, out_buf],
            q.uniform(),
        )?;
        let params = pack_u32s(&[
            element_offset(input, 0) as u32,
            element_offset(indices, 0) as u32,
            element_offset(weights, 0) as u32,
            element_offset(output, 0) as u32,
            outer.max(1) as u32,
            len_in as u32,
            len_out as u32,
            inner.max(1) as u32,
            window as u32,
            0,
            0,
            0,
        ]);
        let dyn_off = q.alloc_uniform(&params)?;
        q.dispatch("resize", &pipeline, &bg, dyn_off, n)
    })
}

/// The plans of a `GpuResize` over exactly the two trailing axes, which the
/// fused kernel serves.
pub struct Resize2dPlan {
    pub op: WgpuResize2d,
    /// row indices, row weights, column indices, column weights
    pub plans: [Arc<Tensor>; 4],
}

pub fn wgpu_resize_2d(source: &TypedModel, node: &TypedNode) -> TractResult<Option<Resize2dPlan>> {
    let Some(op) = node.op_as::<CoreResize>() else { return Ok(None) };
    let facts = source.node_input_facts(node.id)?;
    rule_if!(facts[0].is_plain());
    rule_if!(matches!(facts[0].datum_type, DatumType::F32 | DatumType::F16));
    let Some(baked) = GpuResize::from_core(source, node, op, "Wgpu", wgpu_resize_axis_dispatch)?
    else {
        return Ok(None);
    };
    let rank = facts[0].rank();
    rule_if!(rank >= 2 && baked.axes.as_slice() == [rank - 2, rank - 1]);
    let [(ih, wh), (iw, ww)] = [&baked.plans[0], &baked.plans[1]];
    Ok(Some(Resize2dPlan {
        op: WgpuResize2d {
            window_h: baked.windows[0],
            window_w: baked.windows[1],
            output_shape: baked.output_shape.clone(),
        },
        plans: [ih.clone(), wh.clone(), iw.clone(), ww.clone()],
    }))
}

pub fn wgpu_resize(source: &TypedModel, node: &TypedNode) -> TractResult<Option<Box<dyn TypedOp>>> {
    let Some(op) = node.op_as::<CoreResize>() else { return Ok(None) };
    let facts = source.node_input_facts(node.id)?;
    rule_if!(facts[0].is_plain());
    rule_if!(matches!(facts[0].datum_type, DatumType::F32 | DatumType::F16));
    Ok(GpuResize::from_core(source, node, op, "Wgpu", wgpu_resize_axis_dispatch)?
        .map(|op| Box::new(op) as Box<dyn TypedOp>))
}

pub(crate) fn resize_wgsl(dt: ShaderDtype) -> String {
    let t = dt.wgsl();
    let suf = dt.suffix();
    let mut s = preamble(dt);
    s.push_str(&format!(
        r#"
struct Params {{
    off_in: u32,
    off_idx: u32,
    off_w: u32,
    off_out: u32,
    outer: u32,
    len_in: u32,
    len_out: u32,
    inner: u32,
    window: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}}

@group(0) @binding(0) var<storage, read> inp: array<{t}>;
@group(0) @binding(1) var<storage, read> indices: array<i32>;
@group(0) @binding(2) var<storage, read> weights: array<f32>;
@group(0) @binding(3) var<storage, read_write> outp: array<{t}>;
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size({WORKGROUP})
fn resize_axis_{suf}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.y * {GRID_LIMIT}u * {WORKGROUP}u + gid.x;
    let n = params.outer * params.len_out * params.inner;
    if (i >= n) {{ return; }}
    let inner_i = i % params.inner;
    let t = i / params.inner;
    let xo = t % params.len_out;
    let outer_i = t / params.len_out;
    var acc = 0.0;
    for (var w = 0u; w < params.window; w++) {{
        let ix = indices[params.off_idx + xo * params.window + w];
        let wt = weights[params.off_w + xo * params.window + w];
        let in_i = params.off_in + (outer_i * params.len_in + u32(ix)) * params.inner + inner_i;
        acc += f32(inp[in_i]) * wt;
    }}
    outp[params.off_out + (outer_i * params.len_out) * params.inner + xo * params.inner + inner_i] = {t}(acc);
}}
"#
    ));
    s
}

/// Resize over the two trailing axes in one launch: each output sums its
/// `win_h x win_w` taps, weights being the product of the two axes' plans.
pub fn resize_2d_module(dt: ShaderDtype) -> String {
    let t = dt.wgsl();
    let suf = dt.suffix();
    let mut s = preamble(dt);
    s.push_str(&format!(
        r#"
struct Params {{
    off_in: u32,
    off_ih: u32,
    off_wh: u32,
    off_iw: u32,
    off_ww: u32,
    off_out: u32,
    outer: u32,
    h_in: u32,
    w_in: u32,
    h_out: u32,
    w_out: u32,
    win_h: u32,
    win_w: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}}

@group(0) @binding(0) var<storage, read> inp: array<{t}>;
@group(0) @binding(1) var<storage, read> idx_h: array<i32>;
@group(0) @binding(2) var<storage, read> wt_h: array<f32>;
@group(0) @binding(3) var<storage, read> idx_w: array<i32>;
@group(0) @binding(4) var<storage, read> wt_w: array<f32>;
@group(0) @binding(5) var<storage, read_write> outp: array<{t}>;
@group(0) @binding(6) var<uniform> params: Params;

@compute @workgroup_size({WORKGROUP})
fn resize_2d_{suf}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.y * {GRID_LIMIT}u * {WORKGROUP}u + gid.x;
    let n = params.outer * params.h_out * params.w_out;
    if (i >= n) {{ return; }}
    let xo = i % params.w_out;
    let t = i / params.w_out;
    let yo = t % params.h_out;
    let outer_i = t / params.h_out;
    let in_base = params.off_in + outer_i * params.h_in * params.w_in;
    var acc = 0.0;
    for (var a = 0u; a < params.win_h; a++) {{
        let iy = u32(idx_h[params.off_ih + yo * params.win_h + a]);
        let wy = wt_h[params.off_wh + yo * params.win_h + a];
        let row = in_base + iy * params.w_in;
        for (var b = 0u; b < params.win_w; b++) {{
            let ix = u32(idx_w[params.off_iw + xo * params.win_w + b]);
            let wx = wt_w[params.off_ww + xo * params.win_w + b];
            acc += f32(inp[row + ix]) * wy * wx;
        }}
    }}
    outp[params.off_out + i] = {t}(acc);
}}
"#
    ));
    s
}

/// Workgroup of the two-tap resize: 8 columns x 8 rows x 2 outer slices.
pub const RESIZE_2D_WG: [u32; 3] = [8, 8, 2];

/// [`resize_2d_module`] for two taps per axis — bilinear — with the taps
/// unrolled and a 3-D grid over (columns, rows, outer slices), so a workgroup
/// writes contiguous rows. Found by an OpenEvolve search scored in Chrome
/// (6.8x over the generic kernel there, 144 -> 21 us per frame of the
/// segmenter's three upsamples).
pub fn resize_2d_bilinear_module(dt: ShaderDtype) -> String {
    let t = dt.wgsl();
    let suf = dt.suffix();
    let [wx, wy, wz] = RESIZE_2D_WG;
    let mut s = preamble(dt);
    s.push_str(&format!(
        r#"
struct Params {{
    off_in: u32,
    off_ih: u32,
    off_wh: u32,
    off_iw: u32,
    off_ww: u32,
    off_out: u32,
    outer: u32,
    h_in: u32,
    w_in: u32,
    h_out: u32,
    w_out: u32,
    win_h: u32,
    win_w: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
}}

@group(0) @binding(0) var<storage, read> inp: array<{t}>;
@group(0) @binding(1) var<storage, read> idx_h: array<i32>;
@group(0) @binding(2) var<storage, read> wt_h: array<f32>;
@group(0) @binding(3) var<storage, read> idx_w: array<i32>;
@group(0) @binding(4) var<storage, read> wt_w: array<f32>;
@group(0) @binding(5) var<storage, read_write> outp: array<{t}>;
@group(0) @binding(6) var<uniform> params: Params;

@compute @workgroup_size({wx}, {wy}, {wz})
fn resize_2d_bilinear_{suf}(@builtin(workgroup_id) wgid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    let yo = wgid.y * {wy}u + lid.y;
    let xo = wgid.x * {wx}u + lid.x;
    if (yo >= params.h_out || xo >= params.w_out) {{ return; }}
    let ih0 = u32(idx_h[params.off_ih + yo * 2u]);
    let ih1 = u32(idx_h[params.off_ih + yo * 2u + 1u]);
    let wh0 = wt_h[params.off_wh + yo * 2u];
    let wh1 = wt_h[params.off_wh + yo * 2u + 1u];
    let iw0 = u32(idx_w[params.off_iw + xo * 2u]);
    let iw1 = u32(idx_w[params.off_iw + xo * 2u + 1u]);
    let ww0 = wt_w[params.off_ww + xo * 2u];
    let ww1 = wt_w[params.off_ww + xo * 2u + 1u];
    let row0 = ih0 * params.w_in;
    let row1 = ih1 * params.w_in;
    let out_pos = yo * params.w_out + xo;
    let c = wgid.z * {wz}u + lid.z;
    if (c < params.outer) {{
        let in_base = params.off_in + c * params.h_in * params.w_in;
        let out_base = params.off_out + c * params.h_out * params.w_out;
        let acc = f32(inp[in_base + row0 + iw0]) * wh0 * ww0
            + f32(inp[in_base + row0 + iw1]) * wh0 * ww1
            + f32(inp[in_base + row1 + iw0]) * wh1 * ww0
            + f32(inp[in_base + row1 + iw1]) * wh1 * ww1;
        outp[out_base + out_pos] = {t}(acc);
    }}
}}
"#
    ));
    s
}
