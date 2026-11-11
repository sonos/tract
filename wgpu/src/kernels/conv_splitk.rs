//! Split-K convolution: the implicit GEMM of [`conv_tiled_module`] with its
//! reduction cut into slices, one workgroup per (tile, slice), each writing its
//! partial sums to an f32 scratch; a second pass adds the slices up and applies
//! the epilogue.
//!
//! [`conv_tiled_module`]: crate::kernels::conv::conv_tiled_module

use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

use crate::WgpuQueue;
use crate::kernels::shaders::{
    ChainStep, EntryPoint, GRID_LIMIT, LayoutKind, ShaderDtype, WORKGROUP, binary_ops_wgsl,
    pack_u32s, preamble, unary_ops_wgsl,
};
use crate::kernels::tiled_gemm::{
    MATMUL_TILE, MATMUL_TILE_K, TILED_THREADS, tiled_gemm_core, tiled_gemm_tiles,
};
use crate::utils::get_wgpu_buffer;

/// Workgroups the tiles and slices should add up to.
const TARGET_WORKGROUPS: usize = 64;
/// Most slices a reduction is cut into.
const MAX_SLICES: usize = 16;
/// Fewest reduction values in one slice.
const MIN_SLICE: usize = 64;

/// How a tiled convolution's reduction is cut: `count` slices of `len`
/// values each (a multiple of [`MATMUL_TILE_K`]), the last one possibly shorter.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct Slices {
    pub count: usize,
    pub len: usize,
}

/// The slicing for an f32 or f16 convolution the tiled kernel serves with `tile_m`
/// rows per tile, when its tiles alone are too few workgroups to keep the
/// device busy and its reduction of `k` values is long enough to be cut into
/// at least two slices of [`MIN_SLICE`]; `None` when the tiled kernel should
/// run as is.
pub(crate) fn slices(
    dt: ShaderDtype,
    groups: usize,
    co_pg: usize,
    pixels: usize,
    k: usize,
    tile_m: usize,
) -> Option<Slices> {
    if !matches!(dt, ShaderDtype::F32 | ShaderDtype::F16) {
        return None;
    }
    let tiles = pixels.div_ceil(MATMUL_TILE) * co_pg.div_ceil(tile_m) * groups;
    let wanted = (TARGET_WORKGROUPS / tiles).min(MAX_SLICES).min(k / MIN_SLICE);
    if wanted < 2 {
        return None;
    }
    let len = k.div_ceil(wanted).next_multiple_of(MATMUL_TILE_K);
    let count = k.div_ceil(len);
    (count >= 2 && groups * count <= GRID_LIMIT as usize).then_some(Slices { count, len })
}

/// Runs the convolution `params` describes (the packed `Params` of
/// [`conv_tiled_module`]) in two passes over a scratch of `slices.count` f32
/// copies of the output. `grid` is the tiled kernel's own.
///
/// [`conv_tiled_module`]: crate::kernels::conv::conv_tiled_module
#[allow(clippy::too_many_arguments)]
pub(crate) fn dispatch(
    q: &WgpuQueue,
    dt: ShaderDtype,
    tile_m: usize,
    slices: Slices,
    epilogue: &[ChainStep],
    input: &DeviceTensor,
    weights: &DeviceTensor,
    extras: &[&DeviceTensor],
    output: &DeviceTensor,
    mut params: Vec<u8>,
    grid: [u32; 3],
) -> TractResult<()> {
    let scratch = DeviceTensor::uninitialized_dt(DatumType::F32, &[slices.count, output.len()])?;
    q.retain_tensor(&scratch);
    params.extend_from_slice(&pack_u32s(&[slices.count as u32, slices.len as u32, 0, 0]));
    let dyn_off = q.alloc_uniform(&params)?;
    let context = q.context();

    let layout = LayoutKind::Chain(3);
    let pipeline = context.chain_pipeline(
        ("conv_splitk_partial", tile_m, dt),
        layout,
        EntryPoint::typed("conv2d_splitk_partial", dt),
        || conv_splitk_partial_module(dt, tile_m),
    )?;
    let buffers = [get_wgpu_buffer(input), get_wgpu_buffer(weights), get_wgpu_buffer(&scratch)];
    let bg = context.bind_group(layout, &buffers, q.uniform())?;
    let grid = [grid[0], grid[1], grid[2] * slices.count as u32];
    q.dispatch_grid("conv_splitk_partial", &pipeline, &bg, dyn_off, grid)?;

    let layout = LayoutKind::Chain(2 + extras.len() as u8);
    let pipeline = context.chain_pipeline(
        ("conv_splitk_finish", dt, epilogue, extras.len()),
        layout,
        EntryPoint::typed("conv2d_splitk_finish", dt),
        || conv_splitk_finish_module(dt, epilogue, extras.len()),
    )?;
    let mut buffers = vec![get_wgpu_buffer(&scratch)];
    buffers.extend(extras.iter().map(|t| get_wgpu_buffer(t)));
    buffers.push(get_wgpu_buffer(output));
    let bg = context.bind_group(layout, &buffers, q.uniform())?;
    q.dispatch("conv_splitk_finish", &pipeline, &bg, dyn_off, output.len() as u64)
}

fn params_struct() -> &'static str {
    r#"
struct Params {
    off_in: u32,
    off_w: u32,
    off_out: u32,
    n: u32,
    ci: u32,
    co: u32,
    ih: u32,
    iw: u32,
    oh: u32,
    ow: u32,
    kh: u32,
    kw: u32,
    groups: u32,
    ci_pg: u32,
    co_pg: u32,
    channels_last: u32,
    pad_h: i32,
    pad_w: i32,
    stride_h: i32,
    stride_w: i32,
    dil_h: i32,
    dil_w: i32,
    in_sn: i32,
    in_sc: i32,
    in_sh: i32,
    in_sw: i32,
    w_so: i32,
    w_si: i32,
    w_sh: i32,
    w_sw: i32,
    out_sn: i32,
    out_sc: i32,
    out_sh: i32,
    out_sw: i32,
    _pad: vec2<u32>,
    off_extra: vec4<u32>,
    mode_extra: vec4<u32>,
    slices: u32,
    slice_len: u32,
    _slice_pad: vec2<u32>,
}
"#
}

/// The first pass: workgroup `z` is (group, slice), and each workgroup runs
/// [`tiled_gemm_core`] over its slice of the reduction alone, `k` counting from
/// the slice's start. The sums land unscaled in `scratch`, laid out as
/// `[slice, n, co, oh, ow]`, with no epilogue.
pub fn conv_splitk_partial_module(dt: ShaderDtype, tile_m: usize) -> String {
    let t = dt.wgsl();
    let suf = dt.suffix();
    let tile = MATMUL_TILE;
    let tk = MATMUL_TILE_K;
    let k_prologue = format!(
        "        if (lid < {tk}u) {{
            let k = k0 + lid;
            var info = vec4<i32>(0);
            if (k < k_len) {{
                let ci = (k_first + k) / khw;
                let tap = (k_first + k) % khw;
                let kh = tap / params.kw;
                let kw = tap % params.kw;
                info = vec4<i32>(
                    i32(ci) * params.w_si + i32(kh) * params.w_sh + i32(kw) * params.w_sw,
                    i32(g * params.ci_pg + ci) * params.in_sc,
                    i32(kh) * params.dil_h,
                    i32(kw) * params.dil_w,
                );
            }}
            k_info[lid] = info;
        }}
"
    );
    let mut s = preamble(dt);
    s.push_str(params_struct());
    s.push_str(&format!(
        r#"
@group(0) @binding(0) var<storage, read> inp: array<{t}>;
@group(0) @binding(1) var<storage, read> wgt: array<{t}>;
@group(0) @binding(2) var<storage, read_write> scratch: array<f32>;
@group(0) @binding(3) var<uniform> params: Params;

{tiles}
var<workgroup> k_info: array<vec4<i32>, {tk}>;
var<workgroup> pix_info: array<vec4<i32>, {tile}>;

fn ld_a(ctx: vec4<u32>, r: u32, k: u32, guard: bool) -> {t} {{
    if (guard && (r >= params.co_pg || k >= ctx.y)) {{ return {t}(0.0); }}
    let co = ctx.x * params.co_pg + r;
    return wgt[params.off_w + u32(i32(co) * params.w_so + k_info[k % {tk}u].x)];
}}

fn ld_b(ctx: vec4<u32>, k: u32, pix: u32, guard: bool) -> {t} {{
    if (guard && k >= ctx.y) {{ return {t}(0.0); }}
    let ki = k_info[k % {tk}u];
    let p = pix_info[pix % {tile}u];
    let ih = p.y + ki.z;
    let iw = p.z + ki.w;
    if (ih < 0 || ih >= i32(params.ih) || iw < 0 || iw >= i32(params.iw)) {{ return {t}(0.0); }}
    return inp[params.off_in + u32(p.x + ki.y + ih * params.in_sh + iw * params.in_sw)];
}}

fn store_out(g: u32, slice: u32, row: u32, pix: u32, acc: f32) {{
    let hw = params.oh * params.ow;
    if (row >= params.co_pg || pix >= params.n * hw) {{ return; }}
    let co = g * params.co_pg + row;
    let n = pix / hw;
    scratch[((slice * params.n + n) * params.co + co) * hw + pix % hw] = acc;
}}

@compute @workgroup_size({threads})
fn conv2d_splitk_partial_{suf}(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>) {{
    let g = wid.z / params.slices;
    let slice = wid.z % params.slices;
    let khw = params.kh * params.kw;
    let k_first = slice * params.slice_len;
    let k_len = min(params.slice_len, params.ci_pg * khw - k_first);
    let actx = vec4<u32>(g, k_len, 0u, 0u);
    let bctx = actx;
    let npix = params.n * params.oh * params.ow;
    let row0 = wid.y * {tile_m}u;
    let col0 = wid.x * {tile}u;
    let interior = row0 + {tile_m}u <= params.co_pg && col0 + {tile}u <= npix;
    if (lid < {tile}u) {{
        let pix = col0 + lid;
        let ow = pix % params.ow;
        let rest = pix / params.ow;
        let oh = rest % params.oh;
        let n = rest / params.oh;
        var info = vec4<i32>(0, -0x40000000, 0, 0);
        if (pix < npix) {{
            info = vec4<i32>(
                i32(n) * params.in_sn,
                i32(oh) * params.stride_h - params.pad_h,
                i32(ow) * params.stride_w - params.pad_w,
                0,
            );
        }}
        pix_info[lid] = info;
    }}
{core}}}
"#,
        tiles = tiled_gemm_tiles(t, tile_m),
        threads = TILED_THREADS,
        core = tiled_gemm_core(t, "true", "true", "g, slice, ", &k_prologue, tile_m),
    ));
    s
}

/// The second pass: one thread per output value adds its slices up in f32,
/// applies the epilogue in f32 and stores the result, rounded to the storage
/// type, through the output's strides. `extras` epilogue operands bind between
/// the scratch and the output.
pub fn conv_splitk_finish_module(dt: ShaderDtype, epilogue: &[ChainStep], extras: usize) -> String {
    let t = dt.wgsl();
    let a = "f32";
    let suf = dt.suffix();
    let out_binding = 1 + extras;
    let uniform_binding = 2 + extras;
    let extra_bindings = (0..extras)
        .map(|i| {
            format!("@group(0) @binding({}) var<storage, read> extra{i}: array<{t}>;\n", 1 + i)
        })
        .collect::<String>();
    let lib = if epilogue.is_empty() {
        String::new()
    } else {
        format!("{}{}", unary_ops_wgsl(a), binary_ops_wgsl(a))
    };
    let epilogue = super::conv::conv_direct_epilogue(epilogue, t != a);
    let value = if t == a { "v".to_owned() } else { format!("{t}(v)") };
    let mut s = preamble(dt);
    s.push_str(params_struct());
    s.push_str(&format!(
        r#"
@group(0) @binding(0) var<storage, read> scratch: array<f32>;
{extra_bindings}@group(0) @binding({out_binding}) var<storage, read_write> outp: array<{t}>;
@group(0) @binding({uniform_binding}) var<uniform> params: Params;
{lib}
@compute @workgroup_size({WORKGROUP})
fn conv2d_splitk_finish_{suf}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.y * {GRID_LIMIT}u * {WORKGROUP}u + gid.x;
    let nout = params.n * params.co * params.oh * params.ow;
    if (i >= nout) {{ return; }}
    var rest = i;
    let ow = rest % params.ow; rest = rest / params.ow;
    let oh = rest % params.oh; rest = rest / params.oh;
    let co = rest % params.co;
    let n = rest / params.co;
    var acc = 0.0;
    for (var slice = 0u; slice < params.slices; slice++) {{
        acc += scratch[slice * nout + i];
    }}
    let out_i = params.off_out + u32(
        i32(n) * params.out_sn
        + i32(co) * params.out_sc
        + i32(oh) * params.out_sh
        + i32(ow) * params.out_sw
    );
    var v = {a}(acc);
{epilogue}
    outp[out_i] = {value};
}}
"#
    ));
    s
}
