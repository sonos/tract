//! Prefix-broadcasting GEMM, `PrefixMatMul`'s kernels.
//!
//! The blocked and split-k kernels serve the channels-first f32 products they
//! were tuned on (every transposition set, a unit prefix, a weight `b`); the
//! tiled kernel takes any other product big enough to fill its tiles; an `m`
//! or an `n` of 1 is a GEMV; what is left runs one thread per output element.

use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

use crate::context::RepackKey;
use crate::kernels::shaders::{
    AT8, ChainStep, EntryPoint, GRID_LIMIT, LayoutKind, ModuleKey, ModuleKind, PipelineKey,
    ShaderDtype, WORKGROUP, binary_ops_wgsl, epilogue_body_at, keys_for, pack_u32s, preamble,
    rpad8_dims, rpad8_strides, unary_ops_wgsl,
};
use crate::kernels::tiled_gemm::{
    MATMUL_TILE, TILED_THREADS, tile_rows, tiled_gemm_core, tiled_gemm_tiles,
};
use crate::utils::{element_offset, get_wgpu_buffer};
use crate::with_wgpu_queue;

/// Rank the 8 stride slots can address, two of them being the matmul axes.
pub const MAX_RANK: usize = 8;

pub fn all_pipeline_keys(shader_f16: bool) -> Vec<PipelineKey> {
    keys_for(ModuleKind::MatMul, &["matmul"], shader_f16)
}

/// `m`, `k`, `n` read off the trailing two axes of `a` and `b`.
pub fn mkn(
    a_shape: &[usize],
    b_shape: &[usize],
    transpose_a: bool,
    transpose_b: bool,
) -> TractResult<(usize, usize, usize)> {
    let rank = a_shape.len();
    ensure!(rank >= 2 && rank == b_shape.len(), "matmul wants two tensors of equal rank >= 2");
    let (m, k) = if transpose_a {
        (a_shape[rank - 1], a_shape[rank - 2])
    } else {
        (a_shape[rank - 2], a_shape[rank - 1])
    };
    let (kb, n) = if transpose_b {
        (b_shape[rank - 1], b_shape[rank - 2])
    } else {
        (b_shape[rank - 2], b_shape[rank - 1])
    };
    ensure!(k == kb, "matmul k mismatch: a says {k}, b says {kb}");
    Ok((m, k, n))
}

/// Output shape of `PrefixMatMul`: broadcast prefix, then `m`, `n` (swapped
/// when `transpose_c`).
pub fn output_shape(
    a_shape: &[usize],
    b_shape: &[usize],
    transpose_a: bool,
    transpose_b: bool,
    transpose_c: bool,
) -> TractResult<TVec<usize>> {
    let rank = a_shape.len();
    let (m, _, n) = mkn(a_shape, b_shape, transpose_a, transpose_b)?;
    let mut shape: TVec<usize> =
        (0..rank - 2).map(|ix| if a_shape[ix] == 1 { b_shape[ix] } else { a_shape[ix] }).collect();
    if transpose_c {
        shape.push(n);
        shape.push(m);
    } else {
        shape.push(m);
        shape.push(n);
    }
    Ok(shape)
}

pub fn is_supported_dt(dt: DatumType) -> bool {
    matches!(dt, DatumType::F32 | DatumType::F16)
}

/// Epilogue operands are a scalar or one value per output column.
pub fn epilogue_mode(extra: &DeviceTensor, n: usize) -> TractResult<u32> {
    match extra.len() {
        1 => Ok(0),
        len if len == n => Ok(1),
        len => bail!("epilogue operand of {len} values fits neither a scalar nor {n} columns"),
    }
}

/// As [`epilogue_mode`], plus an operand laid out exactly like the output,
/// read element for element.
fn gemm_epilogue_mode(extra: &DeviceTensor, n: usize, output: &DeviceTensor) -> TractResult<u32> {
    if extra.len() == output.len() && extra.len() != 1 && extra.len() != n {
        ensure!(
            extra.shape() == output.shape() && extra.strides() == output.strides(),
            "an elementwise epilogue operand must share the output's shape and strides"
        );
        return Ok(2);
    }
    epilogue_mode(extra, n)
}

/// The transpose flags a `PrefixMatMul` carries, as the kernel reads them.
#[derive(Debug, Clone, Copy)]
pub struct Transposes {
    pub a: bool,
    pub b: bool,
    pub c: bool,
}

/// `b` transposed once, kept for the frames that read it again. Only reached
/// for an unpooled `b`, so the key cannot outlive the buffer it names.
fn repacked_b(
    q: &crate::context::WgpuQueue,
    b: &DeviceTensor,
    k: usize,
    n: usize,
) -> TractResult<DeviceTensor> {
    let key = RepackKey { buffer: get_wgpu_buffer(b).id, offset: element_offset(b, 0), k, n };
    if let Some(packed) = q.repacked(&key) {
        return Ok(packed);
    }
    let packed = DeviceTensor::uninitialized_dt(b.datum_type(), &[k, n])?;
    // b is (n, k) with k fastest; write it out with n fastest
    crate::kernels::copy::wgpu_copy_nd_dispatch(
        b,
        0,
        &[1, k as isize],
        &packed,
        0,
        &[k, n],
        &[n as isize, 1],
    )?;
    q.store_repacked(key, packed.clone());
    Ok(packed)
}

/// Whether the blocked kernel can serve this product: it reads both operands as
/// `vec4<f32>`, so the data must be f32 and the tile must divide the shape and the fastest axis of each
/// operand must be the one it widens. `b` is repacked once and read back by
/// buffer id, which only a buffer outside the frame pool keeps.
#[allow(clippy::too_many_arguments)]
fn blocked_applies(
    prefix: usize,
    m: usize,
    k: usize,
    n: usize,
    a: &DeviceTensor,
    b: &DeviceTensor,
    out: &DeviceTensor,
    t: Transposes,
) -> bool {
    let rank = a.rank();
    a.datum_type() == f32::datum_type()
        && !get_wgpu_buffer(b).pooled
        && prefix == 1
        && m.is_multiple_of(4)
        && n.is_multiple_of(4)
        && k.is_multiple_of(4)
        && ((m / 4) * (n / 4)).div_ceil(MATMUL_BLOCKED_WG as usize) <= 65535
        && t.a
        && t.b
        && t.c
        && a.strides()[rank - 1] == 1
        && a.strides()[rank - 2] == m as isize
        && out.strides()[out.rank() - 1] == 1
        && element_offset(a, 0).is_multiple_of(4)
        && element_offset(out, 0).is_multiple_of(4)
}

/// Whether the tiled kernel should serve this product: `m`, `n` and `k` are at
/// least 16, with every tile index inside one grid dimension. The number of
/// tiles does not matter: on a thin product with a long `k`, such as a late
/// pointwise layer of a CNN, a few tiles staging `k` through workgroup memory
/// still beat one thread per output reading it from global memory.
fn tiled_applies(prefix: usize, m: usize, k: usize, n: usize) -> bool {
    m >= 16
        && n >= 16
        && k >= 16
        && prefix <= 65535
        && m.div_ceil(MATMUL_TILE) <= 65535
        && n.div_ceil(MATMUL_TILE) <= 65535
}

pub fn wgpu_matmul_dispatch(
    t: Transposes,
    epilogue: &[ChainStep],
    a: &DeviceTensor,
    b: &DeviceTensor,
    extras: &[&DeviceTensor],
    output: &DeviceTensor,
) -> TractResult<()> {
    let (transpose_a, transpose_b, transpose_c) = (t.a, t.b, t.c);
    with_wgpu_queue(|q| {
        q.retain_tensor(a);
        q.retain_tensor(b);
        q.retain_tensor(output);
        ensure!(a.datum_type() == b.datum_type(), "matmul inputs must share a dtype");
        ensure!(a.rank() <= MAX_RANK, "tract-wgpu matmul is limited to rank {MAX_RANK}");
        for t in extras {
            q.retain_tensor(t);
        }
        let dt = ShaderDtype::from_datum(a.datum_type())?;
        let (m, k, n) = mkn(a.shape(), b.shape(), transpose_a, transpose_b)?;
        let layout = LayoutKind::Chain(3 + extras.len() as u8);
        let out_shape = output.shape();
        ensure!(
            out_shape
                == &*output_shape(a.shape(), b.shape(), transpose_a, transpose_b, transpose_c)?,
            "matmul output shape {out_shape:?} does not match its inputs"
        );
        let prefix: usize = out_shape[..out_shape.len() - 2].iter().product();
        // The other kernels tile `m`, so a single row never reaches them.
        let gemv = m == 1 && prefix > 0 && n > 0;
        // The other kernels tile `n`, so a single column never reaches them.
        let gemv_n = n == 1 && prefix > 0 && m > 0;

        if blocked_applies(prefix, m, k, n, a, b, output, t) {
            let packed = repacked_b(q, b, k, n)?;
            let split_k = m < MATMUL_SPLITK_MAX_M;
            let pipeline = if split_k {
                q.context().chain_pipeline(
                    ("matmul_splitk", dt, epilogue, extras.len()),
                    layout,
                    EntryPoint::typed("matmul_splitk", dt),
                    || matmul_splitk_module(dt, epilogue, extras.len()),
                )?
            } else {
                q.context().chain_pipeline(
                    ("matmul_blocked", dt, epilogue, extras.len()),
                    layout,
                    EntryPoint::typed("matmul_blocked", dt),
                    || matmul_blocked_module(dt, epilogue, extras.len()),
                )?
            };
            q.retain_tensor(&packed);
            let mut buffers: Vec<&crate::context::WgpuBuffer> =
                vec![get_wgpu_buffer(a), get_wgpu_buffer(&packed)];
            buffers.extend(extras.iter().map(|t| get_wgpu_buffer(t)));
            buffers.push(get_wgpu_buffer(output));
            let bg = q.context().bind_group(layout, &buffers, q.uniform())?;
            let mut vals = vec![
                m as u32,
                k as u32,
                n as u32,
                (element_offset(a, 0) / 4) as u32,
                (element_offset(&packed, 0) / 4) as u32,
                (element_offset(output, 0) / 4) as u32,
                0,
                0,
            ];
            let mut off_extra = [0u32; 4];
            let mut mode_extra = [0u32; 4];
            for (i, t) in extras.iter().enumerate() {
                off_extra[i] = element_offset(t, 0) as u32;
                mode_extra[i] = gemm_epilogue_mode(t, n, output)?;
            }
            vals.extend_from_slice(&off_extra);
            vals.extend_from_slice(&mode_extra);
            let dyn_off = q.alloc_uniform(&pack_u32s(&vals))?;
            let tiles = (m / 4) * (n / 4);
            let groups = if split_k {
                (tiles * 4).div_ceil(MATMUL_SPLITK_WG as usize)
            } else {
                tiles.div_ceil(MATMUL_BLOCKED_WG as usize)
            } as u32;
            return q.dispatch_grid("matmul_blocked", &pipeline, &bg, dyn_off, [groups, 1, 1]);
        }

        let tiled = tiled_applies(prefix, m, k, n);
        let tile_m = tile_rows(m);
        let pipeline = if gemv {
            q.context().chain_pipeline(
                ("matmul_gemv", dt, epilogue, extras.len()),
                layout,
                EntryPoint::typed("matmul_gemv", dt),
                || matmul_gemv_module(dt, epilogue, extras.len()),
            )?
        } else if gemv_n {
            q.context().chain_pipeline(
                ("matmul_gemv_n", dt, epilogue, extras.len()),
                layout,
                EntryPoint::typed("matmul_gemv_n", dt),
                || matmul_gemv_n_module(dt, epilogue, extras.len()),
            )?
        } else if tiled {
            q.context().chain_pipeline(
                ("matmul_tiled", tile_m, dt, epilogue, extras.len()),
                layout,
                EntryPoint::typed("matmul_tiled", dt),
                || matmul_tiled_module(dt, epilogue, extras.len(), tile_m),
            )?
        } else if epilogue.is_empty() {
            q.context().pipeline(PipelineKey {
                module: ModuleKey { kind: ModuleKind::MatMul, dtype: dt },
                entry: EntryPoint::typed("matmul", dt),
            })?
        } else {
            q.context().chain_pipeline(
                ("matmul", dt, epilogue, extras.len()),
                layout,
                EntryPoint::typed("matmul", dt),
                || matmul_module(dt, epilogue, extras.len()),
            )?
        };
        let mut buffers: Vec<&crate::context::WgpuBuffer> =
            vec![get_wgpu_buffer(a), get_wgpu_buffer(b)];
        buffers.extend(extras.iter().map(|t| get_wgpu_buffer(t)));
        buffers.push(get_wgpu_buffer(output));
        let bg = q.context().bind_group(layout, &buffers, q.uniform())?;

        let mut vals = vec![
            element_offset(a, 0) as u32,
            element_offset(b, 0) as u32,
            element_offset(output, 0) as u32,
            prefix as u32,
            m as u32,
            k as u32,
            n as u32,
            transpose_a as u32,
            transpose_b as u32,
            transpose_c as u32,
            0,
            0,
        ];
        vals.extend_from_slice(&rpad8_strides(a.shape(), a.strides(), out_shape));
        vals.extend_from_slice(&rpad8_strides(b.shape(), b.strides(), out_shape));
        vals.extend_from_slice(&rpad8_strides(out_shape, output.strides(), out_shape));
        vals.extend_from_slice(&rpad8_dims(out_shape));
        let mut off_extra = [0u32; 4];
        let mut mode_extra = [0u32; 4];
        for (i, t) in extras.iter().enumerate() {
            off_extra[i] = element_offset(t, 0) as u32;
            mode_extra[i] = gemm_epilogue_mode(t, n, output)?;
        }
        vals.extend_from_slice(&off_extra);
        vals.extend_from_slice(&mode_extra);
        let dyn_off = q.alloc_uniform(&pack_u32s(&vals))?;
        if gemv {
            let tiles = prefix * n.div_ceil(MATMUL_GEMV_WG as usize);
            let x = tiles.min(GRID_LIMIT as usize) as u32;
            let y = tiles.div_ceil(GRID_LIMIT as usize) as u32;
            return q.dispatch_grid("matmul_gemv", &pipeline, &bg, dyn_off, [x, y, 1]);
        }
        if gemv_n {
            let tiles = prefix * m.div_ceil(MATMUL_GEMV_N_WG as usize);
            let x = tiles.min(GRID_LIMIT as usize) as u32;
            let y = tiles.div_ceil(GRID_LIMIT as usize) as u32;
            return q.dispatch_grid("matmul_gemv_n", &pipeline, &bg, dyn_off, [x, y, 1]);
        }
        if tiled {
            let grid = [n.div_ceil(MATMUL_TILE) as u32, m.div_ceil(tile_m) as u32, prefix as u32];
            return q.dispatch_grid("matmul_tiled", &pipeline, &bg, dyn_off, grid);
        }
        q.dispatch("matmul", &pipeline, &bg, dyn_off, (prefix * m * n) as u64)
    })
}

/// Threads per workgroup of the blocked matmul.
pub const MATMUL_BLOCKED_WG: u32 = 192;

/// Columns one `m = 1` workgroup owns, and the `k` slice it stages for them.
pub const MATMUL_GEMV_WG: u32 = 64;

/// Rows one `n = 1` workgroup owns, and the `k` slice it stages for them.
pub const MATMUL_GEMV_N_WG: u32 = 64;

pub(crate) fn matmul_wgsl(dt: ShaderDtype) -> String {
    matmul_module(dt, &[], 0)
}

/// `extras` epilogue operands bind between B and the output.
pub fn matmul_module(dt: ShaderDtype, epilogue: &[ChainStep], extras: usize) -> String {
    let t = dt.wgsl();
    let suf = dt.suffix();
    let out_binding = 2 + extras;
    let uniform_binding = 3 + extras;
    let extra_bindings = (0..extras)
        .map(|i| {
            format!("@group(0) @binding({}) var<storage, read> extra{i}: array<{t}>;\n", 2 + i)
        })
        .collect::<String>();
    let lib = if epilogue.is_empty() {
        String::new()
    } else {
        format!("{}{}", unary_ops_wgsl(t), binary_ops_wgsl(t))
    };
    let epilogue = epilogue_body_at(epilogue, "col", "oe");
    let mut s = preamble(dt);
    s.push_str(&format!(
        r#"
struct Params {{
    off_a: u32,
    off_b: u32,
    off_out: u32,
    prefix: u32,
    m: u32,
    k: u32,
    n: u32,
    ta: u32,
    tb: u32,
    tc: u32,
    _p0: u32,
    _p1: u32,
    a_s0: vec4<u32>,
    a_s1: vec4<u32>,
    b_s0: vec4<u32>,
    b_s1: vec4<u32>,
    out_s0: vec4<u32>,
    out_s1: vec4<u32>,
    out_sh0: vec4<u32>,
    out_sh1: vec4<u32>,
    off_extra: vec4<u32>,
    mode_extra: vec4<u32>,
}}

@group(0) @binding(0) var<storage, read> a: array<{t}>;
@group(0) @binding(1) var<storage, read> b: array<{t}>;
{extra_bindings}@group(0) @binding({out_binding}) var<storage, read_write> outp: array<{t}>;
@group(0) @binding({uniform_binding}) var<uniform> params: Params;
{lib}
{AT8}

@compute @workgroup_size({WORKGROUP})
fn matmul_{suf}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.y * {GRID_LIMIT}u * {WORKGROUP}u + gid.x;
    let mn = params.m * params.n;
    let nout = params.prefix * mn;
    if (i >= nout) {{ return; }}
    let pref = i / mn;
    let rest = i % mn;
    var row: u32;
    var col: u32;
    if (params.tc != 0u) {{
        row = rest % params.m;
        col = rest / params.m;
    }} else {{
        col = rest % params.n;
        row = rest / params.n;
    }}
    // Shapes and strides are right-aligned in the 8 slots: the two matmul axes
    // land in slots 6 and 7, prefix axes in 0..5, padded with dim 1 stride 0.
    // `pref` is row-major over the prefix, so decode from the fastest axis up.
    var a_i = params.off_a;
    var b_i = params.off_b;
    var o_i = params.off_out;
    var restp = pref;
    for (var j = 0u; j < 6u; j++) {{
        let ax = 5u - j;
        let dim = max(at8(params.out_sh0, params.out_sh1, ax), 1u);
        let c = restp % dim;
        restp = restp / dim;
        a_i += c * at8(params.a_s0, params.a_s1, ax);
        b_i += c * at8(params.b_s0, params.b_s1, ax);
        o_i += c * at8(params.out_s0, params.out_s1, ax);
    }}
    let a_row_s = select(at8(params.a_s0, params.a_s1, 6u), at8(params.a_s0, params.a_s1, 7u), params.ta != 0u);
    let a_col_s = select(at8(params.a_s0, params.a_s1, 7u), at8(params.a_s0, params.a_s1, 6u), params.ta != 0u);
    let b_row_s = select(at8(params.b_s0, params.b_s1, 6u), at8(params.b_s0, params.b_s1, 7u), params.tb != 0u);
    let b_col_s = select(at8(params.b_s0, params.b_s1, 7u), at8(params.b_s0, params.b_s1, 6u), params.tb != 0u);
    var acc = 0.0;
    for (var kk = 0u; kk < params.k; kk++) {{
        let av = f32(a[a_i + row * a_row_s + kk * a_col_s]);
        let bv = f32(b[b_i + kk * b_row_s + col * b_col_s]);
        acc += av * bv;
    }}
    let o_row_s = select(at8(params.out_s0, params.out_s1, 6u), at8(params.out_s0, params.out_s1, 7u), params.tc != 0u);
    let o_col_s = select(at8(params.out_s0, params.out_s1, 7u), at8(params.out_s0, params.out_s1, 6u), params.tc != 0u);
    let oe = o_i - params.off_out + row * o_row_s + col * o_col_s;
    var v = {t}(acc);
{epilogue}
    outp[o_i + row * o_row_s + col * o_col_s] = v;
}}
"#
    ));
    s
}

/// `m = 1` product. One workgroup owns [`MATMUL_GEMV_WG`] columns of one prefix
/// slice: the threads stage that slice of the single row into workgroup memory
/// and each then walks its own column. Same `Params`, strides and epilogue as
/// [`matmul_module`]. `k` past the last partial slice is padded with zeros.
pub fn matmul_gemv_module(dt: ShaderDtype, epilogue: &[ChainStep], extras: usize) -> String {
    let t = dt.wgsl();
    let suf = dt.suffix();
    let wg = MATMUL_GEMV_WG;
    let out_binding = 2 + extras;
    let uniform_binding = 3 + extras;
    let extra_bindings = (0..extras)
        .map(|i| {
            format!("@group(0) @binding({}) var<storage, read> extra{i}: array<{t}>;\n", 2 + i)
        })
        .collect::<String>();
    let lib = if epilogue.is_empty() {
        String::new()
    } else {
        format!("{}{}", unary_ops_wgsl(t), binary_ops_wgsl(t))
    };
    let epilogue = epilogue_body_at(epilogue, "col", "oe");
    let mut s = preamble(dt);
    s.push_str(&format!(
        r#"
struct Params {{
    off_a: u32,
    off_b: u32,
    off_out: u32,
    prefix: u32,
    m: u32,
    k: u32,
    n: u32,
    ta: u32,
    tb: u32,
    tc: u32,
    _p0: u32,
    _p1: u32,
    a_s0: vec4<u32>,
    a_s1: vec4<u32>,
    b_s0: vec4<u32>,
    b_s1: vec4<u32>,
    out_s0: vec4<u32>,
    out_s1: vec4<u32>,
    out_sh0: vec4<u32>,
    out_sh1: vec4<u32>,
    off_extra: vec4<u32>,
    mode_extra: vec4<u32>,
}}

@group(0) @binding(0) var<storage, read> a: array<{t}>;
@group(0) @binding(1) var<storage, read> b: array<{t}>;
{extra_bindings}@group(0) @binding({out_binding}) var<storage, read_write> outp: array<{t}>;
@group(0) @binding({uniform_binding}) var<uniform> params: Params;
{lib}
{AT8}

var<workgroup> avec: array<f32, {wg}>;

@compute @workgroup_size({wg})
fn matmul_gemv_{suf}(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{
    let tiles_n = (params.n + {wg}u - 1u) / {wg}u;
    let tile = wid.y * {GRID_LIMIT}u + wid.x;
    if (tile >= params.prefix * tiles_n) {{ return; }}
    let pref = tile / tiles_n;
    let col = (tile % tiles_n) * {wg}u + lid.x;

    var a_i = params.off_a;
    var b_i = params.off_b;
    var o_i = params.off_out;
    var restp = pref;
    for (var j = 0u; j < 6u; j++) {{
        let ax = 5u - j;
        let dim = max(at8(params.out_sh0, params.out_sh1, ax), 1u);
        let c = restp % dim;
        restp = restp / dim;
        a_i += c * at8(params.a_s0, params.a_s1, ax);
        b_i += c * at8(params.b_s0, params.b_s1, ax);
        o_i += c * at8(params.out_s0, params.out_s1, ax);
    }}
    let a_k_s = select(at8(params.a_s0, params.a_s1, 7u), at8(params.a_s0, params.a_s1, 6u), params.ta != 0u);
    let b_k_s = select(at8(params.b_s0, params.b_s1, 6u), at8(params.b_s0, params.b_s1, 7u), params.tb != 0u);
    let b_n_s = select(at8(params.b_s0, params.b_s1, 7u), at8(params.b_s0, params.b_s1, 6u), params.tb != 0u);
    let o_n_s = select(at8(params.out_s0, params.out_s1, 7u), at8(params.out_s0, params.out_s1, 6u), params.tc != 0u);

    var acc = 0.0;
    for (var k0 = 0u; k0 < params.k; k0 += {wg}u) {{
        let kk = k0 + lid.x;
        if (kk < params.k) {{
            avec[lid.x] = f32(a[a_i + kk * a_k_s]);
        }} else {{
            avec[lid.x] = 0.0;
        }}
        workgroupBarrier();
        if (col < params.n) {{
            let b_base = b_i + col * b_n_s;
            for (var t = 0u; t < {wg}u; t++) {{
                let k = k0 + t;
                if (k < params.k) {{
                    acc += avec[t] * f32(b[b_base + k * b_k_s]);
                }}
            }}
        }}
        workgroupBarrier();
    }}
    if (col < params.n) {{
        let oe = o_i - params.off_out + col * o_n_s;
        var v = {t}(acc);
{epilogue}
        outp[o_i + col * o_n_s] = v;
    }}
}}
"#
    ));
    s
}

/// `n = 1` product. One workgroup owns [`MATMUL_GEMV_N_WG`] rows of one prefix
/// slice: the threads stage that slice of the single column into workgroup
/// memory and each then walks its own row. Same `Params`, strides and epilogue
/// as [`matmul_module`]. `k` past the last partial slice is padded with zeros.
pub fn matmul_gemv_n_module(dt: ShaderDtype, epilogue: &[ChainStep], extras: usize) -> String {
    let t = dt.wgsl();
    let suf = dt.suffix();
    let wg = MATMUL_GEMV_N_WG;
    let out_binding = 2 + extras;
    let uniform_binding = 3 + extras;
    let extra_bindings = (0..extras)
        .map(|i| {
            format!("@group(0) @binding({}) var<storage, read> extra{i}: array<{t}>;\n", 2 + i)
        })
        .collect::<String>();
    let lib = if epilogue.is_empty() {
        String::new()
    } else {
        format!("{}{}", unary_ops_wgsl(t), binary_ops_wgsl(t))
    };
    let epilogue = epilogue_body_at(epilogue, "col", "oe");
    let mut s = preamble(dt);
    s.push_str(&format!(
        r#"
struct Params {{
    off_a: u32,
    off_b: u32,
    off_out: u32,
    prefix: u32,
    m: u32,
    k: u32,
    n: u32,
    ta: u32,
    tb: u32,
    tc: u32,
    _p0: u32,
    _p1: u32,
    a_s0: vec4<u32>,
    a_s1: vec4<u32>,
    b_s0: vec4<u32>,
    b_s1: vec4<u32>,
    out_s0: vec4<u32>,
    out_s1: vec4<u32>,
    out_sh0: vec4<u32>,
    out_sh1: vec4<u32>,
    off_extra: vec4<u32>,
    mode_extra: vec4<u32>,
}}

@group(0) @binding(0) var<storage, read> a: array<{t}>;
@group(0) @binding(1) var<storage, read> b: array<{t}>;
{extra_bindings}@group(0) @binding({out_binding}) var<storage, read_write> outp: array<{t}>;
@group(0) @binding({uniform_binding}) var<uniform> params: Params;
{lib}
{AT8}

var<workgroup> bvec: array<f32, {wg}>;

@compute @workgroup_size({wg})
fn matmul_gemv_n_{suf}(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{
    let tiles_m = (params.m + {wg}u - 1u) / {wg}u;
    let tile = wid.y * {GRID_LIMIT}u + wid.x;
    if (tile >= params.prefix * tiles_m) {{ return; }}
    let pref = tile / tiles_m;
    let row = (tile % tiles_m) * {wg}u + lid.x;

    var a_i = params.off_a;
    var b_i = params.off_b;
    var o_i = params.off_out;
    var restp = pref;
    for (var j = 0u; j < 6u; j++) {{
        let ax = 5u - j;
        let dim = max(at8(params.out_sh0, params.out_sh1, ax), 1u);
        let c = restp % dim;
        restp = restp / dim;
        a_i += c * at8(params.a_s0, params.a_s1, ax);
        b_i += c * at8(params.b_s0, params.b_s1, ax);
        o_i += c * at8(params.out_s0, params.out_s1, ax);
    }}
    let a_m_s = select(at8(params.a_s0, params.a_s1, 6u), at8(params.a_s0, params.a_s1, 7u), params.ta != 0u);
    let a_k_s = select(at8(params.a_s0, params.a_s1, 7u), at8(params.a_s0, params.a_s1, 6u), params.ta != 0u);
    let b_k_s = select(at8(params.b_s0, params.b_s1, 6u), at8(params.b_s0, params.b_s1, 7u), params.tb != 0u);
    let o_m_s = select(at8(params.out_s0, params.out_s1, 6u), at8(params.out_s0, params.out_s1, 7u), params.tc != 0u);
    let o_n_s = select(at8(params.out_s0, params.out_s1, 7u), at8(params.out_s0, params.out_s1, 6u), params.tc != 0u);

    var acc = 0.0;
    for (var k0 = 0u; k0 < params.k; k0 += {wg}u) {{
        let kk = k0 + lid.x;
        if (kk < params.k) {{
            bvec[lid.x] = f32(b[b_i + kk * b_k_s]);
        }} else {{
            bvec[lid.x] = 0.0;
        }}
        workgroupBarrier();
        if (row < params.m) {{
            let a_base = a_i + row * a_m_s;
            for (var t = 0u; t < {wg}u; t++) {{
                let k = k0 + t;
                if (k < params.k) {{
                    acc += f32(a[a_base + k * a_k_s]) * bvec[t];
                }}
            }}
        }}
        workgroupBarrier();
    }}
    if (row < params.m) {{
        let col = 0u;
        let oe = o_i - params.off_out + row * o_m_s + col * o_n_s;
        var v = {t}(acc);
{epilogue}
        outp[o_i + row * o_m_s + col * o_n_s] = v;
    }}
}}
"#
    ));
    s
}

/// The general matmul, staged through workgroup memory: each workgroup owns a
/// `tile_m` by [`MATMUL_TILE`] block of one prefix slice of the output, `tile_m`
/// being [`tile_rows`]`(m)`, and walks `k`
/// in [`MATMUL_TILE_K`]-deep slices, with the loads, layout and accumulation of
/// [`tiled_gemm_core`]. Each operand is loaded along whichever of its two axes
/// is unit-strided, so loads coalesce for any transposition. Same `Params`,
/// strides and epilogue as [`matmul_module`], and no constraint on `m`, `k` or
/// `n`: partial tiles are guarded. Workgroup `x` tiles `n`, `y` tiles `m`, `z`
/// is the prefix index.
///
/// [`MATMUL_TILE_K`]: crate::kernels::tiled_gemm::MATMUL_TILE_K
pub fn matmul_tiled_module(
    dt: ShaderDtype,
    epilogue: &[ChainStep],
    extras: usize,
    tile_m: usize,
) -> String {
    let t = dt.wgsl();
    let suf = dt.suffix();
    let out_binding = 2 + extras;
    let uniform_binding = 3 + extras;
    let extra_bindings = (0..extras)
        .map(|i| {
            format!("@group(0) @binding({}) var<storage, read> extra{i}: array<{t}>;\n", 2 + i)
        })
        .collect::<String>();
    let lib = if epilogue.is_empty() {
        String::new()
    } else {
        format!("{}{}", unary_ops_wgsl(t), binary_ops_wgsl(t))
    };
    let epilogue = epilogue_body_at(epilogue, "col", "oe");
    let tile = MATMUL_TILE;
    let mut s = preamble(dt);
    s.push_str(&format!(
        r#"
struct Params {{
    off_a: u32,
    off_b: u32,
    off_out: u32,
    prefix: u32,
    m: u32,
    k: u32,
    n: u32,
    ta: u32,
    tb: u32,
    tc: u32,
    _p0: u32,
    _p1: u32,
    a_s0: vec4<u32>,
    a_s1: vec4<u32>,
    b_s0: vec4<u32>,
    b_s1: vec4<u32>,
    out_s0: vec4<u32>,
    out_s1: vec4<u32>,
    out_sh0: vec4<u32>,
    out_sh1: vec4<u32>,
    off_extra: vec4<u32>,
    mode_extra: vec4<u32>,
}}

@group(0) @binding(0) var<storage, read> a: array<{t}>;
@group(0) @binding(1) var<storage, read> b: array<{t}>;
{extra_bindings}@group(0) @binding({out_binding}) var<storage, read_write> outp: array<{t}>;
@group(0) @binding({uniform_binding}) var<uniform> params: Params;
{lib}
{AT8}

{tiles}
fn ld_a(ctx: vec4<u32>, r: u32, k: u32, guard: bool) -> {t} {{
    if (guard && (r >= params.m || k >= params.k)) {{ return {t}(0.0); }}
    return a[ctx.x + r * ctx.y + k * ctx.z];
}}

fn ld_b(ctx: vec4<u32>, k: u32, c: u32, guard: bool) -> {t} {{
    if (guard && (k >= params.k || c >= params.n)) {{ return {t}(0.0); }}
    return b[ctx.x + k * ctx.y + c * ctx.z];
}}

fn store_out(o_i: u32, o_row_s: u32, o_col_s: u32, row: u32, col: u32, acc: f32) {{
    if (row >= params.m || col >= params.n) {{ return; }}
    let oe = o_i - params.off_out + row * o_row_s + col * o_col_s;
    var v = {t}(acc);
{epilogue}
    outp[o_i + row * o_row_s + col * o_col_s] = v;
}}

@compute @workgroup_size({threads})
fn matmul_tiled_{suf}(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>) {{
    var a_i = params.off_a;
    var b_i = params.off_b;
    var o_i = params.off_out;
    var restp = wid.z;
    for (var j = 0u; j < 6u; j++) {{
        let ax = 5u - j;
        let dim = max(at8(params.out_sh0, params.out_sh1, ax), 1u);
        let c = restp % dim;
        restp = restp / dim;
        a_i += c * at8(params.a_s0, params.a_s1, ax);
        b_i += c * at8(params.b_s0, params.b_s1, ax);
        o_i += c * at8(params.out_s0, params.out_s1, ax);
    }}
    let a_row_s = select(at8(params.a_s0, params.a_s1, 6u), at8(params.a_s0, params.a_s1, 7u), params.ta != 0u);
    let a_col_s = select(at8(params.a_s0, params.a_s1, 7u), at8(params.a_s0, params.a_s1, 6u), params.ta != 0u);
    let b_row_s = select(at8(params.b_s0, params.b_s1, 6u), at8(params.b_s0, params.b_s1, 7u), params.tb != 0u);
    let b_col_s = select(at8(params.b_s0, params.b_s1, 7u), at8(params.b_s0, params.b_s1, 6u), params.tb != 0u);
    let actx = vec4<u32>(a_i, a_row_s, a_col_s, 0u);
    let bctx = vec4<u32>(b_i, b_row_s, b_col_s, 0u);
    let o_row_s = select(at8(params.out_s0, params.out_s1, 6u), at8(params.out_s0, params.out_s1, 7u), params.tc != 0u);
    let o_col_s = select(at8(params.out_s0, params.out_s1, 7u), at8(params.out_s0, params.out_s1, 6u), params.tc != 0u);

    let row0 = wid.y * {tile_m}u;
    let col0 = wid.x * {tile}u;
    let k_len = params.k;
    let interior = row0 + {tile_m}u <= params.m && col0 + {tile}u <= params.n;
{core}}}
"#,
        tiles = tiled_gemm_tiles(t, tile_m),
        threads = TILED_THREADS,
        core = tiled_gemm_core(
            t,
            "a_col_s == 1u",
            "b_col_s == 1u",
            "o_i, o_row_s, o_col_s, ",
            "",
            tile_m
        ),
    ));
    s
}

/// Threads per workgroup of the split-k matmul: 32 tiles, four lanes each.
pub const MATMUL_SPLITK_WG: u32 = 128;
/// Below this `m` the blocked matmul runs split over k.
pub const MATMUL_SPLITK_MAX_M: usize = 2304;

/// The blocked matmul for small `m`: four lanes share a four-by-four output
/// tile, each summing a quarter of `k`, and reduce through workgroup memory.
/// The squeeze-excitation projections (m = 144, k up to 128) put only a few
/// hundred threads on the device with the plain kernel, so its time is one
/// thread's k loop; four lanes cut that chain and came out of a search scored
/// in Chrome at 2-2.5x on those shapes (m = 144: 56 -> 22 us, m = 576 with
/// k >= 72: 47 -> 23 us). Same contract and epilogue as
/// [`matmul_blocked_module`]; `k` a multiple of four.
pub fn matmul_splitk_module(dt: ShaderDtype, epilogue: &[ChainStep], extras: usize) -> String {
    let t = dt.wgsl();
    let suf = dt.suffix();
    let out_binding = 2 + extras;
    let uniform_binding = 3 + extras;
    let extra_bindings = (0..extras)
        .map(|i| {
            format!("@group(0) @binding({}) var<storage, read> extra{i}: array<{t}>;\n", 2 + i)
        })
        .collect::<String>();
    let lib = format!("{}{}", unary_ops_wgsl("f32"), binary_ops_wgsl("f32"));
    let epi = epilogue_body_at(epilogue, "col", "oe");
    let mut s = preamble(dt);
    s.push_str(&format!(
        r#"
struct Params {{
    m: u32,
    k: u32,
    n: u32,
    off_a: u32,
    off_b: u32,
    off_out: u32,
    _p0: u32,
    _p1: u32,
    off_extra: vec4<u32>,
    mode_extra: vec4<u32>,
}}

@group(0) @binding(0) var<storage, read> a4: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> b4: array<vec4<f32>>;
{extra_bindings}@group(0) @binding({out_binding}) var<storage, read_write> c4: array<vec4<f32>>;
@group(0) @binding({uniform_binding}) var<uniform> params: Params;
{lib}
var<workgroup> sh0: array<vec4<f32>, {MATMUL_SPLITK_WG}>;
var<workgroup> sh1: array<vec4<f32>, {MATMUL_SPLITK_WG}>;
var<workgroup> sh2: array<vec4<f32>, {MATMUL_SPLITK_WG}>;
var<workgroup> sh3: array<vec4<f32>, {MATMUL_SPLITK_WG}>;

fn epi(v_in: f32, col: u32, oe: u32) -> f32 {{
    var v = v_in;
{epi}
    return v;
}}

@compute @workgroup_size({MATMUL_SPLITK_WG})
fn matmul_splitk_{suf}(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    let m4 = params.m / 4u;
    let n4 = params.n / 4u;
    let tile_global = gid.x >> 2u;
    let lane = gid.x & 3u;
    let row_tile_raw = tile_global % m4;
    let col_tile_raw = tile_global / m4;
    let valid = (row_tile_raw < m4) && (col_tile_raw < n4);
    let row_tile = min(row_tile_raw, m4 - 1u);
    let col_tile = min(col_tile_raw, n4 - 1u);

    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);

    let chunk = params.k >> 2u;
    let k_start = lane * chunk;
    let unroll_end = k_start + (chunk & ~3u);
    var kk = k_start;
    loop {{
        if (kk >= unroll_end) {{ break; }}
        let ab0 = params.off_a + kk * m4 + row_tile;
        let bb0 = params.off_b + kk * n4 + col_tile;
        let ab1 = ab0 + m4;
        let bb1 = bb0 + n4;
        let ab2 = ab1 + m4;
        let bb2 = bb1 + n4;
        let ab3 = ab2 + m4;
        let bb3 = bb2 + n4;
        let a0v = a4[ab0]; let b0v = b4[bb0];
        let a1v = a4[ab1]; let b1v = b4[bb1];
        let a2v = a4[ab2]; let b2v = b4[bb2];
        let a3v = a4[ab3]; let b3v = b4[bb3];
        acc0 += a0v * b0v.x + a1v * b1v.x + a2v * b2v.x + a3v * b3v.x;
        acc1 += a0v * b0v.y + a1v * b1v.y + a2v * b2v.y + a3v * b3v.y;
        acc2 += a0v * b0v.z + a1v * b1v.z + a2v * b2v.z + a3v * b3v.z;
        acc3 += a0v * b0v.w + a1v * b1v.w + a2v * b2v.w + a3v * b3v.w;
        kk += 4u;
    }}
    let k_end = k_start + chunk;
    loop {{
        if (kk >= k_end) {{ break; }}
        let ab = params.off_a + kk * m4 + row_tile;
        let bb = params.off_b + kk * n4 + col_tile;
        let av = a4[ab];
        let bv = b4[bb];
        acc0 += av * bv.x;
        acc1 += av * bv.y;
        acc2 += av * bv.z;
        acc3 += av * bv.w;
        kk += 1u;
    }}

    let base = lid.x - lane;
    sh0[lid.x] = acc0;
    sh1[lid.x] = acc1;
    sh2[lid.x] = acc2;
    sh3[lid.x] = acc3;
    workgroupBarrier();

    if (lane == 0u) {{
        let s0 = sh0[base] + sh0[base + 1u] + sh0[base + 2u] + sh0[base + 3u];
        let s1 = sh1[base] + sh1[base + 1u] + sh1[base + 2u] + sh1[base + 3u];
        let s2 = sh2[base] + sh2[base + 1u] + sh2[base + 2u] + sh2[base + 3u];
        let s3 = sh3[base] + sh3[base + 1u] + sh3[base + 2u] + sh3[base + 3u];
        if (valid) {{
            let col0 = col_tile * 4u;
            let cc = params.off_out + col0 * m4 + row_tile;
            let row_base = row_tile * 4u;
            let e0 = col0 * params.m + row_base;
            let e1 = e0 + params.m;
            let e2 = e1 + params.m;
            let e3 = e2 + params.m;
            c4[cc] = vec4<f32>(epi(s0.x, col0, e0), epi(s0.y, col0, e0 + 1u), epi(s0.z, col0, e0 + 2u), epi(s0.w, col0, e0 + 3u));
            c4[cc + m4] = vec4<f32>(epi(s1.x, col0 + 1u, e1), epi(s1.y, col0 + 1u, e1 + 1u), epi(s1.z, col0 + 1u, e1 + 2u), epi(s1.w, col0 + 1u, e1 + 3u));
            c4[cc + 2u * m4] = vec4<f32>(epi(s2.x, col0 + 2u, e2), epi(s2.y, col0 + 2u, e2 + 1u), epi(s2.z, col0 + 2u, e2 + 2u), epi(s2.w, col0 + 2u, e2 + 3u));
            c4[cc + 3u * m4] = vec4<f32>(epi(s3.x, col0 + 3u, e3), epi(s3.y, col0 + 3u, e3 + 1u), epi(s3.z, col0 + 3u, e3 + 2u), epi(s3.w, col0 + 3u, e3 + 3u));
        }}
    }}
}}
"#
    ));
    s
}

pub fn matmul_blocked_module(dt: ShaderDtype, epilogue: &[ChainStep], extras: usize) -> String {
    let t = dt.wgsl();
    let suf = dt.suffix();
    let out_binding = 2 + extras;
    let uniform_binding = 3 + extras;
    let extra_bindings = (0..extras)
        .map(|i| {
            format!("@group(0) @binding({}) var<storage, read> extra{i}: array<{t}>;\n", 2 + i)
        })
        .collect::<String>();
    let lib = format!("{}{}", unary_ops_wgsl("f32"), binary_ops_wgsl("f32"));
    let epi = epilogue_body_at(epilogue, "col", "oe");
    let mut s = preamble(dt);
    s.push_str(&format!(
        r#"
struct Params {{
    m: u32,
    k: u32,
    n: u32,
    off_a: u32,
    off_b: u32,
    off_out: u32,
    _p0: u32,
    _p1: u32,
    off_extra: vec4<u32>,
    mode_extra: vec4<u32>,
}}

@group(0) @binding(0) var<storage, read> a4: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> b4: array<vec4<f32>>;
{extra_bindings}@group(0) @binding({out_binding}) var<storage, read_write> c4: array<vec4<f32>>;
@group(0) @binding({uniform_binding}) var<uniform> params: Params;
{lib}

fn epi(v_in: f32, col: u32, oe: u32) -> f32 {{
    var v = v_in;
{epi}
    return v;
}}

@compute @workgroup_size({MATMUL_BLOCKED_WG})
fn matmul_blocked_{suf}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let m4 = params.m / 4u;
    let n4 = params.n / 4u;
    let tile = gid.x;
    let row_tile = tile % m4;
    let col_tile = tile / m4;
    if (col_tile >= n4) {{ return; }}
    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);
    var kk = 0u;
    for (; kk + 8u <= params.k; kk += 8u) {{
        let ab = params.off_a + kk * m4 + row_tile;
        let bb = params.off_b + kk * n4 + col_tile;
        let a0 = a4[ab];
        let a1 = a4[ab + m4];
        let a2 = a4[ab + 2u * m4];
        let a3 = a4[ab + 3u * m4];
        let a4v = a4[ab + 4u * m4];
        let a5 = a4[ab + 5u * m4];
        let a6 = a4[ab + 6u * m4];
        let a7 = a4[ab + 7u * m4];
        let b0 = b4[bb];
        let b1 = b4[bb + n4];
        let b2 = b4[bb + 2u * n4];
        let b3 = b4[bb + 3u * n4];
        let b4v = b4[bb + 4u * n4];
        let b5 = b4[bb + 5u * n4];
        let b6 = b4[bb + 6u * n4];
        let b7 = b4[bb + 7u * n4];
        acc0 += a0 * b0.x + a1 * b1.x + a2 * b2.x + a3 * b3.x
              + a4v * b4v.x + a5 * b5.x + a6 * b6.x + a7 * b7.x;
        acc1 += a0 * b0.y + a1 * b1.y + a2 * b2.y + a3 * b3.y
              + a4v * b4v.y + a5 * b5.y + a6 * b6.y + a7 * b7.y;
        acc2 += a0 * b0.z + a1 * b1.z + a2 * b2.z + a3 * b3.z
              + a4v * b4v.z + a5 * b5.z + a6 * b6.z + a7 * b7.z;
        acc3 += a0 * b0.w + a1 * b1.w + a2 * b2.w + a3 * b3.w
              + a4v * b4v.w + a5 * b5.w + a6 * b6.w + a7 * b7.w;
    }}
    for (; kk < params.k; kk += 4u) {{
        let ab = params.off_a + kk * m4 + row_tile;
        let bb = params.off_b + kk * n4 + col_tile;
        let a0 = a4[ab];
        let a1 = a4[ab + m4];
        let a2 = a4[ab + 2u * m4];
        let a3 = a4[ab + 3u * m4];
        let b0 = b4[bb];
        let b1 = b4[bb + n4];
        let b2 = b4[bb + 2u * n4];
        let b3 = b4[bb + 3u * n4];
        acc0 += a0 * b0.x + a1 * b1.x + a2 * b2.x + a3 * b3.x;
        acc1 += a0 * b0.y + a1 * b1.y + a2 * b2.y + a3 * b3.y;
        acc2 += a0 * b0.z + a1 * b1.z + a2 * b2.z + a3 * b3.z;
        acc3 += a0 * b0.w + a1 * b1.w + a2 * b2.w + a3 * b3.w;
    }}
    let col0 = col_tile * 4u;
    let cc = params.off_out + col0 * m4 + row_tile;
    let e0 = (col0 * m4 + row_tile) * 4u;
    let e1 = e0 + m4 * 4u;
    let e2 = e1 + m4 * 4u;
    let e3 = e2 + m4 * 4u;
    c4[cc] = vec4<f32>(
        epi(acc0.x, col0, e0), epi(acc0.y, col0, e0 + 1u),
        epi(acc0.z, col0, e0 + 2u), epi(acc0.w, col0, e0 + 3u));
    c4[cc + m4] = vec4<f32>(
        epi(acc1.x, col0 + 1u, e1), epi(acc1.y, col0 + 1u, e1 + 1u),
        epi(acc1.z, col0 + 1u, e1 + 2u), epi(acc1.w, col0 + 1u, e1 + 3u));
    c4[cc + 2u * m4] = vec4<f32>(
        epi(acc2.x, col0 + 2u, e2), epi(acc2.y, col0 + 2u, e2 + 1u),
        epi(acc2.z, col0 + 2u, e2 + 2u), epi(acc2.w, col0 + 2u, e2 + 3u));
    c4[cc + 3u * m4] = vec4<f32>(
        epi(acc3.x, col0 + 3u, e3), epi(acc3.y, col0 + 3u, e3 + 1u),
        epi(acc3.z, col0 + 3u, e3 + 2u), epi(acc3.w, col0 + 3u, e3 + 3u));
}}
"#
    ));
    s
}

#[cfg(test)]
mod tests {
    use tract_core::internal::*;

    use crate::test_utils::{fill, gpu_vs_cpu, transformed};

    /// A residual: the operand is another tensor shaped like the product, read
    /// element for element. Two products chained so the residual is an op's
    /// output. With `transposed` every flag is set, as a channels-first pointwise
    /// convolution lowers, and the sizes are multiples of four so the blocked
    /// kernel takes it.
    fn matmul_residual(m: usize, k: usize, n: usize, transposed: bool) -> TractResult<()> {
        use tract_core::ops::einsum::prefix_matmul::PrefixMatMul;
        let mut model = TypedModel::default();
        let op = PrefixMatMul {
            transpose_a: transposed,
            transpose_b: transposed,
            transpose_c: transposed,
            quantize_output: None,
            operating_dt: None,
        };
        let (a_shape, b_shape, bias_shape, w2_shape) = if transposed {
            (vec![k, m], vec![n, k], vec![n, 1], vec![n, n])
        } else {
            (vec![m, k], vec![k, n], vec![1, n], vec![n, n])
        };
        let a = model.add_source("a", f32::fact(&*a_shape))?;
        let b = model.add_const("b", Tensor::from_shape(&b_shape, &fill(1, k * n))?)?;
        let bias = model.add_const("bias", Tensor::from_shape(&bias_shape, &fill(2, n))?)?;
        let first = model.wire_node("first", op, &[a, b])?[0];
        let first = model.wire_node("first_bias", tract_core::ops::math::add(), &[first, bias])?[0];
        let w2 = model.add_const("w2", Tensor::from_shape(&w2_shape, &fill(3, n * n))?)?;
        let second = model.wire_node("second", op, &[first, w2])?[0];
        let y = model.wire_node("residual", tract_core::ops::math::add(), &[second, first])?[0];
        model.select_output_outlets(&[y])?;
        gpu_vs_cpu(model, Tensor::from_shape(&a_shape, &fill(4, m * k))?)
    }

    /// `k` a multiple of four but not of eight, so the blocked kernel's unrolled
    /// loop leaves a four-step tail to run.
    #[test]
    fn matmul_blocked_k_tail_matches_cpu() -> TractResult<()> {
        use tract_core::ops::einsum::prefix_matmul::PrefixMatMul;
        for (m, k, n) in [(64usize, 12usize, 16usize), (144, 20, 32), (16, 4, 8)] {
            let mut model = TypedModel::default();
            let a = model.add_source("a", f32::fact([k, m]))?;
            let b = model.add_const("b", Tensor::from_shape(&[n, k], &fill(1, n * k))?)?;
            let op = PrefixMatMul {
                transpose_a: true,
                transpose_b: true,
                transpose_c: true,
                quantize_output: None,
                operating_dt: None,
            };
            let out = model.wire_node("mm", op, &[a, b])?[0];
            let bias = model.add_const("bias", Tensor::from_shape(&[n, 1], &fill(2, n))?)?;
            let y = model.wire_node("bias_add", tract_core::ops::math::add(), &[out, bias])?[0];
            model.select_output_outlets(&[y])?;
            gpu_vs_cpu(model, Tensor::from_shape(&[k, m], &fill(3, k * m))?)
                .with_context(|| format!("m={m} k={k} n={n}"))?;
        }
        Ok(())
    }

    #[test]
    fn matmul_residual_matches_cpu() -> TractResult<()> {
        matmul_residual(6, 5, 7, false)?;
        matmul_residual(8, 8, 16, true)
    }

    /// Products the tiled matmul takes: every dimension off the tile size, each
    /// transposition, a prefix, f16, a bias epilogue, and thin products with a long
    /// `k` that fill one or two tiles.
    #[test]
    fn matmul_tiled_matches_cpu() -> TractResult<()> {
        use tract_core::ops::einsum::prefix_matmul::PrefixMatMul;
        for (prefix, m, k, n, ta, tb, tc, f16_, bias) in [
            (1usize, 130usize, 17usize, 513usize, false, false, false, false, false),
            (1, 130, 17, 513, true, false, false, false, false),
            (1, 130, 17, 513, false, true, false, false, false),
            (1, 130, 17, 513, false, false, true, false, false),
            (1, 130, 17, 513, true, true, true, false, false),
            (8, 65, 33, 63, false, false, false, false, false),
            (1, 130, 17, 513, false, false, false, true, false),
            (1, 130, 17, 513, false, false, false, false, true),
            (1, 16, 300, 16, false, false, false, false, false),
            (1, 20, 960, 40, true, false, false, false, false),
            (1, 112, 40, 130, false, false, false, false, false),
            (1, 112, 40, 130, true, true, true, false, false),
        ] {
            let dt = if f16_ { f16::datum_type() } else { f32::datum_type() };
            let a_shape = [prefix, if ta { k } else { m }, if ta { m } else { k }];
            let b_shape = [prefix, if tb { n } else { k }, if tb { k } else { n }];
            let mut model = TypedModel::default();
            let a = model.add_source("a", dt.fact(a_shape))?;
            let b_data = Tensor::from_shape(&b_shape, &fill(1, b_shape.iter().product()))?;
            let b = model.add_const("b", b_data.cast_to_dt(dt)?.into_owned())?;
            let op = PrefixMatMul {
                transpose_a: ta,
                transpose_b: tb,
                transpose_c: tc,
                quantize_output: None,
                operating_dt: None,
            };
            let mut y = model.wire_node("mm", op, &[a, b])?[0];
            if bias {
                let bias = Tensor::from_shape(&[1, 1, n], &fill(2, n))?;
                let bias = model.add_const("bias", bias.cast_to_dt(dt)?.into_owned())?;
                y = model.wire_node("bias_add", tract_core::ops::math::add(), &[y, bias])?[0];
            }
            model.select_output_outlets(&[y])?;
            let input = Tensor::from_shape(&a_shape, &fill(3, a_shape.iter().product()))?;
            gpu_vs_cpu(model, input.cast_to_dt(dt)?.into_owned()).with_context(|| {
                format!("prefix={prefix} m={m} k={k} n={n} ta={ta} tb={tb} tc={tc} f16={f16_}")
            })?;
        }
        Ok(())
    }

    /// `m = 1` products: partial `k` and `n` tiles, every transposition, a prefix,
    /// a broadcast prefix, f16, and a bias-plus-activation epilogue.
    #[test]
    fn matmul_gemv_matches_cpu() -> TractResult<()> {
        use tract_core::ops::einsum::prefix_matmul::PrefixMatMul;
        use tract_core::ops::nn::sigmoid;
        for (prefix, k, n, ta, tb, tc, f16_, bias) in [
            (1usize, 1usize, 1usize, false, false, false, false, false),
            (1, 17, 7, false, false, false, false, false),
            (1, 63, 64, false, false, false, false, false),
            (1, 65, 63, false, false, false, false, false),
            (1, 300, 130, false, false, false, false, false),
            (1, 300, 130, true, false, false, false, false),
            (1, 300, 130, false, true, false, false, false),
            (1, 300, 130, false, false, true, false, false),
            (1, 300, 130, true, true, true, false, false),
            (4, 40, 80, false, false, false, false, false),
            (1, 200, 1, false, false, false, false, false),
            (1, 128, 256, false, false, false, true, false),
            (1, 100, 50, false, false, false, false, true),
            (1, 80, 33, false, false, false, true, true),
        ] {
            let dt = if f16_ { f16::datum_type() } else { f32::datum_type() };
            let a_shape = [prefix, if ta { k } else { 1 }, if ta { 1 } else { k }];
            let b_shape = [prefix, if tb { n } else { k }, if tb { k } else { n }];
            let mut model = TypedModel::default();
            let a = model.add_source("a", dt.fact(a_shape))?;
            let b_data = Tensor::from_shape(&b_shape, &fill(1, b_shape.iter().product()))?;
            let b = model.add_const("b", b_data.cast_to_dt(dt)?.into_owned())?;
            let op = PrefixMatMul {
                transpose_a: ta,
                transpose_b: tb,
                transpose_c: tc,
                quantize_output: None,
                operating_dt: None,
            };
            let mut y = model.wire_node("mm", op, &[a, b])?[0];
            if bias {
                let bias = Tensor::from_shape(&[1, 1, n], &fill(2, n))?;
                let bias = model.add_const("bias", bias.cast_to_dt(dt)?.into_owned())?;
                y = model.wire_node("bias_add", tract_core::ops::math::add(), &[y, bias])?[0];
                if !f16_ {
                    y = model.wire_node("act", sigmoid(), &[y])?[0];
                }
            }
            model.select_output_outlets(&[y])?;
            if bias {
                let gpu = transformed(model.clone())?;
                let gemm = gpu
                    .nodes()
                    .iter()
                    .find_map(|node| node.op_as::<crate::ops::matmul::WgpuGemm>())
                    .context("m=1 product left the device")?;
                let want = if f16_ { 1 } else { 2 };
                ensure!(gemm.epilogue.len() == want, "epilogue {:?}", gemm.epilogue);
            }
            let input = Tensor::from_shape(&a_shape, &fill(3, a_shape.iter().product()))?;
            gpu_vs_cpu(model, input.cast_to_dt(dt)?.into_owned()).with_context(|| {
                format!(
                    "prefix={prefix} k={k} n={n} ta={ta} tb={tb} tc={tc} f16={f16_} bias={bias}"
                )
            })?;
        }

        let k = 24usize;
        let n = 18usize;
        let mut model = TypedModel::default();
        let a = model.add_source("a", f32::fact([1, 1, k]))?;
        let b = Tensor::from_shape(&[4, k, n], &fill(4, 4 * k * n))?;
        let b = model.add_const("b", b)?;
        let op = PrefixMatMul {
            transpose_a: false,
            transpose_b: false,
            transpose_c: false,
            quantize_output: None,
            operating_dt: None,
        };
        let y = model.wire_node("mm", op, &[a, b])?[0];
        model.select_output_outlets(&[y])?;
        gpu_vs_cpu(model, Tensor::from_shape(&[1, 1, k], &fill(5, k))?)
    }

    /// `n = 1` products: partial `k` and `m` tiles, every transposition, a prefix,
    /// a broadcast prefix, f16, a bias-plus-activation epilogue, and a residual
    /// laid out like the output.
    #[test]
    fn matmul_gemv_n_matches_cpu() -> TractResult<()> {
        use tract_core::ops::einsum::prefix_matmul::PrefixMatMul;
        use tract_core::ops::nn::sigmoid;
        for (prefix, m, k, ta, tb, tc, f16_, bias) in [
            (1usize, 1usize, 1usize, false, false, false, false, false),
            (1, 7, 17, false, false, false, false, false),
            (1, 64, 63, false, false, false, false, false),
            (1, 63, 64, false, false, false, false, false),
            (1, 65, 65, false, false, false, false, false),
            (1, 130, 300, false, false, false, false, false),
            (1, 130, 300, true, false, false, false, false),
            (1, 130, 300, false, true, false, false, false),
            (1, 130, 300, false, false, true, false, false),
            (1, 130, 300, true, true, true, false, false),
            (4, 80, 40, false, false, false, false, false),
            (1, 1, 200, false, false, false, false, false),
            (1, 70, 65, false, false, false, true, false),
            (1, 50, 100, false, false, false, false, true),
            (1, 33, 40, false, false, false, true, true),
        ] {
            let dt = if f16_ { f16::datum_type() } else { f32::datum_type() };
            let a_shape = [prefix, if ta { k } else { m }, if ta { m } else { k }];
            let b_shape = [prefix, if tb { 1 } else { k }, if tb { k } else { 1 }];
            let mut model = TypedModel::default();
            let a = model.add_source("a", dt.fact(a_shape))?;
            let b_data = Tensor::from_shape(&b_shape, &fill(1, b_shape.iter().product()))?;
            let b = model.add_const("b", b_data.cast_to_dt(dt)?.into_owned())?;
            let op = PrefixMatMul {
                transpose_a: ta,
                transpose_b: tb,
                transpose_c: tc,
                quantize_output: None,
                operating_dt: None,
            };
            let mut y = model.wire_node("mm", op, &[a, b])?[0];
            if bias {
                let bias = Tensor::from_shape(&[1, 1, 1], &fill(2, 1))?;
                let bias = model.add_const("bias", bias.cast_to_dt(dt)?.into_owned())?;
                y = model.wire_node("bias_add", tract_core::ops::math::add(), &[y, bias])?[0];
                if !f16_ {
                    y = model.wire_node("act", sigmoid(), &[y])?[0];
                }
            }
            model.select_output_outlets(&[y])?;
            if bias {
                let gpu = transformed(model.clone())?;
                let gemm = gpu
                    .nodes()
                    .iter()
                    .find_map(|node| node.op_as::<crate::ops::matmul::WgpuGemm>())
                    .context("n=1 product left the device")?;
                let want = if f16_ { 1 } else { 2 };
                ensure!(gemm.epilogue.len() == want, "epilogue {:?}", gemm.epilogue);
            }
            let input = Tensor::from_shape(&a_shape, &fill(3, a_shape.iter().product()))?;
            gpu_vs_cpu(model, input.cast_to_dt(dt)?.into_owned()).with_context(|| {
                format!(
                    "prefix={prefix} m={m} k={k} ta={ta} tb={tb} tc={tc} f16={f16_} bias={bias}"
                )
            })?;
        }

        for (a_prefix, b_prefix, m, k) in [(1usize, 4usize, 24usize, 18usize), (4, 1, 24, 18)] {
            let mut model = TypedModel::default();
            let a = model.add_source("a", f32::fact([a_prefix, m, k]))?;
            let b = Tensor::from_shape(&[b_prefix, k, 1], &fill(4, b_prefix * k))?;
            let b = model.add_const("b", b)?;
            let op = PrefixMatMul {
                transpose_a: false,
                transpose_b: false,
                transpose_c: false,
                quantize_output: None,
                operating_dt: None,
            };
            let y = model.wire_node("mm", op, &[a, b])?[0];
            model.select_output_outlets(&[y])?;
            gpu_vs_cpu(model, Tensor::from_shape(&[a_prefix, m, k], &fill(5, a_prefix * m * k))?)
                .with_context(|| format!("broadcast a={a_prefix} b={b_prefix}"))?;
        }

        for (ta, tb, tc) in [(false, false, false), (true, false, false), (true, true, true)] {
            let (m, k) = (20usize, 16usize);
            let a_shape = [1usize, if ta { k } else { m }, if ta { m } else { k }];
            let b_shape = [1, if tb { 1 } else { k }, if tb { k } else { 1 }];
            let o_shape = if tc { [1, 1, m] } else { [1, m, 1] };
            let mut model = TypedModel::default();
            let a = model.add_source("a", f32::fact(a_shape))?;
            let b = Tensor::from_shape(&b_shape, &fill(6, b_shape.iter().product()))?;
            let b = model.add_const("b", b)?;
            let op = PrefixMatMul {
                transpose_a: ta,
                transpose_b: tb,
                transpose_c: tc,
                quantize_output: None,
                operating_dt: None,
            };
            let y = model.wire_node("mm", op, &[a, b])?[0];
            let r = model.add_source("r", f32::fact(o_shape))?;
            let y = model.wire_node("residual", tract_core::ops::math::add(), &[y, r])?[0];
            model.select_output_outlets(&[y])?;
            let gpu_model = transformed(model.clone())?;
            let gemm = gpu_model
                .nodes()
                .iter()
                .find_map(|node| node.op_as::<crate::ops::matmul::WgpuGemm>())
                .context("residual left the device")?;
            ensure!(gemm.epilogue.len() == 1, "epilogue {:?}", gemm.epilogue);
            let a_in = Tensor::from_shape(&a_shape, &fill(7, a_shape.iter().product()))?;
            let r_in = Tensor::from_shape(&o_shape, &fill(8, m))?;
            crate::context::wgpu_context();
            let cpu = SimplePlan::new(model.clone())?;
            let rt = tract_core::runtime::runtime_for_name("wgpu")?
                .context("wgpu runtime not registered")?;
            let gpu = rt.prepare(model)?;
            let g =
                gpu.run(tvec!(a_in.clone().into(), r_in.clone().into()))?.remove(0).into_tensor();
            let c = Arc::new(cpu).run(tvec!(a_in.into(), r_in.into()))?.remove(0).into_tensor();
            g.close_enough(&c, Approximation::Approximate)
                .with_context(|| format!("residual ta={ta} tb={tb} tc={tc}"))?;
        }
        Ok(())
    }
}
