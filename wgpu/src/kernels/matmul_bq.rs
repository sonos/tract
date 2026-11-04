//! `Q4_0` and ternary `Q2_0_T` matmul. B is K-inner packed bytes; the shader
//! dequants one block of 32 into f32 and accumulates there. Group 128 is four
//! of those blocks, each with its own scale.

use tract_core::internal::*;
use tract_core::tract_linalg::block_quant::BlockQuantFact;
use tract_core::tract_linalg::block_quant::Q2_0_T;
use tract_core::tract_linalg::block_quant::Q4_0;
use tract_gpu::tensor::DeviceTensor;

use crate::kernels::matmul::Transposes;
use crate::kernels::matmul::gemm_epilogue_mode;
use crate::kernels::matmul::output_shape;
use crate::kernels::shaders::ChainStep;
use crate::kernels::shaders::EntryPoint;
use crate::kernels::shaders::GRID_LIMIT;
use crate::kernels::shaders::LayoutKind;
use crate::kernels::shaders::ShaderDtype;
use crate::kernels::shaders::binary_ops_wgsl;
use crate::kernels::shaders::epilogue_body_at;
use crate::kernels::shaders::pack_u32s;
use crate::kernels::shaders::preamble;
use crate::kernels::shaders::unary_ops_wgsl;
use crate::utils::element_offset;
use crate::utils::get_wgpu_buffer;
use crate::with_wgpu_queue;

const TILE: u32 = 8;

/// `Q4_0` or `Q2_0_T` on this tensor, owned or viewed.
pub(crate) fn is_block_quant_weight(b: &DeviceTensor) -> bool {
    bq_fact(b).is_some()
}

fn bq_fact(t: &DeviceTensor) -> Option<BlockQuantFact> {
    let fact = match t {
        DeviceTensor::Owned(t) => t.exotic_fact(),
        DeviceTensor::View(t) => t.exotic_fact(),
    }?;
    let bqf = fact.downcast_ref::<BlockQuantFact>()?.clone();
    if bqf.format.block_len() == 32 && (bqf.format.dyn_eq(&Q4_0) || bqf.format.dyn_eq(&Q2_0_T)) {
        Some(bqf)
    } else {
        None
    }
}

/// Element step of one row-major prefix index, or 0 when `input` broadcasts.
fn broadcast_prefix_stride(
    out_prefix: &[usize],
    in_prefix: &[usize],
    in_strides: &[isize],
) -> TractResult<u32> {
    ensure!(out_prefix.len() == in_prefix.len() && out_prefix.len() == in_strides.len());
    if out_prefix.iter().copied().product::<usize>() <= 1 {
        return Ok(0);
    }
    let mut rest = 1usize;
    let mut off = 0isize;
    for i in (0..out_prefix.len()).rev() {
        let coord = rest % out_prefix[i];
        rest /= out_prefix[i];
        if in_prefix[i] != 1 && coord != 0 {
            ensure!(in_strides[i] >= 0, "negative prefix stride");
            off += coord as isize * in_strides[i];
        }
    }
    ensure!(rest == 0, "prefix index overflow");
    let mut later = 1usize;
    for i in (0..out_prefix.len()).rev() {
        if out_prefix[i] > 1 {
            let step = if in_prefix[i] == 1 {
                0
            } else {
                ensure!(in_strides[i] >= 0, "negative prefix stride");
                in_strides[i]
            };
            ensure!(step == off * later as isize, "block-quant matmul prefix is not one stride");
        }
        later *= out_prefix[i];
    }
    u32::try_from(off).context("prefix stride does not fit a WGSL u32")
}

fn u32_of(v: usize) -> TractResult<u32> {
    u32::try_from(v).context("matmul size does not fit a WGSL u32")
}

fn u32_stride(v: isize) -> TractResult<u32> {
    ensure!(v >= 0, "negative matmul stride");
    u32::try_from(v).context("matmul stride does not fit a WGSL u32")
}

/// `(stride of M, stride of K)` or, for C, `(stride of M, stride of N)`.
fn logical_strides(strides: &[isize], transposed: bool) -> TractResult<(u32, u32)> {
    let rank = strides.len();
    ensure!(rank >= 2);
    if transposed {
        Ok((u32_stride(strides[rank - 1])?, u32_stride(strides[rank - 2])?))
    } else {
        Ok((u32_stride(strides[rank - 2])?, u32_stride(strides[rank - 1])?))
    }
}

/// Q4_0 (format 0) or Q2_0_T (format 1) times a plain f16/f32 activation.
/// Workgroup is 8×8, K tiled by 32. Threads past M or N still hit both barriers.
pub fn wgpu_bq_dispatch(
    t: Transposes,
    epilogue: &[ChainStep],
    a: &DeviceTensor,
    b: &DeviceTensor,
    extras: &[&DeviceTensor],
    output: &DeviceTensor,
) -> TractResult<()> {
    let fact = bq_fact(b).context("block-quant matmul expects a Q4_0 or Q2_0_T weight")?;
    with_wgpu_queue(|q| {
        q.retain_tensor(a);
        q.retain_tensor(b);
        q.retain_tensor(output);
        for extra in extras {
            q.retain_tensor(extra);
        }
        ensure!(t.b, "block-quant matmul reads a K-inner weight");
        ensure!(bq_fact(a).is_none(), "block-quant matmul packs B only");
        ensure!(a.datum_type() == output.datum_type(), "block-quant matmul keeps A's dtype");
        for extra in extras {
            ensure!(extra.datum_type() == a.datum_type(), "epilogue dtype differs from A");
        }
        let dt = ShaderDtype::from_datum(a.datum_type())?;
        let (m, k, n) = crate::kernels::matmul::mkn(a.shape(), b.shape(), t.a, t.b)?;
        ensure!(fact.k() == k, "packed K is {}, matmul K is {k}", fact.k());
        ensure!(k % 32 == 0, "block-quant matmul needs K a multiple of 32, got {k}");
        let rank = a.rank();
        let out_shape = output.shape();
        ensure!(
            out_shape == &*output_shape(a.shape(), b.shape(), t.a, t.b, t.c)?,
            "matmul output shape {out_shape:?} does not match its inputs"
        );
        for i in 0..rank - 2 {
            let (ad, bd) = (a.shape()[i], b.shape()[i]);
            if ad != 1 && bd != 1 {
                ensure!(ad == bd, "prefix mismatch on axis {i}: {ad} vs {bd}");
            }
        }
        let prefix_dims = &out_shape[..rank - 2];
        let prefix = prefix_dims.iter().product::<usize>();
        ensure!(prefix <= GRID_LIMIT as usize, "prefix {prefix} exceeds the grid");
        ensure!(m.div_ceil(TILE as usize) <= GRID_LIMIT as usize, "m {m} exceeds the grid");
        ensure!(n.div_ceil(TILE as usize) <= GRID_LIMIT as usize, "n {n} exceeds the grid");

        let (a_stride_m, a_stride_k) = logical_strides(a.strides(), t.a)?;
        let (c_stride_m, c_stride_n) = logical_strides(output.strides(), t.c)?;
        let a_prefix =
            broadcast_prefix_stride(prefix_dims, &a.shape()[..rank - 2], &a.strides()[..rank - 2])?;
        let c_prefix = broadcast_prefix_stride(
            prefix_dims,
            &out_shape[..rank - 2],
            &output.strides()[..rank - 2],
        )?;
        ensure!(b.strides()[rank - 1] == 1, "packed weight K axis is not contiguous");
        ensure!(
            k == 0 || b.strides()[rank - 2] == k as isize,
            "packed weight rows are not contiguous"
        );
        let b_elem_prefix =
            broadcast_prefix_stride(prefix_dims, &b.shape()[..rank - 2], &b.strides()[..rank - 2])?;
        let blocks_per_row = k / 32;
        let block_bytes = fact.format.block_bytes();
        let b_prefix = if b_elem_prefix == 0 {
            0
        } else {
            ensure!(k != 0 && (b_elem_prefix as usize).is_multiple_of(k));
            let rows = b_elem_prefix as usize / k;
            u32_of(rows * blocks_per_row * block_bytes)?
        };
        let packed_rows = b.shape()[..rank - 1].iter().product::<usize>();
        ensure!(fact.m() == packed_rows, "packed rows {} do not match the weight shape", fact.m());

        let format = if fact.format.dyn_eq(&Q4_0) { 0 } else { 1 };
        let a_off = u32_of(element_offset(a, 0))?;
        let b_off = u32_of(b.buffer_offset::<usize>())?;
        let tiled = dt == ShaderDtype::F32
            && a_stride_k == 1
            && a_stride_m.is_multiple_of(4)
            && a_off.is_multiple_of(4)
            && a_prefix.is_multiple_of(4)
            && b_off.is_multiple_of(2)
            && b_prefix.is_multiple_of(2);
        let layout = LayoutKind::Chain(3 + extras.len() as u8);
        let pipeline = if tiled {
            q.context().chain_pipeline(
                (
                    if format == 0 { "matmul_bq_q4" } else { "matmul_bq_q2" },
                    dt,
                    epilogue,
                    extras.len(),
                ),
                layout,
                EntryPoint::typed("bq_tiled", dt),
                || bq_tiled_module(format == 0, epilogue, extras.len()),
            )?
        } else {
            q.context().chain_pipeline(
                ("matmul_bq", dt, epilogue, extras.len()),
                layout,
                EntryPoint::typed("bq_matmul", dt),
                || bq_module(dt, epilogue, extras.len()),
            )?
        };
        let mut buffers = vec![get_wgpu_buffer(a), get_wgpu_buffer(b)];
        buffers.extend(extras.iter().map(|t| get_wgpu_buffer(t)));
        buffers.push(get_wgpu_buffer(output));
        let bg = q.context().bind_group(layout, &buffers, q.uniform())?;
        let mut off_extra = [0u32; 4];
        let mut mode_extra = [0u32; 4];
        for (i, extra) in extras.iter().enumerate() {
            off_extra[i] = u32_of(element_offset(extra, 0))?;
            mode_extra[i] = gemm_epilogue_mode(extra, n, output)?;
        }
        let mut vals = vec![
            u32_of(m)?,
            u32_of(k)?,
            u32_of(n)?,
            u32_of(prefix)?,
            a_off,
            a_prefix,
            a_stride_m,
            a_stride_k,
            b_off,
            b_prefix,
            u32_of(blocks_per_row)?,
            u32_of(block_bytes)?,
            format,
            u32_of(element_offset(output, 0))?,
            c_prefix,
            c_stride_m,
            c_stride_n,
            0,
            0,
            0,
        ];
        vals.extend_from_slice(&off_extra);
        vals.extend_from_slice(&mode_extra);
        debug_assert_eq!(vals.len(), 28);
        let dyn_off = q.alloc_uniform(&pack_u32s(&vals))?;
        let grid = if tiled {
            [
                n.div_ceil(4).div_ceil(TILE as usize) as u32,
                m.div_ceil(4).div_ceil(TILE as usize) as u32,
                prefix.max(1) as u32,
            ]
        } else {
            [
                n.div_ceil(TILE as usize) as u32,
                m.div_ceil(TILE as usize) as u32,
                prefix.max(1) as u32,
            ]
        };
        q.dispatch_grid("matmul_bq", &pipeline, &bg, dyn_off, grid)
    })
}

/// `bq_module`'s product for f32 rows laid out contiguously: one 4x4 output tile per invocation,
/// each block's scale and codes read once, and four codes unpacked at a time against a `vec4` of
/// A. Takes no workgroup memory, so it has no barriers.
fn bq_tiled_module(q4: bool, epilogue: &[ChainStep], extras: usize) -> String {
    let dt = ShaderDtype::F32;
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
    let words = if q4 { 4 } else { 2 };
    let load = (0..4)
        .map(|j| {
            let fields = (0..words)
                .map(|w| format!("        let w{j}_{w} = rd(o{j} + {}u);\n", 2 + 4 * w))
                .collect::<String>();
            format!(
                "        let o{j} = base + n{j} * params.blocks_per_row * params.block_bytes + block_bytes;
        let s{j} = unpack2x16float(rd(o{j})).x;
{fields}"
            )
        })
        .collect::<String>();
    // Scalars and `select`: a dynamically indexed `let` array is far slower under Tint.
    let code = |j: usize| {
        if q4 {
            format!(
                "vec4<f32>((vec4<u32>(select(select(w{j}_0, w{j}_1, (c & 1u) == 1u), select(w{j}_2, w{j}_3, (c & 1u) == 1u), (c & 2u) == 2u) >> (4u * (c >> 2u))) >> vec4<u32>(0u, 8u, 16u, 24u)) & vec4<u32>(15u)) - 8.0"
            )
        } else {
            format!(
                "bitcast<vec4<f32>>(((vec4<u32>(select(w{j}_0, w{j}_1, c >= 4u) >> ((c & 3u) * 8u)) >> vec4<u32>(0u, 2u, 4u, 6u)) & vec4<u32>(3u)) | vec4<u32>(0x4b000000u)) - vec4<f32>(8388609.0)"
            )
        }
    };
    let dots = (0..4)
        .map(|j| {
            format!(
                "            let q{j} = {};
            b{j} += vec4<f32>(dot(a0, q{j}), dot(a1, q{j}), dot(a2, q{j}), dot(a3, q{j}));
",
                code(j)
            )
        })
        .collect::<String>();
    let mut s = preamble(dt);
    s.push_str(&format!(
        r#"
struct Params {{
    m: u32,
    k: u32,
    n: u32,
    prefix: u32,
    a_off: u32,
    a_prefix_stride: u32,
    a_stride_m: u32,
    a_stride_k: u32,
    b_off: u32,
    b_prefix_stride: u32,
    blocks_per_row: u32,
    block_bytes: u32,
    format: u32,
    c_off: u32,
    c_prefix_stride: u32,
    c_stride_m: u32,
    c_stride_n: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
    off_extra: vec4<u32>,
    mode_extra: vec4<u32>,
}}

@group(0) @binding(0) var<storage, read> a4: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> b: array<u32>;
{extra_bindings}@group(0) @binding({out_binding}) var<storage, read_write> outp: array<{t}>;
@group(0) @binding({uniform_binding}) var<uniform> params: Params;
{lib}
// A little-endian word at an even byte offset.
fn rd(off: u32) -> u32 {{
    let w = off >> 2u;
    if ((off & 2u) == 0u) {{
        return b[w];
    }}
    return (b[w] >> 16u) | (b[w + 1u] << 16u);
}}

fn emit(value: f32, row: u32, col: u32, pref: u32) {{
    if (row >= params.m || col >= params.n) {{
        return;
    }}
    let oe = pref * params.c_prefix_stride + row * params.c_stride_m + col * params.c_stride_n;
    var v = value;
{epilogue}    outp[params.c_off + oe] = v;
}}

@compute @workgroup_size({TILE}, {TILE})
fn bq_tiled_{suf}(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {{
    let row0 = (wid.y * {TILE}u + lid.y) * 4u;
    let col0 = (wid.x * {TILE}u + lid.x) * 4u;
    let pref = wid.z;
    if (row0 >= params.m || col0 >= params.n || pref >= params.prefix) {{
        return;
    }}
    let base = params.b_off + pref * params.b_prefix_stride;
    let a_base = (params.a_off + pref * params.a_prefix_stride) >> 2u;
    let rows = params.a_stride_m >> 2u;
    let last_row = params.m - 1u;
    let last_col = params.n - 1u;
    let r0 = a_base + min(row0, last_row) * rows;
    let r1 = a_base + min(row0 + 1u, last_row) * rows;
    let r2 = a_base + min(row0 + 2u, last_row) * rows;
    let r3 = a_base + min(row0 + 3u, last_row) * rows;
    let n0 = min(col0, last_col);
    let n1 = min(col0 + 1u, last_col);
    let n2 = min(col0 + 2u, last_col);
    let n3 = min(col0 + 3u, last_col);
    var acc0 = vec4<f32>(0.0);
    var acc1 = vec4<f32>(0.0);
    var acc2 = vec4<f32>(0.0);
    var acc3 = vec4<f32>(0.0);
    for (var block = 0u; block < params.blocks_per_row; block++) {{
        let block_bytes = block * params.block_bytes;
{load}        var b0 = vec4<f32>(0.0);
        var b1 = vec4<f32>(0.0);
        var b2 = vec4<f32>(0.0);
        var b3 = vec4<f32>(0.0);
        for (var c = 0u; c < 8u; c++) {{
            let kv = block * 8u + c;
            let a0 = a4[r0 + kv];
            let a1 = a4[r1 + kv];
            let a2 = a4[r2 + kv];
            let a3 = a4[r3 + kv];
{dots}        }}
        acc0 += b0 * s0;
        acc1 += b1 * s1;
        acc2 += b2 * s2;
        acc3 += b3 * s3;
    }}
    for (var i = 0u; i < 4u; i++) {{
        emit(acc0[i], row0 + i, col0, pref);
        emit(acc1[i], row0 + i, col0 + 1u, pref);
        emit(acc2[i], row0 + i, col0 + 2u, pref);
        emit(acc3[i], row0 + i, col0 + 3u, pref);
    }}
}}
"#
    ));
    s
}

fn bq_module(dt: ShaderDtype, epilogue: &[ChainStep], extras: usize) -> String {
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
    m: u32,
    k: u32,
    n: u32,
    prefix: u32,
    a_off: u32,
    a_prefix_stride: u32,
    a_stride_m: u32,
    a_stride_k: u32,
    b_off: u32,
    b_prefix_stride: u32,
    blocks_per_row: u32,
    block_bytes: u32,
    format: u32,
    c_off: u32,
    c_prefix_stride: u32,
    c_stride_m: u32,
    c_stride_n: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
    off_extra: vec4<u32>,
    mode_extra: vec4<u32>,
}}

@group(0) @binding(0) var<storage, read> a: array<{t}>;
@group(0) @binding(1) var<storage, read> b: array<u32>;
{extra_bindings}@group(0) @binding({out_binding}) var<storage, read_write> outp: array<{t}>;
@group(0) @binding({uniform_binding}) var<uniform> params: Params;

var<workgroup> a_tile: array<f32, 256>;
var<workgroup> b_tile: array<f32, 256>;
{lib}
fn f16_bits_to_f32(bits: u32) -> f32 {{
    let sign = (bits >> 15u) & 1u;
    let exp = (bits >> 10u) & 31u;
    let mant = bits & 1023u;
    if (exp == 0u) {{
        let mag = f32(mant) * 5.960464477539063e-8;
        return select(mag, -mag, sign == 1u);
    }}
    if (exp == 31u) {{
        let inf_bits = (sign << 31u) | (255u << 23u) | (mant << 13u);
        return bitcast<f32>(inf_bits);
    }}
    let fexp = exp + 112u;
    let fbits = (sign << 31u) | (fexp << 23u) | (mant << 13u);
    return bitcast<f32>(fbits);
}}

fn load_u8(off: u32) -> u32 {{
    let word = b[off >> 2u];
    return (word >> ((off & 3u) * 8u)) & 255u;
}}

fn dequant(n_row: u32, kk: u32, pref_bytes: u32) -> f32 {{
    let block = kk >> 5u;
    let inner = kk & 31u;
    let base = params.b_off + pref_bytes + (n_row * params.blocks_per_row + block) * params.block_bytes;
    let scale = f16_bits_to_f32(load_u8(base) | (load_u8(base + 1u) << 8u));
    if (params.format == 0u) {{
        // ggml Q4_0: byte j holds position j in the low nibble and j+16 in the high nibble.
        let byte = load_u8(base + 2u + (inner & 15u));
        let nib = select(byte >> 4u, byte & 15u, inner < 16u);
        return (f32(nib) - 8.0) * scale;
    }}
    let byte = load_u8(base + 2u + (inner >> 2u));
    let code = (byte >> ((inner & 3u) * 2u)) & 3u;
    return (f32(code) - 1.0) * scale;
}}

@compute @workgroup_size(8, 8)
fn bq_matmul_{suf}(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) lin: u32,
) {{
    let row = wid.y * 8u + lid.y;
    let col = wid.x * 8u + lid.x;
    let pref = wid.z;
    let in_range = row < params.m && col < params.n && pref < params.prefix;
    let pref_bytes = pref * params.b_prefix_stride;
    var acc = 0.0;
    for (var kb = 0u; kb < params.k; kb += 32u) {{
        for (var i = 0u; i < 4u; i++) {{
            let elem = lin + i * 64u;
            let local = elem / 32u;
            let k_local = elem % 32u;
            let gk = kb + k_local;
            let gm = wid.y * 8u + local;
            let gn = wid.x * 8u + local;
            var av = 0.0;
            if (gm < params.m && gk < params.k) {{
                let ix = params.a_off + pref * params.a_prefix_stride + gm * params.a_stride_m + gk * params.a_stride_k;
                av = f32(a[ix]);
            }}
            a_tile[elem] = av;
            var bv = 0.0;
            if (gn < params.n && gk < params.k) {{
                bv = dequant(gn, gk, pref_bytes);
            }}
            b_tile[elem] = bv;
        }}
        workgroupBarrier();
        if (in_range) {{
            for (var kk = 0u; kk < 32u; kk++) {{
                acc += a_tile[lid.y * 32u + kk] * b_tile[lid.x * 32u + kk];
            }}
        }}
        workgroupBarrier();
    }}
    if (in_range) {{
        let oe = pref * params.c_prefix_stride + row * params.c_stride_m + col * params.c_stride_n;
        var v = {t}(acc);
{epilogue}        outp[params.c_off + oe] = v;
    }}
}}
"#
    ));
    s
}

#[cfg(test)]
mod tests {
    use tract_core::dyn_clone::clone_box;
    use tract_core::internal::*;
    use tract_core::ops::einsum::EinSum;
    use tract_core::ops::einsum::prefix_matmul::PrefixMatMul;
    use tract_core::ops::konst::Const;
    use tract_core::tract_linalg::block_quant::BlockQuant;
    use tract_core::tract_linalg::block_quant::BlockQuantFact;
    use tract_core::tract_linalg::block_quant::BlockQuantStorage;
    use tract_core::tract_linalg::block_quant::Q2_0_T;
    use tract_core::tract_linalg::block_quant::Q4_0;

    use crate::ops::matmul::WgpuGemm;
    use crate::test_utils::fill;
    use crate::test_utils::gpu_vs_cpu;
    use crate::test_utils::transformed;

    fn pack(format: &dyn BlockQuant, shape: &[usize], data: &[f32]) -> TractResult<Const> {
        let k = *shape.last().unwrap();
        let rows = shape.iter().product::<usize>() / k;
        let blob = format.quant_f32(data)?;
        let storage = BlockQuantStorage::new(clone_box(format), rows, k, Arc::new(blob))?;
        let tensor = storage.into_tensor_with_shape(f32::datum_type(), shape);
        let fact = BlockQuantFact::new(clone_box(format), shape.into());
        Const::new_with_exotic_fact(Arc::new(tensor), Box::new(fact))
    }

    fn mm(transpose_a: bool, transpose_b: bool, transpose_c: bool, dt: DatumType) -> PrefixMatMul {
        PrefixMatMul {
            transpose_a,
            transpose_b,
            transpose_c,
            quantize_output: None,
            operating_dt: Some(dt),
        }
    }

    fn assert_gemm(model: &TypedModel) -> TractResult<()> {
        ensure!(model.nodes().iter().any(|n| n.op_is::<WgpuGemm>()), "matmul stayed off the GPU");
        ensure!(
            !model.nodes().iter().any(|n| n.op_is::<PrefixMatMul>()),
            "a PrefixMatMul was left for the CPU"
        );
        Ok(())
    }

    fn run_packed(
        format: &dyn BlockQuant,
        a_shape: &[usize],
        b_shape: &[usize],
        dt: DatumType,
        transpose_a: bool,
        transpose_c: bool,
        bias: bool,
    ) -> TractResult<()> {
        let mut model = TypedModel::default();
        let b_len = b_shape.iter().product::<usize>();
        let a = model.add_source("a", dt.fact(a_shape))?;
        let w = pack(format, b_shape, &fill(1, b_len))?;
        let w = model.wire_node("w", w, &[])?[0];
        let op = mm(transpose_a, true, transpose_c, dt);
        let mut y = model.wire_node("mm", op, &[a, w])?[0];
        if bias {
            let n = b_shape[b_shape.len() - 2];
            let bias_shape = if transpose_c { vec![n, 1] } else { vec![1, n] };
            let bias = model.add_const(
                "bias",
                Tensor::from_shape(&bias_shape, &fill(2, n))?.cast_to_dt(dt)?.into_owned(),
            )?;
            y = model.wire_node("bias_add", tract_core::ops::math::add(), &[y, bias])?[0];
        }
        model.select_output_outlets(&[y])?;
        let gpu = transformed(model.clone())?;
        assert_gemm(&gpu)?;
        if bias {
            let gemm = gpu.nodes().iter().find_map(|n| n.op_as::<WgpuGemm>()).unwrap();
            ensure!(!gemm.epilogue.is_empty(), "bias was not fused into the matmul");
        }
        let input = Tensor::from_shape(a_shape, &fill(4, a_shape.iter().product()))?
            .cast_to_dt(dt)?
            .into_owned();
        gpu_vs_cpu(model, input)
    }

    #[test]
    fn q4_0_f32_partial_tile_with_bias() -> TractResult<()> {
        run_packed(&Q4_0, &[13, 64], &[17, 64], f32::datum_type(), false, false, true)
    }

    #[test]
    fn q4_0_transposed_activation_and_output() -> TractResult<()> {
        run_packed(&Q4_0, &[64, 13], &[17, 64], f32::datum_type(), true, true, true)
    }

    #[test]
    fn q2_0_t_f32_group128() -> TractResult<()> {
        run_packed(&Q2_0_T, &[9, 128], &[11, 128], f32::datum_type(), false, false, false)
    }

    #[test]
    fn q2_0_t_f16_partial_tile() -> TractResult<()> {
        run_packed(&Q2_0_T, &[5, 32], &[7, 32], f16::datum_type(), false, false, false)
    }

    #[test]
    fn q4_0_batched_broadcast_weight() -> TractResult<()> {
        run_packed(&Q4_0, &[2, 13, 64], &[1, 17, 64], f32::datum_type(), false, false, false)?;
        run_packed(&Q2_0_T, &[2, 9, 32], &[2, 11, 32], f32::datum_type(), false, false, false)
    }

    #[test]
    fn q2_0_t_negative_scale_and_code3() -> TractResult<()> {
        let (n, k) = (5usize, 32usize);
        let mut codes = vec![1u8; n * k];
        for (i, code) in codes.iter_mut().take(k).enumerate() {
            *code = (i % 4) as u8;
        }
        let mut scales = vec![0.5f32; n];
        scales[0] = -1.25;
        scales[1] = 0.0;
        scales[2] = 1.0e-7;
        let blob = Q2_0_T.pack_prequantized(&codes, &scales, n, k)?;
        let storage = BlockQuantStorage::new(Box::new(Q2_0_T), n, k, Arc::new(blob))?;
        let tensor = storage.into_tensor_with_shape(f32::datum_type(), &[n, k]);
        let fact = BlockQuantFact::new(Box::new(Q2_0_T), tvec![n, k]);
        let w = Const::new_with_exotic_fact(Arc::new(tensor), Box::new(fact))?;
        let mut model = TypedModel::default();
        let a_shape = [3usize, k];
        let a = model.add_source("a", f32::fact(a_shape))?;
        let w = model.wire_node("w", w, &[])?[0];
        let y = model.wire_node("mm", mm(false, true, false, f32::datum_type()), &[a, w])?[0];
        model.select_output_outlets(&[y])?;
        assert_gemm(&transformed(model.clone())?)?;
        gpu_vs_cpu(model, Tensor::from_shape(&a_shape, &fill(9, 3 * k))?)
    }

    fn packed_on_a(format: &dyn BlockQuant) -> TractResult<()> {
        let (m, k, n) = (13usize, 64usize, 17usize);
        let mut model = TypedModel::default();
        let x = model.add_source("x", f32::fact([m, k]))?;
        let w = model.wire_node("w", pack(format, &[n, k], &fill(3, n * k))?, &[])?[0];
        let y = model.wire_node("mm", mm(false, true, true, f32::datum_type()), &[w, x])?[0];
        model.select_output_outlets(&[y])?;
        assert_gemm(&transformed(model.clone())?)?;
        gpu_vs_cpu(model, Tensor::from_shape(&[m, k], &fill(8, m * k))?)
    }

    #[test]
    fn q4_0_on_a_moves_to_b() -> TractResult<()> {
        packed_on_a(&Q4_0)
    }

    #[test]
    fn q2_0_t_on_a_moves_to_b() -> TractResult<()> {
        packed_on_a(&Q2_0_T)
    }

    #[test]
    fn einsum_nk_weight_runs_on_gpu() -> TractResult<()> {
        let (m, k, n) = (13usize, 64usize, 17usize);
        let mut model = TypedModel::default();
        let a = model.add_source("a", f32::fact([m, k]))?;
        let w = pack(&Q2_0_T, &[n, k], &fill(3, n * k))?;
        let w = model.wire_node("w", w, &[])?[0];
        let op = EinSum::new("mk,nk->mn".parse()?, f32::datum_type());
        let y = model.wire_node("mm", op, &[a, w])?[0];
        model.select_output_outlets(&[y])?;
        assert_gemm(&transformed(model.clone())?)?;
        gpu_vs_cpu(model, Tensor::from_shape(&[m, k], &fill(8, m * k))?)
    }
}
