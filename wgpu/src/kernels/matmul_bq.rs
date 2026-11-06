//! `Q4_0` and ternary `Q2_0_T` matmul. B is K-inner. The shader dequants one
//! block of 32 into f32 and accumulates there.
//!
//! The uploaded buffer is not the host interleave. Scale words come first, one
//! little-endian f16 in the low half of a `u32` per run of `group` blocks, then
//! every block's code bytes. `group` is 4 when each row's blocks, taken four at
//! a time, share one scale bit pattern, and 1 otherwise. A run does not cross a
//! row. Host facts stay [`BlockQuantFact`].

use tract_core::internal::*;
use tract_core::tract_linalg::block_quant::BlockQuant;
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

/// Uniform block shared by both kernels. Twenty scalars, then two `vec4`s:
/// a `vec4` in a uniform buffer starts on a 16-byte boundary.
const BQ_PARAMS: &str = "
struct Params {
    m: u32,
    k: u32,
    n: u32,
    prefix: u32,
    a_off: u32,
    a_prefix_stride: u32,
    a_stride_m: u32,
    a_stride_k: u32,
    scale_off: u32,
    scale_prefix: u32,
    blocks_per_row: u32,
    code_off: u32,
    format: u32,
    c_off: u32,
    c_prefix_stride: u32,
    c_stride_m: u32,
    c_stride_n: u32,
    code_prefix: u32,
    bq_group: u32,
    _pad: u32,
    off_extra: vec4<u32>,
    mode_extra: vec4<u32>,
}
";

/// Uploaded `Q4_0` or `Q2_0_T` weight.
///
/// `inner` is the host fact. `group` is 4 when every in-row run of four blocks
/// repeats one scale bit pattern, and 1 otherwise. The buffer is one scale word
/// per run, low 16 bits little-endian f16, then each block's code bytes.
#[derive(Clone, Hash, PartialEq, Eq)]
pub(crate) struct WgpuBqFact {
    pub(crate) inner: BlockQuantFact,
    pub(crate) group: u32,
}

impl std::fmt::Debug for WgpuBqFact {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "WgpuBq({:?}, group={})", self.inner, self.group)
    }
}

impl ExoticFact for WgpuBqFact {
    fn buffer_sizes(&self) -> TVec<TDim> {
        tvec!(packed_len(&self.inner, self.group).to_dim())
    }
}

impl WgpuBqFact {
    /// Host interleave `[scale:2][codes]` for this uploaded buffer.
    pub(crate) fn host_bytes(&self, packed: &[u8]) -> TractResult<Vec<u8>> {
        let (rows, blocks_per_row, block_bytes) = geometry(&self.inner)?;
        unpack_bq(packed, block_bytes, blocks_per_row, rows, self.group)
    }
}

fn is_repackable(format: &dyn BlockQuant) -> bool {
    format.block_len() == 32 && (format.dyn_eq(&Q4_0) || format.dyn_eq(&Q2_0_T))
}

fn geometry(fact: &BlockQuantFact) -> TractResult<(usize, usize, usize)> {
    let block_len = fact.format.block_len();
    let block_bytes = fact.format.block_bytes();
    ensure!(block_len == 32, "block length is {block_len}");
    ensure!(fact.k().is_multiple_of(block_len), "K {} is not a multiple of {block_len}", fact.k());
    ensure!(block_bytes >= 2 && (block_bytes - 2).is_multiple_of(4), "code bytes are not words");
    let blocks_per_row = fact.k() / block_len;
    Ok((fact.m(), blocks_per_row, block_bytes))
}

fn packed_len(fact: &BlockQuantFact, group: u32) -> usize {
    let Ok((rows, blocks_per_row, block_bytes)) = geometry(fact) else {
        return 0;
    };
    let group = group.max(1) as usize;
    let total = rows * blocks_per_row;
    total.div_ceil(group) * 4 + total * (block_bytes - 2)
}

/// Repacks host `Q4_0` / `Q2_0_T` bytes for the wgpu kernels.
///
/// `shape` is the weight, K last. Returns the device bytes and `group`.
pub(crate) fn repack_weight(
    bytes: &[u8],
    format: &dyn BlockQuant,
    shape: &[usize],
) -> TractResult<(Vec<u8>, u32)> {
    ensure!(is_repackable(format), "only Q4_0 and Q2_0_T are repacked");
    let k = *shape.last().context("block-quant weight has no K")?;
    let block_len = format.block_len();
    ensure!(k.is_multiple_of(block_len), "K {k} is not a multiple of {block_len}");
    let rows = shape.iter().product::<usize>().checked_div(k).unwrap_or(0);
    repack_bq(bytes, format.block_bytes(), k / block_len, rows)
}

fn repack_bq(
    bytes: &[u8],
    block_bytes: usize,
    blocks_per_row: usize,
    rows: usize,
) -> TractResult<(Vec<u8>, u32)> {
    let code_bytes = block_bytes - 2;
    ensure!(block_bytes >= 2 && code_bytes.is_multiple_of(4), "code bytes are not words");
    let total_blocks = blocks_per_row * rows;
    ensure!(
        bytes.len() >= total_blocks * block_bytes,
        "block-quant blob is {} bytes, {total_blocks} blocks need {}",
        bytes.len(),
        total_blocks * block_bytes
    );
    let scale_at = |i: usize| -> u16 {
        let off = i * block_bytes;
        u16::from_le_bytes([bytes[off], bytes[off + 1]])
    };
    let group = if blocks_per_row.is_multiple_of(4)
        && (0..rows).all(|r| {
            (0..blocks_per_row).step_by(4).all(|c| {
                let base = r * blocks_per_row + c;
                let s0 = scale_at(base);
                (1..4).all(|j| scale_at(base + j) == s0)
            })
        }) {
        4u32
    } else {
        1
    };
    let scale_words = total_blocks.div_ceil(group as usize);
    let mut out = vec![0u8; scale_words * 4 + total_blocks * code_bytes];
    for i in 0..total_blocks {
        if i % group as usize == 0 {
            let w = i / group as usize * 4;
            out[w..w + 2].copy_from_slice(&scale_at(i).to_le_bytes());
        }
        let src = i * block_bytes + 2;
        let dst = scale_words * 4 + i * code_bytes;
        out[dst..dst + code_bytes].copy_from_slice(&bytes[src..src + code_bytes]);
    }
    Ok((out, group))
}

fn unpack_bq(
    bytes: &[u8],
    block_bytes: usize,
    blocks_per_row: usize,
    rows: usize,
    group: u32,
) -> TractResult<Vec<u8>> {
    ensure!(group == 1 || group == 4, "scale group is {group}");
    let code_bytes = block_bytes - 2;
    let total_blocks = blocks_per_row * rows;
    let scale_words = total_blocks.div_ceil(group as usize);
    let need = scale_words * 4 + total_blocks * code_bytes;
    ensure!(bytes.len() >= need, "repacked buffer is {} bytes, layout needs {need}", bytes.len());
    let mut out = vec![0u8; total_blocks * block_bytes];
    for i in 0..total_blocks {
        let sw = i / group as usize * 4;
        out[i * block_bytes..i * block_bytes + 2].copy_from_slice(&bytes[sw..sw + 2]);
        let src = scale_words * 4 + i * code_bytes;
        let dst = i * block_bytes + 2;
        out[dst..dst + code_bytes].copy_from_slice(&bytes[src..src + code_bytes]);
    }
    Ok(out)
}

/// `Q4_0` or `Q2_0_T` on this tensor, owned or viewed.
pub(crate) fn is_block_quant_weight(b: &DeviceTensor) -> bool {
    bq_fact(b).is_some()
}

fn bq_fact(t: &DeviceTensor) -> Option<WgpuBqFact> {
    let fact = match t {
        DeviceTensor::Owned(t) => t.exotic_fact(),
        DeviceTensor::View(t) => t.exotic_fact(),
    }?;
    let wgpu = fact.downcast_ref::<WgpuBqFact>()?;
    if is_repackable(&*wgpu.inner.format) && (wgpu.group == 1 || wgpu.group == 4) {
        Some(wgpu.clone())
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
/// The weight is the [`WgpuBqFact`] buffer: scale words, then code bytes.
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
        ensure!(fact.inner.k() == k, "packed K is {}, matmul K is {k}", fact.inner.k());
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
        let (packed_rows, fact_blocks_per_row, block_bytes) = geometry(&fact.inner)?;
        ensure!(
            fact_blocks_per_row == blocks_per_row,
            "packed blocks per row {fact_blocks_per_row} do not match K {k}"
        );
        let shape_rows = b.shape()[..rank - 1].iter().product::<usize>();
        ensure!(
            packed_rows == shape_rows,
            "packed rows {packed_rows} do not match the weight shape"
        );
        ensure!(
            b.buffer_offset::<usize>() == 0,
            "block-quant matmul reads a weight at the start of its buffer"
        );
        let group = fact.group;
        ensure!(group == 1 || blocks_per_row.is_multiple_of(group as usize));
        let words_per_block = (block_bytes - 2) / 4;
        let rows_per_prefix = if b_elem_prefix == 0 {
            0
        } else {
            ensure!(k != 0 && (b_elem_prefix as usize).is_multiple_of(k));
            b_elem_prefix as usize / k
        };
        let scale_prefix = if rows_per_prefix == 0 {
            0
        } else {
            u32_of(rows_per_prefix * blocks_per_row / group as usize)?
        };
        let code_prefix = if rows_per_prefix == 0 {
            0
        } else {
            u32_of(rows_per_prefix * blocks_per_row * words_per_block)?
        };
        let total_blocks = packed_rows * blocks_per_row;
        let scale_off = 0u32;
        let code_off = u32_of(total_blocks.div_ceil(group as usize))?;

        let format = if fact.inner.format.dyn_eq(&Q4_0) { 0 } else { 1 };
        let a_off = u32_of(element_offset(a, 0))?;
        let tiled = dt == ShaderDtype::F32
            && a_stride_k == 1
            && a_stride_m.is_multiple_of(4)
            && a_off.is_multiple_of(4)
            && a_prefix.is_multiple_of(4);
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
            scale_off,
            scale_prefix,
            u32_of(blocks_per_row)?,
            code_off,
            format,
            u32_of(element_offset(output, 0))?,
            c_prefix,
            c_stride_m,
            c_stride_n,
            code_prefix,
            group,
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
                .map(|w| format!("        let w{j}_{w} = b[cw{j} + {w}u];\n"))
                .collect::<String>();
            format!(
                "        let bi{j} = n{j} * params.blocks_per_row + block;
        let s{j} = unpack2x16float(b[scale_base + bi{j} / params.bq_group]).x;
        let cw{j} = code_base + bi{j} * {words}u;
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
    let params = BQ_PARAMS;
    let mut s = preamble(dt);
    s.push_str(&format!(
        r#"
{params}
@group(0) @binding(0) var<storage, read> a4: array<vec4<f32>>;
@group(0) @binding(1) var<storage, read> b: array<u32>;
{extra_bindings}@group(0) @binding({out_binding}) var<storage, read_write> outp: array<{t}>;
@group(0) @binding({uniform_binding}) var<uniform> params: Params;
{lib}
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
    let scale_base = params.scale_off + pref * params.scale_prefix;
    let code_base = params.code_off + pref * params.code_prefix;
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
    let params = BQ_PARAMS;
    let mut s = preamble(dt);
    s.push_str(&format!(
        r#"
{params}
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

fn dequant(n_row: u32, kk: u32, pref: u32) -> f32 {{
    let block = kk >> 5u;
    let inner = kk & 31u;
    let bi = n_row * params.blocks_per_row + block;
    let scale_word = b[params.scale_off + pref * params.scale_prefix + bi / params.bq_group];
    let scale = f16_bits_to_f32(scale_word & 0xffffu);
    let words = select(2u, 4u, params.format == 0u);
    let code_byte = (params.code_off + pref * params.code_prefix + bi * words) * 4u;
    if (params.format == 0u) {{
        // ggml Q4_0: byte j holds position j in the low nibble and j+16 in the high nibble.
        let byte = load_u8(code_byte + (inner & 15u));
        let nib = select(byte >> 4u, byte & 15u, inner < 16u);
        return (f32(nib) - 8.0) * scale;
    }}
    let byte = load_u8(code_byte + (inner >> 2u));
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
                bv = dequant(gn, gk, pref);
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
    use tract_gpu::fact::DeviceFact;
    use tract_gpu::tensor::DeviceTensor;
    use tract_gpu::tensor::DeviceTensorExt;

    use super::WgpuBqFact;
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
    fn rank4_broadcast_weight() -> TractResult<()> {
        run_packed(
            &Q2_0_T,
            &[2, 3, 13, 64],
            &[1, 1, 17, 64],
            f32::datum_type(),
            false,
            false,
            false,
        )?;
        run_packed(&Q4_0, &[2, 3, 13, 64], &[1, 1, 17, 64], f16::datum_type(), false, false, false)
    }

    #[test]
    fn rank4_and_5_batched_weight() -> TractResult<()> {
        run_packed(
            &Q2_0_T,
            &[2, 3, 9, 32],
            &[2, 3, 11, 32],
            f32::datum_type(),
            false,
            false,
            false,
        )?;
        run_packed(
            &Q4_0,
            &[2, 2, 2, 32, 9],
            &[2, 2, 2, 11, 32],
            f32::datum_type(),
            true,
            true,
            false,
        )
    }

    #[test]
    fn rank4_partly_broadcast_weight_stays_on_cpu() -> TractResult<()> {
        let (a_shape, b_shape) = ([2usize, 3, 9, 32], [1usize, 3, 11, 32]);
        let mut model = TypedModel::default();
        let a = model.add_source("a", f32::fact(a_shape))?;
        let w = model.wire_node("w", pack(&Q2_0_T, &b_shape, &fill(1, 3 * 11 * 32))?, &[])?[0];
        let y = model.wire_node("mm", mm(false, true, false, f32::datum_type()), &[a, w])?[0];
        model.select_output_outlets(&[y])?;
        let gpu = transformed(model.clone())?;
        ensure!(
            !gpu.nodes().iter().any(|n| n.op_is::<WgpuGemm>()),
            "partial broadcast went to the GPU"
        );
        gpu_vs_cpu(model, Tensor::from_shape(&a_shape, &fill(4, 2 * 3 * 9 * 32))?)
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

    fn q2_interleaved(scales: &[u16]) -> Vec<u8> {
        let mut bytes = Vec::new();
        for (i, scale) in scales.iter().enumerate() {
            bytes.extend_from_slice(&scale.to_le_bytes());
            bytes.extend(std::iter::repeat_n(i as u8, 8));
        }
        bytes
    }

    #[test]
    fn repack_keeps_groups_inside_rows() -> TractResult<()> {
        let blocks_per_row = 8usize;
        let scales = [1u16, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3, 4, 4, 4, 4];
        let bytes = q2_interleaved(&scales);
        let (packed, group) = super::repack_bq(&bytes, 10, blocks_per_row, 2)?;
        ensure!(group == 4, "group is {group}");
        ensure!(packed.len() == 16 + 16 * 8, "packed len {}", packed.len());
        ensure!(u16::from_le_bytes([packed[0], packed[1]]) == 1);
        ensure!(u16::from_le_bytes([packed[4], packed[5]]) == 2);
        ensure!(u16::from_le_bytes([packed[8], packed[9]]) == 3);
        ensure!(u16::from_le_bytes([packed[12], packed[13]]) == 4);
        ensure!(packed[16] == 0 && packed[24] == 1 && packed[16 + 15 * 8] == 15);
        ensure!(super::unpack_bq(&packed, 10, blocks_per_row, 2, group)? == bytes);

        let mut broken = scales;
        broken[7] = 9;
        let bytes = q2_interleaved(&broken);
        let (packed, group) = super::repack_bq(&bytes, 10, blocks_per_row, 2)?;
        ensure!(group == 1, "a broken run of four stayed group {group}");
        ensure!(super::unpack_bq(&packed, 10, blocks_per_row, 2, group)? == bytes);

        let same = [7u16; 12];
        let bytes = q2_interleaved(&same);
        let (packed, group) = super::repack_bq(&bytes, 10, 6, 2)?;
        ensure!(group == 1, "a row that is not a multiple of 4 became group {group}");
        ensure!(super::unpack_bq(&packed, 10, 6, 2, group)? == bytes);
        Ok(())
    }

    fn assert_uploaded(model: &TypedModel, original: &[u8], group: u32) -> TractResult<()> {
        let mut found = false;
        for node in model.nodes() {
            let Some(op) = node.op_as::<Const>() else { continue };
            let Some(device) = op.val().as_device_tensor() else { continue };
            let stored = match device {
                DeviceTensor::Owned(t) => t.exotic_fact(),
                DeviceTensor::View(t) => t.exotic_fact(),
            };
            let Some(wq) = stored.and_then(|f| f.downcast_ref::<WgpuBqFact>()) else { continue };
            let host = op
                .exotic_fact()
                .and_then(|f| f.downcast_ref::<DeviceFact>())
                .context("weight const is not a device fact")?;
            ensure!(
                host.fact
                    .exotic_fact()
                    .is_some_and(|e| e.downcast_ref::<BlockQuantFact>().is_some()),
                "graph fact is no longer BlockQuantFact"
            );
            ensure!(wq.group == group, "scale group is {}, wanted {group}", wq.group);
            let back = device.to_host()?;
            let got = back
                .storage_as::<BlockQuantStorage>()
                .context("readback is not block-quant")?
                .value()
                .as_bytes();
            ensure!(got == original, "readback bytes differ from the host weight");
            found = true;
        }
        ensure!(found, "no repacked weight");
        Ok(())
    }

    fn run_prequantized(
        format: &dyn BlockQuant,
        blob: Blob,
        a_shape: &[usize],
        b_shape: &[usize],
        dt: DatumType,
        expect_group: u32,
    ) -> TractResult<()> {
        let k = *b_shape.last().unwrap();
        let rows = b_shape.iter().product::<usize>() / k;
        let original = blob.as_bytes().to_vec();
        let storage = BlockQuantStorage::new(clone_box(format), rows, k, Arc::new(blob))?;
        let tensor = storage.into_tensor_with_shape(f32::datum_type(), b_shape);
        let fact = BlockQuantFact::new(clone_box(format), b_shape.into());
        let w = Const::new_with_exotic_fact(Arc::new(tensor), Box::new(fact))?;
        let mut model = TypedModel::default();
        let a = model.add_source("a", dt.fact(a_shape))?;
        let w = model.wire_node("w", w, &[])?[0];
        let y = model.wire_node("mm", mm(false, true, false, dt), &[a, w])?[0];
        model.select_output_outlets(&[y])?;
        let gpu = transformed(model.clone())?;
        assert_gemm(&gpu)?;
        assert_uploaded(&gpu, &original, expect_group)?;
        let input = Tensor::from_shape(a_shape, &fill(4, a_shape.iter().product()))?
            .cast_to_dt(dt)?
            .into_owned();
        gpu_vs_cpu(model, input)
    }

    #[test]
    fn q2_0_t_repeated_scale_is_group4() -> TractResult<()> {
        let (n, k) = (5usize, 256usize);
        let blocks = k / 32;
        let codes: Vec<u8> = (0..n * k).map(|i| (i % 4) as u8).collect();
        let mut scales = vec![0f32; n * blocks];
        for row in 0..n {
            for blk in 0..blocks {
                scales[row * blocks + blk] = if blk < 4 { -1.25 } else { 0.5 + row as f32 };
            }
        }
        let blob = Q2_0_T.pack_prequantized(&codes, &scales, n, k)?;
        run_prequantized(&Q2_0_T, blob, &[7, k], &[n, k], f32::datum_type(), 4)
    }

    #[test]
    fn q2_0_t_distinct_scale_is_group1() -> TractResult<()> {
        let (n, k) = (4usize, 128usize);
        let blocks = k / 32;
        let codes: Vec<u8> = (0..n * k).map(|i| (i % 4) as u8).collect();
        let scales: Vec<f32> = (0..n * blocks).map(|i| 0.25 * (i + 1) as f32).collect();
        let blob = Q2_0_T.pack_prequantized(&codes, &scales, n, k)?;
        run_prequantized(&Q2_0_T, blob, &[3, k], &[n, k], f32::datum_type(), 1)
    }

    #[test]
    fn q4_0_repeated_scale_f16_is_group4() -> TractResult<()> {
        let (n, k) = (6usize, 128usize);
        let blocks = k / 32;
        let codes: Vec<u8> = (0..n * k).map(|i| (i % 16) as u8).collect();
        let scales = (0..n * blocks)
            .map(|i| if i / blocks % 2 == 0 { 0.5 } else { -2.0 })
            .collect::<Vec<_>>();
        let blob = Q4_0.pack_prequantized(&codes, &scales, n, k)?;
        run_prequantized(&Q4_0, blob, &[5, k], &[n, k], f16::datum_type(), 4)
    }

    #[test]
    fn q2_0_t_group4_broadcast_and_batched() -> TractResult<()> {
        let (n, k) = (5usize, 128usize);
        let blocks = k / 32;
        let codes: Vec<u8> = (0..n * k).map(|i| (i % 4) as u8).collect();
        let scales = vec![0.5f32; n * blocks];
        let blob = Q2_0_T.pack_prequantized(&codes, &scales, n, k)?;
        run_prequantized(&Q2_0_T, blob, &[2, 3, k], &[1, n, k], f32::datum_type(), 4)?;

        let rows = 2 * n;
        let codes: Vec<u8> = (0..rows * k).map(|i| (i % 4) as u8).collect();
        let scales =
            (0..rows * blocks).map(|i| if i / blocks < n { -0.5 } else { 1.5 }).collect::<Vec<_>>();
        let blob = Q2_0_T.pack_prequantized(&codes, &scales, rows, k)?;
        run_prequantized(&Q2_0_T, blob, &[2, 3, k], &[2, n, k], f32::datum_type(), 4)
    }
}
