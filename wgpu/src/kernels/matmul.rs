//! Prefix-broadcasting GEMM, `PrefixMatMul`'s kernels.
//!
//! The blocked and split-k kernels serve the aligned channels-first products they
//! were tuned on (every transposition set, a unit prefix, a weight `b`); the
//! tiled kernel takes any other product big enough to fill its tiles; an `m`
//! or an `n` of 1 is a GEMV; what is left runs one thread per output element.

use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

use crate::context::{RepackKey, RepackLayout};
use crate::kernels::shaders::{
    AT8, ChainStep, EntryPoint, GRID_LIMIT, LayoutKind, ModuleKey, ModuleKind, PipelineKey,
    ShaderDtype, WORKGROUP, binary_ops_wgsl, epilogue_body_at, keys_for, pack_u32s, preamble,
    rpad8_dims, rpad8_strides, unary_ops_wgsl,
};
use crate::kernels::tiled_gemm::{
    MATMUL_TILE, TILED_THREADS, tile_rows, tiled_gemm_core_vec4, tiled_gemm_tiles,
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
pub(crate) fn gemm_epilogue_mode(
    extra: &DeviceTensor,
    n: usize,
    output: &DeviceTensor,
) -> TractResult<u32> {
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
    let key = RepackKey {
        buffer: get_wgpu_buffer(b).id,
        offset: element_offset(b, 0),
        k,
        n,
        layout: RepackLayout::Transposed,
    };
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
/// four-wide vectors, so the tile must divide the shape and the fastest axis of each
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
    is_supported_dt(a.datum_type())
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

/// Whether the lane-parallel `gemv_kc` kernel applies: `b` contiguous along `k`
/// with a few rows of `a` and `k` of at least `MATMUL_GEMV_KC_MIN_K`, so a slice of
/// lanes walks it side by side for one output column — or, an `m = 1` product too thin to fill the plain kernel's
/// grid, whatever stride `b` has, since a grid of lane reductions beats a
/// handful of serial workgroups.
fn gemv_kc_applies(
    m: usize,
    k: usize,
    n: usize,
    prefix: usize,
    gemv_small: bool,
    b_k_stride: isize,
) -> bool {
    !gemv_small
        && prefix > 0
        && n > 0
        && ((1..=MATMUL_GEMV_KC_MAX_M).contains(&m) && b_k_stride == 1 && k >= MATMUL_GEMV_KC_MIN_K
            || (m == 1 && k >= 256 && prefix * n.div_ceil(MATMUL_GEMV_WG as usize) < 16))
}

/// Whether the four-rows-a-thread `gemv_n` kernel applies: a vec4 of `a` is one
/// load and a vec4 of the output one store, which needs the `m` axis
/// contiguous and four-aligned on both sides. One staging pass of `b` covers
/// `k`, so `k` may not exceed the workgroup.
fn gemv_n_v4_applies(
    prefix: usize,
    m: usize,
    k: usize,
    a: &DeviceTensor,
    output: &DeviceTensor,
    transpose_a: bool,
    transpose_c: bool,
) -> bool {
    let a_m_stride = a.strides()[a.rank() - if transpose_a { 1 } else { 2 }];
    let a_k_stride = a.strides()[a.rank() - if transpose_a { 2 } else { 1 }];
    let o_m_stride = output.strides()[output.rank() - if transpose_c { 1 } else { 2 }];
    prefix == 1
        && k <= MATMUL_GEMV_N_WG as usize
        && a_m_stride == 1
        && a_k_stride % 4 == 0
        && o_m_stride == 1
        && m.is_multiple_of(4)
        && element_offset(a, 0).is_multiple_of(4)
        && element_offset(output, 0).is_multiple_of(4)
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

pub(crate) fn apple_webgpu(_device: &wgpu::Device) -> bool {
    #[cfg(target_arch = "wasm32")]
    if let Some(raw) = _device.as_webgpu()
        && let Ok(info) =
            js_sys::Reflect::get(raw.as_ref(), &wasm_bindgen::JsValue::from_str("adapterInfo"))
    {
        let lookup = |key| js_sys::Reflect::get(&info, &wasm_bindgen::JsValue::from_str(key)).ok();
        return lookup("vendor").and_then(|v| v.as_string()).as_deref() == Some("apple")
            && lookup("isFallbackAdapter").and_then(|v| v.as_bool()) != Some(true);
    }
    false
}

pub fn wgpu_matmul_dispatch(
    t: Transposes,
    epilogue: &[ChainStep],
    a: &DeviceTensor,
    b: &DeviceTensor,
    extras: &[&DeviceTensor],
    output: &DeviceTensor,
    f16_accumulation: bool,
) -> TractResult<()> {
    wgpu_matmul_dispatch_impl(t, epilogue, a, b, extras, output, f16_accumulation, None, None)
}

#[allow(clippy::too_many_arguments)]
fn wgpu_matmul_dispatch_impl(
    t: Transposes,
    epilogue: &[ChainStep],
    a: &DeviceTensor,
    b: &DeviceTensor,
    extras: &[&DeviceTensor],
    output: &DeviceTensor,
    f16_accumulation: bool,
    small_gemv_override: Option<bool>,
    column_first_override: Option<bool>,
) -> TractResult<()> {
    if crate::kernels::matmul_bq::is_block_quant_weight(b) {
        return crate::kernels::matmul_bq::wgpu_bq_dispatch(t, epilogue, a, b, extras, output);
    }
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
        if crate::kernels::subgroup::try_dispatch(
            q,
            t,
            epilogue,
            a,
            b,
            extras,
            output,
            dt,
            f16_accumulation,
            m,
            k,
            n,
            prefix,
            out_shape,
        )? {
            return Ok(());
        }
        // The other kernels tile `m`, so a single row never reaches them.
        let gemv = m == 1 && prefix > 0 && n > 0;
        // The other kernels tile `n`, so a single column never reaches them.
        let gemv_n = n == 1 && prefix > 0 && m > 0;

        if blocked_applies(prefix, m, k, n, a, b, output, t) {
            let packed = repacked_b(q, b, k, n)?;
            let split_k = m < MATMUL_SPLITK_MAX_M;
            let f16_acc = dt == ShaderDtype::F16 && f16_accumulation;
            let column_first = split_k
                && f16_acc
                && matches!((m, k, n), (920, 160, 960) | (920, 960, 160))
                && column_first_override
                    .unwrap_or_else(|| apple_splitk_device(q.context().device()));
            let splitk_wg = if column_first { 64 } else { MATMUL_SPLITK_WG };
            let splitk_name = if column_first { "matmul_splitk_col_fast" } else { "matmul_splitk" };
            let pipeline = if split_k {
                q.context().chain_pipeline(
                    (splitk_name, dt, f16_acc, epilogue, extras.len()),
                    layout,
                    EntryPoint::typed(splitk_name, dt),
                    || {
                        if column_first {
                            matmul_splitk_module_for_layout(
                                dt,
                                f16_acc,
                                epilogue,
                                extras.len(),
                                true,
                            )
                        } else {
                            matmul_splitk_module(dt, f16_acc, epilogue, extras.len())
                        }
                    },
                )?
            } else {
                q.context().chain_pipeline(
                    ("matmul_blocked", dt, f16_acc, epilogue, extras.len()),
                    layout,
                    EntryPoint::typed("matmul_blocked", dt),
                    || matmul_blocked_module(dt, f16_acc, epilogue, extras.len()),
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
                (tiles * 4).div_ceil(splitk_wg as usize)
            } else {
                tiles.div_ceil(MATMUL_BLOCKED_WG as usize)
            } as u32;
            let label = if column_first { splitk_name } else { "matmul_blocked" };
            return q.dispatch_grid(label, &pipeline, &bg, dyn_off, [groups, 1, 1]);
        }

        let gemv_small = gemv
            && dt == ShaderDtype::F16
            && prefix == 1
            && matches!((k, n), (480, 120) | (672, 168) | (960, 240))
            && !t.a
            && t.b
            && !t.c
            && a.strides()[a.rank() - 1] == 1
            && b.strides()[b.rank() - 1] == 1
            && b.strides()[b.rank() - 2] == k as isize
            && output.strides()[output.rank() - 1] == 1
            && element_offset(a, 0).is_multiple_of(4)
            && element_offset(b, 0).is_multiple_of(4)
            && small_gemv_override.unwrap_or_else(|| apple_webgpu(q.context().device()));
        let b_k_stride = b.strides()[b.rank() - if t.b { 1 } else { 2 }];
        let gemv_kc = gemv_kc_applies(m, k, n, prefix, gemv_small, b_k_stride);
        let gemv_kc4 = gemv_kc
            && k.is_multiple_of(4)
            && vec4_along_k(a, if t.a { a.rank() - 2 } else { a.rank() - 1 })
            && vec4_along_k(b, if t.b { b.rank() - 1 } else { b.rank() - 2 });
        let gemv_kc4b = gemv_kc
            && !gemv_kc4
            && k.is_multiple_of(4)
            && vec4_along_k(b, if t.b { b.rank() - 1 } else { b.rank() - 2 });
        let gemv = gemv || gemv_kc;
        let gemv_name = if gemv_small {
            "matmul_gemv_small"
        } else if gemv_kc4 {
            "matmul_gemv_kc4"
        } else if gemv_kc4b {
            "matmul_gemv_kc4b"
        } else if gemv_kc {
            "matmul_gemv_kc"
        } else {
            "matmul_gemv"
        };
        let gemv_n_v4 =
            gemv_n && gemv_n_v4_applies(prefix, m, k, a, output, transpose_a, transpose_c);
        let gemv_n_name = if gemv_n_v4 { "matmul_gemv_n_v4" } else { "matmul_gemv_n" };
        let tiled = tiled_applies(prefix, m, k, n);
        let tile_m = tile_rows(m);
        let pipeline = if gemv {
            q.context().chain_pipeline(
                (gemv_name, dt, epilogue, extras.len()),
                layout,
                EntryPoint::typed(gemv_name, dt),
                || {
                    let variant = if gemv_small {
                        GemvVariant::Small
                    } else if gemv_kc4 {
                        GemvVariant::LaneParallelVec4
                    } else if gemv_kc4b {
                        GemvVariant::LaneParallelVec4B
                    } else if gemv_kc {
                        GemvVariant::LaneParallel
                    } else {
                        GemvVariant::Staged
                    };
                    matmul_gemv_module_for_layout(dt, epilogue, extras.len(), variant)
                },
            )?
        } else if gemv_n {
            q.context().chain_pipeline(
                (gemv_n_name, dt, epilogue, extras.len()),
                layout,
                EntryPoint::typed(gemv_n_name, dt),
                || matmul_gemv_n_module_for_layout(dt, epilogue, extras.len(), gemv_n_v4),
            )?
        } else if tiled {
            q.context().chain_pipeline(
                ("matmul_tiled", tile_m, dt, epilogue, extras.len()),
                LayoutKind::Chain(5 + extras.len() as u8),
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
            let columns = if gemv_small {
                MATMUL_GEMV_SMALL_COLS
            } else if gemv_kc {
                MATMUL_GEMV_KC_COLS
            } else {
                MATMUL_GEMV_WG
            };
            let tiles = prefix * n.div_ceil(columns as usize);
            let x = tiles.min(GRID_LIMIT as usize) as u32;
            let y = tiles.div_ceil(GRID_LIMIT as usize) as u32;
            return q.dispatch_grid(gemv_name, &pipeline, &bg, dyn_off, [x, y, 1]);
        }
        if gemv_n {
            let rows =
                if gemv_n_v4 { 4 * MATMUL_GEMV_N_WG as usize } else { MATMUL_GEMV_N_WG as usize };
            let tiles = prefix * m.div_ceil(rows);
            let x = tiles.min(GRID_LIMIT as usize) as u32;
            let y = tiles.div_ceil(GRID_LIMIT as usize) as u32;
            return q.dispatch_grid(gemv_n_name, &pipeline, &bg, dyn_off, [x, y, 1]);
        }
        if tiled {
            let mut tiled_buffers: Vec<&crate::context::WgpuBuffer> = vec![
                get_wgpu_buffer(a),
                get_wgpu_buffer(b),
                get_wgpu_buffer(a),
                get_wgpu_buffer(b),
            ];
            tiled_buffers.extend(extras.iter().map(|t| get_wgpu_buffer(t)));
            tiled_buffers.push(get_wgpu_buffer(output));
            let tiled_bg = q.context().bind_group(
                LayoutKind::Chain(5 + extras.len() as u8),
                &tiled_buffers,
                q.uniform(),
            )?;
            let grid = [n.div_ceil(MATMUL_TILE) as u32, m.div_ceil(tile_m) as u32, prefix as u32];
            return q.dispatch_grid("matmul_tiled", &pipeline, &tiled_bg, dyn_off, grid);
        }
        q.dispatch("matmul", &pipeline, &bg, dyn_off, (prefix * m * n) as u64)
    })
}

/// Threads per workgroup of the blocked matmul.
pub const MATMUL_BLOCKED_WG: u32 = 192;

/// Columns one `m = 1` workgroup owns, and the `k` slice it stages for them.
pub const MATMUL_GEMV_WG: u32 = 64;
const MATMUL_GEMV_SMALL_COLS: u32 = 8;
/// Columns one workgroup of the lane-parallel GEMV kernel owns, each reduced by
/// `MATMUL_GEMV_KC_LANES` lanes.
const MATMUL_GEMV_KC_COLS: u32 = 8;
const MATMUL_GEMV_KC_LANES: u32 = 32;
/// Rows of `a` the lane-parallel GEMV kernel serves, each `b` load feeding all of them.
const MATMUL_GEMV_KC_MAX_M: usize = 8;
/// Shortest `k` the lane-parallel GEMV kernel takes for a `b` contiguous along `k`: below it
/// a product is too small for coalescing to pay for the per-column reduction.
const MATMUL_GEMV_KC_MIN_K: usize = 128;

/// The m = 1 (or few rows) kernels, by how they walk `k`.
#[derive(Clone, Copy, PartialEq, Eq)]
enum GemvVariant {
    /// Stages `a` through workgroup memory, one thread per column.
    Staged,
    /// Apple's aligned unit-prefix kernel for a few measured shapes.
    Small,
    /// Lanes walking `k` side by side, one column per slice of lanes.
    LaneParallel,
    /// As `LaneParallel`, loading four `k` steps of `a` and `b` at once.
    LaneParallelVec4,
    /// As `LaneParallelVec4` for `b` alone, reading `a` through its strides.
    LaneParallelVec4B,
}

/// Whether `t` reads as `vec4`s along its `k_axis`: contiguous along it, with every other
/// stride and its offset whole multiples of four elements.
fn vec4_along_k(t: &DeviceTensor, k_axis: usize) -> bool {
    t.strides()[k_axis] == 1
        && element_offset(t, 0).is_multiple_of(4)
        && t.strides().iter().enumerate().all(|(ax, s)| ax == k_axis || s % 4 == 0)
}

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
    matmul_gemv_module_for_layout(dt, epilogue, extras, GemvVariant::Staged)
}

fn matmul_gemv_module_for_layout(
    dt: ShaderDtype,
    epilogue: &[ChainStep],
    extras: usize,
    variant: GemvVariant,
) -> String {
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
    let vec4_t = format!("vec4<{t}>");
    let a_t = if matches!(variant, GemvVariant::Small | GemvVariant::LaneParallelVec4) {
        vec4_t.clone()
    } else {
        t.to_string()
    };
    let b_t = if matches!(
        variant,
        GemvVariant::Small | GemvVariant::LaneParallelVec4 | GemvVariant::LaneParallelVec4B
    ) {
        vec4_t.clone()
    } else {
        t.to_string()
    };
    let body = if variant == GemvVariant::Small {
        matmul_gemv_small_body(t, suf, &epilogue)
    } else if variant == GemvVariant::LaneParallel {
        matmul_gemv_kc_body(t, suf, &epilogue, false)
    } else if variant == GemvVariant::LaneParallelVec4 {
        matmul_gemv_kc_body(t, suf, &epilogue, true)
    } else if variant == GemvVariant::LaneParallelVec4B {
        matmul_gemv_kc4b_body(t, suf, &epilogue)
    } else {
        format!(
            r#"var<workgroup> avec: array<f32, {wg}>;

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
        )
    };
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

@group(0) @binding(0) var<storage, read> a: array<{a_t}>;
@group(0) @binding(1) var<storage, read> b: array<{b_t}>;
{extra_bindings}@group(0) @binding({out_binding}) var<storage, read_write> outp: array<{t}>;
@group(0) @binding({uniform_binding}) var<uniform> params: Params;
{lib}
{AT8}

{body}
"#
    ));
    s
}

/// Product of up to `MATMUL_GEMV_KC_MAX_M` rows of `a` by `b`: each column is
/// reduced by a slice of lanes walking `k` side by side — consecutive addresses
/// of `b` when it is contiguous along `k`, as a linear layer's weight is — then
/// summed across the slice in workgroup memory. When `k` is a multiple of 4 and both
/// operands read as `vec4`s along it, each lane loads four `k` steps at a time. Accumulates
/// in f32. Same `Params`, strides and epilogue as [`matmul_module`].
/// [`matmul_gemv_kc_body`] with `b` read as `vec4`s along `k` and `a`, which may be
/// strided along `k` (a transposed activation), as four scalars per step.
fn matmul_gemv_kc4b_body(t: &str, suf: &str, epilogue: &str) -> String {
    let walk_k = format!(
        r#"for (var k4 = lane; k4 < params.k / 4u; k4 += {lanes}u) {{
            let bv = vec4<f32>(b[b_base / 4u + k4]);
            for (var r = 0u; r < params.m; r++) {{
                let ai = a_i + r * a_m_s + k4 * 4u * a_k_s;
                let av = vec4<f32>(f32(a[ai]), f32(a[ai + a_k_s]), f32(a[ai + 2u * a_k_s]), f32(a[ai + 3u * a_k_s]));
                acc[r] += dot(av, bv);
            }}
        }}"#,
        lanes = MATMUL_GEMV_KC_LANES
    );
    matmul_gemv_kc_body_walking(t, suf, epilogue, "matmul_gemv_kc4b", &walk_k)
}

fn matmul_gemv_kc_body(t: &str, suf: &str, epilogue: &str, vec4: bool) -> String {
    let entry = if vec4 { "matmul_gemv_kc4" } else { "matmul_gemv_kc" };
    let walk_k = if vec4 {
        format!(
            r#"for (var k4 = lane; k4 < params.k / 4u; k4 += {lanes}u) {{
            let bv = vec4<f32>(b[b_base / 4u + k4]);
            for (var r = 0u; r < params.m; r++) {{
                acc[r] += dot(vec4<f32>(a[(a_i + r * a_m_s) / 4u + k4]), bv);
            }}
        }}"#,
            lanes = MATMUL_GEMV_KC_LANES
        )
    } else {
        format!(
            r#"for (var k = lane; k < params.k; k += {lanes}u) {{
            let bv = f32(b[b_base + k * b_k_s]);
            for (var r = 0u; r < params.m; r++) {{
                acc[r] += f32(a[a_i + r * a_m_s + k * a_k_s]) * bv;
            }}
        }}"#,
            lanes = MATMUL_GEMV_KC_LANES
        )
    };
    matmul_gemv_kc_body_walking(t, suf, epilogue, entry, &walk_k)
}

fn matmul_gemv_kc_body_walking(
    t: &str,
    suf: &str,
    epilogue: &str,
    entry: &str,
    walk_k: &str,
) -> String {
    let cols = MATMUL_GEMV_KC_COLS;
    let lanes = MATMUL_GEMV_KC_LANES;
    let max_m = MATMUL_GEMV_KC_MAX_M;
    let wg = cols * lanes;
    format!(
        r#"var<workgroup> partial: array<f32, {wg}>;

@compute @workgroup_size({wg})
fn {entry}_{suf}(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{
    let tiles_n = (params.n + {cols}u - 1u) / {cols}u;
    let tile = wid.y * {GRID_LIMIT}u + wid.x;
    if (tile >= params.prefix * tiles_n) {{ return; }}
    let pref = tile / tiles_n;
    let lane = lid.x % {lanes}u;
    let slot = lid.x / {lanes}u;
    let col = (tile % tiles_n) * {cols}u + slot;

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
    let b_n_s = select(at8(params.b_s0, params.b_s1, 7u), at8(params.b_s0, params.b_s1, 6u), params.tb != 0u);
    let b_k_s = select(at8(params.b_s0, params.b_s1, 6u), at8(params.b_s0, params.b_s1, 7u), params.tb != 0u);
    let o_n_s = select(at8(params.out_s0, params.out_s1, 7u), at8(params.out_s0, params.out_s1, 6u), params.tc != 0u);
    let a_m_s = select(at8(params.a_s0, params.a_s1, 6u), at8(params.a_s0, params.a_s1, 7u), params.ta != 0u);
    let o_m_s = select(at8(params.out_s0, params.out_s1, 6u), at8(params.out_s0, params.out_s1, 7u), params.tc != 0u);
    var acc: array<f32, {max_m}>;
    for (var r = 0u; r < {max_m}u; r++) {{ acc[r] = 0.0; }}
    if (col < params.n) {{
        let b_base = b_i + col * b_n_s;
        {walk_k}
    }}
    for (var r = 0u; r < params.m; r++) {{
        partial[lid.x] = acc[r];
        workgroupBarrier();
        for (var s = {lanes}u / 2u; s > 0u; s = s / 2u) {{
            if (lane < s) {{
                partial[lid.x] += partial[lid.x + s];
            }}
            workgroupBarrier();
        }}
        if (lane == 0u && col < params.n) {{
            let oe = o_i - params.off_out + r * o_m_s + col * o_n_s;
            var v = {t}(partial[lid.x]);
{epilogue}
            outp[o_i + r * o_m_s + col * o_n_s] = v;
        }}
        workgroupBarrier();
    }}
}}
"#
    )
}

/// Aligned unit-prefix GEMV with contiguous vector loads and ordered f32 dots.
fn matmul_gemv_small_body(t: &str, suf: &str, epilogue: &str) -> String {
    let cols = MATMUL_GEMV_SMALL_COLS;
    format!(
        r#"
@compute @workgroup_size({cols})
fn matmul_gemv_small_{suf}(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{
    let col = wid.x * {cols}u + lid.x;
    let a_base = params.off_a / 4u;
    let b_base = (params.off_b + col * params.k) / 4u;
    var acc = 0.0;
    for (var q = 0u; q < params.k / 4u; q += 1u) {{
        let av = vec4<f32>(a[a_base + q]);
        let bv = vec4<f32>(b[b_base + q]);
        acc += av.x * bv.x;
        acc += av.y * bv.y;
        acc += av.z * bv.z;
        acc += av.w * bv.w;
    }}
    let oe = col;
    var v = {t}(acc);
{epilogue}
    outp[params.off_out + col] = v;
}}
"#
    )
}

/// `n = 1` product. One workgroup owns [`MATMUL_GEMV_N_WG`] rows of one prefix
/// slice: the threads stage that slice of the single column into workgroup
/// memory and each then walks its own row. Same `Params`, strides and epilogue
/// as [`matmul_module`]. `k` past the last partial slice is padded with zeros.
pub fn matmul_gemv_n_module(dt: ShaderDtype, epilogue: &[ChainStep], extras: usize) -> String {
    matmul_gemv_n_module_for_layout(dt, epilogue, extras, false)
}

/// [`matmul_gemv_n_module`]'s `v4` form: a thread owns four adjacent rows of a
/// contiguous `a` and reaches them as `vec4`s, so a fourth of the threads does
/// the same taps. The dispatch only picks it when `a` and the output are
/// unit-strided along `m` and four-aligned.
fn matmul_gemv_n_module_for_layout(
    dt: ShaderDtype,
    epilogue: &[ChainStep],
    extras: usize,
    v4: bool,
) -> String {
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
    let row_t = if v4 { format!("vec4<{t}>") } else { t.to_string() };
    let body = if v4 {
        let epilogue = epilogue_body_at(epilogue, "col", "oe + r");
        format!(
            r#"var<workgroup> bvec: array<f32, {wg}>;

@compute @workgroup_size({wg})
fn matmul_gemv_n_v4_{suf}(@builtin(local_invocation_id) lid: vec3<u32>, @builtin(workgroup_id) wid: vec3<u32>) {{
    let rows = {wg}u * 4u;
    let tiles_m = (params.m + rows - 1u) / rows;
    let tile = wid.y * {GRID_LIMIT}u + wid.x;
    if (tile >= params.prefix * tiles_m) {{ return; }}
    let pref = tile / tiles_m;
    let row = (tile % tiles_m) * rows + lid.x * 4u;

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

    var acc = vec4<f32>(0.0);
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
                    acc += vec4<f32>(a[(a_base + k * a_k_s) / 4u]) * bvec[t];
                }}
            }}
        }}
        workgroupBarrier();
    }}
    if (row < params.m) {{
        let col = 0u;
        let oe = o_i - params.off_out + row * o_m_s + col * o_n_s;
        var outv: vec4<{t}>;
        for (var r = 0u; r < 4u; r++) {{
            var v = {t}(acc[r]);
{epilogue}
            outv[r] = v;
        }}
        outp[(o_i + row * o_m_s + col * o_n_s) / 4u] = outv;
    }}
}}
"#
        )
    } else {
        let epilogue = epilogue_body_at(epilogue, "col", "oe");
        format!(
            r#"var<workgroup> bvec: array<f32, {wg}>;

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
        )
    };
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

@group(0) @binding(0) var<storage, read> a: array<{row_t}>;
@group(0) @binding(1) var<storage, read> b: array<{t}>;
{extra_bindings}@group(0) @binding({out_binding}) var<storage, read_write> outp: array<{row_t}>;
@group(0) @binding({uniform_binding}) var<uniform> params: Params;
{lib}
{AT8}

{body}
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
    let out_binding = 4 + extras;
    let uniform_binding = 5 + extras;
    let extra_bindings = (0..extras)
        .map(|i| {
            format!("@group(0) @binding({}) var<storage, read> extra{i}: array<{t}>;\n", 4 + i)
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
@group(0) @binding(2) var<storage, read> a4: array<vec4<{t}>>;
@group(0) @binding(3) var<storage, read> b4: array<vec4<{t}>>;
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
        core = tiled_gemm_core_vec4(
            t,
            "a_col_s == 1u",
            "b_col_s == 1u",
            "o_i, o_row_s, o_col_s, ",
            tile_m,
            "f32",
            [
                "a_col_s == 1u && (a_i & 3u) == 0u && (a_row_s & 3u) == 0u",
                "a_row_s == 1u && (a_i & 3u) == 0u && (a_col_s & 3u) == 0u",
                "b_col_s == 1u && (b_i & 3u) == 0u && (b_row_s & 3u) == 0u",
                "b_row_s == 1u && (b_i & 3u) == 0u && (b_col_s & 3u) == 0u",
            ],
        ),
    ));
    s
}

/// Threads per workgroup of the split-k matmul: 32 tiles, four lanes each.
pub const MATMUL_SPLITK_WG: u32 = 128;
/// Below this `m` the blocked matmul runs split over k.
pub const MATMUL_SPLITK_MAX_M: usize = 2304;

fn apple_splitk_device(_device: &wgpu::Device) -> bool {
    #[cfg(target_arch = "wasm32")]
    if let Some(raw) = _device.as_webgpu()
        && let Ok(info) =
            js_sys::Reflect::get(raw.as_ref(), &wasm_bindgen::JsValue::from_str("adapterInfo"))
    {
        let lookup = |key| js_sys::Reflect::get(&info, &wasm_bindgen::JsValue::from_str(key)).ok();
        return lookup("vendor").and_then(|v| v.as_string()).as_deref() == Some("apple")
            && lookup("isFallbackAdapter").and_then(|v| v.as_bool()) != Some(true);
    }
    false
}

/// The blocked matmul for small `m`: four lanes share a four-by-four output
/// tile, each summing a quarter of `k`, and reduce through workgroup memory.
/// The squeeze-excitation projections (m = 144, k up to 128) put only a few
/// hundred threads on the device with the plain kernel, so its time is one
/// thread's k loop; four lanes cut that chain and came out of a search scored
/// in Chrome at 2-2.5x on those shapes (m = 144: 56 -> 22 us, m = 576 with
/// k >= 72: 47 -> 23 us). Same contract and epilogue as
/// [`matmul_blocked_module`]; `k` a multiple of four.
pub fn matmul_splitk_module(
    dt: ShaderDtype,
    f16_accumulation: bool,
    epilogue: &[ChainStep],
    extras: usize,
) -> String {
    matmul_splitk_module_for_layout(dt, f16_accumulation, epilogue, extras, false)
}

fn matmul_splitk_module_for_layout(
    dt: ShaderDtype,
    f16_accumulation: bool,
    epilogue: &[ChainStep],
    extras: usize,
    column_first: bool,
) -> String {
    let (entry, wg, row_tile, col_tile) = if column_first {
        ("matmul_splitk_col_fast", 64, "tile_global / n4", "tile_global % n4")
    } else {
        ("matmul_splitk", MATMUL_SPLITK_WG, "tile_global % m4", "tile_global / m4")
    };
    let t = dt.wgsl();
    let acc = if dt == ShaderDtype::F16 && f16_accumulation { "f16" } else { "f32" };
    let (widen, widen_end) = if t == acc { ("", "") } else { ("vec4<f32>(", ")") };
    let (narrow, narrow_end) = if t == acc { ("", "") } else { ("vec4<f16>(", ")") };
    let suf = dt.suffix();
    let out_binding = 2 + extras;
    let uniform_binding = 3 + extras;
    let extra_bindings = (0..extras)
        .map(|i| {
            format!("@group(0) @binding({}) var<storage, read> extra{i}: array<{t}>;\n", 2 + i)
        })
        .collect::<String>();
    let lib = format!("{}{}", unary_ops_wgsl(acc), binary_ops_wgsl(acc));
    let epi = epilogue_body_at(epilogue, "col", "oe");
    let epi = if t != acc {
        epi.lines()
            .map(|line| {
                if line.contains(" = extra") {
                    line.replace(" = ", " = f32(").replace("];", "]);")
                } else {
                    line.to_string()
                }
            })
            .map(|line| line + "\n")
            .collect::<String>()
    } else {
        epi
    };
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

@group(0) @binding(0) var<storage, read> a4: array<vec4<{t}>>;
@group(0) @binding(1) var<storage, read> b4: array<vec4<{t}>>;
{extra_bindings}@group(0) @binding({out_binding}) var<storage, read_write> c4: array<vec4<{t}>>;
@group(0) @binding({uniform_binding}) var<uniform> params: Params;
{lib}
var<workgroup> sh0: array<vec4<{acc}>, {wg}>;
var<workgroup> sh1: array<vec4<{acc}>, {wg}>;
var<workgroup> sh2: array<vec4<{acc}>, {wg}>;
var<workgroup> sh3: array<vec4<{acc}>, {wg}>;

fn epi(v_in: {acc}, col: u32, oe: u32) -> {acc} {{
    var v = v_in;
{epi}
    return v;
}}

@compute @workgroup_size({wg})
fn {entry}_{suf}(@builtin(global_invocation_id) gid: vec3<u32>, @builtin(local_invocation_id) lid: vec3<u32>) {{
    let m4 = params.m / 4u;
    let n4 = params.n / 4u;
    let tile_global = gid.x >> 2u;
    let lane = gid.x & 3u;
    let row_tile_raw = {row_tile};
    let col_tile_raw = {col_tile};
    let valid = (row_tile_raw < m4) && (col_tile_raw < n4);
    let row_tile = min(row_tile_raw, m4 - 1u);
    let col_tile = min(col_tile_raw, n4 - 1u);

    var acc0 = vec4<{acc}>(0.0);
    var acc1 = vec4<{acc}>(0.0);
    var acc2 = vec4<{acc}>(0.0);
    var acc3 = vec4<{acc}>(0.0);

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
        let a0v = {widen}a4[ab0]{widen_end}; let b0v = {widen}b4[bb0]{widen_end};
        let a1v = {widen}a4[ab1]{widen_end}; let b1v = {widen}b4[bb1]{widen_end};
        let a2v = {widen}a4[ab2]{widen_end}; let b2v = {widen}b4[bb2]{widen_end};
        let a3v = {widen}a4[ab3]{widen_end}; let b3v = {widen}b4[bb3]{widen_end};
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
        let av = {widen}a4[ab]{widen_end};
        let bv = {widen}b4[bb]{widen_end};
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
            c4[cc] = {narrow}vec4<{acc}>(epi(s0.x, col0, e0), epi(s0.y, col0, e0 + 1u), epi(s0.z, col0, e0 + 2u), epi(s0.w, col0, e0 + 3u)){narrow_end};
            c4[cc + m4] = {narrow}vec4<{acc}>(epi(s1.x, col0 + 1u, e1), epi(s1.y, col0 + 1u, e1 + 1u), epi(s1.z, col0 + 1u, e1 + 2u), epi(s1.w, col0 + 1u, e1 + 3u)){narrow_end};
            c4[cc + 2u * m4] = {narrow}vec4<{acc}>(epi(s2.x, col0 + 2u, e2), epi(s2.y, col0 + 2u, e2 + 1u), epi(s2.z, col0 + 2u, e2 + 2u), epi(s2.w, col0 + 2u, e2 + 3u)){narrow_end};
            c4[cc + 3u * m4] = {narrow}vec4<{acc}>(epi(s3.x, col0 + 3u, e3), epi(s3.y, col0 + 3u, e3 + 1u), epi(s3.z, col0 + 3u, e3 + 2u), epi(s3.w, col0 + 3u, e3 + 3u)){narrow_end};
        }}
    }}
}}
"#
    ));
    s
}

pub fn matmul_blocked_module(
    dt: ShaderDtype,
    f16_accumulation: bool,
    epilogue: &[ChainStep],
    extras: usize,
) -> String {
    let t = dt.wgsl();
    let acc = if dt == ShaderDtype::F16 && f16_accumulation { "f16" } else { "f32" };
    let (widen, widen_end) = if t == acc { ("", "") } else { ("vec4<f32>(", ")") };
    let (narrow, narrow_end) = if t == acc { ("", "") } else { ("vec4<f16>(", ")") };
    let suf = dt.suffix();
    let out_binding = 2 + extras;
    let uniform_binding = 3 + extras;
    let extra_bindings = (0..extras)
        .map(|i| {
            format!("@group(0) @binding({}) var<storage, read> extra{i}: array<{t}>;\n", 2 + i)
        })
        .collect::<String>();
    let lib = format!("{}{}", unary_ops_wgsl(acc), binary_ops_wgsl(acc));
    let epi = epilogue_body_at(epilogue, "col", "oe");
    let epi = if t != acc {
        epi.lines()
            .map(|line| {
                if line.contains(" = extra") {
                    line.replace(" = ", " = f32(").replace("];", "]);")
                } else {
                    line.to_string()
                }
            })
            .map(|line| line + "\n")
            .collect::<String>()
    } else {
        epi
    };
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

@group(0) @binding(0) var<storage, read> a4: array<vec4<{t}>>;
@group(0) @binding(1) var<storage, read> b4: array<vec4<{t}>>;
{extra_bindings}@group(0) @binding({out_binding}) var<storage, read_write> c4: array<vec4<{t}>>;
@group(0) @binding({uniform_binding}) var<uniform> params: Params;
{lib}

fn epi(v_in: {acc}, col: u32, oe: u32) -> {acc} {{
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
    var acc0 = vec4<{acc}>(0.0);
    var acc1 = vec4<{acc}>(0.0);
    var acc2 = vec4<{acc}>(0.0);
    var acc3 = vec4<{acc}>(0.0);
    var kk = 0u;
    for (; kk + 8u <= params.k; kk += 8u) {{
        let ab = params.off_a + kk * m4 + row_tile;
        let bb = params.off_b + kk * n4 + col_tile;
        let a0 = {widen}a4[ab]{widen_end};
        let a1 = {widen}a4[ab + m4]{widen_end};
        let a2 = {widen}a4[ab + 2u * m4]{widen_end};
        let a3 = {widen}a4[ab + 3u * m4]{widen_end};
        let a4v = {widen}a4[ab + 4u * m4]{widen_end};
        let a5 = {widen}a4[ab + 5u * m4]{widen_end};
        let a6 = {widen}a4[ab + 6u * m4]{widen_end};
        let a7 = {widen}a4[ab + 7u * m4]{widen_end};
        let b0 = {widen}b4[bb]{widen_end};
        let b1 = {widen}b4[bb + n4]{widen_end};
        let b2 = {widen}b4[bb + 2u * n4]{widen_end};
        let b3 = {widen}b4[bb + 3u * n4]{widen_end};
        let b4v = {widen}b4[bb + 4u * n4]{widen_end};
        let b5 = {widen}b4[bb + 5u * n4]{widen_end};
        let b6 = {widen}b4[bb + 6u * n4]{widen_end};
        let b7 = {widen}b4[bb + 7u * n4]{widen_end};
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
        let a0 = {widen}a4[ab]{widen_end};
        let a1 = {widen}a4[ab + m4]{widen_end};
        let a2 = {widen}a4[ab + 2u * m4]{widen_end};
        let a3 = {widen}a4[ab + 3u * m4]{widen_end};
        let b0 = {widen}b4[bb]{widen_end};
        let b1 = {widen}b4[bb + n4]{widen_end};
        let b2 = {widen}b4[bb + 2u * n4]{widen_end};
        let b3 = {widen}b4[bb + 3u * n4]{widen_end};
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
    c4[cc] = {narrow}vec4<{acc}>(
        epi(acc0.x, col0, e0), epi(acc0.y, col0, e0 + 1u),
        epi(acc0.z, col0, e0 + 2u), epi(acc0.w, col0, e0 + 3u)){narrow_end};
    c4[cc + m4] = {narrow}vec4<{acc}>(
        epi(acc1.x, col0 + 1u, e1), epi(acc1.y, col0 + 1u, e1 + 1u),
        epi(acc1.z, col0 + 1u, e1 + 2u), epi(acc1.w, col0 + 1u, e1 + 3u)){narrow_end};
    c4[cc + 2u * m4] = {narrow}vec4<{acc}>(
        epi(acc2.x, col0 + 2u, e2), epi(acc2.y, col0 + 2u, e2 + 1u),
        epi(acc2.z, col0 + 2u, e2 + 2u), epi(acc2.w, col0 + 2u, e2 + 3u)){narrow_end};
    c4[cc + 3u * m4] = {narrow}vec4<{acc}>(
        epi(acc3.x, col0 + 3u, e3), epi(acc3.y, col0 + 3u, e3 + 1u),
        epi(acc3.z, col0 + 3u, e3 + 2u), epi(acc3.w, col0 + 3u, e3 + 3u)){narrow_end};
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

    /// Products the k-contiguous GEMV kernel takes, `b` transposed so its rows run along `k`:
    /// one to eight rows of `a`, `k` past the kernel's minimum and `k`, `n` off the lane and
    /// column counts, a prefix, `a` and `c` transposed, f16 and a bias epilogue, with `k`
    /// a multiple of four or not, for the `vec4` and scalar loads, `a` transposed or not.
    #[test]
    fn matmul_gemv_k_contiguous_matches_cpu() -> TractResult<()> {
        use tract_core::ops::einsum::prefix_matmul::PrefixMatMul;
        for (prefix, m, k, n, ta, tc, f16_, bias) in [
            (1usize, 1usize, 170usize, 13usize, false, false, false, false),
            (1, 3, 170, 13, false, false, false, false),
            (1, 8, 1000, 37, false, false, false, false),
            (3, 4, 133, 9, false, false, false, false),
            (1, 4, 170, 13, true, false, false, false),
            (1, 4, 170, 13, false, true, false, false),
            (1, 2, 130, 21, false, false, true, false),
            (1, 4, 170, 13, false, false, false, true),
            (1, 1, 256, 13, false, false, false, false),
            (1, 5, 256, 13, false, false, false, false),
            (3, 4, 132, 9, false, false, false, false),
            (1, 2, 128, 21, false, false, true, false),
            (1, 4, 256, 13, false, false, false, true),
            (1, 4, 256, 13, true, false, false, false),
            (3, 3, 132, 9, true, false, false, false),
            (1, 4, 128, 21, true, false, true, false),
            (1, 4, 256, 13, true, false, false, true),
        ] {
            let dt = if f16_ { f16::datum_type() } else { f32::datum_type() };
            let a_shape = [prefix, if ta { k } else { m }, if ta { m } else { k }];
            let b_shape = [prefix, n, k];
            let mut model = TypedModel::default();
            let a = model.add_source("a", dt.fact(a_shape))?;
            let b_data = Tensor::from_shape(&b_shape, &fill(1, b_shape.iter().product()))?;
            let b = model.add_const("b", b_data.cast_to_dt(dt)?.into_owned())?;
            let op = PrefixMatMul {
                transpose_a: ta,
                transpose_b: true,
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
                format!("prefix={prefix} m={m} k={k} n={n} ta={ta} tc={tc} f16={f16_} bias={bias}")
            })?;
        }
        Ok(())
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

    /// f16 operands take the blocked and split-K kernels, which accumulate and
    /// run the epilogue in f32 and round once on store.
    #[test]
    fn matmul_blocked_f16_accumulates_in_f32() -> TractResult<()> {
        use tract_gpu::tensor::DeviceTensor;
        use tract_gpu::tensor::IntoDevice;

        use super::Transposes;
        use super::blocked_applies;
        use super::wgpu_matmul_dispatch;
        use crate::kernels::shaders::ChainStep;

        crate::context::wgpu_context();
        let transposes = Transposes { a: true, b: true, c: true };
        for (m, k, n) in [(12usize, 4usize, 12usize), (144, 20, 32), (2304, 12, 16)] {
            let values = |len: usize, seed: usize| {
                (0..len)
                    .map(|i| f16::from_f32(((i * 17 + seed) % 9) as f32 / 16.0 - 0.25))
                    .collect::<Vec<_>>()
            };
            let av = values(k * m, 1);
            let bv = values(n * k, 3);
            let a = Tensor::from_shape(&[k, m], &av)?.into_device()?;
            let b = Tensor::from_shape(&[n, k], &bv)?.into_device()?;
            let output = DeviceTensor::uninitialized_dt(DatumType::F16, &[n, m])?;
            ensure!(blocked_applies(1, m, k, n, &a, &b, &output, transposes));
            for mode in 0..3 {
                let extra_len = [1, n, n * m][mode];
                let extra_values = values(extra_len, 5);
                let extra_shape = [vec![1], vec![n, 1], vec![n, m]][mode].clone();
                let extra = Tensor::from_shape(&extra_shape, &extra_values)?.into_device()?;
                let epilogue = [ChainStep::Binary { op: "add".into(), rhs: 1, swapped: false }];
                let mut reference = vec![];
                for col in 0..n {
                    for row in 0..m {
                        let sum: f64 = (0..k)
                            .map(|kk| av[kk * m + row].to_f64() * bv[col * k + kk].to_f64())
                            .sum();
                        let extra_index = [0, col, col * m + row][mode];
                        reference.push(f16::from_f64(sum + extra_values[extra_index].to_f64()));
                    }
                }
                wgpu_matmul_dispatch(transposes, &epilogue, &a, &b, &[&extra], &output, false)?;
                crate::with_wgpu_queue(|q| q.flush())?;
                let actual = output.to_host()?;
                for (i, (actual, expected)) in
                    actual.try_as_plain_ram()?.as_slice::<f16>()?.iter().zip(&reference).enumerate()
                {
                    ensure!(
                        actual == expected,
                        "m={m} k={k} n={n} mode={mode} index={i}: {actual} != {expected}"
                    );
                }
            }
        }
        let a = Tensor::from_shape(&[4, 4], &[f16::from_f32(300.0); 16])?.into_device()?;
        let b_values = [300.0f32, -300.0, 0.0, 0.0].repeat(4);
        let b_values = b_values.into_iter().map(f16::from_f32).collect::<Vec<_>>();
        let b = Tensor::from_shape(&[4, 4], &b_values)?.into_device()?;
        let output = DeviceTensor::uninitialized_dt(DatumType::F16, &[4, 4])?;
        wgpu_matmul_dispatch(transposes, &[], &a, &b, &[], &output, false)?;
        let actual = output.to_host()?;
        for value in actual.try_as_plain_ram()?.as_slice::<f16>()? {
            ensure!(*value == f16::ZERO, "partial sum overflowed f16: {value}");
        }
        Ok(())
    }

    /// With f16 accumulation the blocked and split-K kernels multiply in f16, so
    /// products that overflow f16 no longer cancel; each mode keeps its own
    /// cached pipeline.
    #[test]
    fn matmul_blocked_f16_accumulation_modes() -> TractResult<()> {
        use tract_gpu::tensor::DeviceTensor;
        use tract_gpu::tensor::IntoDevice;

        use super::Transposes;
        use super::wgpu_matmul_dispatch;

        crate::context::wgpu_context();
        let transposes = Transposes { a: true, b: true, c: true };
        for m in [4usize, 2304] {
            let a = Tensor::from_shape(&[4, m], &vec![f16::from_f32(300.0); 4 * m])?;
            let b_values = [300.0f32, -300.0, 0.0, 0.0].repeat(4);
            let b_values = b_values.into_iter().map(f16::from_f32).collect::<Vec<_>>();
            let b = Tensor::from_shape(&[4, 4], &b_values)?.into_device()?;
            let a = a.into_device()?;
            let output = DeviceTensor::uninitialized_dt(DatumType::F16, &[4, m])?;
            for f16_accumulation in [false, true, false] {
                wgpu_matmul_dispatch(transposes, &[], &a, &b, &[], &output, f16_accumulation)?;
                let actual = output.to_host()?;
                for value in actual.try_as_plain_ram()?.as_slice::<f16>()? {
                    ensure!(
                        if f16_accumulation { !value.is_finite() } else { *value == f16::ZERO },
                        "m={m} f16_accumulation={f16_accumulation}: {value}"
                    );
                }
            }
        }
        Ok(())
    }

    #[test]
    fn splitk_column_first_preserves_fp16_projection_outputs() -> TractResult<()> {
        use tract_gpu::tensor::DeviceTensor;
        use tract_gpu::tensor::IntoDevice;

        use super::Transposes;
        use super::wgpu_matmul_dispatch_impl;
        use crate::kernels::shaders::ChainStep;

        crate::wgpu_context();
        let m = 920;
        let t = Transposes { a: true, b: true, c: true };
        let epilogue = [ChainStep::Binary { op: "add".into(), rhs: 1, swapped: false }];
        let make = |shape: &[usize], seed| -> TractResult<DeviceTensor> {
            let values: Vec<_> = fill(seed, shape.iter().product())
                .into_iter()
                .map(|x| f16::from_f32(x * 0.125))
                .collect();
            Tensor::from_shape(shape, &values)?.into_device()
        };
        // The knob would let the cooperative GEMM serve these shapes first.
        let off = tract_data::knobs::KnobScope::new([("TRACT_WGPU_SUBGROUPS", "false")])?;
        let check = |k: usize, n: usize| -> TractResult<()> {
            let a = make(&[k, m], 1)?;
            let b = make(&[n, k], 2)?;
            let bias = make(&[n], 3)?;
            let normal = DeviceTensor::uninitialized_dt(DatumType::F16, &[n, m])?;
            let column = DeviceTensor::uninitialized_dt(DatumType::F16, &[n, m])?;
            wgpu_matmul_dispatch_impl(
                t,
                &epilogue,
                &a,
                &b,
                &[&bias],
                &normal,
                true,
                None,
                Some(false),
            )?;
            let (_result, profile) = crate::with_wgpu_queue(|queue| {
                queue.profile(|| {
                    wgpu_matmul_dispatch_impl(
                        t,
                        &epilogue,
                        &a,
                        &b,
                        &[&bias],
                        &column,
                        true,
                        None,
                        Some(true),
                    )
                })
            })?;
            ensure!(profile.iter().any(|(name, _)| *name == "matmul_splitk_col_fast"));
            assert_eq!(normal.to_host()?, column.to_host()?, "k={k} n={n}");
            Ok(())
        };
        for (k, n) in [(160, 960), (960, 160)] {
            off.enter(|| check(k, n))?;
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
            (1, 64, 32, 96, true, true, true, true, false),
            (1, 64, 32, 96, false, false, false, true, false),
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

    /// `a` transposed so the `m` axis is contiguous: four rows a thread, a
    /// scalar bias and a clamp riding the epilogue. Shapes that miss a
    /// constraint fall back to the plain `gemv_n` kernel.
    #[test]
    fn matmul_gemv_n_v4_matches_cpu() -> TractResult<()> {
        use tract_core::ops::einsum::prefix_matmul::PrefixMatMul;
        use tract_core::ops::math::add;
        use tract_core::ops::math::max;
        use tract_core::ops::math::min;
        use tract_gpu::tensor::IntoDevice;
        for (m, k, v4) in [(256usize, 16usize, true), (130, 16, false), (256, 100, false)] {
            let mut model = TypedModel::default();
            let a = model.add_source("a", f32::fact([1, k, m]))?;
            let b = model.add_const("b", Tensor::from_shape(&[1, k, 1], &fill(1, k))?)?;
            let op = PrefixMatMul {
                transpose_a: true,
                transpose_b: false,
                transpose_c: false,
                quantize_output: None,
                operating_dt: None,
            };
            let mut y = model.wire_node("mm", op, &[a, b])?[0];
            for (name, ew, seed) in
                [("bias", add(), 2u64), ("lo", max(), 3u64), ("hi", min(), 4u64)]
            {
                let c = Tensor::from_shape(&[1, 1, 1], &fill(seed, 1))?;
                let c = model.add_const(format!("{name}_c"), c)?;
                y = model.wire_node(name, ew, &[y, c])?[0];
            }
            model.select_output_outlets(&[y])?;
            let fused = transformed(model.clone())?;
            let gemm = fused
                .nodes()
                .iter()
                .find_map(|node| node.op_as::<crate::ops::matmul::WgpuGemm>())
                .context("n=1 product left the device")?;
            ensure!(gemm.epilogue.len() == 3, "epilogue {:?}", gemm.epilogue);
            let a_dev = Tensor::from_shape(&[1, k, m], &fill(9, k * m))?.into_device()?;
            let out_dev = Tensor::from_shape(&[1, m, 1], &fill(9, m))?.into_device()?;
            ensure!(
                super::gemv_n_v4_applies(1, m, k, &a_dev, &out_dev, true, false) == v4,
                "v4 selection m={m} k={k}"
            );
            gpu_vs_cpu(model, Tensor::from_shape(&[1, k, m], &fill(5, k * m))?)
                .with_context(|| format!("m={m} k={k} v4={v4}"))?;
        }
        Ok(())
    }

    /// An `m = 1` product too thin to fill the plain GEMV's grid rides the
    /// lane-parallel kernel even when `b` is not contiguous along `k`.
    #[test]
    fn matmul_gemv_lane_parallel_matches_cpu() -> TractResult<()> {
        use tract_core::ops::einsum::prefix_matmul::PrefixMatMul;
        let (k, n) = (960usize, 64usize);
        ensure!(
            super::gemv_kc_applies(1, k, n, 1, false, n as isize),
            "a thin product should ride gemv_kc"
        );
        ensure!(
            !super::gemv_kc_applies(1, k, 4096, 1, false, 4096),
            "a full grid should keep the plain kernel"
        );
        let mut model = TypedModel::default();
        let a = model.add_source("a", f32::fact([1, 1, k]))?;
        let b = model.add_const("b", Tensor::from_shape(&[1, k, n], &fill(1, k * n))?)?;
        let op = PrefixMatMul {
            transpose_a: false,
            transpose_b: false,
            transpose_c: false,
            quantize_output: None,
            operating_dt: None,
        };
        let y = model.wire_node("mm", op, &[a, b])?[0];
        model.select_output_outlets(&[y])?;
        gpu_vs_cpu(model, Tensor::from_shape(&[1, 1, k], &fill(3, k))?)
    }

    fn small_gemv_fixture(
        k: usize,
        n: usize,
    ) -> TractResult<(
        tract_gpu::tensor::DeviceTensor,
        tract_gpu::tensor::DeviceTensor,
        tract_gpu::tensor::DeviceTensor,
    )> {
        use tract_gpu::tensor::IntoDevice;

        let values = |count: usize, seed: u64| {
            fill(seed, count)
                .into_iter()
                .enumerate()
                .map(|(i, x)| f16::from_f32(x * 2.0_f32.powi((i % 17) as i32 - 12)))
                .collect::<Vec<_>>()
        };
        let a = Tensor::from_shape(&[1, 1, k], &values(k, 1))?.into_device()?;
        let b = Tensor::from_shape(&[1, n, k], &values(k * n, 3))?.into_device()?;
        let bias = Tensor::from_shape(&[1, 1, n], &values(n, 5))?.into_device()?;
        Ok((a, b, bias))
    }

    #[test]
    fn matmul_gemv_small_matches_original() -> TractResult<()> {
        use tract_gpu::tensor::DeviceTensor;

        use super::Transposes;
        use super::wgpu_matmul_dispatch_impl;
        use crate::kernels::shaders::ChainStep;

        crate::context::wgpu_context();
        for (k, n) in [(480, 120), (672, 168), (960, 240)] {
            let (a, b, bias) = small_gemv_fixture(k, n)?;
            let output_old = DeviceTensor::uninitialized_dt(DatumType::F16, &[1, 1, n])?;
            let output_new = DeviceTensor::uninitialized_dt(DatumType::F16, &[1, 1, n])?;
            let t = Transposes { a: false, b: true, c: false };
            let epilogue = [ChainStep::Binary { op: "add".into(), rhs: 1, swapped: false }];
            for (new, output) in [(false, &output_old), (true, &output_new)] {
                wgpu_matmul_dispatch_impl(
                    t,
                    &epilogue,
                    &a,
                    &b,
                    &[&bias],
                    output,
                    false,
                    Some(new),
                    None,
                )?;
            }
            crate::with_wgpu_queue(|q| q.flush())?;
            let old = output_old.to_host()?;
            let new = output_new.to_host()?;
            assert_eq!(
                old.try_as_plain_ram()?.as_slice::<f16>()?,
                new.try_as_plain_ram()?.as_slice::<f16>()?,
                "k={k} n={n}"
            );
        }
        Ok(())
    }
}
