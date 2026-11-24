//! Cooperative-matrix 3x3 convolution: stride 1, one pad, channel-first, zero padded.
//!
//! A subgroup owns a tile of rows x 8-pixel blocks x 8-channel blocks and runs one 8x8 product per
//! tap, reading eight input channels by eight neighbouring pixels straight from the map. Weights are
//! repacked once into contiguous 8x8 blocks. Rows above and below the map are zero matrices; the
//! tap column that leaves the map at the left and right edge reuses the centre tile and is added
//! one pixel over. A map too small to fill the device splits its input channels over several
//! subgroups. The sum is accumulated in what [`accumulator`] names.

use std::fmt::Write;

use tract_core::internal::*;
use tract_core::ops::cnn::KernelFormat;
use tract_core::ops::cnn::conv::Conv;
use tract_core::ops::nn::DataFormat;
use tract_gpu::tensor::DeviceTensor;

use crate::context::{RepackKey, RepackLayout, WgpuQueue};
use crate::kernels::conv::conv_direct_epilogue;
use crate::kernels::coop_dialect::{Dialect, Side, accumulator};
use crate::kernels::matmul::epilogue_mode;
use crate::kernels::shaders::{
    ChainStep, EntryPoint, GRID_LIMIT, LayoutKind, ShaderDtype, binary_ops_wgsl, pack_u32s,
    unary_ops_wgsl,
};
use crate::kernels::subgroup::{chosen_subgroup, enables, knob_on, limits_of};
use crate::utils::{element_offset, get_wgpu_buffer};

const LANES: usize = 32;
const BLOCK: usize = 8;
const TAPS: usize = 9;
/// Blocks of 8x8 a subgroup accumulates at most. More spill registers.
const ACCUMULATORS: usize = 10;
/// Workgroups under which the rows of a tile are given up for occupancy.
const MIN_GROUPS: usize = 14;
/// Workgroups times subgroups the channel split aims for.
const TARGET_SUBGROUPS: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Tile {
    rows: usize,
    /// 8-pixel blocks along W.
    pixels: usize,
    /// 8-channel output blocks.
    channels: usize,
    /// Subgroups the input channels are split over.
    split: usize,
}

impl Tile {
    fn width(self) -> usize {
        BLOCK * self.pixels
    }

    fn groups(self, co: usize, h: usize, w: usize) -> [usize; 3] {
        [w / self.width(), h.div_ceil(self.rows), co / BLOCK / self.channels]
    }

    fn row_span(self) -> usize {
        self.channels * BLOCK * self.width()
    }

    fn span(self) -> usize {
        self.rows * self.row_span()
    }

    fn workgroup_bytes(self) -> usize {
        2 * self.span() * self.split * size_of::<f32>()
    }
}

fn tile(ci: usize, co: usize, h: usize, w: usize) -> Option<Tile> {
    let fits = ci >= BLOCK && co.is_multiple_of(BLOCK) && co > 0 && h >= BLOCK;
    if !fits || w < 2 * BLOCK || !w.is_multiple_of(BLOCK) {
        return None;
    }
    let blocks = co / BLOCK;
    let channels = [5, 4, 3, 2, 1].into_iter().find(|t| blocks.is_multiple_of(*t))?;
    let pixels = if w > 2 * BLOCK && w.is_multiple_of(2 * BLOCK) { 2 } else { 1 };
    let mut tile =
        Tile { rows: (ACCUMULATORS / (channels * pixels)).clamp(1, h), pixels, channels, split: 1 };
    let total = |t: Tile| t.groups(co, h, w).iter().product::<usize>();
    while total(tile) < MIN_GROUPS && tile.rows > 1 {
        tile.rows -= 1;
    }
    let total = total(tile);
    let split = [4, 2]
        .into_iter()
        .find(|s| total * s <= TARGET_SUBGROUPS && ci.div_ceil(BLOCK) >= *s)
        .unwrap_or(1);
    Some(Tile { split, ..tile })
}

/// Contiguous `[co/8][ceil(ci/8)][taps][8][8]` weights. The last block of channels starts at
/// `ci - 8`; the channels it shares with the block before are halved in both, which keeps an
/// infinite input infinite where a zero weight would make it `NaN`. Only reached for unpooled
/// weights, so the key cannot outlive the buffer it names.
fn packed_weights(
    q: &WgpuQueue,
    weights: &DeviceTensor,
    dt: ShaderDtype,
    ci: usize,
    co: usize,
) -> TractResult<DeviceTensor> {
    let key = RepackKey {
        buffer: get_wgpu_buffer(weights).id,
        offset: element_offset(weights, 0),
        k: ci,
        n: co,
        layout: RepackLayout::CoopConv3x3,
    };
    if let Some(packed) = q.repacked(&key) {
        return Ok(packed);
    }
    let in_blocks = ci.div_ceil(BLOCK);
    let len = co * TAPS * in_blocks * BLOCK;
    let packed = DeviceTensor::uninitialized_dt(weights.datum_type(), &[len])?;
    let t = dt.wgsl();
    let layout = LayoutKind::Chain(2);
    let enable = if dt == ShaderDtype::F16 { "enable f16;\n" } else { "" };
    let pipeline = q.context().chain_pipeline(
        ("conv_coop_pack", dt, ci, co),
        layout,
        EntryPoint::typed("conv2d_coop_pack", dt),
        || {
            format!(
                r#"{enable}@group(0) @binding(0) var<storage, read> inp: array<{t}>;
@group(0) @binding(1) var<storage, read_write> outp: array<{t}>;
@group(0) @binding(2) var<uniform> params: vec4<u32>;

@compute @workgroup_size(64)
fn conv2d_coop_pack_{}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.x;
    if (i >= {len}u) {{ return; }}
    let lane = i % 8u;
    let o = (i / {}u) * 8u + (i / 8u) % 8u;
    let tap = (i / 64u) % {TAPS}u;
    let b = (i / {}u) % {in_blocks}u;
    let last = b == {}u;
    let c = select(b * 8u, {}u, last) + lane;
    let twice = c >= {}u && c < {}u;
    let w = inp[params.x + (o * {ci}u + c) * {TAPS}u + tap];
    outp[i] = select(w, w * {t}(0.5), twice);
}}
"#,
                dt.suffix(),
                64 * TAPS * in_blocks,
                64 * TAPS,
                in_blocks - 1,
                ci - BLOCK,
                ci - BLOCK,
                BLOCK * (in_blocks - 1),
            )
        },
    )?;
    ensure!(len.div_ceil(64) <= GRID_LIMIT as usize, "weights too large for one repack dispatch");
    let bg = q.context().bind_group(
        layout,
        &[get_wgpu_buffer(weights), get_wgpu_buffer(&packed)],
        q.uniform(),
    )?;
    let offset = element_offset(weights, 0) as u32;
    let dyn_off = q.alloc_uniform(&pack_u32s(&[offset, 0, 0, 0]))?;
    q.retain_tensor(weights);
    q.retain_tensor(&packed);
    q.dispatch_grid("conv_coop_pack", &pipeline, &bg, dyn_off, [len.div_ceil(64) as u32, 1, 1])?;
    q.store_repacked(key, packed.clone());
    Ok(packed)
}

fn shader(
    dialect: Dialect,
    dt: ShaderDtype,
    acc: ShaderDtype,
    epilogue: &[ChainStep],
    extras: usize,
    [ci, h, w]: [usize; 3],
    tile: Tile,
) -> String {
    let Tile { rows, pixels, channels, split } = tile;
    let (t, st) = (dt.wgsl(), acc.wgsl());
    let (map, width) = (h * w, tile.width());
    let in_blocks = ci.div_ceil(BLOCK);
    let (row_span, span) = (tile.row_span(), tile.span());
    let threads = LANES * split;
    let blocks: Vec<_> = (0..rows)
        .flat_map(|d| (0..channels).flat_map(move |c| (0..pixels).map(move |p| (d, c, p))))
        .collect();
    let enable = enables(dt, true, false);
    let input = match dialect {
        Dialect::Naga => format!("array<array<{t}, {map}>>"),
        Dialect::Dawn => format!("array<{t}>"),
    };
    let extra_bindings = (0..extras)
        .map(|i| {
            format!("@group(0) @binding({}) var<storage, read> extra{i}: array<{t}>;\n", 2 + i)
        })
        .collect::<String>();
    let mut s = format!(
        r#"{enable}struct Params {{
    off_out: u32,
    _p0: u32,
    _p1: u32,
    _p2: u32,
    off_extra: vec4<u32>,
    mode_extra: vec4<u32>,
}}

@group(0) @binding(0) var<storage, read> inp: {input};
@group(0) @binding(1) var<storage, read> wgt: array<array<array<array<array<{t}, 64>, {TAPS}>, {in_blocks}>, {channels}>>;
{extra_bindings}@group(0) @binding({}) var<storage, read_write> outp: array<{t}>;
@group(0) @binding({}) var<uniform> params: Params;
var<workgroup> stage: array<array<{st}, {span}>, {split}>;
var<workgroup> edge_stage: array<array<{st}, {span}>, {split}>;
{}{}
fn epi(acc: {st}, co: u32) -> {t} {{
    var v = acc;
{}    return {t}(v);
}}

"#,
        2 + extras,
        3 + extras,
        unary_ops_wgsl(st),
        binary_ops_wgsl(st),
        conv_direct_epilogue(epilogue, acc != dt),
    );
    let subgroup_id = if split > 1 { ", @builtin(subgroup_id) sg: u32" } else { "" };
    writeln!(
        s,
        "@compute @workgroup_size({threads})\nfn conv2d_coop_{}(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32{subgroup_id}) {{",
        dt.suffix()
    )
    .unwrap();
    writeln!(
        s,
        "    let lane = li & 31u;\n    let x0 = wid.x * {width}u;\n    let y0 = min(wid.y * {rows}u, {}u);\n    let g = wid.z;\n    let left = x0 == 0u;\n    let right = x0 == {}u;\n    let top = y0 == 0u;\n    let bottom = y0 == {}u;",
        h - rows,
        w - width,
        h - rows
    )
    .unwrap();
    for u in 0..rows + 2 {
        writeln!(s, "    let r{u} = (y0 + {u}u - 1u) * {w}u + x0;").unwrap();
    }
    for (d, c, p) in &blocks {
        writeln!(s, "    var a{d}_{c}_{p} = {};", dialect.accumulator(st)).unwrap();
    }
    // One pass over the input channels for the tap columns `ks` and their pixel shifts. `edge` is
    // the tap column that leaves the map, accumulated apart on the centre tile.
    let pass = |s: &mut String, ks: &[(usize, i32)], guarded: bool, edge: Option<usize>| {
        for (k, shift) in ks {
            for (u, p) in (0..rows + 2).flat_map(|u| (0..pixels).map(move |p| (u, p))) {
                let at = BLOCK as i32 * p as i32 + shift;
                let op = if at < 0 { "-" } else { "+" };
                writeln!(s, "        let o{u}_{k}_{p} = r{u} {op} {}u;", at.abs()).unwrap();
            }
        }
        let first = if split > 1 { "sg" } else { "0u" };
        writeln!(
            s,
            "        for (var b = {first}; b < {in_blocks}u; b += {split}u) {{\n        let cs = min(b * 8u, {}u);",
            ci - BLOCK
        )
        .unwrap();
        for (u, p) in (0..rows + 2).flat_map(|u| (0..pixels).map(move |p| (u, p))) {
            for (k, _) in ks {
                let load = dialect.load(
                    Side::Right,
                    t,
                    &format!("inp[cs][o{u}_{k}_{p}]"),
                    ("inp", &format!("cs * {map}u + o{u}_{k}_{p}")),
                    map,
                );
                let name = format!("i{u}_{k}_{p}");
                if guarded && (u == 0 || u == rows + 1) {
                    let inside = if u == 0 { "!top" } else { "!bottom" };
                    writeln!(
                        s,
                        "        var {name} = {};\n        if ({inside}) {{ {name} = {load}; }}",
                        dialect.zero(Side::Right, t)
                    )
                    .unwrap();
                } else {
                    writeln!(s, "        let {name} = {load};").unwrap();
                }
            }
        }
        let weight = |tap: usize, c: usize| {
            let block = format!("wgt[g][{c}][b][{tap}]");
            dialect.load(Side::Left, t, &format!("{block}[0]"), (&block, "0u"), BLOCK)
        };
        for (c, row) in (0..channels).flat_map(|c| (0..3).map(move |row| (c, row))) {
            for (k, _) in ks {
                writeln!(s, "        {{ let wm = {};", weight(row * 3 + k, c)).unwrap();
                for (d, p) in (0..rows).flat_map(|d| (0..pixels).map(move |p| (d, p))) {
                    let acc = format!("a{d}_{c}_{p}");
                    let sum = dialect.multiply_add("wm", &format!("i{}_{k}_{p}", d + row), &acc);
                    writeln!(s, "            {acc} = {sum};").unwrap();
                }
                s.push_str("        }\n");
            }
            if let Some(tap) = edge {
                writeln!(s, "        {{ let we = {};", weight(row * 3 + tap, c)).unwrap();
                for (d, p) in (0..rows).flat_map(|d| (0..pixels).map(move |p| (d, p))) {
                    let acc = format!("e{d}_{c}_{p}");
                    let sum = dialect.multiply_add("we", &format!("i{}_1_{p}", d + row), &acc);
                    writeln!(s, "            {acc} = {sum};").unwrap();
                }
                s.push_str("        }\n");
            }
        }
        s.push_str("        }\n");
    };
    let index = if split > 1 { "sg" } else { "0u" };
    let store = |s: &mut String, name: &str, stage: &str| {
        for (d, c, p) in &blocks {
            let at = d * row_span + c * BLOCK * width + p * BLOCK;
            let array = format!("{stage}[{index}]");
            let store =
                dialect.store(&format!("{name}{d}_{c}_{p}"), &array, &format!("{at}u"), width);
            writeln!(s, "    {store}").unwrap();
        }
    };
    let edge_accumulators = |s: &mut String| {
        for (d, c, p) in &blocks {
            writeln!(s, "    var e{d}_{c}_{p} = {};", dialect.accumulator(st)).unwrap();
        }
    };
    s.push_str("    if (left | right | top | bottom) {\n    if (left) {\n");
    edge_accumulators(&mut s);
    pass(&mut s, &[(1, 0), (2, 1)], true, Some(0));
    store(&mut s, "a", "stage");
    store(&mut s, "e", "edge_stage");
    s.push_str("    } else if (right) {\n");
    edge_accumulators(&mut s);
    pass(&mut s, &[(0, -1), (1, 0)], true, Some(2));
    store(&mut s, "a", "stage");
    store(&mut s, "e", "edge_stage");
    s.push_str("    } else {\n");
    pass(&mut s, &[(0, -1), (1, 0), (2, 1)], true, None);
    store(&mut s, "a", "stage");
    s.push_str("    }\n    } else {\n");
    pass(&mut s, &[(0, -1), (1, 0), (2, 1)], false, None);
    store(&mut s, "a", "stage");
    s.push_str("    }\n    workgroupBarrier();\n    if (left | right) {\n");
    let sum = |stage: &str, at: &str| {
        (0..split).map(|q| format!("{stage}[{q}u][{at}]")).collect::<Vec<_>>().join(" + ")
    };
    for d in 0..rows {
        let base = d * row_span;
        let at = format!("{base}u + e");
        writeln!(
            s,
            "    for (var e = li; e < {row_span}u; e += {threads}u) {{\n        let co = g * {}u + e / {width}u;\n        let dx = e % {width}u;\n        var v = {};\n        if (left && dx > 0u) {{ v += {}; }}\n        if (right && dx < {}u) {{ v += {}; }}\n        outp[params.off_out + co * {map}u + (y0 + {d}u) * {w}u + x0 + dx] = epi(v, co);\n    }}",
            channels * BLOCK,
            sum("stage", &at),
            sum("edge_stage", &format!("{base}u + e - 1u")),
            width - 1,
            sum("edge_stage", &format!("{base}u + e + 1u")),
        )
        .unwrap();
    }
    s.push_str(
        "    } else if (li < 32u) {\n    let col = lane & 7u;\n    let rows8 = lane >> 3u;\n",
    );
    for (d, c, p) in &blocks {
        let base = d * row_span + c * BLOCK * width + p * BLOCK;
        for half in 0..2 {
            writeln!(
                s,
                "    {{\n        let r = rows8 + {}u;\n        let e = {base}u + r * {width}u + col;\n        let co = g * {}u + {}u + r;\n        outp[params.off_out + co * {map}u + (y0 + {d}u) * {w}u + x0 + {}u + col] = epi({}, co);\n    }}",
                4 * half,
                channels * BLOCK,
                c * BLOCK,
                p * BLOCK,
                sum("stage", "e"),
            )
            .unwrap();
        }
    }
    s.push_str("    }\n}\n");
    s
}

/// Runs the convolution when the device, the op and the shape allow the kernel.
#[allow(clippy::too_many_arguments)]
pub(crate) fn try_dispatch(
    q: &WgpuQueue,
    op: &Conv,
    epilogue: &[ChainStep],
    input: &DeviceTensor,
    weights: &DeviceTensor,
    extras: &[&DeviceTensor],
    output: &DeviceTensor,
    dt: ShaderDtype,
    f16_accumulation: bool,
) -> TractResult<bool> {
    let caps = *q.context().subgroup_caps();
    let subgroup =
        chosen_subgroup(&caps).filter(|c| c.size as usize == LANES && !c.varying && !c.pin);
    let runs = knob_on(&caps)
        && caps.matrix
        && subgroup.is_some()
        && op.group == 1
        && op.kernel_fmt == KernelFormat::OIHW
        && op.pool_spec.data_format == DataFormat::NCHW
        && op.pool_spec.kernel_shape[..] == [3, 3]
        && op.pool_spec.strides()[..] == [1, 1]
        && op.pool_spec.dilations()[..] == [1, 1];
    let Some(acc) = accumulator(&caps, dt, f16_accumulation).filter(|_| runs) else {
        return Ok(false);
    };
    let pads = op.pool_spec.computed_padding(&input.shape()[2..]);
    let [ci, co] = [op.pool_spec.input_channels, op.pool_spec.output_channels];
    let &[n, _, ih, iw] = input.shape() else { return Ok(false) };
    let &[_, _, h, w] = output.shape() else { return Ok(false) };
    let natural = |t: &DeviceTensor, shape: &[usize]| {
        t.shape() == shape && t.strides() == &*Tensor::natural_strides(shape)
    };
    let plain = n == 1
        && (ih, iw) == (h, w)
        && pads.iter().all(|p| p.pad_before == 1)
        && natural(input, &[1, ci, h, w])
        && natural(output, &[1, co, h, w])
        && natural(weights, &[co, ci, 3, 3])
        && !get_wgpu_buffer(weights).pooled
        && element_offset(input, 0) == 0;
    let Some(tile) = tile(ci, co, h, w).filter(|_| plain) else { return Ok(false) };
    let limits = q.context().device().limits();
    let (max_invocations, max_x) = limits_of(q);
    if (LANES * tile.split) as u32 > max_invocations.min(max_x)
        || tile.workgroup_bytes() as u32 > limits.max_compute_workgroup_storage_size
        || extras.len() > 4
    {
        return Ok(false);
    }
    let mut vals = vec![element_offset(output, 0) as u32, 0, 0, 0];
    let (mut offsets, mut modes) = ([0u32; 4], [0u32; 4]);
    for (i, extra) in extras.iter().enumerate() {
        offsets[i] = element_offset(extra, 0) as u32;
        modes[i] = epilogue_mode(extra, co)?;
    }
    vals.extend_from_slice(&offsets);
    vals.extend_from_slice(&modes);
    let packed = packed_weights(q, weights, dt, ci, co)?;
    let layout = LayoutKind::Chain(3 + extras.len() as u8);
    let shape = [ci, co, h, w];
    let pipeline = q.context().chain_pipeline(
        ("conv_coop", dt, acc, shape, tile, epilogue.to_vec(), extras.len()),
        layout,
        EntryPoint::typed("conv2d_coop", dt),
        || shader(Dialect::CURRENT, dt, acc, epilogue, extras.len(), [ci, h, w], tile),
    )?;
    let mut buffers = vec![get_wgpu_buffer(input), get_wgpu_buffer(&packed)];
    buffers.extend(extras.iter().map(|t| get_wgpu_buffer(t)));
    buffers.push(get_wgpu_buffer(output));
    let bg = q.context().bind_group(layout, &buffers, q.uniform())?;
    let dyn_off = q.alloc_uniform(&pack_u32s(&vals))?;
    let grid = tile.groups(co, h, w).map(|g| g as u32);
    q.dispatch_grid("conv_coop", &pipeline, &bg, dyn_off, grid)?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::TRACT_WGPU_SUBGROUPS;
    use tract_core::ops::cnn::{PaddingSpec, PoolSpec};

    #[test]
    fn tile_covers_the_map_and_keeps_the_device_busy() {
        for (ci, co, h, w) in
            [(35, 16, 720, 1280), (59, 32, 360, 640), (107, 40, 180, 320), (64, 64, 24, 40)]
        {
            let t = tile(ci, co, h, w).unwrap();
            assert!(t.rows * t.pixels * t.channels <= ACCUMULATORS.max(t.channels * t.pixels));
            assert!(w.is_multiple_of(t.width()) && (co / BLOCK).is_multiple_of(t.channels));
            assert!(t.rows <= h && t.workgroup_bytes() <= 32768);
        }
    }

    #[test]
    fn small_maps_split_the_channels_over_subgroups() {
        assert_eq!(tile(35, 16, 720, 1280).unwrap().split, 1);
        assert_eq!(tile(176, 64, 16, 16).unwrap().split, 2);
        assert!(tile(8, 8, 8, 16).unwrap().split == 1);
    }

    #[test]
    fn tile_declines_what_the_kernel_cannot_run() {
        assert!(tile(16, 16, 64, 64).is_some());
        assert!(tile(7, 16, 64, 64).is_none());
        assert!(tile(16, 20, 64, 64).is_none());
        assert!(tile(16, 16, 64, 60).is_none());
        assert!(tile(16, 16, 64, 8).is_none());
        assert!(tile(16, 16, 7, 64).is_none());
        assert!(tile(16, 0, 64, 64).is_none());
    }

    #[test]
    fn edge_columns_and_rows_have_their_own_paths() {
        let t = tile(16, 16, 32, 32).unwrap();
        let source =
            shader(Dialect::Naga, ShaderDtype::F16, ShaderDtype::F32, &[], 0, [16, 32, 32], t);
        assert!(source.contains("left | right | top | bottom"));
        assert!(source.contains("if (!top)") && source.contains("if (!bottom)"));
        assert!(source.contains("coop_mat8x8<f32, C>()"));
        assert!(!source.contains("left || right"));
    }

    #[test]
    fn dawn_reads_the_flat_map_and_accumulates_in_f16() {
        let t = tile(16, 16, 32, 32).unwrap();
        let source =
            shader(Dialect::Dawn, ShaderDtype::F16, ShaderDtype::F16, &[], 0, [16, 32, 32], t);
        assert!(source.contains("var<storage, read> inp: array<f16>;"));
        assert!(source.contains("subgroup_matrix_result<f16, 8, 8>()"));
        assert!(source.contains("fn epi(acc: f16"));
        assert!(!source.contains("coop_mat") && !source.contains("coopLoad"));
    }

    /// Products of 300 x 300 overflow `f16` and cancel only when they are added in `f32`.
    #[test]
    fn a_sum_that_overflows_f16_still_cancels() -> TractResult<()> {
        use tract_gpu::tensor::IntoDevice;
        if !TRACT_WGPU_SUBGROUPS.get()
            || accumulator(crate::wgpu_context().subgroup_caps(), ShaderDtype::F16, false).is_none()
        {
            return Ok(());
        }
        let (ci, co, h, w) = (16, 16, 16, 32);
        let weights = (0..co * ci * TAPS)
            .map(|i| f16::from_f32(if ((i / TAPS) % ci).is_multiple_of(2) { 300. } else { -300. }))
            .collect::<Vec<_>>();
        let x = vec![f16::from_f32(300.); ci * h * w];
        let x = Tensor::from_shape(&[1, ci, h, w], &x)?.into_device()?;
        let weights = Tensor::from_shape(&[co, ci, 3, 3], &weights)?.into_device()?;
        let bias = Tensor::from_shape(&[co], &vec![f16::from_f32(0.); co])?.into_device()?;
        let output = DeviceTensor::uninitialized_dt(DatumType::F16, &[1, co, h, w])?;
        let epilogue = [ChainStep::Binary { op: "add".into(), rhs: 1, swapped: false }];
        crate::kernels::conv::wgpu_conv_dispatch(
            &conv3x3(ci, co),
            &epilogue,
            &x,
            &weights,
            &[&bias],
            &output,
            true,
            None,
            None,
            None,
        )?;
        let got = pollster::block_on(crate::to_host_async(&output))?;
        let got = got.to_plain_array_view::<f16>()?;
        ensure!(got.iter().all(|v| *v == f16::from_f32(0.)), "the sum did not cancel");
        Ok(())
    }

    fn run(x: &[f32], w: &DeviceTensor, [ci, co, hh, ww]: [usize; 4]) -> TractResult<Vec<f32>> {
        use tract_gpu::tensor::IntoDevice;
        let x = Tensor::from_shape(&[1, ci, hh, ww], x)?.into_device()?;
        let output = DeviceTensor::uninitialized_dt(DatumType::F32, &[1, co, hh, ww])?;
        crate::kernels::conv::wgpu_conv_dispatch(
            &conv3x3(ci, co),
            &[],
            &x,
            w,
            &[],
            &output,
            false,
            None,
            None,
            None,
        )?;
        let got = pollster::block_on(crate::to_host_async(&output))?;
        Ok(got.to_plain_array_view::<f32>()?.iter().copied().collect())
    }

    fn engages() -> bool {
        TRACT_WGPU_SUBGROUPS.get()
            && accumulator(crate::wgpu_context().subgroup_caps(), ShaderDtype::F32, false).is_some()
    }

    /// The last block of channels shares channels with the block before when `ci` is not a multiple
    /// of 8. A shared infinite channel stays infinite.
    #[test]
    fn an_infinite_input_stays_infinite_in_a_shared_channel() -> TractResult<()> {
        use tract_gpu::tensor::IntoDevice;
        if !engages() {
            return Ok(());
        }
        let shape @ [ci, co, h, w] = [12, 8, 8, 16];
        let weights = Tensor::from_shape(&[co, ci, 3, 3], &vec![0.5f32; co * ci * TAPS])?;
        let mut x = vec![1f32; ci * h * w];
        x[5 * h * w + 3 * w + 4] = f32::INFINITY;
        let got = run(&x, &weights.into_device()?, shape)?;
        ensure!(got.iter().all(|v| !v.is_nan()), "an infinite input became NaN");
        ensure!(got[3 * w + 4] == f32::INFINITY, "the infinity was lost: {}", got[3 * w + 4]);
        Ok(())
    }

    /// A pooled buffer is handed to another tensor, so weights packed from it would go stale.
    #[test]
    fn pooled_weights_are_not_packed() -> TractResult<()> {
        if !engages() {
            return Ok(());
        }
        let shape @ [ci, co, h, w] = [16, 8, 8, 16];
        let weights = DeviceTensor::uninitialized_dt(DatumType::F32, &[co, ci, 3, 3])?;
        ensure!(get_wgpu_buffer(&weights).pooled, "the weights are not pooled");
        crate::set_dispatch_capture(true);
        let _ = crate::take_dispatch_labels();
        run(&vec![1f32; ci * h * w], &weights, shape)?;
        let labels = crate::take_dispatch_labels();
        ensure!(!labels.contains(&"conv_coop"), "pooled weights were packed: {labels:?}");
        Ok(())
    }

    fn conv3x3(ci: usize, co: usize) -> Conv {
        Conv {
            pool_spec: PoolSpec {
                data_format: DataFormat::NCHW,
                kernel_shape: tvec![3, 3],
                padding: PaddingSpec::Explicit(tvec![1, 1], tvec![1, 1]),
                dilations: None,
                strides: None,
                input_channels: ci,
                output_channels: co,
            },
            kernel_fmt: KernelFormat::OIHW,
            group: 1,
            q_params: None,
        }
    }

    /// Device values. Needs `TRACT_WGPU_SUBGROUPS=1` set before the first device is created:
    /// `TRACT_WGPU_SUBGROUPS=1 cargo test -p tract-wgpu --lib coop_conv -- --test-threads=1`.
    #[test]
    fn coop_conv_matches_a_plain_convolution() -> TractResult<()> {
        if !TRACT_WGPU_SUBGROUPS.get() {
            eprintln!("cooperative conv test skipped: TRACT_WGPU_SUBGROUPS is off");
            return Ok(());
        }
        let caps = *crate::wgpu_context().subgroup_caps();
        if accumulator(&caps, ShaderDtype::F16, false).is_none() {
            eprintln!("cooperative conv test skipped: {caps:?}");
            return Ok(());
        }
        // Wide channels, ci not a multiple of 8, an odd height, a channel split, one tile of 8 rows,
        // and a 16-wide map whose two edge tiles meet.
        for (ci, co, h, w) in [
            (16, 16, 24, 32),
            (19, 24, 33, 40),
            (12, 40, 17, 48),
            (176, 64, 16, 16),
            (64, 8, 12, 24),
            (8, 16, 8, 16),
        ] {
            check::<f16>(ci, co, h, w).with_context(|| format!("f16 {ci}>{co} {h}x{w}"))?;
            check::<f32>(ci, co, h, w).with_context(|| format!("f32 {ci}>{co} {h}x{w}"))?;
        }
        Ok(())
    }

    fn check<T: Datum + num_traits::Float + num_traits::FromPrimitive>(
        ci: usize,
        co: usize,
        h: usize,
        w: usize,
    ) -> TractResult<()> {
        use tract_gpu::tensor::IntoDevice;
        let dt = T::datum_type();
        let fill = |seed: u64, len: usize, scale: f64| -> Vec<T> {
            let mut s = seed ^ 0x9e37_79b9_7f4a_7c15;
            (0..len)
                .map(|_| {
                    s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    T::from_f64(((s >> 33) as f64 / u32::MAX as f64 - 0.5) * scale).unwrap()
                })
                .collect()
        };
        let x_h = fill(11, ci * h * w, 1.0);
        let w_h = fill(12, co * ci * TAPS, (4.0 / (TAPS * ci) as f64).sqrt());
        let bias_h = fill(13, co, 0.5);
        let x = Tensor::from_shape(&[1, ci, h, w], &x_h)?.into_device()?;
        let weights = Tensor::from_shape(&[co, ci, 3, 3], &w_h)?.into_device()?;
        let bias = Tensor::from_shape(&[co], &bias_h)?.into_device()?;
        let output = DeviceTensor::uninitialized_dt(dt, &[1, co, h, w])?;
        let op = conv3x3(ci, co);
        let epilogue = [
            ChainStep::Binary { op: "add".into(), rhs: 1, swapped: false },
            ChainStep::Unary("abs".into()),
        ];
        crate::set_dispatch_capture(true);
        let _ = crate::take_dispatch_labels();
        crate::kernels::conv::wgpu_conv_dispatch(
            &op,
            &epilogue,
            &x,
            &weights,
            &[&bias],
            &output,
            false,
            None,
            None,
            None,
        )?;
        let got = pollster::block_on(crate::to_host_async(&output))?;
        let labels = crate::take_dispatch_labels();
        ensure!(labels.contains(&"conv_coop"), "the kernel did not run: {labels:?}");
        let got = got.to_plain_array_view::<T>()?;
        let got = got.as_slice().context("the output is not contiguous")?;
        let to = |v: T| v.to_f64().unwrap();
        for (o, y, xx) in
            (0..co).flat_map(|o| (0..h).flat_map(move |y| (0..w).map(move |xx| (o, y, xx))))
        {
            let mut sum = to(bias_h[o]);
            for c in 0..ci {
                for tap in 0..TAPS {
                    let (iy, ix) = (y + tap / 3, xx + tap % 3);
                    if iy == 0 || ix == 0 || iy > h || ix > w {
                        continue;
                    }
                    sum +=
                        to(w_h[(o * ci + c) * TAPS + tap]) * to(x_h[(c * h + iy - 1) * w + ix - 1]);
                }
            }
            let want = sum.abs();
            let have = to(got[(o * h + y) * w + xx]);
            let tolerance =
                if dt == DatumType::F16 { (want * 2f64.powi(-10)).max(5e-4) } else { 1e-5 };
            ensure!(
                (have - want).abs() <= tolerance,
                "channel {o} at {y},{xx}: {have} instead of {want}"
            );
        }
        Ok(())
    }
}
