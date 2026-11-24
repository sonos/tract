//! Cooperative-matrix 3x3 and 5x5 convolution of small maps with many channels: stride 1, same
//! padding, channel-first, zero padded.
//!
//! A first pass copies the map with its zero border, so no tap leaves the copy. A subgroup owns a
//! tile of rows x 8 pixels x 8-channel blocks and runs one 8x8 product per tap. Several subgroups
//! split the input channels, because such a map has too few tiles to fill the device. A width that
//! is not a multiple of 8 makes the last tile overlap the one before.

use std::fmt::Write;

use tract_core::internal::*;
use tract_core::ops::cnn::conv::Conv;
use tract_gpu::tensor::DeviceTensor;

use crate::context::WgpuQueue;
use crate::kernels::coop_conv::{
    BLOCK, LANES, head, map_type, packed_weights, site, uniform, weights_type,
};
use crate::kernels::coop_dialect::{Dialect, Side};
use crate::kernels::shaders::{ChainStep, EntryPoint, LayoutKind, ShaderDtype};
use crate::kernels::subgroup::limits_of;
use crate::utils::get_wgpu_buffer;

/// Workgroups from which the input channels stay with one subgroup, and with two.
const UNSPLIT_GROUPS: usize = 512;
const HALVED_GROUPS: usize = 256;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
struct Tile {
    rows: usize,
    /// 8-channel output blocks.
    channels: usize,
    /// Subgroups the input channels are split over.
    split: usize,
}

impl Tile {
    fn groups(self, co: usize, h: usize, w: usize) -> [usize; 3] {
        [w.div_ceil(BLOCK), h.div_ceil(self.rows), co / BLOCK / self.channels]
    }

    fn span(self) -> usize {
        self.rows * BLOCK * self.channels * BLOCK
    }

    fn workgroup_bytes(self) -> usize {
        self.span() * self.split * size_of::<f32>()
    }
}

fn tile(ci: usize, co: usize, h: usize, w: usize) -> Option<Tile> {
    if ci < BLOCK || co == 0 || !co.is_multiple_of(BLOCK) || h == 0 || w < BLOCK {
        return None;
    }
    let channels = if (co / BLOCK).is_multiple_of(2) { 2 } else { 1 };
    let tile = Tile { rows: if h >= 3 { 3 } else { 1 }, channels, split: 1 };
    let groups = tile.groups(co, h, w).iter().product::<usize>();
    let split = match groups {
        UNSPLIT_GROUPS.. => 1,
        HALVED_GROUPS.. => 2,
        _ => 4,
    };
    Some(Tile { split, ..tile })
}

/// Copies a `[ci, h, w]` map with `radius` zero rows and columns around it.
fn pad_shader(dt: ShaderDtype, h: usize, w: usize, radius: usize) -> String {
    let (ph, pw, t) = (h + 2 * radius, w + 2 * radius, dt.wgsl());
    let enable = if dt == ShaderDtype::F16 { "enable f16;\n" } else { "" };
    format!(
        r#"{enable}@group(0) @binding(0) var<storage, read> inp: array<{t}>;
@group(0) @binding(1) var<storage, read_write> outp: array<{t}>;
@group(0) @binding(2) var<uniform> params: vec4<u32>;

@compute @workgroup_size(8, 8)
fn conv2d_pad_{}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    if (gid.x >= {pw}u || gid.y >= {ph}u) {{ return; }}
    let x = gid.x - {radius}u;
    let y = gid.y - {radius}u;
    var v = {t}(0.0);
    if (x < {w}u && y < {h}u) {{ v = inp[(gid.z * {h}u + y) * {w}u + x]; }}
    outp[(gid.z * {ph}u + gid.y) * {pw}u + gid.x] = v;
}}
"#,
        dt.suffix(),
    )
}

#[allow(clippy::too_many_arguments)]
fn shader(
    dialect: Dialect,
    dt: ShaderDtype,
    acc: ShaderDtype,
    epilogue: &[ChainStep],
    extras: usize,
    kernel: usize,
    [ci, h, w]: [usize; 3],
    tile: Tile,
) -> String {
    let Tile { rows, channels, split } = tile;
    let (t, st) = (dt.wgsl(), acc.wgsl());
    let pw = w + kernel - 1;
    let map = (h + kernel - 1) * pw;
    let (in_blocks, span, threads) = (ci.div_ceil(BLOCK), tile.span(), LANES * split);
    let row_span = rows * BLOCK;
    let blocks: Vec<_> = (0..rows).flat_map(|d| (0..channels).map(move |c| (d, c))).collect();
    let mut s = head(
        dt,
        acc,
        epilogue,
        extras,
        &map_type(dialect, dt, map),
        &weights_type(dt, kernel * kernel, in_blocks, channels),
    );
    writeln!(s, "var<workgroup> stage: array<{st}, {}>;", span * split).unwrap();
    writeln!(
        s,
        "@compute @workgroup_size({threads})\nfn conv2d_coop_pad_{}(@builtin(workgroup_id) wid: vec3<u32>, @builtin(local_invocation_index) li: u32, @builtin(subgroup_id) sg: u32) {{\n    let x0 = min(wid.x * 8u, {}u);\n    let y0 = min(wid.y * {rows}u, {}u);\n    let g = wid.z;",
        dt.suffix(),
        w - BLOCK,
        h - rows,
    )
    .unwrap();
    for u in 0..rows + kernel - 1 {
        for k in 0..kernel {
            writeln!(s, "    let o{u}_{k} = (y0 + {u}u) * {pw}u + x0 + {k}u;").unwrap();
        }
    }
    for (d, c) in &blocks {
        writeln!(s, "    var a{d}_{c} = {};", dialect.accumulator(st)).unwrap();
    }
    writeln!(
        s,
        "    for (var b = sg; b < {in_blocks}u; b += {split}u) {{\n        let cs = min(b * 8u, {}u);",
        ci - BLOCK
    )
    .unwrap();
    for k in 0..kernel {
        s.push_str("        {\n");
        for u in 0..rows + kernel - 1 {
            let load = dialect.load(
                Side::Right,
                t,
                &format!("inp[cs][o{u}_{k}]"),
                ("inp", &format!("cs * {map}u + o{u}_{k}")),
                map,
            );
            writeln!(s, "        let i{u} = {load};").unwrap();
        }
        for (c, row) in (0..channels).flat_map(|c| (0..kernel).map(move |row| (c, row))) {
            let block = format!("wgt[g][{c}][b][{}]", row * kernel + k);
            let load = dialect.load(Side::Left, t, &format!("{block}[0]"), (&block, "0u"), BLOCK);
            writeln!(s, "        {{ let wm = {load};").unwrap();
            for d in 0..rows {
                let sum = format!("a{d}_{c}");
                let add = dialect.multiply_add("wm", &format!("i{}", d + row), &sum);
                writeln!(s, "            {sum} = {add};").unwrap();
            }
            s.push_str("        }\n");
        }
        s.push_str("        }\n");
    }
    writeln!(s, "    }}\n    let base = sg * {span}u;").unwrap();
    for (d, c) in &blocks {
        let at = format!("base + {}u", (c * BLOCK * rows + d) * BLOCK);
        let store = dialect.store(&format!("a{d}_{c}"), "stage", &at, row_span);
        writeln!(s, "    {store}").unwrap();
    }
    let sum =
        (0..split).map(|q| format!("stage[e + {}u]", span * q)).collect::<Vec<_>>().join(" + ");
    writeln!(
        s,
        "    workgroupBarrier();\n    for (var e = li; e < {span}u; e += {threads}u) {{\n        let co = g * {}u + e / {row_span}u;\n        let d = (e / 8u) % {rows}u;\n        outp[params.off_out + (co * {h}u + y0 + d) * {w}u + x0 + e % 8u] = epi({sum}, co);\n    }}\n}}",
        channels * BLOCK,
    )
    .unwrap();
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
    let kernel = match op.pool_spec.kernel_shape[..] {
        [3, 3] => 3,
        [5, 5] => 5,
        _ => return Ok(false),
    };
    let Some((acc, [ci, co, h, w])) =
        site(q, op, kernel, input, weights, extras, output, dt, f16_accumulation)
    else {
        return Ok(false);
    };
    let Some(tile) = tile(ci, co, h, w) else { return Ok(false) };
    let (max_invocations, max_x) = limits_of(q);
    let storage = q.context().device().limits().max_compute_workgroup_storage_size;
    if (LANES * tile.split) as u32 > max_invocations.min(max_x)
        || tile.workgroup_bytes() as u32 > storage
    {
        return Ok(false);
    }
    let radius = kernel / 2;
    let (ph, pw) = (h + 2 * radius, w + 2 * radius);
    let dyn_off = q.alloc_uniform(&uniform(output, extras, co)?)?;
    let packed = packed_weights(q, weights, dt, ci, co, kernel * kernel)?;
    let padded = DeviceTensor::uninitialized_dt(input.datum_type(), &[ci, ph, pw])?;
    q.retain_tensor(&padded);
    let two = LayoutKind::Chain(2);
    let pad = q.context().chain_pipeline(
        ("conv_pad", dt, h, w, radius),
        two,
        EntryPoint::typed("conv2d_pad", dt),
        || pad_shader(dt, h, w, radius),
    )?;
    let buffers = [get_wgpu_buffer(input), get_wgpu_buffer(&padded)];
    let bg = q.context().bind_group(two, &buffers, q.uniform())?;
    let grid = [pw.div_ceil(8) as u32, ph.div_ceil(8) as u32, ci as u32];
    q.dispatch_grid("conv_pad", &pad, &bg, dyn_off, grid)?;
    let layout = LayoutKind::Chain(3 + extras.len() as u8);
    let pipeline = q.context().chain_pipeline(
        ("conv_coop_pad", dt, acc, kernel, [ci, co, h, w], tile, epilogue.to_vec(), extras.len()),
        layout,
        EntryPoint::typed("conv2d_coop_pad", dt),
        || shader(Dialect::CURRENT, dt, acc, epilogue, extras.len(), kernel, [ci, h, w], tile),
    )?;
    let mut buffers = vec![get_wgpu_buffer(&padded), get_wgpu_buffer(&packed)];
    buffers.extend(extras.iter().map(|t| get_wgpu_buffer(t)));
    buffers.push(get_wgpu_buffer(output));
    let bg = q.context().bind_group(layout, &buffers, q.uniform())?;
    let grid = tile.groups(co, h, w).map(|g| g as u32);
    q.dispatch_grid("conv_coop_pad", &pipeline, &bg, dyn_off, grid)?;
    Ok(true)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::kernels::TRACT_WGPU_SUBGROUPS;
    use crate::kernels::coop_dialect::accumulator;
    use crate::with_wgpu_queue;
    use tract_core::ops::cnn::{KernelFormat, PaddingSpec, PoolSpec};
    use tract_core::ops::nn::DataFormat;
    use tract_gpu::tensor::IntoDevice;

    #[test]
    fn small_maps_split_the_channels_over_subgroups() {
        assert_eq!(tile(1280, 96, 9, 16).unwrap(), Tile { rows: 3, channels: 2, split: 4 });
        assert_eq!(tile(96, 32, 72, 48).unwrap().split, 2);
        assert_eq!(tile(96, 64, 72, 64).unwrap().split, 1);
        assert_eq!(tile(16, 24, 2, 8).unwrap(), Tile { rows: 1, channels: 1, split: 4 });
    }

    #[test]
    fn tile_declines_what_the_kernel_cannot_run() {
        assert!(tile(7, 16, 9, 16).is_none());
        assert!(tile(16, 12, 9, 16).is_none());
        assert!(tile(16, 0, 9, 16).is_none());
        assert!(tile(16, 16, 9, 7).is_none());
        assert!(tile(16, 16, 0, 16).is_none());
    }

    #[test]
    fn dawn_reads_the_flat_map() {
        let (dt, t) = (ShaderDtype::F16, tile(96, 32, 18, 32).unwrap());
        let source = shader(Dialect::Dawn, dt, dt, &[], 0, 5, [96, 18, 32], t);
        assert!(source.contains("var<storage, read> inp: array<f16>;"));
        assert!(source.contains("subgroup_matrix_result<f16, 8, 8>()"));
        assert!(!source.contains("coop_mat") && !source.contains("coopLoad"));
    }

    fn engages(dt: ShaderDtype) -> bool {
        TRACT_WGPU_SUBGROUPS.get()
            && accumulator(crate::wgpu_context().subgroup_caps(), dt, false).is_some()
    }

    fn same_conv(kernel: usize, ci: usize, co: usize) -> Conv {
        let radius = kernel / 2;
        Conv {
            pool_spec: PoolSpec {
                data_format: DataFormat::NCHW,
                kernel_shape: tvec![kernel, kernel],
                padding: PaddingSpec::Explicit(tvec![radius, radius], tvec![radius, radius]),
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

    /// The kernel's output, and whether it took the convolution.
    fn run<T: Datum + Copy>(
        kernel: usize,
        [ci, co, h, w]: [usize; 4],
        x: &[T],
        weights: &DeviceTensor,
        bias: Option<&[T]>,
    ) -> TractResult<Option<Vec<T>>> {
        let x = Tensor::from_shape(&[1, ci, h, w], x)?.into_device()?;
        let bias = bias.map(|b| Tensor::from_shape(&[co], b)?.into_device()).transpose()?;
        let output = DeviceTensor::uninitialized_dt(T::datum_type(), &[1, co, h, w])?;
        let epilogue = [
            ChainStep::Binary { op: "add".into(), rhs: 1, swapped: false },
            ChainStep::Unary("abs".into()),
        ];
        let epilogue = if bias.is_some() { &epilogue[..] } else { &[] };
        let extras: Vec<&DeviceTensor> = bias.iter().collect();
        let dt = ShaderDtype::from_datum(T::datum_type())?;
        let op = same_conv(kernel, ci, co);
        let ran = with_wgpu_queue(|q| {
            for t in [&x, weights, &output].into_iter().chain(extras.iter().copied()) {
                q.retain_tensor(t);
            }
            try_dispatch(q, &op, epilogue, &x, weights, &extras, &output, dt, false)
        })?;
        if !ran {
            return Ok(None);
        }
        let got = pollster::block_on(crate::to_host_async(&output))?;
        Ok(Some(got.to_plain_array_view::<T>()?.iter().cloned().collect()))
    }

    fn check<T: Datum + num_traits::Float + num_traits::FromPrimitive>(
        kernel: usize,
        shape @ [ci, co, h, w]: [usize; 4],
    ) -> TractResult<()> {
        let taps = kernel * kernel;
        let radius = kernel / 2;
        let fill = |seed: u64, len: usize, scale: f64| -> Vec<T> {
            let mut s = seed ^ 0x9e37_79b9_7f4a_7c15;
            (0..len)
                .map(|_| {
                    s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
                    T::from_f64(((s >> 33) as f64 / u32::MAX as f64 - 0.5) * scale).unwrap()
                })
                .collect()
        };
        let x = fill(11, ci * h * w, 1.0);
        let wgt = fill(12, co * ci * taps, (4.0 / (taps * ci) as f64).sqrt());
        let bias = fill(13, co, 0.5);
        let weights = Tensor::from_shape(&[co, ci, kernel, kernel], &wgt)?.into_device()?;
        let got = run(kernel, shape, &x, &weights, Some(&bias))?.context("the kernel declined")?;
        let to = |v: T| v.to_f64().unwrap();
        for (o, y, xx) in
            (0..co).flat_map(|o| (0..h).flat_map(move |y| (0..w).map(move |xx| (o, y, xx))))
        {
            let mut sum = to(bias[o]);
            for (c, tap) in (0..ci).flat_map(|c| (0..taps).map(move |tap| (c, tap))) {
                let (iy, ix) = (y + tap / kernel, xx + tap % kernel);
                if iy < radius || ix < radius || iy >= h + radius || ix >= w + radius {
                    continue;
                }
                sum += to(wgt[(o * ci + c) * taps + tap])
                    * to(x[(c * h + iy - radius) * w + ix - radius]);
            }
            let (want, have) = (sum.abs(), to(got[(o * h + y) * w + xx]));
            let tolerance = if T::datum_type() == DatumType::F16 {
                (want * 2f64.powi(-10)).max(5e-4)
            } else {
                want.max(1.) * 1e-5
            };
            ensure!((have - want).abs() <= tolerance, "{o} at {y},{xx}: {have} instead of {want}");
        }
        Ok(())
    }

    /// Device values. Needs `TRACT_WGPU_SUBGROUPS=1` set before the first device is created:
    /// `TRACT_WGPU_SUBGROUPS=1 cargo test -p tract-wgpu --lib coop_conv -- --test-threads=1`.
    #[test]
    fn padded_coop_conv_matches_a_plain_convolution() -> TractResult<()> {
        if !engages(ShaderDtype::F16) {
            eprintln!("padded cooperative conv test skipped");
            return Ok(());
        }
        // Many channels, ci and w not multiples of 8, one row, an odd channel block count, fewer
        // input blocks than subgroups, and no split.
        for (kernel, shape) in [
            (5, [320, 96, 9, 16]),
            (5, [19, 24, 7, 13]),
            (5, [12, 16, 1, 9]),
            (5, [16, 64, 24, 40]),
            (5, [8, 64, 72, 64]),
            (3, [67, 32, 5, 11]),
            (3, [64, 16, 2, 8]),
        ] {
            check::<f16>(kernel, shape).with_context(|| format!("f16 {kernel} {shape:?}"))?;
            check::<f32>(kernel, shape).with_context(|| format!("f32 {kernel} {shape:?}"))?;
        }
        Ok(())
    }

    /// Products of 300 x 300 overflow `f16` and cancel only when they are added in `f32`.
    #[test]
    fn a_sum_that_overflows_f16_still_cancels() -> TractResult<()> {
        if !engages(ShaderDtype::F16) {
            return Ok(());
        }
        let shape @ [ci, co, h, w] = [16, 16, 9, 16];
        let weights = (0..co * ci * 25)
            .map(|i: usize| {
                f16::from_f32(if ((i / 25) % ci).is_multiple_of(2) { 300. } else { -300. })
            })
            .collect::<Vec<_>>();
        let weights = Tensor::from_shape(&[co, ci, 5, 5], &weights)?.into_device()?;
        let x = vec![f16::from_f32(300.); ci * h * w];
        let got = run(5, shape, &x, &weights, None)?.context("the kernel declined")?;
        ensure!(got.iter().all(|v| *v == f16::from_f32(0.)), "the sum did not cancel");
        Ok(())
    }

    /// The last block of channels shares channels with the block before when `ci` is not a multiple
    /// of 8. A shared infinite channel stays infinite.
    #[test]
    fn an_infinite_input_stays_infinite_in_a_shared_channel() -> TractResult<()> {
        if !engages(ShaderDtype::F32) {
            return Ok(());
        }
        let shape @ [ci, co, h, w] = [12, 8, 9, 16];
        let weights = Tensor::from_shape(&[co, ci, 5, 5], &vec![0.5f32; co * ci * 25])?;
        let mut x = vec![1f32; ci * h * w];
        x[(5 * h + 3) * w + 4] = f32::INFINITY;
        let got = run(5, shape, &x, &weights.into_device()?, None)?.context("declined")?;
        ensure!(got.iter().all(|v| !v.is_nan()), "an infinite input became NaN");
        ensure!(got[3 * w + 4] == f32::INFINITY, "the infinity was lost: {}", got[3 * w + 4]);
        Ok(())
    }

    /// A pooled buffer is handed to another tensor, so weights packed from it would go stale.
    #[test]
    fn pooled_weights_are_declined() -> TractResult<()> {
        if !engages(ShaderDtype::F32) {
            return Ok(());
        }
        let shape @ [ci, co, h, w] = [16, 8, 9, 16];
        let weights = DeviceTensor::uninitialized_dt(DatumType::F32, &[co, ci, 5, 5])?;
        ensure!(get_wgpu_buffer(&weights).pooled, "the weights are not pooled");
        ensure!(run(5, shape, &vec![1f32; ci * h * w], &weights, None)?.is_none());
        Ok(())
    }

    /// The small maps the tiled kernel serves take this one.
    #[test]
    fn fat_small_map_conv_routes_here() -> TractResult<()> {
        if !engages(ShaderDtype::F32) {
            return Ok(());
        }
        let [ci, co, h, w] = [1280, 96, 9, 16];
        let x = Tensor::zero::<f32>(&[1, ci, h, w])?.into_device()?;
        let weights = Tensor::zero::<f32>(&[co, ci, 5, 5])?.into_device()?;
        let output = DeviceTensor::uninitialized_dt(DatumType::F32, &[1, co, h, w])?;
        crate::set_dispatch_capture(true);
        let _ = crate::take_dispatch_labels();
        crate::kernels::conv::wgpu_conv_dispatch(
            &same_conv(5, ci, co),
            &[],
            &x,
            &weights,
            &[],
            &output,
            false,
            None,
            None,
            None,
        )?;
        let labels = crate::take_dispatch_labels();
        ensure!(labels.contains(&"conv_coop_pad"), "the kernel did not run: {labels:?}");
        Ok(())
    }
}
