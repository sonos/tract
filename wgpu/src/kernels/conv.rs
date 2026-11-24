use tract_core::internal::*;
use tract_core::ops::cnn::Conv;
use tract_gpu::tensor::DeviceTensor;

use crate::kernels::conv_splitk;
use crate::kernels::shaders::{
    ChainStep, EntryPoint, GRID_LIMIT, LayoutKind, ModuleKey, ModuleKind, PipelineKey, ShaderDtype,
    WORKGROUP, binary_ops_wgsl, epilogue_body, keys_for, pack_u32s, preamble, unary_ops_wgsl,
};
use crate::kernels::tiled_gemm::{
    MATMUL_TILE, MATMUL_TILE_K, TILED_THREADS, tile_rows, tiled_gemm_core, tiled_gemm_tiles,
};
use crate::utils::{element_offset, get_wgpu_buffer};
use crate::with_wgpu_queue;

pub fn all_pipeline_keys(shader_f16: bool) -> Vec<PipelineKey> {
    keys_for(ModuleKind::Conv, &["conv2d"], shader_f16)
}

/// Whether the implicit-GEMM kernel should serve this convolution: per group,
/// enough output channels, pixels and reduction to fill a good part of its
/// tiles, with every tile index inside one grid dimension. Sixteen output
/// channels are enough once the map and the reduction are large. The number of
/// tiles does not matter: a few tiles staging a long reduction through
/// workgroup memory still beat one thread per output reading it from global
/// memory.
fn tiled_applies(groups: usize, co_pg: usize, pixels: usize, k: usize) -> bool {
    (co_pg >= 32 || (co_pg >= 16 && pixels >= 14400 && k >= 256))
        && pixels >= 32
        && k >= 16
        && groups <= 65535
        && co_pg.div_ceil(MATMUL_TILE) <= 65535
        && pixels.div_ceil(MATMUL_TILE) <= 65535
}

fn as_u32_i32(v: i32) -> u32 {
    v as u32
}

/// Channels-last, ungrouped 3x3, dilation 1, stride 1 or 2, `co <= 20`, and a
/// pixel count that fits one dispatch dimension.
#[allow(clippy::too_many_arguments)]
fn thin_nhwc_pixel(
    channels_last: u32,
    groups: usize,
    kh: usize,
    kw: usize,
    dilations: &[usize],
    strides: &[usize],
    co: usize,
    pixels: usize,
) -> bool {
    let stride_ok = matches!(strides, [1, 1] | [2, 2]);
    let dil_ok = matches!(dilations, [1, 1]);
    channels_last == 1
        && groups == 1
        && kh == 3
        && kw == 3
        && dil_ok
        && stride_ok
        && co <= 20
        && pixels.div_ceil(CONV_PIXEL_WG as usize) <= 65535
}

/// Whether the vec4 pixel kernel applies: packed `[co, 9, ci]` weights,
/// `ci >= 16`, `ci` and `co` multiples of 4, `co <= 20`, channel stride 1,
/// and all other strides and buffer offsets 4-aligned.
fn pixel_vec_ready(
    packed: Option<&DeviceTensor>,
    input: &DeviceTensor,
    output: &DeviceTensor,
    ci: usize,
    in_strides: [usize; 4],
    out_strides: [usize; 4],
) -> bool {
    let Some(packed) = packed else { return false };
    let &[co, taps, packed_ci] = packed.shape() else { return false };
    let aligned = |strides: [usize; 4]| {
        strides[1] == 1
            && strides[3].is_multiple_of(4)
            && (strides[0] == 0 || strides[0].is_multiple_of(4))
            && strides[2].is_multiple_of(4)
    };
    ci >= 16
        && ci.is_multiple_of(4)
        && packed_ci == ci
        && co.is_multiple_of(4)
        && co <= 20
        && taps == 9
        && aligned(in_strides)
        && aligned(out_strides)
        && element_offset(input, 0).is_multiple_of(4)
        && element_offset(output, 0).is_multiple_of(4)
        && element_offset(packed, 0).is_multiple_of(4)
}

/// One filter per channel, no cross-channel sum: the tiled kernel applies, in
/// either data format, since it addresses input and output through their strides.
fn depthwise_shape(op: &Conv) -> Option<DepthwiseShape> {
    let spec = &op.pool_spec;
    if op.group == 1
        || op.group != spec.input_channels
        || spec.output_channels != spec.input_channels
        || spec.kernel_shape.len() != 2
    {
        return None;
    }
    let strides = spec.strides();
    let dilations = spec.dilations();
    Some(DepthwiseShape {
        kh: spec.kernel_shape[0] as u32,
        kw: spec.kernel_shape[1] as u32,
        stride_h: strides[0] as u32,
        stride_w: strides[1] as u32,
        dil_h: dilations[0] as u32,
        dil_w: dilations[1] as u32,
    })
}

#[allow(clippy::too_many_arguments)]
fn dispatch_depthwise(
    q: &crate::WgpuQueue,
    op: &Conv,
    shape: DepthwiseShape,
    epilogue: &[ChainStep],
    input: &DeviceTensor,
    weights: &DeviceTensor,
    extras: &[&DeviceTensor],
    output: &DeviceTensor,
    f16_accumulation: bool,
) -> TractResult<()> {
    let dt = ShaderDtype::from_datum(input.datum_type())?;
    let layout = LayoutKind::Chain(3 + extras.len() as u8);
    let in_shape = op.pool_spec.data_format.shape(input.shape())?;
    let out_shape = op.pool_spec.data_format.shape(output.shape())?;
    let n = *in_shape.n().unwrap_or(&1);
    let co = *out_shape.c();
    let (oh, ow) = (out_shape.hw_dims()[0], out_shape.hw_dims()[1]);
    let padding = op.pool_spec.computed_padding(in_shape.hw_dims());
    let ws = weights.strides();
    let in_strides = [
        *in_shape.n_stride().unwrap_or(&0),
        *in_shape.c_stride(),
        in_shape.hw_strides()[0],
        in_shape.hw_strides()[1],
    ];
    let f16_acc = dt == ShaderDtype::F16 && f16_accumulation;
    let pad = [padding[0].pad_before as i32, padding[1].pad_before as i32];
    let out_strides = [
        *out_shape.n_stride().unwrap_or(&0),
        *out_shape.c_stride(),
        out_shape.hw_strides()[0],
        out_shape.hw_strides()[1],
    ];
    let channel_groups = depthwise_channel_tiling(
        op.pool_spec.data_format.c_is_last(),
        shape,
        co,
        [n, oh, ow],
        in_strides,
        out_strides,
        [element_offset(input, 0), element_offset(output, 0)],
    );
    let vec_rows = depthwise_vec_tiling(
        op,
        shape,
        pad,
        in_shape.hw_dims()[1],
        in_strides,
        element_offset(input, 0),
    )
    .filter(|_| dt != ShaderDtype::F32 || (shape.kh < 9 && shape.kw < 9));
    let pipeline = if channel_groups {
        let key = ("conv_dw_ch", shape, dt, f16_acc, epilogue, extras.len());
        q.context().chain_pipeline(key, layout, EntryPoint::typed("conv_dw_ch", dt), || {
            conv_depthwise_channel_module(dt, f16_acc, shape, epilogue, extras.len())
        })?
    } else if let Some(rows) = vec_rows {
        let key = ("conv_dw_vec", shape, pad, rows, dt, f16_acc, epilogue, extras.len());
        q.context().chain_pipeline(key, layout, EntryPoint::typed("conv_dw_vec", dt), || {
            conv_depthwise_vec_module(dt, f16_acc, shape, pad, rows, epilogue, extras.len())
        })?
    } else {
        let key = ("conv_dw", shape, dt, epilogue, extras.len());
        q.context().chain_pipeline(key, layout, EntryPoint::typed("conv_dw", dt), || {
            conv_depthwise_module(dt, shape, epilogue, extras.len())
        })?
    };

    let mut buffers: Vec<&crate::context::WgpuBuffer> =
        vec![get_wgpu_buffer(input), get_wgpu_buffer(weights)];
    buffers.extend(extras.iter().map(|t| get_wgpu_buffer(t)));
    buffers.push(get_wgpu_buffer(output));
    let bg = q.context().bind_group(layout, &buffers, q.uniform())?;

    let mut off_extra = [0u32; 4];
    let mut mode_extra = [0u32; 4];
    for (i, t) in extras.iter().enumerate() {
        off_extra[i] = element_offset(t, 0) as u32;
        mode_extra[i] = crate::kernels::matmul::epilogue_mode(t, co)?;
    }
    let mut vals = vec![
        element_offset(input, 0) as u32,
        element_offset(weights, 0) as u32,
        element_offset(output, 0) as u32,
        n as u32,
        co as u32,
        in_shape.hw_dims()[0] as u32,
        in_shape.hw_dims()[1] as u32,
        oh as u32,
        ow as u32,
        padding[0].pad_before as u32,
        padding[1].pad_before as u32,
        0,
        as_u32_i32(*in_shape.n_stride().unwrap_or(&0) as i32),
        as_u32_i32(*in_shape.c_stride() as i32),
        as_u32_i32(in_shape.hw_strides()[0] as i32),
        as_u32_i32(in_shape.hw_strides()[1] as i32),
        as_u32_i32(ws[0] as i32),
        as_u32_i32(ws[2] as i32),
        as_u32_i32(ws[3] as i32),
        0,
        as_u32_i32(*out_shape.n_stride().unwrap_or(&0) as i32),
        as_u32_i32(*out_shape.c_stride() as i32),
        as_u32_i32(out_shape.hw_strides()[0] as i32),
        as_u32_i32(out_shape.hw_strides()[1] as i32),
    ];
    vals.extend_from_slice(&off_extra);
    vals.extend_from_slice(&mode_extra);
    let dyn_off = q.alloc_uniform(&pack_u32s(&vals))?;
    if channel_groups {
        let threads = n * oh * ow.div_ceil(DIRECT_PX) * (co / DIRECT_PX);
        let groups = [threads.div_ceil(CONV_DIRECT_WG as usize) as u32, 1, 1];
        return q.dispatch_grid("conv_depthwise_ch", &pipeline, &bg, dyn_off, groups);
    }
    if let Some(rows) = vec_rows {
        let threads = n * oh.div_ceil(rows) * ow.div_ceil(DIRECT_PX);
        let groups = [threads.div_ceil(CONV_DIRECT_WG as usize) as u32, co as u32, 1];
        return q.dispatch_grid("conv_depthwise", &pipeline, &bg, dyn_off, groups);
    }
    let wg = shape.wg();
    let groups = [(ow as u32).div_ceil(wg), (oh as u32).div_ceil(wg), (n * co) as u32];
    q.dispatch_grid("conv_depthwise", &pipeline, &bg, dyn_off, groups)
}

#[allow(clippy::too_many_arguments)]
pub fn wgpu_conv_dispatch(
    op: &Conv,
    epilogue: &[ChainStep],
    input: &DeviceTensor,
    weights: &DeviceTensor,
    extras: &[&DeviceTensor],
    output: &DeviceTensor,
    f16_accumulation: bool,
    packed_weights: Option<&DeviceTensor>,
    packed_tile: Option<DirectTile>,
    pixel_packed: Option<&DeviceTensor>,
) -> TractResult<()> {
    with_wgpu_queue(|q| {
        q.retain_tensor(input);
        q.retain_tensor(weights);
        for t in extras {
            q.retain_tensor(t);
        }
        q.retain_tensor(output);
        ensure!(matches!(input.datum_type(), DatumType::F32 | DatumType::F16));
        if let Some(shape) = depthwise_shape(op) {
            return dispatch_depthwise(
                q,
                op,
                shape,
                epilogue,
                input,
                weights,
                extras,
                output,
                f16_accumulation,
            );
        }
        let dt = ShaderDtype::from_datum(input.datum_type())?;
        if packed_weights.is_none()
            && crate::kernels::coop_conv::try_dispatch(
                q,
                op,
                epilogue,
                input,
                weights,
                extras,
                output,
                dt,
                f16_accumulation,
            )?
        {
            return Ok(());
        }
        let f16_acc = dt == ShaderDtype::F16 && f16_accumulation;
        let layout = LayoutKind::Chain(3 + extras.len() as u8);
        let in_shape = op.pool_spec.data_format.shape(input.shape())?;
        let out_shape = op.pool_spec.data_format.shape(output.shape())?;
        ensure!(in_shape.hw_rank() == 2, "tract-wgpu conv is 2D only");
        let n = *in_shape.n().unwrap_or(&1);
        let ci = *in_shape.c();
        let co = *out_shape.c();
        let ih = in_shape.hw_dims()[0];
        let iw = in_shape.hw_dims()[1];
        let oh = out_shape.hw_dims()[0];
        let ow = out_shape.hw_dims()[1];
        let kh = op.pool_spec.kernel_shape[0];
        let kw = op.pool_spec.kernel_shape[1];
        let groups = op.group;
        let ci_pg = op.pool_spec.input_channels / groups;
        let co_pg = op.pool_spec.output_channels / groups;
        let padding = op.pool_spec.computed_padding(in_shape.hw_dims());
        let strides = op.pool_spec.strides();
        let dilations = op.pool_spec.dilations();
        let channels_last = op.pool_spec.data_format.c_is_last() as u32;

        let in_strides = [
            *in_shape.n_stride().unwrap_or(&0),
            *in_shape.c_stride(),
            in_shape.hw_strides()[0],
            in_shape.hw_strides()[1],
        ];
        let direct = direct_tiling(
            op,
            dt,
            f16_acc,
            input,
            [padding[0].pad_before, padding[1].pad_before],
            [n, co, ih, iw, oh, ow],
            in_strides,
        );
        // Packed rows are a vec4 with a zero in `.w`. Native tests may request
        // them; the browser uses them only on Apple, where that load was measured.
        let packed_weights = packed_weights
            .filter(|_| cfg!(test) || crate::kernels::matmul::apple_webgpu(q.context().device()));
        let packed = packed_weights.is_some();
        if packed {
            ensure!(
                direct.is_some() && f16_acc,
                "packed direct weights require the f16 direct kernel"
            );
        }
        let weights = packed_weights.unwrap_or(weights);
        if packed {
            q.retain_tensor(weights);
            ensure!(
                dt == ShaderDtype::F16
                    && weights.datum_type() == DatumType::F16
                    && weights.shape() == [co, ci_pg, 3, 4]
                    && weights.strides() == [(ci_pg * 12) as isize, 12, 4, 1]
                    && element_offset(weights, 0).is_multiple_of(4),
                "packed direct weights must be contiguous f16 [co, ci, 3, 4]"
            );
        }
        let direct = direct.map(|tile| if packed { packed_tile.unwrap_or(tile) } else { tile });
        let pixels = n * oh * ow;
        let out_sn = *out_shape.n_stride().unwrap_or(&0);
        let out_sc = *out_shape.c_stride();
        let out_sh = out_shape.hw_strides()[0];
        let out_sw = out_shape.hw_strides()[1];
        let out_strides = [out_sn, out_sc, out_sh, out_sw];
        // Thin channels-last 3x3 layers use the per-pixel kernel. Only the
        // packed vec4 variant takes priority over tiled GEMM.
        let thin_nhwc = thin_nhwc_pixel(
            channels_last,
            groups,
            kh,
            kw,
            dilations.as_ref(),
            strides.as_ref(),
            co_pg,
            pixels,
        );
        let pixel_vec = thin_nhwc
            && pixel_vec_ready(pixel_packed, input, output, ci, in_strides, out_strides)
            && (pixels * (co / 4)).div_ceil(CONV_PIXEL_WG as usize) <= 65535;
        let tiled =
            direct.is_none() && !pixel_vec && tiled_applies(groups, co_pg, pixels, ci_pg * kh * kw);
        let tile_m = tile_rows(co_pg);
        // The per-pixel kernel unrolls 3x3 taps at dilation 1 and wants every
        // output channel of a pixel from one thread. With one thread per pixel
        // it only keeps the device busy when that thread's work is a stem's,
        // or when the pixels alone are enough threads.
        let per_pixel = direct.is_none()
            && !pixel_vec
            && !tiled
            && (thin_nhwc
                || (channels_last == 0
                    && groups == 1
                    && kh == 3
                    && kw == 3
                    && dilations[0] == 1
                    && dilations[1] == 1
                    && (ci * co <= 64 || pixels >= 32768)))
            && pixels.div_ceil(CONV_PIXEL_WG as usize) <= 65535;
        let in_sn = *in_shape.n_stride().unwrap_or(&0);
        let in_sc = *in_shape.c_stride();
        let in_sh = in_shape.hw_strides()[0];
        let in_sw = in_shape.hw_strides()[1];

        // OIHW: [co, ci_pg, kh, kw]. The vec4 pixel kernel replaces that buffer
        // with weights packed [co, 9, ci].
        let weight_buf = if pixel_vec {
            let packed = pixel_packed.unwrap();
            q.retain_tensor(packed);
            packed
        } else {
            weights
        };
        let (w_so, w_si, w_sh, w_sw) = if pixel_vec {
            (0isize, 0, 0, 0)
        } else {
            let ws = weights.strides();
            ensure!(ws.len() >= 4, "expected 4D OIHW kernel, got {:?}", weights.shape());
            (ws[0], ws[1], ws[2], ws[3])
        };

        let params = pack_u32s(&[
            element_offset(input, 0) as u32,
            element_offset(weight_buf, 0) as u32,
            element_offset(output, 0) as u32,
            n as u32,
            ci as u32,
            co as u32,
            ih as u32,
            iw as u32,
            oh as u32,
            ow as u32,
            kh as u32,
            kw as u32,
            groups as u32,
            ci_pg as u32,
            co_pg as u32,
            channels_last,
            padding[0].pad_before as u32,
            padding[1].pad_before as u32,
            strides[0] as u32,
            strides[1] as u32,
            dilations[0] as u32,
            dilations[1] as u32,
            as_u32_i32(in_sn as i32),
            as_u32_i32(in_sc as i32),
            as_u32_i32(in_sh as i32),
            as_u32_i32(in_sw as i32),
            as_u32_i32(w_so as i32),
            as_u32_i32(w_si as i32),
            as_u32_i32(w_sh as i32),
            as_u32_i32(w_sw as i32),
            as_u32_i32(out_sn as i32),
            as_u32_i32(out_sc as i32),
            as_u32_i32(out_sh as i32),
            as_u32_i32(out_sw as i32),
        ]);
        let mut tail = vec![0u32; 2];
        let mut off_extra = [0u32; 4];
        let mut mode_extra = [0u32; 4];
        for (i, t) in extras.iter().enumerate() {
            off_extra[i] = element_offset(t, 0) as u32;
            mode_extra[i] = crate::kernels::matmul::epilogue_mode(t, co)?;
        }
        tail.extend_from_slice(&off_extra);
        tail.extend_from_slice(&mode_extra);
        let mut params = params;
        params.extend_from_slice(&pack_u32s(&tail));
        let tiled_grid =
            [pixels.div_ceil(MATMUL_TILE) as u32, co_pg.div_ceil(tile_m) as u32, groups as u32];
        if tiled
            && crate::kernels::coop_conv_padded::try_dispatch(
                q,
                op,
                epilogue,
                input,
                weights,
                extras,
                output,
                dt,
                f16_accumulation,
            )?
        {
            return Ok(());
        }
        if tiled
            && let Some(slices) =
                conv_splitk::slices(dt, groups, co_pg, pixels, ci_pg * kh * kw, tile_m)
        {
            return conv_splitk::dispatch(
                q, dt, f16_acc, tile_m, slices, epilogue, input, weights, extras, output, params,
                tiled_grid,
            );
        }
        let pipeline = if let Some(tile) = direct {
            let entry = if packed { "conv2d_direct_packed_weights" } else { "conv2d_direct" };
            q.context().chain_pipeline(
                (entry, dt, f16_acc, tile.co, tile.rows, epilogue, extras.len()),
                layout,
                EntryPoint::typed(entry, dt),
                || conv_direct_module(dt, f16_acc, epilogue, extras.len(), tile, packed),
            )?
        } else if pixel_vec {
            q.context().chain_pipeline(
                ("conv_pixel_vec", dt, f16_acc, epilogue, extras.len()),
                layout,
                EntryPoint::typed("conv2d_pixel_vec", dt),
                || conv_pixel_vec_module(dt, f16_acc, epilogue, extras.len()),
            )?
        } else if tiled {
            q.context().chain_pipeline(
                ("conv_tiled", tile_m, dt, epilogue, extras.len()),
                layout,
                EntryPoint::typed("conv2d_tiled", dt),
                || conv_tiled_module(dt, epilogue, extras.len(), tile_m),
            )?
        } else if per_pixel {
            // Four output channels re-read the same 3×3 patch. Holding every
            // channel (at most 16) loads that patch once per input channel.
            let wide = ci <= 4 && co <= 16;
            let vec4_acc = co >= 6;
            let entry = if wide { "conv2d_pixel_wide" } else { "conv2d_pixel" };
            q.context().chain_pipeline(
                ("conv_pixel", dt, epilogue, extras.len(), wide, vec4_acc),
                layout,
                EntryPoint::typed(entry, dt),
                || conv_pixel_module(dt, epilogue, extras.len(), wide, vec4_acc),
            )?
        } else if epilogue.is_empty() {
            q.context().pipeline(PipelineKey {
                module: ModuleKey { kind: ModuleKind::Conv, dtype: dt },
                entry: EntryPoint::typed("conv2d", dt),
            })?
        } else {
            let key = ("conv", dt, epilogue, extras.len());
            q.context().chain_pipeline(key, layout, EntryPoint::typed("conv2d", dt), || {
                conv_module(dt, epilogue, extras.len())
            })?
        };

        let mut buffers: Vec<&crate::context::WgpuBuffer> =
            vec![get_wgpu_buffer(input), get_wgpu_buffer(weight_buf)];
        buffers.extend(extras.iter().map(|t| get_wgpu_buffer(t)));
        buffers.push(get_wgpu_buffer(output));
        let bg = q.context().bind_group(layout, &buffers, q.uniform())?;
        let dyn_off = q.alloc_uniform(&params)?;
        if let Some(tile) = direct {
            let grid = [
                tile.threads(n, oh, ow).div_ceil(CONV_DIRECT_WG as usize) as u32,
                co.div_ceil(tile.co) as u32,
                1,
            ];
            let label = if packed { "conv_direct_packed_weights" } else { "conv_direct" };
            return q.dispatch_grid(label, &pipeline, &bg, dyn_off, grid);
        }
        if pixel_vec {
            let threads = pixels * (co / 4);
            let groups = threads.div_ceil(CONV_PIXEL_WG as usize) as u32;
            return q.dispatch_grid("conv_pixel_vec", &pipeline, &bg, dyn_off, [groups, 1, 1]);
        }
        if tiled {
            return q.dispatch_grid("conv_tiled", &pipeline, &bg, dyn_off, tiled_grid);
        }
        if per_pixel {
            let groups = pixels.div_ceil(CONV_PIXEL_WG as usize) as u32;
            let label = if ci <= 4 && co <= 16 { "conv_pixel_wide" } else { "conv_pixel" };
            return q.dispatch_grid(label, &pipeline, &bg, dyn_off, [groups, 1, 1]);
        }
        q.dispatch("conv", &pipeline, &bg, dyn_off, (n * co * oh * ow) as u64)
    })
}

/// Workgroup edge of the depthwise tile: 16x16 output values per workgroup.
pub const DW_WG: u32 = 16;

/// Workgroup edge when the halo is left to the cache instead of staged.
pub const DW_DIRECT_WG: u32 = 8;

/// Shape of one depthwise convolution, baked into its program so the filter
/// loops unroll and the halo tile is sized exactly.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct DepthwiseShape {
    pub kh: u32,
    pub kw: u32,
    pub stride_h: u32,
    pub stride_w: u32,
    pub dil_h: u32,
    pub dil_w: u32,
}

impl DepthwiseShape {
    /// Whether to read the input directly rather than stage a halo tile. Taps
    /// this few reuse too little for the staging and its barrier to repay
    /// themselves and the overlap is served from cache; from 7x7 up the tile
    /// wins instead.
    pub fn reads_direct(&self) -> bool {
        self.kh <= 3 && self.kw <= 3
    }

    /// Output values along each edge of one workgroup's tile.
    pub fn wg(&self) -> u32 {
        if self.reads_direct() { DW_DIRECT_WG } else { DW_WG }
    }

    fn tile_h(&self) -> u32 {
        (DW_WG - 1) * self.stride_h + (self.kh - 1) * self.dil_h + 1
    }

    fn tile_w(&self) -> u32 {
        (DW_WG - 1) * self.stride_w + (self.kw - 1) * self.dil_w + 1
    }
}

/// The workgroup arrays the staged body reads through.
fn depthwise_tile_decls(shape: DepthwiseShape) -> String {
    let DepthwiseShape { kh, kw, .. } = shape;
    let (tile_h, tile_w) = (shape.tile_h(), shape.tile_w());
    format!(
        "var<workgroup> x_tile: array<array<f32, {tile_w}>, {tile_h}>;\n\
         var<workgroup> w_tile: array<array<f32, {kw}>, {kh}>;\n\n"
    )
}

/// Every thread stages its share of the workgroup's input halo and the filter
/// in workgroup memory, so each input value is read once rather than once per
/// tap.
fn depthwise_staged_body(shape: DepthwiseShape) -> String {
    let DepthwiseShape { kh, kw, stride_h, stride_w, dil_h, dil_w } = shape;
    let (tile_h, tile_w) = (shape.tile_h(), shape.tile_w());
    let wg = shape.wg();
    format!(
        r#"
    for (var r = lid.y; r < {tile_h}u; r = r + {wg}u) {{
        for (var c = lid.x; c < {tile_w}u; c = c + {wg}u) {{
            let ih = ih0 + i32(r);
            let iw = iw0 + i32(c);
            var v = 0.0;
            if (ih >= 0 && ih < i32(params.ih) && iw >= 0 && iw < i32(params.iw)) {{
                let idx = params.off_in + u32(
                    i32(n) * params.in_sn
                    + i32(co) * params.in_sc
                    + ih * params.in_sh
                    + iw * params.in_sw
                );
                v = f32(inp[idx]);
            }}
            x_tile[r][c] = v;
        }}
    }}
    if (lidx < {kh}u * {kw}u) {{
        let wr = lidx / {kw}u;
        let wc = lidx % {kw}u;
        let idx = params.off_w + u32(
            i32(co) * params.w_so + i32(wr) * params.w_sh + i32(wc) * params.w_sw
        );
        w_tile[wr][wc] = f32(wgt[idx]);
    }}
    workgroupBarrier();

    let oh = oh0 + lid.y;
    let ow = ow0 + lid.x;
    if (oh >= params.oh || ow >= params.ow) {{ return; }}
    var acc = 0.0;
    for (var wr = 0u; wr < {kh}u; wr++) {{
        for (var wc = 0u; wc < {kw}u; wc++) {{
            acc = fma(
                x_tile[lid.y * {stride_h}u + wr * {dil_h}u][lid.x * {stride_w}u + wc * {dil_w}u],
                w_tile[wr][wc],
                acc
            );
        }}
    }}
"#
    )
}

/// Each thread reads its own taps straight from the input, leaving the overlap
/// between neighbours to the cache: with few taps there is too little reuse to
/// repay staging it, and dropping the workgroup arrays and their barrier frees
/// the occupancy that pays instead.
fn depthwise_direct_body(shape: DepthwiseShape) -> String {
    let DepthwiseShape { kh, kw, stride_h, stride_w, dil_h, dil_w } = shape;
    format!(
        r#"
    let oh = oh0 + lid.y;
    let ow = ow0 + lid.x;
    if (oh >= params.oh || ow >= params.ow) {{ return; }}
    var acc = 0.0;
    for (var wr = 0u; wr < {kh}u; wr++) {{
        for (var wc = 0u; wc < {kw}u; wc++) {{
            let ih = ih0 + i32(lid.y) * {stride_h} + i32(wr) * {dil_h};
            let iw = iw0 + i32(lid.x) * {stride_w} + i32(wc) * {dil_w};
            var v = 0.0;
            if (ih >= 0 && ih < i32(params.ih) && iw >= 0 && iw < i32(params.iw)) {{
                let idx = params.off_in + u32(
                    i32(n) * params.in_sn
                    + i32(co) * params.in_sc
                    + ih * params.in_sh
                    + iw * params.in_sw
                );
                v = f32(inp[idx]);
            }}
            let w_idx = params.off_w + u32(
                i32(co) * params.w_so + i32(wr) * params.w_sh + i32(wc) * params.w_sw
            );
            acc = fma(v, f32(wgt[w_idx]), acc);
        }}
    }}
"#
    )
}

/// Depthwise convolution, ported from tfjs-backend-webgpu
/// `DepthwiseConv2DNCHWSharedProgram`
/// (tfjs-backend-webgpu/src/depthwise_conv2d_nchw_shared_webgpu.ts, Apache-2.0,
/// Copyright 2021 Google LLC): a workgroup owns one 16x16 output tile of one
/// channel, staging that tile's input halo and the filter in workgroup memory,
/// so each input value is read once instead of once per filter tap. Strides and
/// dilations size the halo here, where tfjs assumed one; indices come from
/// tract's strides, and the epilogue replaces their bias/activation snippet.
pub fn conv_depthwise_module(
    dt: ShaderDtype,
    shape: DepthwiseShape,
    epilogue: &[ChainStep],
    extras: usize,
) -> String {
    let t = dt.wgsl();
    let suf = dt.suffix();
    let DepthwiseShape { stride_h, stride_w, .. } = shape;
    let wg = shape.wg();
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
    let epilogue = epilogue_body(epilogue, "co");
    let (tile_decls, accumulate) = if shape.reads_direct() {
        (String::new(), depthwise_direct_body(shape))
    } else {
        (depthwise_tile_decls(shape), depthwise_staged_body(shape))
    };
    let mut s = preamble(dt);
    s.push_str(&format!(
        r#"
struct Params {{
    off_in: u32,
    off_w: u32,
    off_out: u32,
    n: u32,
    co: u32,
    ih: u32,
    iw: u32,
    oh: u32,
    ow: u32,
    pad_h: i32,
    pad_w: i32,
    _p0: u32,
    in_sn: i32,
    in_sc: i32,
    in_sh: i32,
    in_sw: i32,
    w_so: i32,
    w_sh: i32,
    w_sw: i32,
    _p1: u32,
    out_sn: i32,
    out_sc: i32,
    out_sh: i32,
    out_sw: i32,
    off_extra: vec4<u32>,
    mode_extra: vec4<u32>,
}}

@group(0) @binding(0) var<storage, read> inp: array<{t}>;
@group(0) @binding(1) var<storage, read> wgt: array<{t}>;
{extra_bindings}@group(0) @binding({out_binding}) var<storage, read_write> outp: array<{t}>;
@group(0) @binding({uniform_binding}) var<uniform> params: Params;
{lib}
{tile_decls}@compute @workgroup_size({wg}, {wg}, 1)
fn conv_dw_{suf}(
    @builtin(workgroup_id) wid: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(local_invocation_index) lidx: u32,
) {{
    let co = wid.z % params.co;
    let n = wid.z / params.co;
    let oh0 = wid.y * {wg}u;
    let ow0 = wid.x * {wg}u;
    let ih0 = i32(oh0) * {stride_h} - params.pad_h;
    let iw0 = i32(ow0) * {stride_w} - params.pad_w;

{accumulate}    var v = {t}(acc);
{epilogue}
    let out_i = params.off_out + u32(
        i32(n) * params.out_sn
        + i32(co) * params.out_sc
        + i32(oh) * params.out_sh
        + i32(ow) * params.out_sw
    );
    outp[out_i] = v;
}}
"#
    ));
    s
}

/// Widest input window, in `vec4`s, the vector depthwise kernel keeps in
/// registers per input row.
const DW_VEC_MAX_LOADS: i32 = 4;

impl DepthwiseShape {
    /// Input-row offsets, relative to the first output row's stride origin,
    /// that `rows` output rows read, in first-use order.
    fn vec_row_offsets(&self, rows: usize, pad_h: i32) -> Vec<i32> {
        let mut out = vec![];
        for d in 0..rows {
            for r in 0..self.kh as usize {
                let ry = self.vec_row_offset(d, r, pad_h);
                if !out.contains(&ry) {
                    out.push(ry);
                }
            }
        }
        out
    }

    fn vec_row_offset(&self, d: usize, r: usize, pad_h: i32) -> i32 {
        d as i32 * self.stride_h as i32 + r as i32 * self.dil_h as i32 - pad_h
    }

    /// Lane of input, relative to `x0 * stride_w`, that output pixel `j` reads
    /// for tap column `k`.
    fn vec_lane(&self, j: usize, k: usize, pad_w: i32) -> i32 {
        j as i32 * self.stride_w as i32 + k as i32 * self.dil_w as i32 - pad_w
    }

    /// Range of `vec4` indices, relative to `x0 * stride_w / 4`, one row's
    /// window spans.
    fn vec_load_range(&self, pad_w: i32) -> (i32, i32) {
        let lo = self.vec_lane(0, 0, pad_w).div_euclid(4);
        let hi = self.vec_lane(DIRECT_PX - 1, self.kw as usize - 1, pad_w).div_euclid(4);
        (lo, hi)
    }
}

/// Channels-last depthwise whose threads walk channels. One thread owns four
/// adjacent channels and four output columns, so a subgroup reads one pixel's
/// channels contiguously. Below 32 channels that subgroup is only partly full
/// and the spatial kernel is faster, which is the full-resolution stage of
/// these segmenters.
fn depthwise_channel_tiling(
    channels_last: bool,
    shape: DepthwiseShape,
    co: usize,
    grid: [usize; 3],
    in_strides: [usize; 4],
    out_strides: [usize; 4],
    offsets: [usize; 2],
) -> bool {
    if !channels_last
        || shape.kh > 5
        || shape.kw > 5
        || co < 32
        || !co.is_multiple_of(DIRECT_PX)
        || in_strides[1] != 1
        || out_strides[1] != 1
        || !offsets[0].is_multiple_of(DIRECT_PX)
        || !offsets[1].is_multiple_of(DIRECT_PX)
    {
        return false;
    }
    let aligned = |stride: usize| stride == 0 || stride.is_multiple_of(DIRECT_PX);
    if !aligned(in_strides[0])
        || !aligned(out_strides[0])
        || !in_strides[2].is_multiple_of(DIRECT_PX)
        || !in_strides[3].is_multiple_of(DIRECT_PX)
        || !out_strides[2].is_multiple_of(DIRECT_PX)
        || !out_strides[3].is_multiple_of(DIRECT_PX)
    {
        return false;
    }
    let [n, oh, ow] = grid;
    let threads = n * oh * ow.div_ceil(DIRECT_PX) * (co / DIRECT_PX);
    threads.div_ceil(CONV_DIRECT_WG as usize) <= GRID_LIMIT as usize
}

/// Register-blocked channels-last depthwise. Input and the four owned channels
/// are contiguous, so each tap is one `vec4` load per output column. A tap
/// outside the input contributes zero through `select` with an in-range index,
/// so a padded read never fetches the neighbour. Serves the convolutions
/// [`depthwise_channel_tiling`] admits. The uniform block matches
/// [`conv_depthwise_vec_module`].
fn conv_depthwise_channel_module(
    dt: ShaderDtype,
    f16_accumulation: bool,
    shape: DepthwiseShape,
    epilogue: &[ChainStep],
    extras: usize,
) -> String {
    let t = dt.wgsl();
    let a = if dt == ShaderDtype::F16 && f16_accumulation { "f16" } else { "f32" };
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
        format!("{}{}", unary_ops_wgsl(a), binary_ops_wgsl(a))
    };
    let epilogue = conv_direct_epilogue(epilogue, t != a);
    let result = if t == a { "v".to_owned() } else { format!("{t}(v)") };
    let widen = if t == a { String::new() } else { a.to_string() };
    let DepthwiseShape { kh, kw, stride_h, stride_w, dil_h, dil_w } = shape;
    let weight = |lane: u32| {
        let loaded = format!("wgt[u32(wbase + i32(c0 + {lane}u) * params.w_so)]");
        if widen.is_empty() { loaded } else { format!("{widen}({loaded})") }
    };
    let mut taps = String::new();
    let mut stores = String::new();
    for j in 0..DIRECT_PX {
        taps.push_str(&format!(
            r#"
        let iw{j} = i32(x0 + {j}u) * {stride_w} + i32(wc) * {dil_w} - params.pad_w;
        let xok{j} = yok && iw{j} >= 0 && iw{j} < i32(params.iw) && x0 + {j}u < params.ow;
        let iw_s{j} = select(0, iw{j}, xok{j});
        let idx{j} = i32(params.off_in) + i32(n) * params.in_sn + i32(c0) * params.in_sc + ih_s * params.in_sh + iw_s{j} * params.in_sw;
        let raw{j} = select(vec4<{t}>(0.0), inp[u32(idx{j} / 4)], xok{j});
"#
        ));
        let value = if widen.is_empty() { format!("raw{j}") } else { format!("vec4<{a}>(raw{j})") };
        taps.push_str(&format!("        acc{j} = fma(wv, {value}, acc{j});\n"));
        stores.push_str(&format!(
            r#"    if (x0 + {j}u < params.ow) {{
        let o{j} = u32(obase + i32(x0 + {j}u) * params.out_sw);
        outp[o{j}] = epi(acc{j}.x, c0);
        outp[o{j} + 1u] = epi(acc{j}.y, c0 + 1u);
        outp[o{j} + 2u] = epi(acc{j}.z, c0 + 2u);
        outp[o{j} + 3u] = epi(acc{j}.w, c0 + 3u);
    }}
"#
        ));
    }
    format!(
        r#"
{preamble}
struct Params {{
    off_in: u32,
    off_w: u32,
    off_out: u32,
    n: u32,
    co: u32,
    ih: u32,
    iw: u32,
    oh: u32,
    ow: u32,
    pad_h: i32,
    pad_w: i32,
    _p0: u32,
    in_sn: i32,
    in_sc: i32,
    in_sh: i32,
    in_sw: i32,
    w_so: i32,
    w_sh: i32,
    w_sw: i32,
    _p1: u32,
    out_sn: i32,
    out_sc: i32,
    out_sh: i32,
    out_sw: i32,
    off_extra: vec4<u32>,
    mode_extra: vec4<u32>,
}}

@group(0) @binding(0) var<storage, read> inp: array<vec4<{t}>>;
@group(0) @binding(1) var<storage, read> wgt: array<{t}>;
{extra_bindings}@group(0) @binding({out_binding}) var<storage, read_write> outp: array<{t}>;
@group(0) @binding({uniform_binding}) var<uniform> params: Params;
{lib}
fn epi(acc: {a}, co: u32) -> {t} {{
    var v = acc;
{epilogue}
    return {result};
}}

@compute @workgroup_size({wg})
fn conv_dw_ch_{suf}(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>) {{
    let c_threads = params.co / 4u;
    let x_blocks = (params.ow + 3u) / 4u;
    let i = wid.x * {wg}u + lid;
    if (i >= params.n * params.oh * x_blocks * c_threads) {{ return; }}
    let cv = i % c_threads;
    let t1 = i / c_threads;
    let xq = t1 % x_blocks;
    let t2 = t1 / x_blocks;
    let y = t2 % params.oh;
    let n = t2 / params.oh;
    let c0 = cv * 4u;
    let x0 = xq * 4u;
    var acc0 = vec4<{a}>(0.0);
    var acc1 = vec4<{a}>(0.0);
    var acc2 = vec4<{a}>(0.0);
    var acc3 = vec4<{a}>(0.0);
    for (var wr = 0u; wr < {kh}u; wr++) {{
        let ih = i32(y) * {stride_h} + i32(wr) * {dil_h} - params.pad_h;
        let yok = ih >= 0 && ih < i32(params.ih);
        let ih_s = select(0, ih, yok);
        for (var wc = 0u; wc < {kw}u; wc++) {{
            let wbase = i32(params.off_w) + i32(wr) * params.w_sh + i32(wc) * params.w_sw;
            let wv = vec4<{a}>({w0}, {w1}, {w2}, {w3});
{taps}        }}
    }}
    let obase = i32(params.off_out) + i32(n) * params.out_sn + i32(y) * params.out_sh + i32(c0) * params.out_sc;
{stores}}}
"#,
        preamble = preamble(dt),
        wg = CONV_DIRECT_WG,
        w0 = weight(0),
        w1 = weight(1),
        w2 = weight(2),
        w3 = weight(3),
        taps = taps,
        stores = stores,
    )
}

/// Output rows per thread of the vector depthwise kernel, if it can serve this
/// convolution: NCHW, the input `vec4`-aligned along W, and a window narrow
/// enough to hold in registers. Unit stride and dilation along H let two
/// output rows share their loaded rows; otherwise a thread owns one.
fn depthwise_vec_tiling(
    op: &Conv,
    shape: DepthwiseShape,
    pad: [i32; 2],
    iw: usize,
    in_strides: [usize; 4],
    in_offset: usize,
) -> Option<usize> {
    if op.pool_spec.data_format.c_is_last()
        || in_strides[3] != 1
        || !iw.is_multiple_of(DIRECT_PX)
        || !in_strides[..3].iter().all(|s| s.is_multiple_of(DIRECT_PX))
        || !in_offset.is_multiple_of(DIRECT_PX)
    {
        return None;
    }
    let (lo, hi) = shape.vec_load_range(pad[1]);
    if hi - lo + 1 > DW_VEC_MAX_LOADS {
        return None;
    }
    Some(if shape.stride_h == 1 && shape.dil_h == 1 { 2 } else { 1 })
}

fn vec_name(v: i32) -> String {
    if v < 0 { format!("m{}", -v) } else { v.to_string() }
}

/// Register-blocked depthwise convolution: each thread owns `DIRECT_PX`
/// consecutive output pixels of `rows` rows for one channel, loads every input
/// row its taps touch once as `vec4`s along W, and gathers each tap's window
/// from those registers. Padded taps read zero through `select`, so an `inf`
/// or `NaN` outside the input never leaks into a padded output. f16 inputs are
/// widened as they are read and the accumulator and epilogue run in f32; with
/// `f16_accumulation`, f16 stays f16 throughout. Serves the convolutions
/// [`depthwise_vec_tiling`] admits.
pub fn conv_depthwise_vec_module(
    dt: ShaderDtype,
    f16_accumulation: bool,
    shape: DepthwiseShape,
    pad: [i32; 2],
    rows: usize,
    epilogue: &[ChainStep],
    extras: usize,
) -> String {
    use std::fmt::Write;
    let t = dt.wgsl();
    let a = if dt == ShaderDtype::F16 && f16_accumulation { "f16" } else { "f32" };
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
        format!("{}{}", unary_ops_wgsl(a), binary_ops_wgsl(a))
    };
    let epilogue = conv_direct_epilogue(epilogue, t != a);
    let result = if t == a { "v".to_owned() } else { format!("{t}(v)") };
    let DepthwiseShape { kh, kw, stride_h, stride_w, .. } = shape;
    let [pad_h, pad_w] = pad;
    let (lo, hi) = shape.vec_load_range(pad_w);
    let lanes = ["x", "y", "z", "w"];

    let mut body = String::new();
    for d in 0..rows {
        writeln!(body, "    var acc{d} = vec4<{a}>(0.0);").unwrap();
    }
    for r in 0..kh as usize {
        for k in 0..kw as usize {
            let value = format!(
                "wgt[params.off_w + u32(i32(co) * params.w_so + {r} * params.w_sh + {k} * params.w_sw)]"
            );
            let value = if t == a { value } else { format!("{a}({value})") };
            writeln!(body, "    let w{r}_{k} = vec4<{a}>({value});").unwrap();
        }
    }
    writeln!(body, "    let xv = i32(xq) * {stride_w};").unwrap();
    writeln!(body, "    let xvw = i32(params.iw / 4u);").unwrap();
    for v in lo..=hi {
        let vn = vec_name(v);
        writeln!(body, "    let xok{vn} = xv + {v} >= 0 && xv + {v} < xvw;").unwrap();
    }
    writeln!(
        body,
        "    let in_base = i32(params.off_in) + i32(n) * params.in_sn + i32(co) * params.in_sc;"
    )
    .unwrap();
    writeln!(body, "    let ry0 = i32(y0) * {stride_h};").unwrap();
    let offsets = shape.vec_row_offsets(rows, pad_h);
    for (u, ry) in offsets.iter().enumerate() {
        writeln!(body, "    let iy{u} = ry0 + {ry};").unwrap();
        writeln!(body, "    let ok{u} = iy{u} >= 0 && iy{u} < i32(params.ih);").unwrap();
        writeln!(
            body,
            "    let rb{u} = i32(in_base + select(0, iy{u}, ok{u}) * params.in_sh) / 4 + xv;"
        )
        .unwrap();
        for v in lo..=hi {
            let vn = vec_name(v);
            let load = format!("select(vec4<{t}>(0.0), inp[u32(rb{u} + {v})], ok{u} && xok{vn})");
            let load = if t == a { load } else { format!("vec4<{a}>({load})") };
            writeln!(body, "    let a{u}_{vn} = {load};").unwrap();
        }
    }
    for d in 0..rows {
        for r in 0..kh as usize {
            let u = offsets.iter().position(|o| *o == shape.vec_row_offset(d, r, pad_h)).unwrap();
            for k in 0..kw as usize {
                let picks: Vec<(i32, usize)> = (0..DIRECT_PX)
                    .map(|j| {
                        let lane = shape.vec_lane(j, k, pad_w);
                        (lane.div_euclid(4), lane.rem_euclid(4) as usize)
                    })
                    .collect();
                let window =
                    if picks.iter().enumerate().all(|(j, (v, c))| *v == picks[0].0 && *c == j) {
                        format!("a{u}_{}", vec_name(picks[0].0))
                    } else {
                        let parts: Vec<String> = picks
                            .iter()
                            .map(|(v, c)| format!("a{u}_{}.{}", vec_name(*v), lanes[*c]))
                            .collect();
                        format!("vec4<{a}>({})", parts.join(", "))
                    };
                writeln!(body, "    acc{d} = fma(w{r}_{k}, {window}, acc{d});").unwrap();
            }
        }
    }
    for d in 0..rows {
        writeln!(body, "    if (y0 + {d}u < params.oh) {{").unwrap();
        writeln!(body, "        let o = i32(params.off_out) + i32(n) * params.out_sn + i32(co) * params.out_sc + i32(y0 + {d}u) * params.out_sh + i32(x0) * params.out_sw;").unwrap();
        for (j, lane) in lanes.iter().enumerate() {
            writeln!(body, "        if (x0 + {j}u < params.ow) {{ outp[u32(o + {j} * params.out_sw)] = epi(acc{d}.{lane}, co); }}").unwrap();
        }
        writeln!(body, "    }}").unwrap();
    }

    let mut s = preamble(dt);
    s.push_str(&format!(
        r#"
struct Params {{
    off_in: u32,
    off_w: u32,
    off_out: u32,
    n: u32,
    co: u32,
    ih: u32,
    iw: u32,
    oh: u32,
    ow: u32,
    pad_h: i32,
    pad_w: i32,
    _p0: u32,
    in_sn: i32,
    in_sc: i32,
    in_sh: i32,
    in_sw: i32,
    w_so: i32,
    w_sh: i32,
    w_sw: i32,
    _p1: u32,
    out_sn: i32,
    out_sc: i32,
    out_sh: i32,
    out_sw: i32,
    off_extra: vec4<u32>,
    mode_extra: vec4<u32>,
}}

@group(0) @binding(0) var<storage, read> inp: array<vec4<{t}>>;
@group(0) @binding(1) var<storage, read> wgt: array<{t}>;
{extra_bindings}@group(0) @binding({out_binding}) var<storage, read_write> outp: array<{t}>;
@group(0) @binding({uniform_binding}) var<uniform> params: Params;
{lib}
fn epi(acc: {a}, co: u32) -> {t} {{
    var v = acc;
{epilogue}
    return {result};
}}

@compute @workgroup_size({CONV_DIRECT_WG})
fn conv_dw_vec_{suf}(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>) {{
    let xqw = (params.ow + 3u) / 4u;
    let row_tiles = (params.oh + {rows}u - 1u) / {rows}u;
    let i = wid.x * {CONV_DIRECT_WG}u + lid;
    if (i >= params.n * row_tiles * xqw) {{ return; }}
    let xq = i % xqw;
    let rt = i / xqw;
    let y0 = (rt % row_tiles) * {rows}u;
    let n = rt / row_tiles;
    let co = wid.y;
    let x0 = xq * 4u;
{body}}}
"#
    ));
    s
}

pub(crate) fn conv_wgsl(dt: ShaderDtype) -> String {
    conv_module(dt, &[], 0)
}

/// `extras` epilogue operands bind between the weights and the output.
pub fn conv_module(dt: ShaderDtype, epilogue: &[ChainStep], extras: usize) -> String {
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
    let epilogue = epilogue_body(epilogue, "co");
    let mut s = preamble(dt);
    // Packed params: see kernels/conv.rs. 2D NCHW/NHWC, OIHW weights.
    s.push_str(&format!(
        r#"
struct Params {{
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
}}

@group(0) @binding(0) var<storage, read> inp: array<{t}>;
@group(0) @binding(1) var<storage, read> wgt: array<{t}>;
{extra_bindings}@group(0) @binding({out_binding}) var<storage, read_write> outp: array<{t}>;
@group(0) @binding({uniform_binding}) var<uniform> params: Params;
{lib}
@compute @workgroup_size({WORKGROUP})
fn conv2d_{suf}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let i = gid.y * {GRID_LIMIT}u * {WORKGROUP}u + gid.x;
    let nout = params.n * params.co * params.oh * params.ow;
    if (i >= nout) {{ return; }}
    var rest = i;
    let ow = rest % params.ow; rest = rest / params.ow;
    let oh = rest % params.oh; rest = rest / params.oh;
    let co = rest % params.co;
    let n = rest / params.co;
    let g = co / params.co_pg;
    let ci0 = g * params.ci_pg;
    var acc = 0.0;
    // Where a tap reads does not depend on the channel, so the channels run
    // innermost and the window is walked once rather than once per channel.
    for (var kh = 0u; kh < params.kh; kh++) {{
        let ih = i32(oh) * params.stride_h + i32(kh) * params.dil_h - params.pad_h;
        if (ih < 0 || ih >= i32(params.ih)) {{ continue; }}
        for (var kw = 0u; kw < params.kw; kw++) {{
            let iw = i32(ow) * params.stride_w + i32(kw) * params.dil_w - params.pad_w;
            if (iw < 0 || iw >= i32(params.iw)) {{ continue; }}
            let in_base = i32(n) * params.in_sn
                + i32(ci0) * params.in_sc
                + ih * params.in_sh
                + iw * params.in_sw;
            let w_base = i32(co) * params.w_so
                + i32(kh) * params.w_sh
                + i32(kw) * params.w_sw;
            for (var ci = 0u; ci < params.ci_pg; ci++) {{
                let in_i = params.off_in + u32(in_base + i32(ci) * params.in_sc);
                let w_i = params.off_w + u32(w_base + i32(ci) * params.w_si);
                acc += f32(inp[in_i]) * f32(wgt[w_i]);
            }}
        }}
    }}
    let out_i = params.off_out + u32(
        i32(n) * params.out_sn
        + i32(co) * params.out_sc
        + i32(oh) * params.out_sh
        + i32(ow) * params.out_sw
    );
    var v = {t}(acc);
{epilogue}
    outp[out_i] = v;
}}
"#
    ));
    s
}

/// The `k_info` of a tiled convolution's current `k` slice: for each `k`, the
/// weight offset of its (input channel, kernel row, kernel column), the input
/// offset of its channel, and its row and column displacement in the input.
/// The divisions happen here, sixteen per slice, rather than in every load.
fn conv_k_prologue() -> String {
    let tk = MATMUL_TILE_K;
    format!(
        "        if (lid < {tk}u) {{
            let k = k0 + lid;
            var info = vec4<i32>(0);
            if (k < k_len) {{
                let ci = k / khw;
                let tap = k % khw;
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
    )
}

/// A 2-D convolution as an implicit GEMM, staged through workgroup memory the
/// way [`matmul_tiled_module`] stages a product. Per group, rows are output
/// channels, columns are output pixels (batch, row, column) and the reduction
/// runs over (input channel, kernel row, kernel column) in the weights' OIHW
/// order; the input slice is gathered on load, taps outside the window reading
/// zero. Same `Params`, strides and per-channel epilogue as [`conv_module`].
/// Workgroup `x` tiles the pixels, `y` the output channels of a group in
/// [`tile_rows`]`(co_pg)` steps, `z` is the group.
///
/// [`matmul_tiled_module`]: crate::kernels::matmul::matmul_tiled_module
pub fn conv_tiled_module(
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
    let epilogue = epilogue_body(epilogue, "co");
    let tile = MATMUL_TILE;
    let mut s = preamble(dt);
    s.push_str(&format!(
        r#"
struct Params {{
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
}}

@group(0) @binding(0) var<storage, read> inp: array<{t}>;
@group(0) @binding(1) var<storage, read> wgt: array<{t}>;
{extra_bindings}@group(0) @binding({out_binding}) var<storage, read_write> outp: array<{t}>;
@group(0) @binding({uniform_binding}) var<uniform> params: Params;
{lib}

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

fn store_out(g: u32, row: u32, pix: u32, acc: f32) {{
    if (row >= params.co_pg || pix >= params.n * params.oh * params.ow) {{ return; }}
    let co = g * params.co_pg + row;
    let ow = pix % params.ow;
    let rest = pix / params.ow;
    let oh = rest % params.oh;
    let n = rest / params.oh;
    let out_i = params.off_out + u32(
        i32(n) * params.out_sn
        + i32(co) * params.out_sc
        + i32(oh) * params.out_sh
        + i32(ow) * params.out_sw
    );
    var v = {t}(acc);
{epilogue}
    outp[out_i] = v;
}}

@compute @workgroup_size({threads})
fn conv2d_tiled_{suf}(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>) {{
    let g = wid.z;
    let khw = params.kh * params.kw;
    let k_len = params.ci_pg * khw;
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
        tk = MATMUL_TILE_K,
        core = tiled_gemm_core(t, "true", "true", "g, ", &conv_k_prologue(), tile_m),
    ));
    s
}

/// Threads per workgroup of the per-pixel convolution.
pub const CONV_PIXEL_WG: u32 = 256;

/// One thread per output pixel computing every output channel, for a dense 3x3
/// convolution over few input channels — the MobileNet stem: the 27 inputs a
/// pixel reads are shared by all its channels, so they are read once per group
/// of four rather than once per output element. The kernel came out of an
/// OpenEvolve search over tract's own conv contract (3.0x on the stem, seed
/// 115us); the taps are unrolled for kh = kw = 3 at dilation 1. Channels-last
/// layers of at most 20 output channels take it too.
pub fn conv_pixel_module(
    dt: ShaderDtype,
    epilogue: &[ChainStep],
    extras: usize,
    wide: bool,
    vec4_acc: bool,
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
    let epilogue = epilogue_body(epilogue, "co");
    let entry = if wide { "conv2d_pixel_wide" } else { "conv2d_pixel" };
    let store_one = |co_expr: &str, acc: &str| {
        format!(
            r#"            let co = {co_expr};
            let out_i = params.off_out + u32(i32(n) * params.out_sn + i32(co) * params.out_sc + i32(oh) * params.out_sh + i32(ow) * params.out_sw);
            var v = {t}({acc});
{epilogue}
            outp[out_i] = v;
"#
        )
    };
    // Wide loads the nine taps once per input channel. The grouped form re-reads
    // them for every four output channels, which is the whole stem at co = 16.
    let pixel_loops = if vec4_acc {
        let lanes = ["x", "y", "z", "w"];
        // Weight offsets of the nine 3x3 taps from a channel's filter base.
        let woffs = [
            "",
            " + u32(params.w_sw)",
            " + 2u * u32(params.w_sw)",
            " + u32(params.w_sh)",
            " + u32(params.w_sh) + u32(params.w_sw)",
            " + u32(params.w_sh) + 2u * u32(params.w_sw)",
            " + 2u * u32(params.w_sh)",
            " + 2u * u32(params.w_sh) + u32(params.w_sw)",
            " + 2u * u32(params.w_sh) + 2u * u32(params.w_sw)",
        ];
        let itaps = ["i00", "i01", "i02", "i10", "i11", "i12", "i20", "i21", "i22"];
        // Four output channels share one vec4 accumulator; each lane performs
        // the same ops the scalar form did for its channel. Channel indices
        // clamp to `co - 1` so out-of-range lanes read valid weights; their
        // stores are guarded per lane.
        let vec_group = |acc: &str, chans: [&str; 4]| {
            let mut s = String::new();
            for (l, ch) in chans.iter().enumerate() {
                s.push_str(&format!(
                    "            let wc{l} = params.off_w + min({ch}, params.co - 1u) * u32(params.w_so) + ci * u32(params.w_si);\n"
                ));
            }
            for (p, off) in woffs.iter().enumerate() {
                let loads =
                    (0..4).map(|l| format!("f32(wgt[wc{l}{off}])")).collect::<Vec<_>>().join(", ");
                s.push_str(&format!("            let w{p}v = vec4<f32>({loads});\n"));
            }
            let terms = itaps
                .iter()
                .enumerate()
                .map(|(p, it)| format!("vec4<f32>({it}) * w{p}v"))
                .collect::<Vec<_>>()
                .join(" + ");
            s.push_str(&format!("            {acc} = {acc} + ({terms});\n"));
            s
        };
        if wide {
            let acc_decls: String =
                (0..4).map(|g| format!("    var acc{g} = vec4<f32>(0.0);\n")).collect();
            let groups: String = (0..4)
                .map(|g| {
                    let ch: Vec<String> = (0..4).map(|l| format!("{}u", g * 4 + l)).collect();
                    let ch: [&str; 4] =
                        [ch[0].as_str(), ch[1].as_str(), ch[2].as_str(), ch[3].as_str()];
                    format!(
                        "        if ({}u < params.co) {{\n{}\n        }}\n",
                        g * 4,
                        vec_group(&format!("acc{g}"), ch)
                    )
                })
                .collect();
            let stores: String = (0..16)
                .map(|c| {
                    format!(
                        "        if ({c}u < params.co) {{\n{}}}\n",
                        store_one(&format!("{c}u"), &format!("acc{}.{}", c / 4, lanes[c % 4]))
                    )
                })
                .collect();
            format!(
                "{acc_decls}    for (var ci = 0u; ci < params.ci; ci++) {{\n        \
let i00 = read_inp(ih0, iw0, ci, n);\n        \
let i01 = read_inp(ih0, iw0 + 1, ci, n);\n        \
let i02 = read_inp(ih0, iw0 + 2, ci, n);\n        \
let i10 = read_inp(ih0 + 1, iw0, ci, n);\n        \
let i11 = read_inp(ih0 + 1, iw0 + 1, ci, n);\n        \
let i12 = read_inp(ih0 + 1, iw0 + 2, ci, n);\n        \
let i20 = read_inp(ih0 + 2, iw0, ci, n);\n        \
let i21 = read_inp(ih0 + 2, iw0 + 1, ci, n);\n        \
let i22 = read_inp(ih0 + 2, iw0 + 2, ci, n);\n\
{groups}    }}\n{stores}"
            )
        } else {
            let group_body =
                vec_group("acc", ["co_grp", "co_grp + 1u", "co_grp + 2u", "co_grp + 3u"]);
            let group_stores: String = (0..4)
                .map(|l| {
                    format!(
                        "        if (co_grp + {l}u < params.co) {{\n{}}}\n",
                        store_one(&format!("co_grp + {l}u"), &format!("acc.{}", lanes[l]))
                    )
                })
                .collect();
            format!(
                "    for (var co_grp = 0u; co_grp < params.co; co_grp += 4u) {{\n        \
var acc = vec4<f32>(0.0);\n        \
for (var ci = 0u; ci < params.ci; ci++) {{\n            \
let i00 = read_inp(ih0, iw0, ci, n);\n            \
let i01 = read_inp(ih0, iw0 + 1, ci, n);\n            \
let i02 = read_inp(ih0, iw0 + 2, ci, n);\n            \
let i10 = read_inp(ih0 + 1, iw0, ci, n);\n            \
let i11 = read_inp(ih0 + 1, iw0 + 1, ci, n);\n            \
let i12 = read_inp(ih0 + 1, iw0 + 2, ci, n);\n            \
let i20 = read_inp(ih0 + 2, iw0, ci, n);\n            \
let i21 = read_inp(ih0 + 2, iw0 + 1, ci, n);\n            \
let i22 = read_inp(ih0 + 2, iw0 + 2, ci, n);\n\
{group_body}        }}\n{group_stores}    }}\n"
            )
        }
    } else {
        let nacc: usize = if wide { 16 } else { 4 };
        let acc_decls: String = (0..nacc).map(|j| format!("        var acc{j} = 0.0;\n")).collect();
        let tap = |j: usize, co_expr: &str| {
            format!(
                r#"                let wc0 = params.off_w + ({co_expr}) * u32(params.w_so) + ci * u32(params.w_si);
                let w00 = f32(wgt[wc0]);
                let w01 = f32(wgt[wc0 + u32(params.w_sw)]);
                let w02 = f32(wgt[wc0 + 2u * u32(params.w_sw)]);
                let w10 = f32(wgt[wc0 + u32(params.w_sh)]);
                let w11 = f32(wgt[wc0 + u32(params.w_sh) + u32(params.w_sw)]);
                let w12 = f32(wgt[wc0 + u32(params.w_sh) + 2u * u32(params.w_sw)]);
                let w20 = f32(wgt[wc0 + 2u * u32(params.w_sh)]);
                let w21 = f32(wgt[wc0 + 2u * u32(params.w_sh) + u32(params.w_sw)]);
                let w22 = f32(wgt[wc0 + 2u * u32(params.w_sh) + 2u * u32(params.w_sw)]);
                acc{j} += i00 * w00 + i01 * w01 + i02 * w02 + i10 * w10 + i11 * w11 + i12 * w12 + i20 * w20 + i21 * w21 + i22 * w22;
"#
            )
        };
        if wide {
            let macs: String = (0..nacc)
                .map(|j| {
                    format!(
                        "            if ({j}u < params.co) {{\n{}\n            }}\n",
                        tap(j, &format!("{j}u"))
                    )
                })
                .collect();
            let stores: String = (0..nacc)
                .map(|j| {
                    format!(
                        "        if ({j}u < params.co) {{\n{}}}\n",
                        store_one(&format!("{j}u"), &format!("acc{j}"))
                    )
                })
                .collect();
            format!(
                "{acc_decls}    for (var ci = 0u; ci < params.ci; ci++) {{\n        \
let i00 = read_inp(ih0, iw0, ci, n);\n        \
let i01 = read_inp(ih0, iw0 + 1, ci, n);\n        \
let i02 = read_inp(ih0, iw0 + 2, ci, n);\n        \
let i10 = read_inp(ih0 + 1, iw0, ci, n);\n        \
let i11 = read_inp(ih0 + 1, iw0 + 1, ci, n);\n        \
let i12 = read_inp(ih0 + 1, iw0 + 2, ci, n);\n        \
let i20 = read_inp(ih0 + 2, iw0, ci, n);\n        \
let i21 = read_inp(ih0 + 2, iw0 + 1, ci, n);\n        \
let i22 = read_inp(ih0 + 2, iw0 + 2, ci, n);\n\
{macs}    }}\n{stores}"
            )
        } else {
            let group_macs: String = (0..4)
                .map(|j| {
                    format!(
                        "            if (co_grp + {j}u < params.co) {{\n{}\n            }}\n",
                        tap(j, &format!("co_grp + {j}u"))
                    )
                })
                .collect();
            let group_stores: String = (0..4)
                .map(|j| {
                    format!(
                        "        if (co_grp + {j}u < params.co) {{\n{}}}\n",
                        store_one(&format!("co_grp + {j}u"), &format!("acc{j}"))
                    )
                })
                .collect();
            format!(
                "    for (var co_grp = 0u; co_grp < params.co; co_grp += 4u) {{\n        \
var acc0 = 0.0;\n        var acc1 = 0.0;\n        var acc2 = 0.0;\n        var acc3 = 0.0;\n        \
for (var ci = 0u; ci < params.ci; ci++) {{\n            \
let i00 = read_inp(ih0, iw0, ci, n);\n            \
let i01 = read_inp(ih0, iw0 + 1, ci, n);\n            \
let i02 = read_inp(ih0, iw0 + 2, ci, n);\n            \
let i10 = read_inp(ih0 + 1, iw0, ci, n);\n            \
let i11 = read_inp(ih0 + 1, iw0 + 1, ci, n);\n            \
let i12 = read_inp(ih0 + 1, iw0 + 2, ci, n);\n            \
let i20 = read_inp(ih0 + 2, iw0, ci, n);\n            \
let i21 = read_inp(ih0 + 2, iw0 + 1, ci, n);\n            \
let i22 = read_inp(ih0 + 2, iw0 + 2, ci, n);\n\
{group_macs}        }}\n{group_stores}    }}\n"
            )
        }
    };
    let mut s = preamble(dt);
    // Same packed params as `conv_module`, see kernels/conv.rs.
    s.push_str(&format!(
        r#"
struct Params {{
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
}}

@group(0) @binding(0) var<storage, read> inp: array<{t}>;
@group(0) @binding(1) var<storage, read> wgt: array<{t}>;
{extra_bindings}@group(0) @binding({out_binding}) var<storage, read_write> outp: array<{t}>;
@group(0) @binding({uniform_binding}) var<uniform> params: Params;
{lib}
fn read_inp(ih: i32, iw: i32, ic: u32, n: u32) -> f32 {{
    if (ih < 0 || ih >= i32(params.ih) || iw < 0 || iw >= i32(params.iw)) {{ return 0.0; }}
    return f32(inp[params.off_in + u32(i32(n) * params.in_sn + ih * params.in_sh + iw * params.in_sw + i32(ic) * params.in_sc)]);
}}

@compute @workgroup_size({CONV_PIXEL_WG})
fn {entry}_{suf}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    let npix = params.n * params.oh * params.ow;
    if (idx >= npix) {{ return; }}
    var rest = idx;
    let ow = rest % params.ow; rest /= params.ow;
    let oh = rest % params.oh; rest /= params.oh;
    let n = rest;
    let ih0 = i32(oh) * params.stride_h - params.pad_h;
    let iw0 = i32(ow) * params.stride_w - params.pad_w;
    // Named accumulators: an indexed private array lands in memory under Tint.
{pixel_loops}
}}
"#
    ));
    s
}

/// One thread owns four adjacent output channels at one pixel. The nine taps
/// are outside the channel loop: a tap that misses the input is skipped, and
/// a tap that hits streams `vec4`s along C, one input load dotted into all
/// four accumulators. Weights are `[co, 9, ci]`, the same order as a
/// channels-last `[co, 3, 3, ci]` filter. Out-of-bounds taps are skipped, not
/// multiplied by zero, so non-finite weights do not reach border outputs.
/// Serves the shapes [`pixel_vec_ready`] admits. f16 sums in f16 when the
/// accumulation knob is on, and in f32 otherwise.
fn conv_pixel_vec_module(
    dt: ShaderDtype,
    f16_accumulation: bool,
    epilogue: &[ChainStep],
    extras: usize,
) -> String {
    let t = dt.wgsl();
    let a = if dt == ShaderDtype::F16 && f16_accumulation { t } else { "f32" };
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
    let mut taps = String::new();
    for (tap, (dy, dx)) in
        [(0i32, 0i32), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)]
            .iter()
            .enumerate()
    {
        taps.push_str(&format!(
            r#"
        {{
            let ih = ih0 + {dy};
            let iw = iw0 + {dx};
            if (ih >= 0 && ih < i32(params.ih) && iw >= 0 && iw < i32(params.iw)) {{
                let s = u32(i32(params.off_in) + i32(n) * params.in_sn + ih * params.in_sh + iw * params.in_sw);
                let base = s / 4u;
                for (var ci4 = 0u; ci4 < params.ci / 4u; ci4++) {{
                    let x = vec4<{a}>(inp[base + ci4]);
                    acc0 += dot(x, w4(co_grp + 0u, {tap}u, ci4));
                    acc1 += dot(x, w4(co_grp + 1u, {tap}u, ci4));
                    acc2 += dot(x, w4(co_grp + 2u, {tap}u, ci4));
                    acc3 += dot(x, w4(co_grp + 3u, {tap}u, ci4));
                }}
            }}
        }}
"#
        ));
    }
    let mut store = String::new();
    for j in 0..4 {
        let body = epilogue_body(epilogue, "co");
        store.push_str(&format!(
            r#"
        {{
            let co = co_grp + {j}u;
            var v = v{j};
{body}            v{j} = v;
        }}
"#
        ));
    }
    let mut s = preamble(dt);
    s.push_str(&format!(
        r#"
struct Params {{
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
}}

@group(0) @binding(0) var<storage, read> inp: array<vec4<{t}>>;
@group(0) @binding(1) var<storage, read> wgt: array<vec4<{t}>>;
{extra_bindings}@group(0) @binding({out_binding}) var<storage, read_write> outp: array<vec4<{t}>>;
@group(0) @binding({uniform_binding}) var<uniform> params: Params;
{lib}
fn w4(oc: u32, tap: u32, ci4: u32) -> vec4<{a}> {{
    let i = params.off_w / 4u + (oc * 9u + tap) * (params.ci / 4u) + ci4;
    return vec4<{a}>(wgt[i]);
}}

@compute @workgroup_size({CONV_PIXEL_WG})
fn conv2d_pixel_vec_{suf}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    let nvec = params.co / 4u;
    let npix = params.n * params.oh * params.ow;
    let pix = idx / nvec;
    if (pix >= npix) {{ return; }}
    let co_grp = (idx % nvec) * 4u;
    var rest = pix;
    let ow = rest % params.ow; rest /= params.ow;
    let oh = rest % params.oh; rest /= params.oh;
    let n = rest;
    let ih0 = i32(oh) * params.stride_h - params.pad_h;
    let iw0 = i32(ow) * params.stride_w - params.pad_w;
    var acc0 = {a}(0.0);
    var acc1 = {a}(0.0);
    var acc2 = {a}(0.0);
    var acc3 = {a}(0.0);
{taps}
    var v0 = {t}(acc0);
    var v1 = {t}(acc1);
    var v2 = {t}(acc2);
    var v3 = {t}(acc3);
{store}
    let out_i = params.off_out + u32(i32(n) * params.out_sn + i32(co_grp) * params.out_sc + i32(oh) * params.out_sh + i32(ow) * params.out_sw);
    outp[out_i / 4u] = vec4<{t}>(v0, v1, v2, v3);
}}
"#
    ));
    s
}

/// Threads per workgroup of the register-blocked direct convolution.
pub const CONV_DIRECT_WG: u32 = 64;

/// Spatial threads below which a long 3×3 reduction leaves the register-blocked
/// kernel idle. A 10×16 map launches twenty threads and a 20×32 map eighty;
/// each thread then walks every input channel. The tiled kernel splits that
/// reduction across more workgroups than this kernel has threads.
const DIRECT_MIN_THREADS: usize = 96;

/// Reduction length, input channels times the nine taps, past which those idle
/// launches lose. Shorter reductions stay here, including the small shapes the
/// packed-weight test dispatches by hand.
const DIRECT_LONG_REDUCTION: usize = 32 * 9;

/// Output pixels along a row that one direct-convolution thread owns: one
/// `vec4` of input along W feeds all of them.
const DIRECT_PX: usize = 4;

/// Register tile of one direct-convolution thread: `co` output channels by
/// `rows` consecutive output rows, each row `DIRECT_PX` pixels wide. The
/// `rows + 2` input rows the tile needs are loaded once per input channel and
/// shared across the `rows` outputs.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct DirectTile {
    pub co: usize,
    pub rows: usize,
}

impl DirectTile {
    /// Tile for the accumulator type: f16 accumulation spends its eight
    /// `vec4<f16>` accumulators on two rows of four channels; f32 accumulation
    /// keeps one row of up to eight channels.
    fn for_accumulator(f16_acc: bool, co: usize) -> DirectTile {
        if f16_acc {
            DirectTile { co: 4, rows: 2 }
        } else {
            DirectTile { co: if co.is_multiple_of(8) { 8 } else { 4 }, rows: 1 }
        }
    }

    fn threads(self, n: usize, oh: usize, ow: usize) -> usize {
        n * oh.div_ceil(self.rows) * (ow / DIRECT_PX)
    }
}

/// Tile measured for the eight RVM f16 direct 3×3 layers: eight output channels,
/// two rows at 180p and 360p, one row on the smaller maps. Each filter row is
/// stored as a vec4 so the kernel loads it in one read.
pub(crate) fn measured_packed_tile(ci: usize, co: usize, h: usize, w: usize) -> Option<DirectTile> {
    let rows = match (ci, co, h, w) {
        (35, 16, 360, 640) | (16, 16, 360, 640) | (59, 32, 180, 320) | (32, 32, 180, 320) => 2,
        (107, 40, 90, 160) | (171, 80, 45, 80) | (128, 64, 23, 40) | (32, 16, 180, 320) => 1,
        _ => return None,
    };
    Some(DirectTile { co: 8, rows })
}

/// Register tile of the direct convolution, if the register-blocked kernel can
/// serve this convolution: dense 3x3, stride 1, dilation 1, NCHW f32/f16, one
/// pad on each side so the output keeps the input's size, and every input
/// offset and stride a multiple of four elements so the input reads as `vec4`.
/// Winograd takes the large channel counts before translation; this takes up
/// to 128 output channels.
#[allow(clippy::too_many_arguments)]
fn direct_tiling(
    op: &Conv,
    dt: ShaderDtype,
    f16_acc: bool,
    input: &DeviceTensor,
    pad_before: [usize; 2],
    [n, co, ih, iw, oh, ow]: [usize; 6],
    in_strides: [usize; 4],
) -> Option<DirectTile> {
    let spec = &op.pool_spec;
    if !matches!(dt, ShaderDtype::F32 | ShaderDtype::F16)
        || op.group != 1
        || spec.data_format.c_is_last()
        || spec.kernel_shape[..] != [3, 3]
        || spec.strides()[..] != [1, 1]
        || spec.dilations()[..] != [1, 1]
        || pad_before != [1, 1]
        || (ih, iw) != (oh, ow)
        || co > 128
        || n * oh * ow < 128
        || !iw.is_multiple_of(DIRECT_PX)
        || in_strides[3] != 1
        || !in_strides[..3].iter().all(|s| s.is_multiple_of(DIRECT_PX))
        || !element_offset(input, 0).is_multiple_of(DIRECT_PX)
    {
        return None;
    }
    let tile = DirectTile::for_accumulator(f16_acc, co);
    let threads = tile.threads(n, oh, ow);
    let reduction = spec.input_channels * 9;
    // Only step aside when the tiled kernel can take the shape. A short map
    // with few output channels would otherwise fall through to the scalar kernel.
    if threads < DIRECT_MIN_THREADS
        && reduction >= DIRECT_LONG_REDUCTION
        && tiled_applies(1, co, n * oh * ow, reduction)
    {
        return None;
    }
    (threads.div_ceil(CONV_DIRECT_WG as usize) <= GRID_LIMIT as usize).then_some(tile)
}

/// Register-blocked direct 3x3 convolution: each thread owns `DIRECT_PX`
/// consecutive output pixels of `tile.rows` rows for `tile.co` output
/// channels, loads the `tile.rows + 2` input rows it needs once per input
/// channel as `vec4` along W, and reads its weights as workgroup-uniform
/// scalars, with no workgroup memory and no barrier. Padded taps read zero
/// through `select`, so an `inf` or `NaN` at an edge never leaks into a padded
/// output. Serves the convolutions [`direct_tiling`] admits. f16 inputs and
/// weights are widened as they are read: the accumulators and the epilogue run
/// in f32 and the result is rounded to the storage type once. With
/// `f16_accumulation`, f16 stays f16 throughout. `packed` reads each 3-tap
/// filter row from a `vec4`, whose last lane is zero and unused.
pub fn conv_direct_module(
    dt: ShaderDtype,
    f16_accumulation: bool,
    epilogue: &[ChainStep],
    extras: usize,
    tile: DirectTile,
    packed: bool,
) -> String {
    let t = dt.wgsl();
    let a = if dt == ShaderDtype::F16 && f16_accumulation { "f16" } else { "f32" };
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
        format!("{}{}", unary_ops_wgsl(a), binary_ops_wgsl(a))
    };
    let epilogue = conv_direct_epilogue(epilogue, t != a);
    let result = if t == a { "v".to_owned() } else { format!("{t}(v)") };
    let DirectTile { co: co_tile, rows } = tile;
    let (decls, rows_setup, taps, stores) = conv_direct_body(tile, t, a, packed);
    let weight_type = if packed { format!("vec4<{t}>") } else { t.to_owned() };
    let entry = if packed { "conv2d_direct_packed_weights" } else { "conv2d_direct" };
    let mut s = preamble(dt);
    s.push_str(&format!(
        r#"
struct Params {{
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
}}

@group(0) @binding(0) var<storage, read> inp: array<vec4<{t}>>;
@group(0) @binding(1) var<storage, read> wgt: array<{weight_type}>;
{extra_bindings}@group(0) @binding({out_binding}) var<storage, read_write> outp: array<{t}>;
@group(0) @binding({uniform_binding}) var<uniform> params: Params;
{lib}
fn epi(acc: {a}, co: u32) -> {t} {{
    var v = acc;
{epilogue}
    return {result};
}}

@compute @workgroup_size({CONV_DIRECT_WG})
fn {entry}_{suf}(@builtin(local_invocation_index) lid: u32, @builtin(workgroup_id) wid: vec3<u32>) {{
    let xqw = params.ow / 4u;
    let row_tiles = (params.oh + {rows}u - 1u) / {rows}u;
    let i = wid.x * {CONV_DIRECT_WG}u + lid;
    if (i >= params.n * row_tiles * xqw) {{ return; }}
    let xq = i % xqw;
    let rt = i / xqw;
    let y0 = (rt % row_tiles) * {rows}u;
    let n = rt / row_tiles;
    let co0 = wid.y * {co_tile}u;
    let x0 = xq * 4u;
{decls}    let in_row_base = i32(params.off_in) + i32(n) * params.in_sn;
    let x_left_ok = xq > 0u;
    let x_right_ok = x0 + 4u < params.iw;
{rows_setup}    for (var ci = 0u; ci < params.ci_pg; ci++) {{
        let ci_base = in_row_base + i32(ci) * params.in_sc;
{taps}    }}
{stores}}}
"#
    ));
    s
}

/// The unrolled pieces of [`conv_direct_module`]: one named `vec4`
/// accumulator per output row and channel, the `rows + 2` input rows of each
/// input channel, and the nine taps and the stores of each output, so the
/// register tile never becomes an indexed private array.
fn conv_direct_body(
    tile: DirectTile,
    t: &str,
    a: &str,
    packed: bool,
) -> (String, String, String, String) {
    use std::fmt::Write;
    let DirectTile { co: co_tile, rows } = tile;
    let mut decls = String::new();
    for d in 0..rows {
        for c in 0..co_tile {
            writeln!(decls, "    var acc{d}_{c} = vec4<{a}>(0.0);").unwrap();
        }
    }
    let mut rows_setup = String::new();
    for u in 0..rows + 2 {
        let iy = if u == 0 { "y0 - 1u".to_owned() } else { format!("y0 + {}u", u - 1) };
        writeln!(rows_setup, "    let iy{u} = {iy};").unwrap();
        writeln!(rows_setup, "    let row_ok{u} = iy{u} < params.ih;").unwrap();
    }
    let mut taps = String::new();
    for u in 0..rows + 2 {
        let load = |which: &str, guard: &str| {
            let load = format!("select(vec4<{t}>(0.0), inp[base{u} + {which}], {guard})");
            if t == a { load } else { format!("vec4<{a}>({load})") }
        };
        writeln!(taps, "        let base{u} = u32((ci_base + i32(select(0u, iy{u}, row_ok{u})) * params.in_sh) / 4);").unwrap();
        writeln!(
            taps,
            "        let a0_{u} = {};",
            load("xq - 1u", &format!("row_ok{u} && x_left_ok"))
        )
        .unwrap();
        writeln!(taps, "        let a1_{u} = {};", load("xq", &format!("row_ok{u}"))).unwrap();
        writeln!(
            taps,
            "        let a2_{u} = {};",
            load("xq + 1u", &format!("row_ok{u} && x_right_ok"))
        )
        .unwrap();
        writeln!(taps, "        let left{u} = vec4<{a}>(a0_{u}.w, a1_{u}.xyz);").unwrap();
        writeln!(taps, "        let mid{u} = vec4<{a}>(a1_{u});").unwrap();
        writeln!(taps, "        let right{u} = vec4<{a}>(a1_{u}.yzw, a2_{u}.x);").unwrap();
    }
    for c in 0..co_tile {
        writeln!(taps, "        {{").unwrap();
        if packed {
            writeln!(
                taps,
                "            let wi = params.off_w / 4u + ((co0 + {c}u) * params.ci_pg + ci) * 3u;"
            )
            .unwrap();
        } else {
            writeln!(taps, "            let wi = u32(i32(params.off_w) + i32(co0 + {c}u) * params.w_so + i32(ci) * params.w_si);").unwrap();
        }
        for r in 0..3 {
            if packed {
                writeln!(taps, "            let wr{r} = wgt[wi + {r}u];").unwrap();
            }
            for (k, lane) in ["x", "y", "z"].iter().enumerate() {
                let value = if packed {
                    format!("wr{r}.{lane}")
                } else {
                    format!("wgt[wi + {r}u * u32(params.w_sh) + {k}u * u32(params.w_sw)]")
                };
                let value = if t == a { value } else { format!("{a}({value})") };
                writeln!(taps, "            let w{r}{k} = vec4<{a}>({value});").unwrap();
            }
        }
        for d in 0..rows {
            for r in 0..3 {
                let u = d + r;
                for (k, win) in ["left", "mid", "right"].iter().enumerate() {
                    writeln!(taps, "            acc{d}_{c} = fma(w{r}{k}, {win}{u}, acc{d}_{c});")
                        .unwrap();
                }
            }
        }
        writeln!(taps, "        }}").unwrap();
    }
    let mut stores = String::new();
    for d in 0..rows {
        for c in 0..co_tile {
            let row_guard =
                if rows == 1 { String::new() } else { format!("y0 + {d}u < params.oh && ") };
            writeln!(stores, "    if ({row_guard}co0 + {c}u < params.co) {{").unwrap();
            writeln!(stores, "        let co = co0 + {c}u;").unwrap();
            writeln!(stores, "        let o = i32(params.off_out) + i32(n) * params.out_sn + i32(co) * params.out_sc + i32(y0 + {d}u) * params.out_sh + i32(x0) * params.out_sw;").unwrap();
            for (x, lane) in ["x", "y", "z", "w"].iter().enumerate() {
                writeln!(
                    stores,
                    "        outp[u32(o + {x} * params.out_sw)] = epi(acc{d}_{c}.{lane}, co);"
                )
                .unwrap();
            }
            writeln!(stores, "    }}").unwrap();
        }
    }
    (decls, rows_setup, taps, stores)
}

pub(crate) fn conv_direct_epilogue(steps: &[ChainStep], widen: bool) -> String {
    if !widen {
        return epilogue_body(steps, "co");
    }
    let mut result = String::new();
    for step in steps {
        match step {
            ChainStep::Unary(op) => result.push_str(&format!("    v = op_{op}(v);\n")),
            ChainStep::Binary { op, rhs, swapped } => {
                let i = rhs - 1;
                result.push_str(&format!(
                    "    let e{i} = f32(extra{i}[params.off_extra[{i}u] + select(0u, co, params.mode_extra[{i}u] == 1u)]);\n"
                ));
                if *swapped {
                    result.push_str(&format!("    v = op_{op}(e{i}, v);\n"));
                } else {
                    result.push_str(&format!("    v = op_{op}(v, e{i});\n"));
                }
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use tract_core::internal::*;
    use tract_core::transform::ModelTransform;

    use crate::WgpuTransform;
    use crate::test_utils::{OneWeight, f16_vs_f32, fill, gpu_vs_cpu, transformed};

    /// A dense 3x3 convolution takes the per-pixel kernel, every output channel
    /// of a pixel from one thread; the shapes cover the stem, a batch, a stride-1
    /// layer with more input channels, and an odd size, against the CPU.
    #[test]
    fn conv_3x3_per_pixel() -> TractResult<()> {
        use tract_core::ops::cnn::{Conv, KernelFormat, PaddingSpec, PoolSpec};
        use tract_core::ops::nn::DataFormat;
        for (n, ci, co, h, w, stride, pad) in [
            (1usize, 3usize, 16usize, 144usize, 256usize, 2usize, 1usize),
            (2, 3, 8, 20, 30, 2, 1),
            (1, 7, 5, 13, 17, 1, 1),
            (1, 4, 6, 9, 9, 1, 0),
            (1, 3, 4, 18, 26, 2, 1),
            (1, 8, 8, 15, 21, 1, 1),
        ] {
            let mut model = TypedModel::default();
            let x = model.add_source("x", f32::fact([n, ci, h, w]))?;
            let k = model
                .add_const("k", Tensor::from_shape(&[co, ci, 3, 3], &fill(3, co * ci * 9))?)?;
            let b = model.add_const("b", Tensor::from_shape(&[co], &fill(4, co))?)?;
            let conv = Conv {
                pool_spec: PoolSpec {
                    data_format: DataFormat::NCHW,
                    kernel_shape: tvec![3, 3],
                    padding: PaddingSpec::Explicit(tvec![pad, pad], tvec![pad, pad]),
                    dilations: None,
                    strides: Some(tvec![stride, stride]),
                    input_channels: ci,
                    output_channels: co,
                },
                kernel_fmt: KernelFormat::OIHW,
                group: 1,
                q_params: None,
            };
            let y = model.wire_node("conv", conv, &[x, k, b])?[0];
            model.select_output_outlets(&[y])?;
            gpu_vs_cpu(model, Tensor::from_shape(&[n, ci, h, w], &fill(5, n * ci * h * w))?)
                .with_context(|| format!("n{n} ci{ci} co{co} {h}x{w} s{stride} p{pad}"))?;
        }
        Ok(())
    }

    /// Channels-last 3x3 layers with few output channels take the same per-pixel
    /// kernel, including a 16-channel map large enough that the tiled GEMM would
    /// otherwise claim it. Stride 1 and 2, f32 and f16.
    #[test]
    fn conv_3x3_per_pixel_nhwc() -> TractResult<()> {
        use tract_core::ops::cnn::{Conv, KernelFormat, PaddingSpec, PoolSpec};
        use tract_core::ops::nn::DataFormat;
        let build = |dt: DatumType, ci: usize, co: usize, h: usize, w: usize, stride: usize| {
            let mut model = TypedModel::default();
            let x = model.add_source("x", dt.fact([1, h, w, ci]))?;
            let k = model.add_const(
                "k",
                Tensor::from_shape(&[co, ci, 3, 3], &fill(3, co * ci * 9))?
                    .cast_to_dt(dt)?
                    .into_owned(),
            )?;
            let b = model.add_const(
                "b",
                Tensor::from_shape(&[co], &fill(4, co))?.cast_to_dt(dt)?.into_owned(),
            )?;
            let conv = Conv {
                pool_spec: PoolSpec {
                    data_format: DataFormat::NHWC,
                    kernel_shape: tvec![3, 3],
                    padding: PaddingSpec::Explicit(tvec![1, 1], tvec![1, 1]),
                    dilations: None,
                    strides: Some(tvec![stride, stride]),
                    input_channels: ci,
                    output_channels: co,
                },
                kernel_fmt: KernelFormat::OIHW,
                group: 1,
                q_params: None,
            };
            let y = model.wire_node("conv", conv, &[x, k, b])?[0];
            model.select_output_outlets(&[y])?;
            Ok(model)
        };
        for (ci, co, h, w, stride) in [
            (40usize, 20usize, 18usize, 20usize, 1usize),
            (3, 4, 24, 32, 2),
            (32, 16, 120, 120, 1),
            (16, 8, 33, 31, 2),
        ] {
            let model = build(DatumType::F32, ci, co, h, w, stride)?;
            let mut inspected = model.clone();
            WgpuTransform.transform(&mut inspected)?;
            let convs = inspected
                .nodes()
                .iter()
                .filter_map(|n| n.op_as::<crate::ops::conv::WgpuConv>())
                .collect::<Vec<_>>();
            ensure!(convs.len() == 1, "expected one WgpuConv for {ci}->{co} {h}x{w} s{stride}");
            let vec4 = ci >= 16 && ci % 4 == 0 && co % 4 == 0 && co <= 20;
            ensure!(
                convs[0].pixel_packed.is_some() == vec4,
                "packed pixel weights mismatch for {ci}->{co} {h}x{w} s{stride}"
            );
            gpu_vs_cpu(model, Tensor::from_shape(&[1, h, w, ci], &fill(5, h * w * ci))?)
                .with_context(|| format!("f32 {ci}->{co} {h}x{w} s{stride}"))?;
        }
        let (ci, co, h, w) = (40usize, 20usize, 18usize, 20usize);
        let input = Tensor::from_shape(&[1, h, w, ci], &fill(5, h * w * ci))?;
        f16_vs_f32(|dt| build(dt, ci, co, h, w, 1), input.clone(), Approximation::Approximate)?;
        let scope =
            tract_data::knobs::KnobScope::new([("TRACT_WGPU_ALLOW_F16_ACCUMULATION", "true")])?;
        scope.enter(|| {
            f16_vs_f32(|dt| build(dt, ci, co, h, w, 1), input, Approximation::Approximate)
        })
    }

    /// A qualifying dense stride-1 3x3 conv rewrites to Winograd F(4,3): a
    /// tile-extracting grouped conv plus two batched GEMMs. The shapes, all at the
    /// smallest tile count the rewrite takes, cover aligned tiles, odd sizes (tail
    /// tiles cropped) with a batch and no bias, and no padding.
    #[test]
    fn conv_winograd_matches_cpu() -> TractResult<()> {
        use tract_core::ops::cnn::{Conv, KernelFormat, PaddingSpec, PoolSpec};
        use tract_core::ops::nn::DataFormat;
        for (n, ci, co, h, w, pad, bias) in [
            (1usize, 128usize, 128usize, 16usize, 16usize, 1usize, true),
            (2, 128, 136, 13, 15, 1, false),
            (1, 128, 128, 18, 18, 0, true),
        ] {
            let mut model = TypedModel::default();
            let x = model.add_source("x", f32::fact([n, ci, h, w]))?;
            let k = model
                .add_const("k", Tensor::from_shape(&[co, ci, 3, 3], &fill(3, co * ci * 9))?)?;
            let b = if bias {
                model.add_const("b", Tensor::from_shape(&[co], &fill(4, co))?)?
            } else {
                model.add_const("b", Tensor::zero::<f32>(&[co])?)?
            };
            let conv = Conv {
                pool_spec: PoolSpec {
                    data_format: DataFormat::NCHW,
                    kernel_shape: tvec![3, 3],
                    padding: PaddingSpec::Explicit(tvec![pad, pad], tvec![pad, pad]),
                    dilations: None,
                    strides: Some(tvec![1, 1]),
                    input_channels: ci,
                    output_channels: co,
                },
                kernel_fmt: KernelFormat::OIHW,
                group: 1,
                q_params: None,
            };
            let y = model.wire_node("conv", conv, &[x, k, b])?[0];
            model.select_output_outlets(&[y])?;
            let mut transformed = model.clone();
            WgpuTransform.transform(&mut transformed)?;
            ensure!(
                transformed.nodes().iter().any(|node| node.name.contains(".wino.")),
                "winograd rewrite did not fire for {ci}->{co} {h}x{w}"
            );
            gpu_vs_cpu(model, Tensor::from_shape(&[n, ci, h, w], &fill(5, n * ci * h * w))?)
                .with_context(|| format!("n{n} ci{ci} co{co} {h}x{w} p{pad}"))?;
        }
        Ok(())
    }

    /// An f16 conv whose kernel is a named weight, not yet loaded, rewrites to
    /// Winograd too, and matches the CPU in f32 on the same f16-rounded values.
    #[test]
    fn conv_winograd_f16_named_weight_matches_cpu() -> TractResult<()> {
        use tract_core::ops::cnn::{Conv, KernelFormat, PaddingSpec, PoolSpec};
        use tract_core::ops::nn::DataFormat;
        crate::context::wgpu_context();
        let (ci, co, h, w) = (128usize, 128usize, 16usize, 16usize);
        let rounded = |t: Tensor| -> TractResult<Tensor> {
            Ok(t.cast_to::<f16>()?.cast_to::<f32>()?.into_owned())
        };
        let kernel: Vec<f32> = fill(3, co * ci * 9).iter().map(|v| v * 0.05).collect();
        let kernel = rounded(Tensor::from_shape(&[co, ci, 3, 3], &kernel)?)?;
        let bias = rounded(Tensor::from_shape(&[co], &fill(4, co))?)?;
        let build = |dt: DatumType| -> TractResult<TypedModel> {
            let mut model = TypedModel::default();
            let x = model.add_source("x", dt.fact([1, ci, h, w]))?;
            let k = if dt == DatumType::F16 {
                let source = Arc::new(OneWeight(kernel.cast_to::<f16>()?.into_owned()));
                NamedStorage::new("k", dt, kernel.shape(), Some(source)).into_tensor()
            } else {
                kernel.clone()
            };
            let k = model.add_const("k", k)?;
            let b = model.add_const("b", bias.cast_to_dt(dt)?.into_owned())?;
            let conv = Conv {
                pool_spec: PoolSpec {
                    data_format: DataFormat::NCHW,
                    kernel_shape: tvec![3, 3],
                    padding: PaddingSpec::Explicit(tvec![1, 1], tvec![1, 1]),
                    dilations: None,
                    strides: Some(tvec![1, 1]),
                    input_channels: ci,
                    output_channels: co,
                },
                kernel_fmt: KernelFormat::OIHW,
                group: 1,
                q_params: None,
            };
            let y = model.wire_node("conv", conv, &[x, k, b])?[0];
            model.select_output_outlets(&[y])?;
            Ok(model)
        };
        let mut transformed = build(DatumType::F16)?;
        WgpuTransform.transform(&mut transformed)?;
        ensure!(
            transformed.nodes().iter().any(|node| node.name.contains(".wino.")),
            "winograd rewrite did not fire on the f16 named-weight conv"
        );
        let input = Tensor::from_shape(&[1, ci, h, w], &fill(5, ci * h * w))?;
        f16_vs_f32(build, input, Approximation::Approximate)
    }

    /// The rewrite declines below its channel-product and tile gates, and on a
    /// stride-2 conv: those keep the plain wgpu conv.
    #[test]
    fn conv_winograd_skips_unfit() -> TractResult<()> {
        use tract_core::ops::cnn::{Conv, KernelFormat, PaddingSpec, PoolSpec};
        use tract_core::ops::nn::DataFormat;
        for (n, ci, co, h, w, stride) in [
            (1usize, 64usize, 64usize, 16usize, 16usize, 2usize),
            (1, 8, 8, 64, 64, 1),
            (1, 16, 16, 12, 12, 1),
            (1, 64, 64, 16, 16, 1),
            (1, 64, 64, 64, 64, 1),
            (1, 128, 64, 32, 32, 1),
            (1, 128, 128, 8, 8, 1),
        ] {
            let mut model = TypedModel::default();
            let x = model.add_source("x", f32::fact([n, ci, h, w]))?;
            let k = model
                .add_const("k", Tensor::from_shape(&[co, ci, 3, 3], &fill(3, co * ci * 9))?)?;
            let b = model.add_const("b", Tensor::from_shape(&[co], &fill(4, co))?)?;
            let conv = Conv {
                pool_spec: PoolSpec {
                    data_format: DataFormat::NCHW,
                    kernel_shape: tvec![3, 3],
                    padding: PaddingSpec::Explicit(tvec![1, 1], tvec![1, 1]),
                    dilations: None,
                    strides: Some(tvec![stride, stride]),
                    input_channels: ci,
                    output_channels: co,
                },
                kernel_fmt: KernelFormat::OIHW,
                group: 1,
                q_params: None,
            };
            let y = model.wire_node("conv", conv, &[x, k, b])?[0];
            model.select_output_outlets(&[y])?;
            WgpuTransform.transform(&mut model)?;
            ensure!(
                !model.nodes().iter().any(|node| node.name.contains(".wino.")),
                "winograd rewrite fired for {ci}->{co} {h}x{w} s{stride}"
            );
            ensure!(
                model
                    .nodes()
                    .iter()
                    .any(|node| node.op_as::<crate::ops::conv::WgpuConv>().is_some()),
                "expected a plain WgpuConv for {ci}->{co} {h}x{w} s{stride}"
            );
        }
        Ok(())
    }

    /// Convolutions the implicit-GEMM kernel takes: strides, dilation, groups, a
    /// batch, channels-last, an uneven kernel and padding, f16, a bias, whole
    /// tiles, a long reduction over a small map that fills only a few tiles, and
    /// 16 to 31 output channels over a large map.
    #[test]
    fn conv_tiled_matches_cpu() -> TractResult<()> {
        use tract_core::ops::cnn::{Conv, KernelFormat, PaddingSpec, PoolSpec};
        use tract_core::ops::nn::DataFormat;
        #[rustfmt::skip]
        let cases: [(DataFormat, usize, usize, usize, [usize; 2], [usize; 2], [usize; 2], [usize; 2], usize, bool); 16] = [
            // fmt, n, ci, co, hw, kernel, stride, dilation, pad, f16
            (DataFormat::NCHW, 1, 8, 40, [46, 46], [3, 3], [1, 1], [1, 1], 1, false),
            (DataFormat::NCHW, 1, 8, 40, [90, 90], [3, 3], [2, 2], [1, 1], 1, false),
            (DataFormat::NCHW, 1, 8, 40, [49, 49], [3, 3], [1, 1], [2, 2], 0, false),
            (DataFormat::NCHW, 1, 16, 80, [33, 33], [3, 3], [1, 1], [1, 1], 1, false),
            (DataFormat::NCHW, 2, 8, 40, [33, 33], [3, 3], [1, 1], [1, 1], 1, false),
            (DataFormat::NHWC, 1, 8, 40, [46, 46], [3, 3], [1, 1], [1, 1], 1, false),
            (DataFormat::NCHW, 1, 8, 40, [45, 47], [1, 5], [1, 1], [1, 1], 2, false),
            (DataFormat::NCHW, 1, 8, 40, [46, 46], [3, 3], [1, 1], [1, 1], 1, true),
            (DataFormat::NCHW, 1, 32, 64, [46, 46], [3, 3], [1, 1], [1, 1], 1, false),
            (DataFormat::NCHW, 1, 32, 64, [46, 46], [3, 3], [1, 1], [1, 1], 1, true),
            (DataFormat::NCHW, 1, 64, 40, [7, 12], [5, 5], [1, 1], [1, 1], 2, false),
            (DataFormat::NCHW, 1, 8, 96, [20, 20], [3, 3], [1, 1], [1, 1], 1, false),
            (DataFormat::NHWC, 1, 8, 36, [20, 20], [3, 3], [1, 1], [1, 1], 1, false),
            (DataFormat::NCHW, 1, 35, 16, [120, 121], [3, 3], [1, 1], [1, 1], 1, false),
            (DataFormat::NCHW, 1, 40, 20, [120, 121], [3, 3], [1, 1], [1, 1], 1, false),
            (DataFormat::NCHW, 1, 29, 24, [121, 121], [3, 3], [1, 1], [1, 1], 1, false),
        ];
        for (i, (fmt, n, ci, co, hw, kernel, stride, dilation, pad, f16_)) in
            cases.into_iter().enumerate()
        {
            let group = if ci == 16 { 2 } else { 1 };
            let shape = fmt.from_n_c_hw(n, ci, hw)?.shape.to_vec();
            let k_shape = [co, ci / group, kernel[0], kernel[1]];
            // An f16 case runs on the GPU in f16 and is checked against the CPU in f32 on the
            // same f16-rounded values: the CPU's own f16 convolution is the less exact of the two.
            let values = |seed, shape: &[usize]| -> TractResult<Tensor> {
                let t = Tensor::from_shape(shape, &fill(seed, shape.iter().product()))?;
                Ok(if f16_ { t.cast_to::<f16>()?.cast_to::<f32>()?.into_owned() } else { t })
            };
            let input = values(3, &shape)?;
            let build = |dt: DatumType| -> TractResult<TypedModel> {
                let mut model = TypedModel::default();
                let x = model.add_source("x", dt.fact(&shape))?;
                let k = model.add_const("k", values(1, &k_shape)?.cast_to_dt(dt)?.into_owned())?;
                let b = model.add_const("b", values(2, &[co])?.cast_to_dt(dt)?.into_owned())?;
                let conv = Conv {
                    pool_spec: PoolSpec {
                        data_format: fmt,
                        kernel_shape: kernel.iter().copied().collect(),
                        padding: PaddingSpec::Explicit(tvec![pad, pad], tvec![pad, 0]),
                        dilations: Some(dilation.iter().copied().collect()),
                        strides: Some(stride.iter().copied().collect()),
                        input_channels: ci,
                        output_channels: co,
                    },
                    kernel_fmt: KernelFormat::OIHW,
                    group,
                    q_params: None,
                };
                let y = model.wire_node("conv", conv, &[x, k, b])?[0];
                model.select_output_outlets(&[y])?;
                Ok(model)
            };
            if f16_ {
                let rt =
                    tract_core::runtime::runtime_for_name("wgpu")?.context("no wgpu runtime")?;
                let gpu = rt.prepare(build(f16::datum_type())?)?;
                let g = gpu.run(tvec!(input.cast_to::<f16>()?.into_owned().into()))?.remove(0);
                let cpu = Arc::new(SimplePlan::new(build(f32::datum_type())?)?);
                let c = cpu.run(tvec!(input.clone().into()))?.remove(0);
                let c = c.cast_to::<f16>()?.into_owned();
                g.into_tensor().close_enough(&c, Approximation::Approximate)
            } else {
                gpu_vs_cpu(build(f32::datum_type())?, input)
            }
            .with_context(|| format!("case {i}"))?;
        }
        Ok(())
    }

    /// Convolutions whose few output tiles leave the device idle and whose long
    /// reduction the split-K path cuts into slices: odd channel counts, a
    /// reduction that is not a whole number of `k` steps, pixel and row tails, a
    /// batch, channels-last, strides, dilation, groups, a 1x1 kernel, and a bias,
    /// a bias then a sigmoid, a sigmoid alone, or no epilogue.
    #[test]
    fn conv_splitk_matches_cpu() -> TractResult<()> {
        use tract_core::ops::cnn::{Conv, KernelFormat, PaddingSpec, PoolSpec};
        use tract_core::ops::nn::{DataFormat, sigmoid};
        #[rustfmt::skip]
        let cases = [
            // fmt, n, ci, co, hw, kernel, stride, dilation, group, bias, sigmoid
            (DataFormat::NCHW, 1, 200, 96, [7, 9], [5, 5], 1, 1, 1, true, false),
            (DataFormat::NCHW, 1, 237, 37, [11, 13], [3, 3], 1, 1, 1, true, true),
            (DataFormat::NCHW, 2, 300, 50, [15, 15], [3, 3], 2, 1, 1, false, true),
            (DataFormat::NHWC, 1, 231, 70, [9, 10], [3, 3], 1, 2, 1, true, false),
            (DataFormat::NCHW, 1, 512, 64, [10, 10], [3, 3], 1, 1, 2, true, false),
            (DataFormat::NCHW, 1, 4100, 33, [6, 6], [1, 1], 1, 1, 1, false, false),
            (DataFormat::NHWC, 1, 1280, 96, [14, 24], [5, 5], 1, 1, 1, true, false),
            (DataFormat::NCHW, 1, 1280, 96, [18, 32], [5, 5], 1, 1, 1, true, true),
        ];
        for (i, (fmt, n, ci, co, hw, kernel, stride, dilation, group, bias, act)) in
            cases.into_iter().enumerate()
        {
            let pad = (kernel[0] - 1) / 2 * dilation;
            let shape = fmt.from_n_c_hw(n, ci, hw)?.shape.to_vec();
            let k_shape = [co, ci / group, kernel[0], kernel[1]];
            let mut model = TypedModel::default();
            let x = model.add_source("x", f32::fact(&shape))?;
            let weights: Vec<f32> =
                fill(1, k_shape.iter().product()).iter().map(|v| v * 0.02).collect();
            let k = model.add_const("k", Tensor::from_shape(&k_shape, &weights)?)?;
            let b = if bias {
                model.add_const("b", Tensor::from_shape(&[co], &fill(2, co))?)?
            } else {
                model.add_const("b", Tensor::zero::<f32>(&[co])?)?
            };
            let pool_spec = PoolSpec {
                data_format: fmt,
                kernel_shape: kernel.iter().copied().collect(),
                padding: PaddingSpec::Explicit(tvec![pad, pad], tvec![pad, 0]),
                dilations: Some(tvec![dilation, dilation]),
                strides: Some(tvec![stride, stride]),
                input_channels: ci,
                output_channels: co,
            };
            let out_shape = pool_spec.output_shape(&shape)?;
            let pixels = n * out_shape.hw_dims().iter().product::<usize>();
            let (co_pg, k_len) = (co / group, ci / group * kernel[0] * kernel[1]);
            let tile_m = super::tile_rows(co_pg);
            ensure!(
                super::tiled_applies(group, co_pg, pixels, k_len)
                    && super::conv_splitk::slices(
                        super::ShaderDtype::F32,
                        group,
                        co_pg,
                        pixels,
                        k_len,
                        tile_m
                    )
                    .is_some(),
                "case {i} does not take the split-K path"
            );
            let conv = Conv { pool_spec, kernel_fmt: KernelFormat::OIHW, group, q_params: None };
            let mut y = model.wire_node("conv", conv, &[x, k, b])?[0];
            if act {
                y = model.wire_node("act", sigmoid(), &[y])?[0];
            }
            model.select_output_outlets(&[y])?;
            let total: usize = shape.iter().product();
            gpu_vs_cpu(model, Tensor::from_shape(&shape, &fill(3, total))?)
                .with_context(|| format!("case {i}"))?;
        }
        Ok(())
    }

    /// A NaN in the input and an infinite weight reach the split-K result as
    /// they reach a plain evaluation's.
    #[test]
    fn conv_splitk_propagates_nan_and_inf() -> TractResult<()> {
        use tract_core::ops::cnn::{Conv, KernelFormat, PaddingSpec, PoolSpec};
        use tract_core::ops::nn::DataFormat;
        let (ci, co, h, w) = (256usize, 40usize, 8usize, 8usize);
        let mut model = TypedModel::default();
        let x = model.add_source("x", f32::fact([1, ci, h, w]))?;
        let mut weights = fill(1, co * ci * 9);
        weights[5 * ci * 9 + 100] = f32::INFINITY;
        let k = model.add_const("k", Tensor::from_shape(&[co, ci, 3, 3], &weights)?)?;
        let b = model.add_const("b", Tensor::from_shape(&[co], &fill(2, co))?)?;
        let conv = Conv {
            pool_spec: PoolSpec {
                data_format: DataFormat::NCHW,
                kernel_shape: tvec![3, 3],
                padding: PaddingSpec::Valid,
                dilations: None,
                strides: None,
                input_channels: ci,
                output_channels: co,
            },
            kernel_fmt: KernelFormat::OIHW,
            group: 1,
            q_params: None,
        };
        ensure!(
            super::conv_splitk::slices(super::ShaderDtype::F32, 1, co, 36, ci * 9, 40).is_some()
        );
        let y = model.wire_node("conv", conv, &[x, k, b])?[0];
        model.select_output_outlets(&[y])?;
        let mut input = fill(3, ci * h * w);
        input[200 * h * w + 3 * w + 4] = f32::NAN;
        gpu_vs_cpu(model, Tensor::from_shape(&[1, ci, h, w], &input)?)
    }

    /// Convolutions over one spatial axis run as two-dimensional ones on the device: dilated,
    /// padded, strided, channels-last.
    #[test]
    fn conv_1d_runs_on_device() -> TractResult<()> {
        use tract_core::ops::cnn::{Conv, KernelFormat, PaddingSpec, PoolSpec};
        use tract_core::ops::nn::DataFormat;
        #[rustfmt::skip]
        let cases = [
            // fmt, ci, co, len, kernel, stride, dilation, pad
            (DataFormat::NCHW, 8, 16, 50, 3, 1, 2, 2),
            (DataFormat::NCHW, 16, 40, 200, 2, 1, 4, 0),
            (DataFormat::NHWC, 8, 16, 50, 3, 2, 1, 1),
        ];
        for (i, (fmt, ci, co, len, kernel, stride, dilation, pad)) in cases.into_iter().enumerate()
        {
            let shape = fmt.from_n_c_hw(1, ci, [len])?.shape.to_vec();
            let mut model = TypedModel::default();
            let x = model.add_source("x", f32::fact(&shape))?;
            let k = model.add_const(
                "k",
                Tensor::from_shape(&[co, ci, kernel], &fill(1, co * ci * kernel))?,
            )?;
            let b = model.add_const("b", Tensor::from_shape(&[co], &fill(2, co))?)?;
            let conv = Conv {
                pool_spec: PoolSpec {
                    data_format: fmt,
                    kernel_shape: tvec![kernel],
                    padding: PaddingSpec::Explicit(tvec![pad], tvec![pad]),
                    dilations: Some(tvec![dilation]),
                    strides: Some(tvec![stride]),
                    input_channels: ci,
                    output_channels: co,
                },
                kernel_fmt: KernelFormat::OIHW,
                group: 1,
                q_params: None,
            };
            let y = model.wire_node("conv", conv, &[x, k, b])?[0];
            model.select_output_outlets(&[y])?;
            let translated = transformed(model.clone())?;
            ensure!(
                translated.nodes().iter().any(|n| n.op.name() == "WgpuConv"),
                "case {i}: the convolution did not reach the device"
            );
            let n: usize = shape.iter().product();
            gpu_vs_cpu(model, Tensor::from_shape(&shape, &fill(3, n))?)
                .with_context(|| format!("case {i}"))?;
        }
        Ok(())
    }

    /// `TRACT_WGPU_ALLOW_F16_ACCUMULATION` is captured when the model is
    /// prepared: products overflow f16 but cancel in pairs of input channels,
    /// so f32 accumulation gives exact zeros and f16 accumulation does not. The
    /// shapes take the direct and the split-K kernels.
    /// Runs a check of kernels the cooperative convolution would serve first.
    fn without_subgroups(check: fn() -> TractResult<()>) -> TractResult<()> {
        tract_data::knobs::KnobScope::new([("TRACT_WGPU_SUBGROUPS", "false")])?.enter(check)
    }

    #[test]
    fn conv_f16_accumulation_follows_knob() -> TractResult<()> {
        without_subgroups(conv_f16_accumulation_follows_knob_body)
    }

    fn conv_f16_accumulation_follows_knob_body() -> TractResult<()> {
        use tract_core::ops::cnn::{Conv, KernelFormat, PaddingSpec, PoolSpec};
        use tract_core::ops::nn::DataFormat;
        use tract_data::knobs::KnobScope;
        crate::context::wgpu_context();
        let rt = tract_core::runtime::runtime_for_name("wgpu")?.context("no wgpu runtime")?;
        for (n, ci, co, h, w) in [(1usize, 8usize, 16usize, 8usize, 16usize), (2, 24, 64, 5, 6)] {
            let mut model = TypedModel::default();
            let x = model.add_source("x", f16::fact([n, ci, h, w]))?;
            let weights = (0..co * ci * 9)
                .map(|i| f16::from_f32(if (i / 9) % ci % 2 == 0 { 300. } else { -300. }))
                .collect::<Vec<_>>();
            let k = model.add_const("k", Tensor::from_shape(&[co, ci, 3, 3], &weights)?)?;
            let b = model.add_const("b", rctensor0(f16::ZERO))?;
            let conv = Conv {
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
            };
            let y = model.wire_node("conv", conv, &[x, k, b])?[0];
            model.select_output_outlets(&[y])?;
            let input =
                Tensor::from_shape(&[n, ci, h, w], &vec![f16::from_f32(300.); n * ci * h * w])?;
            for f16_accumulation in [false, true, false] {
                let value = if f16_accumulation { "true" } else { "false" };
                let scope = KnobScope::new([("TRACT_WGPU_ALLOW_F16_ACCUMULATION", value)])?;
                let runnable = scope.enter(|| rt.prepare(model.clone()))?;
                let output = runnable.run(tvec![input.clone().into_tvalue()])?.remove(0);
                let output = output.into_tensor();
                let values = output.try_as_plain_ram()?.as_slice::<f16>()?;
                let all_zero = values.iter().all(|v| *v == f16::ZERO);
                ensure!(
                    all_zero != f16_accumulation,
                    "ci{ci} co{co} {h}x{w}: f16_accumulation={f16_accumulation}, all zero: {all_zero}"
                );
            }
        }
        Ok(())
    }

    #[test]
    fn measured_packed_tiles_cover_the_rvm_layers() {
        use super::DirectTile;
        use super::measured_packed_tile;
        for (ci, co, h, w, rows) in [
            (35usize, 16usize, 360usize, 640usize, 2usize),
            (16, 16, 360, 640, 2),
            (59, 32, 180, 320, 2),
            (32, 32, 180, 320, 2),
            (107, 40, 90, 160, 1),
            (171, 80, 45, 80, 1),
            (128, 64, 23, 40, 1),
            (32, 16, 180, 320, 1),
        ] {
            assert_eq!(measured_packed_tile(ci, co, h, w), Some(DirectTile { co: 8, rows }));
        }
        assert_eq!(measured_packed_tile(35, 16, 384, 640), None);
        assert_eq!(measured_packed_tile(3, 16, 360, 640), None);
    }

    /// Packed vec4 filter rows and the measured eight-channel tile match the
    /// scalar direct kernel bit for bit, including a per-channel bias epilogue.
    #[test]
    fn packed_direct_weights_match_scalar() -> TractResult<()> {
        use tract_core::ops::cnn::{Conv, KernelFormat, PaddingSpec, PoolSpec};
        use tract_core::ops::nn::DataFormat;
        use tract_gpu::tensor::DeviceTensor;
        use tract_gpu::tensor::IntoDevice;

        use super::DirectTile;
        use super::wgpu_conv_dispatch;
        use crate::kernels::shaders::ChainStep;

        crate::context::wgpu_context();
        for (ci, co, h, w, rows, bias) in
            [(4usize, 8usize, 8usize, 16usize, 2usize, true), (5, 8, 8, 16, 1, false)]
        {
            let halves = |seed: u64, len: usize, scale: f32| {
                fill(seed, len).into_iter().map(|v| f16::from_f32(v * scale)).collect::<Vec<_>>()
            };
            let input =
                Tensor::from_shape(&[1, ci, h, w], &halves(1, ci * h * w, 0.25))?.into_device()?;
            let weight_vals = halves(2, co * ci * 9, 0.05);
            let weights = Tensor::from_shape(&[co, ci, 3, 3], &weight_vals)?.into_device()?;
            let packed_vals: Vec<f16> = weight_vals
                .chunks_exact(3)
                .flat_map(|row| [row[0], row[1], row[2], f16::ZERO])
                .collect();
            let packed = Tensor::from_shape(&[co, ci, 3, 4], &packed_vals)?.into_device()?;
            let bias_t = if bias {
                Some(Tensor::from_shape(&[co], &halves(3, co, 0.1))?.into_device()?)
            } else {
                None
            };
            let op = Conv {
                pool_spec: PoolSpec {
                    data_format: DataFormat::NCHW,
                    kernel_shape: tvec![3, 3],
                    padding: PaddingSpec::Explicit(tvec![1, 1], tvec![1, 1]),
                    dilations: None,
                    strides: Some(tvec![1, 1]),
                    input_channels: ci,
                    output_channels: co,
                },
                kernel_fmt: KernelFormat::OIHW,
                group: 1,
                q_params: None,
            };
            let epilogue = if bias {
                vec![ChainStep::Binary { op: "add".into(), rhs: 1, swapped: false }]
            } else {
                vec![]
            };
            let extras: Vec<&DeviceTensor> = bias_t.iter().collect();
            let scalar_out = DeviceTensor::uninitialized_dt(DatumType::F16, &[1, co, h, w])?;
            let packed_out = DeviceTensor::uninitialized_dt(DatumType::F16, &[1, co, h, w])?;
            wgpu_conv_dispatch(
                &op,
                &epilogue,
                &input,
                &weights,
                &extras,
                &scalar_out,
                true,
                None,
                None,
                None,
            )?;
            wgpu_conv_dispatch(
                &op,
                &epilogue,
                &input,
                &weights,
                &extras,
                &packed_out,
                true,
                Some(&packed),
                Some(DirectTile { co: 8, rows }),
                None,
            )?;
            crate::with_wgpu_queue(|q| q.flush())?;
            let scalar = scalar_out.to_host()?;
            let packed_host = packed_out.to_host()?;
            assert_eq!(
                scalar.try_as_plain_ram()?.as_slice::<f16>()?,
                packed_host.try_as_plain_ram()?.as_slice::<f16>()?,
                "ci={ci} rows={rows} bias={bias}"
            );
        }
        Ok(())
    }

    /// Channels-last depthwise from 32 channels up walks channels four at a time.
    /// Sixteen channels stay on the spatial kernel. Both match the CPU, including
    /// a short width, a stride, a dilation, and f16.
    #[test]
    fn conv_depthwise_channels_matches_cpu() -> TractResult<()> {
        use tract_core::ops::cnn::{Conv, KernelFormat, PaddingSpec, PoolSpec};
        use tract_core::ops::nn::DataFormat;
        let shape =
            super::DepthwiseShape { kh: 3, kw: 3, stride_h: 1, stride_w: 1, dil_h: 1, dil_w: 1 };
        let strides = |h, w, c| [h * w * c, 1, w * c, c];
        ensure!(super::depthwise_channel_tiling(
            true,
            shape,
            32,
            [1, 15, 18],
            strides(15, 18, 32),
            strides(15, 18, 32),
            [0, 0],
        ));
        ensure!(!super::depthwise_channel_tiling(
            true,
            shape,
            16,
            [1, 15, 18],
            strides(15, 18, 16),
            strides(15, 18, 16),
            [0, 0],
        ));
        ensure!(!super::depthwise_channel_tiling(
            false,
            shape,
            32,
            [1, 15, 18],
            strides(15, 18, 32),
            strides(15, 18, 32),
            [0, 0],
        ));
        let wide = super::DepthwiseShape { kh: 5, kw: 5, ..shape };
        ensure!(super::depthwise_channel_tiling(
            true,
            wide,
            72,
            [1, 8, 8],
            strides(8, 8, 72),
            strides(8, 8, 72),
            [0, 0],
        ));
        let wider = super::DepthwiseShape { kh: 7, kw: 7, ..shape };
        ensure!(!super::depthwise_channel_tiling(
            true,
            wider,
            72,
            [1, 8, 8],
            strides(8, 8, 72),
            strides(8, 8, 72),
            [0, 0],
        ));

        // ks, n, c, h, w, stride, dilation, pad, f16
        let cases = [
            (3usize, 1usize, 32usize, 15usize, 18usize, 1usize, 1usize, 1usize, false),
            (3, 1, 32, 17, 13, 2, 1, 1, false),
            (3, 2, 64, 12, 20, 1, 1, 1, false),
            (3, 1, 36, 9, 8, 1, 1, 1, false),
            (3, 1, 72, 8, 7, 1, 1, 0, false),
            (3, 1, 32, 10, 10, 1, 2, 2, false),
            (3, 1, 48, 11, 14, 2, 1, 1, true),
            (3, 1, 16, 15, 18, 1, 1, 1, false),
            (5, 1, 32, 15, 18, 1, 1, 2, false),
            (5, 1, 40, 12, 16, 2, 1, 2, true),
        ];
        for (i, (ks, n, c, h, w, stride, dilation, pad, f16_)) in cases.into_iter().enumerate() {
            let hw = [h, w];
            let shape = DataFormat::NHWC.from_n_c_hw(n, c, hw)?.shape.to_vec();
            let k_shape = [c, 1, ks, ks];
            let values = |seed, dims: &[usize]| -> TractResult<Tensor> {
                let t = Tensor::from_shape(dims, &fill(seed, dims.iter().product()))?;
                Ok(if f16_ { t.cast_to::<f16>()?.cast_to::<f32>()?.into_owned() } else { t })
            };
            let input = values(3, &shape)?;
            let build = |dt: DatumType| -> TractResult<TypedModel> {
                let mut model = TypedModel::default();
                let x = model.add_source("x", dt.fact(&shape))?;
                let k = model.add_const("k", values(1, &k_shape)?.cast_to_dt(dt)?.into_owned())?;
                let b = model.add_const("b", values(2, &[c])?.cast_to_dt(dt)?.into_owned())?;
                let conv = Conv {
                    pool_spec: PoolSpec {
                        data_format: DataFormat::NHWC,
                        kernel_shape: tvec![ks, ks],
                        padding: PaddingSpec::Explicit(tvec![pad, pad], tvec![pad, pad]),
                        dilations: Some(tvec![dilation, dilation]),
                        strides: Some(tvec![stride, stride]),
                        input_channels: c,
                        output_channels: c,
                    },
                    kernel_fmt: KernelFormat::OIHW,
                    group: c,
                    q_params: None,
                };
                let y = model.wire_node("conv", conv, &[x, k, b])?[0];
                model.select_output_outlets(&[y])?;
                Ok(model)
            };
            if f16_ {
                f16_vs_f32(build, input, Approximation::Approximate)
                    .with_context(|| format!("case {i}"))?;
            } else {
                gpu_vs_cpu(build(f32::datum_type())?, input)
                    .with_context(|| format!("case {i}"))?;
            }
        }
        Ok(())
    }

    /// A fat 3×3 on a 10×16 or 20×32 map, the Longpipe decoder shapes, leaves the
    /// register-blocked kernel for split-K and still matches the CPU in f32 and
    /// f16 accumulation. A 28×48 map and a wide map keep the direct kernel.
    #[test]
    fn fat_small_map_conv_routes_splitk() -> TractResult<()> {
        without_subgroups(fat_small_map_conv_routes_splitk_body)
    }

    fn fat_small_map_conv_routes_splitk_body() -> TractResult<()> {
        use tract_core::ops::cnn::{Conv, KernelFormat, PaddingSpec, PoolSpec};
        use tract_core::ops::nn::DataFormat;
        use tract_data::knobs::KnobScope;
        use tract_gpu::tensor::IntoDevice;

        use super::wgpu_conv_dispatch;

        crate::context::wgpu_context();
        let names = |ci: usize, co: usize, h: usize, w: usize| -> TractResult<Vec<&'static str>> {
            let halves = |seed: u64, len: usize, scale: f32| {
                fill(seed, len).into_iter().map(|v| f16::from_f32(v * scale)).collect::<Vec<_>>()
            };
            let input =
                Tensor::from_shape(&[1, ci, h, w], &halves(1, ci * h * w, 0.05))?.into_device()?;
            let weights = Tensor::from_shape(&[co, ci, 3, 3], &halves(2, co * ci * 9, 0.02))?
                .into_device()?;
            let bias = Tensor::from_shape(&[co], &halves(3, co, 0.1))?.into_device()?;
            let output =
                tract_gpu::tensor::DeviceTensor::uninitialized_dt(DatumType::F16, &[1, co, h, w])?;
            let op = Conv {
                pool_spec: PoolSpec {
                    data_format: DataFormat::NCHW,
                    kernel_shape: tvec![3, 3],
                    padding: PaddingSpec::Explicit(tvec![1, 1], tvec![1, 1]),
                    dilations: None,
                    strides: Some(tvec![1, 1]),
                    input_channels: ci,
                    output_channels: co,
                },
                kernel_fmt: KernelFormat::OIHW,
                group: 1,
                q_params: None,
            };
            let epilogue = [crate::kernels::shaders::ChainStep::Binary {
                op: "add".into(),
                rhs: 1,
                swapped: false,
            }];
            let (_, profile) = crate::with_wgpu_queue(|queue| {
                queue.profile(|| {
                    wgpu_conv_dispatch(
                        &op,
                        &epilogue,
                        &input,
                        &weights,
                        &[&bias],
                        &output,
                        true,
                        None,
                        None,
                        None,
                    )
                })
            })?;
            Ok(profile.into_iter().map(|(name, _)| name).collect())
        };
        for (ci, co, h, w) in [
            (168usize, 64usize, 10usize, 16usize),
            (64, 64, 10, 16),
            (88, 32, 20, 32),
            (32, 32, 20, 32),
        ] {
            let got = names(ci, co, h, w)?;
            ensure!(
                got.iter().any(|name| name.starts_with("conv_splitk")),
                "{ci}->{co} {h}x{w} stayed on {got:?}"
            );
        }
        for (ci, co) in [(88usize, 32usize), (32, 32)] {
            let got = names(ci, co, 28, 48)?;
            ensure!(
                got.contains(&"conv_direct"),
                "{ci}->{co} at 28×48 left the direct kernel: {got:?}"
            );
        }
        let wide = names(32, 16, 180, 320)?;
        ensure!(wide.contains(&"conv_direct"), "wide map left the direct kernel: {wide:?}");

        let rt = tract_core::runtime::runtime_for_name("wgpu")?.context("no wgpu runtime")?;
        for (ci, co, h, w, f16_acc) in [
            (168usize, 64usize, 10usize, 16usize, false),
            (168, 64, 10, 16, true),
            (88, 32, 20, 32, true),
            (32, 32, 20, 32, true),
        ] {
            let values = |seed, shape: &[usize]| -> TractResult<Tensor> {
                let scaled: Vec<f32> =
                    fill(seed, shape.iter().product()).iter().map(|v| v * 0.02).collect();
                Ok(Tensor::from_shape(shape, &scaled)?
                    .cast_to::<f16>()?
                    .cast_to::<f32>()?
                    .into_owned())
            };
            let input = values(3, &[1, ci, h, w])?;
            let build = |dt: DatumType| -> TractResult<TypedModel> {
                let mut model = TypedModel::default();
                let x = model.add_source("x", dt.fact([1, ci, h, w]))?;
                let k = model
                    .add_const("k", values(1, &[co, ci, 3, 3])?.cast_to_dt(dt)?.into_owned())?;
                let b = model.add_const("b", values(2, &[co])?.cast_to_dt(dt)?.into_owned())?;
                let conv = Conv {
                    pool_spec: PoolSpec {
                        data_format: DataFormat::NCHW,
                        kernel_shape: tvec![3, 3],
                        padding: PaddingSpec::Explicit(tvec![1, 1], tvec![1, 1]),
                        dilations: None,
                        strides: Some(tvec![1, 1]),
                        input_channels: ci,
                        output_channels: co,
                    },
                    kernel_fmt: KernelFormat::OIHW,
                    group: 1,
                    q_params: None,
                };
                let y = model.wire_node("conv", conv, &[x, k, b])?[0];
                model.select_output_outlets(&[y])?;
                Ok(model)
            };
            let allow = if f16_acc { "true" } else { "false" };
            let scope = KnobScope::new([("TRACT_WGPU_ALLOW_F16_ACCUMULATION", allow)])?;
            let gpu = scope.enter(|| rt.prepare(build(f16::datum_type())?))?;
            let got = gpu.run(tvec!(input.cast_to::<f16>()?.into_owned().into()))?.remove(0);
            let cpu = Arc::new(SimplePlan::new(build(f32::datum_type())?)?);
            let expected = cpu.run(tvec!(input.into()))?.remove(0);
            let expected = expected.cast_to::<f16>()?.into_owned();
            got.into_tensor()
                .close_enough(&expected, Approximation::Approximate)
                .with_context(|| format!("{ci}->{co} {h}x{w} f16_accumulation={f16_acc}"))?;
        }
        Ok(())
    }
}
