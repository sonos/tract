use tract_core::internal::*;
use tract_core::ops::cnn::Conv;
use tract_gpu::tensor::DeviceTensor;

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
/// tiles, with every tile index inside one grid dimension. The number of tiles
/// does not matter: a few tiles staging a long reduction through workgroup
/// memory still beat one thread per output reading it from global memory.
fn tiled_applies(groups: usize, co_pg: usize, pixels: usize, k: usize) -> bool {
    co_pg >= 32
        && pixels >= 32
        && k >= 16
        && groups <= 65535
        && co_pg.div_ceil(MATMUL_TILE) <= 65535
        && pixels.div_ceil(MATMUL_TILE) <= 65535
}

fn as_u32_i32(v: i32) -> u32 {
    v as u32
}

/// One filter per channel, no cross-channel sum: the tiled kernel applies.
fn depthwise_shape(op: &Conv, channels_last: bool) -> Option<DepthwiseShape> {
    let spec = &op.pool_spec;
    if channels_last
        || op.group == 1
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
) -> TractResult<()> {
    let dt = ShaderDtype::from_datum(input.datum_type())?;
    let layout = LayoutKind::Chain(3 + extras.len() as u8);
    let key = ("conv_dw", shape, dt, epilogue, extras.len());
    let pipeline =
        q.context().chain_pipeline(key, layout, EntryPoint::typed("conv_dw", dt), || {
            conv_depthwise_module(dt, shape, epilogue, extras.len())
        })?;

    let in_shape = op.pool_spec.data_format.shape(input.shape())?;
    let out_shape = op.pool_spec.data_format.shape(output.shape())?;
    let n = *in_shape.n().unwrap_or(&1);
    let co = *out_shape.c();
    let (oh, ow) = (out_shape.hw_dims()[0], out_shape.hw_dims()[1]);
    let padding = op.pool_spec.computed_padding(in_shape.hw_dims());
    let ws = weights.strides();

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
    let wg = shape.wg();
    let groups = [(ow as u32).div_ceil(wg), (oh as u32).div_ceil(wg), (n * co) as u32];
    q.dispatch_grid("conv_depthwise", &pipeline, &bg, dyn_off, groups)
}

pub fn wgpu_conv_dispatch(
    op: &Conv,
    epilogue: &[ChainStep],
    input: &DeviceTensor,
    weights: &DeviceTensor,
    extras: &[&DeviceTensor],
    output: &DeviceTensor,
) -> TractResult<()> {
    with_wgpu_queue(|q| {
        q.retain_tensor(input);
        q.retain_tensor(weights);
        for t in extras {
            q.retain_tensor(t);
        }
        q.retain_tensor(output);
        ensure!(matches!(input.datum_type(), DatumType::F32 | DatumType::F16));
        let channels_last = op.pool_spec.data_format.c_is_last();
        if let Some(shape) = depthwise_shape(op, channels_last) {
            return dispatch_depthwise(q, op, shape, epilogue, input, weights, extras, output);
        }
        let dt = ShaderDtype::from_datum(input.datum_type())?;
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

        // The per-pixel kernel unrolls 3x3 taps at dilation 1 and wants every
        // output channel of a pixel from one thread. With one thread per pixel
        // it only keeps the device busy when that thread's work is a stem's,
        // or when the pixels alone are enough threads.
        let pixels = n * oh * ow;
        let tiled = tiled_applies(groups, co_pg, pixels, ci_pg * kh * kw);
        let per_pixel = !tiled
            && channels_last == 0
            && groups == 1
            && kh == 3
            && kw == 3
            && dilations[0] == 1
            && dilations[1] == 1
            && (ci * co <= 64 || pixels >= 32768)
            && pixels.div_ceil(CONV_PIXEL_WG as usize) <= 65535;
        let tile_m = tile_rows(co_pg);
        let pipeline = if tiled {
            q.context().chain_pipeline(
                ("conv_tiled", tile_m, dt, epilogue, extras.len()),
                layout,
                EntryPoint::typed("conv2d_tiled", dt),
                || conv_tiled_module(dt, epilogue, extras.len(), tile_m),
            )?
        } else if per_pixel {
            q.context().chain_pipeline(
                ("conv_pixel", dt, epilogue, extras.len()),
                layout,
                EntryPoint::typed("conv2d_pixel", dt),
                || conv_pixel_module(dt, epilogue, extras.len()),
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

        let in_sn = *in_shape.n_stride().unwrap_or(&0);
        let in_sc = *in_shape.c_stride();
        let in_sh = in_shape.hw_strides()[0];
        let in_sw = in_shape.hw_strides()[1];
        let out_sn = *out_shape.n_stride().unwrap_or(&0);
        let out_sc = *out_shape.c_stride();
        let out_sh = out_shape.hw_strides()[0];
        let out_sw = out_shape.hw_strides()[1];

        // OIHW: [co, ci_pg, kh, kw]
        let ws = weights.strides();
        ensure!(ws.len() >= 4, "expected 4D OIHW kernel, got {:?}", weights.shape());
        let w_so = ws[0];
        let w_si = ws[1];
        let w_sh = ws[2];
        let w_sw = ws[3];

        let mut buffers: Vec<&crate::context::WgpuBuffer> =
            vec![get_wgpu_buffer(input), get_wgpu_buffer(weights)];
        buffers.extend(extras.iter().map(|t| get_wgpu_buffer(t)));
        buffers.push(get_wgpu_buffer(output));
        let bg = q.context().bind_group(layout, &buffers, q.uniform())?;
        let params = pack_u32s(&[
            element_offset(input, 0) as u32,
            element_offset(weights, 0) as u32,
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
        let dyn_off = q.alloc_uniform(&params)?;
        if tiled {
            let grid =
                [pixels.div_ceil(MATMUL_TILE) as u32, co_pg.div_ceil(tile_m) as u32, groups as u32];
            return q.dispatch_grid("conv_tiled", &pipeline, &bg, dyn_off, grid);
        }
        if per_pixel {
            let groups = pixels.div_ceil(CONV_PIXEL_WG as usize) as u32;
            return q.dispatch_grid("conv_pixel", &pipeline, &bg, dyn_off, [groups, 1, 1]);
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
/// 115us); the taps are unrolled for kh = kw = 3 at dilation 1.
pub fn conv_pixel_module(dt: ShaderDtype, epilogue: &[ChainStep], extras: usize) -> String {
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
fn conv2d_pixel_{suf}(@builtin(global_invocation_id) gid: vec3<u32>) {{
    let idx = gid.x;
    let npix = params.n * params.oh * params.ow;
    if (idx >= npix) {{ return; }}
    var rest = idx;
    let ow = rest % params.ow; rest /= params.ow;
    let oh = rest % params.oh; rest /= params.oh;
    let n = rest;
    let ih0 = i32(oh) * params.stride_h - params.pad_h;
    let iw0 = i32(ow) * params.stride_w - params.pad_w;
    // Four channels at a time, in four named accumulators: an indexed private
    // array here lands in memory under Tint, which costs ~20% in Chrome.
    for (var co_grp = 0u; co_grp < params.co; co_grp += 4u) {{
        var acc0 = 0.0;
        var acc1 = 0.0;
        var acc2 = 0.0;
        var acc3 = 0.0;
        for (var ci = 0u; ci < params.ci; ci++) {{
            let i00 = read_inp(ih0, iw0, ci, n);
            let i01 = read_inp(ih0, iw0 + 1, ci, n);
            let i02 = read_inp(ih0, iw0 + 2, ci, n);
            let i10 = read_inp(ih0 + 1, iw0, ci, n);
            let i11 = read_inp(ih0 + 1, iw0 + 1, ci, n);
            let i12 = read_inp(ih0 + 1, iw0 + 2, ci, n);
            let i20 = read_inp(ih0 + 2, iw0, ci, n);
            let i21 = read_inp(ih0 + 2, iw0 + 1, ci, n);
            let i22 = read_inp(ih0 + 2, iw0 + 2, ci, n);
            if (co_grp + 0u < params.co) {{
                let co = co_grp + 0u;
                let wc0 = params.off_w + co * u32(params.w_so) + ci * u32(params.w_si);
                let w00 = f32(wgt[wc0]);
                let w01 = f32(wgt[wc0 + u32(params.w_sw)]);
                let w02 = f32(wgt[wc0 + 2u * u32(params.w_sw)]);
                let w10 = f32(wgt[wc0 + u32(params.w_sh)]);
                let w11 = f32(wgt[wc0 + u32(params.w_sh) + u32(params.w_sw)]);
                let w12 = f32(wgt[wc0 + u32(params.w_sh) + 2u * u32(params.w_sw)]);
                let w20 = f32(wgt[wc0 + 2u * u32(params.w_sh)]);
                let w21 = f32(wgt[wc0 + 2u * u32(params.w_sh) + u32(params.w_sw)]);
                let w22 = f32(wgt[wc0 + 2u * u32(params.w_sh) + 2u * u32(params.w_sw)]);
                acc0 += i00 * w00 + i01 * w01 + i02 * w02 + i10 * w10 + i11 * w11 + i12 * w12 + i20 * w20 + i21 * w21 + i22 * w22;
            }}
            if (co_grp + 1u < params.co) {{
                let co = co_grp + 1u;
                let wc0 = params.off_w + co * u32(params.w_so) + ci * u32(params.w_si);
                let w00 = f32(wgt[wc0]);
                let w01 = f32(wgt[wc0 + u32(params.w_sw)]);
                let w02 = f32(wgt[wc0 + 2u * u32(params.w_sw)]);
                let w10 = f32(wgt[wc0 + u32(params.w_sh)]);
                let w11 = f32(wgt[wc0 + u32(params.w_sh) + u32(params.w_sw)]);
                let w12 = f32(wgt[wc0 + u32(params.w_sh) + 2u * u32(params.w_sw)]);
                let w20 = f32(wgt[wc0 + 2u * u32(params.w_sh)]);
                let w21 = f32(wgt[wc0 + 2u * u32(params.w_sh) + u32(params.w_sw)]);
                let w22 = f32(wgt[wc0 + 2u * u32(params.w_sh) + 2u * u32(params.w_sw)]);
                acc1 += i00 * w00 + i01 * w01 + i02 * w02 + i10 * w10 + i11 * w11 + i12 * w12 + i20 * w20 + i21 * w21 + i22 * w22;
            }}
            if (co_grp + 2u < params.co) {{
                let co = co_grp + 2u;
                let wc0 = params.off_w + co * u32(params.w_so) + ci * u32(params.w_si);
                let w00 = f32(wgt[wc0]);
                let w01 = f32(wgt[wc0 + u32(params.w_sw)]);
                let w02 = f32(wgt[wc0 + 2u * u32(params.w_sw)]);
                let w10 = f32(wgt[wc0 + u32(params.w_sh)]);
                let w11 = f32(wgt[wc0 + u32(params.w_sh) + u32(params.w_sw)]);
                let w12 = f32(wgt[wc0 + u32(params.w_sh) + 2u * u32(params.w_sw)]);
                let w20 = f32(wgt[wc0 + 2u * u32(params.w_sh)]);
                let w21 = f32(wgt[wc0 + 2u * u32(params.w_sh) + u32(params.w_sw)]);
                let w22 = f32(wgt[wc0 + 2u * u32(params.w_sh) + 2u * u32(params.w_sw)]);
                acc2 += i00 * w00 + i01 * w01 + i02 * w02 + i10 * w10 + i11 * w11 + i12 * w12 + i20 * w20 + i21 * w21 + i22 * w22;
            }}
            if (co_grp + 3u < params.co) {{
                let co = co_grp + 3u;
                let wc0 = params.off_w + co * u32(params.w_so) + ci * u32(params.w_si);
                let w00 = f32(wgt[wc0]);
                let w01 = f32(wgt[wc0 + u32(params.w_sw)]);
                let w02 = f32(wgt[wc0 + 2u * u32(params.w_sw)]);
                let w10 = f32(wgt[wc0 + u32(params.w_sh)]);
                let w11 = f32(wgt[wc0 + u32(params.w_sh) + u32(params.w_sw)]);
                let w12 = f32(wgt[wc0 + u32(params.w_sh) + 2u * u32(params.w_sw)]);
                let w20 = f32(wgt[wc0 + 2u * u32(params.w_sh)]);
                let w21 = f32(wgt[wc0 + 2u * u32(params.w_sh) + u32(params.w_sw)]);
                let w22 = f32(wgt[wc0 + 2u * u32(params.w_sh) + 2u * u32(params.w_sw)]);
                acc3 += i00 * w00 + i01 * w01 + i02 * w02 + i10 * w10 + i11 * w11 + i12 * w12 + i20 * w20 + i21 * w21 + i22 * w22;
            }}
        }}
        for (var j = 0u; j < 4u; j++) {{
            let co = co_grp + j;
            if (co >= params.co) {{ break; }}
            let acc = select(select(select(acc0, acc1, j == 1u), acc2, j == 2u), acc3, j == 3u);
            let out_i = params.off_out + u32(i32(n) * params.out_sn + i32(co) * params.out_sc + i32(oh) * params.out_sh + i32(ow) * params.out_sw);
            var v = {t}(acc);
{epilogue}
            outp[out_i] = v;
        }}
    }}
}}
"#
    ));
    s
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
    /// tiles, and a long reduction over a small map that fills only a few tiles.
    #[test]
    fn conv_tiled_matches_cpu() -> TractResult<()> {
        use tract_core::ops::cnn::{Conv, KernelFormat, PaddingSpec, PoolSpec};
        use tract_core::ops::nn::DataFormat;
        #[rustfmt::skip]
        let cases: [(DataFormat, usize, usize, usize, [usize; 2], [usize; 2], [usize; 2], [usize; 2], usize, bool); 13] = [
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
}
