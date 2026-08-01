//! Implicit-GEMM 2D convolution via the ported MLX kernel (see mlx_conv.metal).
//!
//! Covers NHWC f16/f32 2D convolutions with a single group and an `OHWI`
//! kernel — the layout the kernel indexes directly. Everything else is left to
//! the direct kernel by `mlx_conv_supported`.

use crate::encoder::EncoderExt;
use crate::{ConstantValues, LibraryName, MetalStream, Value};
use anyhow::ensure;
use metal::MTLSize;
use tract_core::internal::*;
use tract_core::ops::cnn::{Conv, KernelFormat};
use tract_gpu::tensor::DeviceTensor;

/// Mirror of MLX `MLXConvParams<2>` (steel/conv/params.h) — keep field order in sync.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct MlxConvParams2D {
    n: i32,
    c: i32,
    o: i32,
    i_s: [i32; 2],
    w_s: [i32; 2],
    o_s: [i32; 2],
    str: [i32; 2],
    pad: [i32; 2],
    kdil: [i32; 2],
    idil: [i32; 2],
    in_strides: [i64; 4],
    wt_strides: [i64; 4],
    out_strides: [i64; 4],
    groups: i32,
    flip: bool,
}

/// Mirror of MLX `ImplicitGemmConv2DParams`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct ImplicitGemmConv2DParams {
    m: i32,
    n: i32,
    k: i32,
    gemm_k_iterations: i32,
    inp_jump_w: i32,
    inp_jump_h: i32,
    inp_jump_c: i32,
    tiles_n: i32,
    tiles_m: i32,
    swizzle_log: i32,
}

/// Mirror of MLX `Conv2DGeneralJumpParams`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct Conv2DGeneralJumpParams {
    f_wgt_jump_h: i32,
    f_wgt_jump_w: i32,
    f_out_jump_h: i32,
    f_out_jump_w: i32,
    adj_out_h: i32,
    adj_out_w: i32,
    adj_out_hw: i32,
    adj_implicit_m: i32,
}

/// Mirror of MLX `Conv2DGeneralBaseInfo`.
#[repr(C)]
#[derive(Debug, Clone, Copy)]
struct Conv2DGeneralBaseInfo {
    weight_base: i32,
    weight_size: i32,
}

fn gcd(a: i32, b: i32) -> i32 {
    if b == 0 { a } else { gcd(b, a % b) }
}

fn lcm(a: i32, b: i32) -> i32 {
    a / gcd(a, b) * b
}

/// Whether the ported kernel can take this convolution: NHWC f16/f32, one
/// group, rank-2 spatial, `OHWI` weights, no output padding tricks.
pub fn mlx_conv_eligible(op: &Conv, in_facts: &[&TypedFact]) -> bool {
    if op.group != 1 || op.q_params.is_some() {
        return false;
    }
    if !matches!(in_facts[0].datum_type, DatumType::F16 | DatumType::F32) {
        return false;
    }
    if in_facts[0].datum_type != in_facts[1].datum_type {
        return false;
    }
    let Ok(shape) = op.pool_spec.data_format.shape(in_facts[0].shape.to_tvec()) else {
        return false;
    };
    if shape.hw_rank() != 2 || !op.pool_spec.data_format.c_is_last() {
        return false;
    }
    in_facts.iter().all(|f| f.shape.as_concrete().is_some())
}

/// `out[N, oH, oW, O] = conv(in[N, iH, iW, C], wt[O, kH, kW, C])`. mlx keeps two
/// implicit-GEMM kernels and prefers the specialised one whenever the channel
/// counts are aligned, falling back to the general one otherwise.
pub fn dispatch_mlx_conv_2d(
    stream: &MetalStream,
    op: &Conv,
    input: &DeviceTensor,
    weights: &DeviceTensor,
    output: &DeviceTensor,
) -> TractResult<()> {
    let c = op.pool_spec.input_channels;
    let o = op.pool_spec.output_channels;
    if (c <= 4 || c.is_multiple_of(16)) && (o <= 16 || o.is_multiple_of(16)) {
        dispatch_mlx_conv_2d_specialized(stream, op, input, weights, output)
    } else {
        dispatch_mlx_conv_2d_general(stream, op, input, weights, output)
    }
}

/// Mirrors mlx `implicit_gemm_conv_2D_general_gpu`.
pub fn dispatch_mlx_conv_2d_general(
    stream: &MetalStream,
    op: &Conv,
    input: &DeviceTensor,
    weights: &DeviceTensor,
    output: &DeviceTensor,
) -> TractResult<()> {
    let dt = input.datum_type();
    let tname = match dt {
        DatumType::F32 => "float32",
        DatumType::F16 => "float16",
        _ => bail!("MLX conv: F32/F16 only, got {dt:?}"),
    };
    let in_shape = op.pool_spec.data_format.shape(input.shape())?;
    let out_shape = op.pool_spec.data_format.shape(output.shape())?;
    ensure!(in_shape.hw_rank() == 2, "MLX conv is 2D only");

    let n = *in_shape.n().unwrap_or(&1) as i32;
    let c = *in_shape.c() as i32;
    let o = *out_shape.c() as i32;
    let i_s = [in_shape.hw_dims()[0] as i32, in_shape.hw_dims()[1] as i32];
    let w_s = [weights.shape()[1] as i32, weights.shape()[2] as i32];
    let o_s = [out_shape.hw_dims()[0] as i32, out_shape.hw_dims()[1] as i32];
    let strides = op.pool_spec.strides();
    let dilations = op.pool_spec.dilations();
    let str = [strides[0] as i32, strides[1] as i32];
    let kdil = [dilations[0] as i32, dilations[1] as i32];
    let idil = [1i32, 1];
    let padding = op.pool_spec.computed_padding(in_shape.hw_dims());
    let pad = [padding[0].pad_before as i32, padding[1].pad_before as i32];

    let to4 = |s: &[isize]| -> [i64; 4] { [s[0] as i64, s[1] as i64, s[2] as i64, s[3] as i64] };
    let ceil_div = |a: i32, b: i32| (a + b - 1) / b;
    let conv_params = MlxConvParams2D {
        n,
        c,
        o,
        i_s,
        w_s,
        o_s,
        str,
        pad,
        kdil,
        idil,
        in_strides: to4(input.strides()),
        wt_strides: to4(weights.strides()),
        out_strides: to4(output.strides()),
        groups: 1,
        flip: false,
    };

    let implicit_m = n * o_s[0] * o_s[1];
    let implicit_n = o;
    let (wm, wn) = (2i32, 2i32);

    let f_wgt_jump_h = lcm(idil[0], kdil[0]) / kdil[0];
    let f_wgt_jump_w = lcm(idil[1], kdil[1]) / kdil[1];
    let f_out_jump_h = lcm(idil[0], str[0]) / str[0];
    let f_out_jump_w = lcm(idil[1], str[1]) / str[1];
    let adj_out_h = ceil_div(o_s[0], f_out_jump_h);
    let adj_out_w = ceil_div(o_s[1], f_out_jump_w);
    let adj_out_hw = adj_out_h * adj_out_w;
    let adj_implicit_m = n * adj_out_hw;
    let jump_params = Conv2DGeneralJumpParams {
        f_wgt_jump_h,
        f_wgt_jump_w,
        f_out_jump_h,
        f_out_jump_w,
        adj_out_h,
        adj_out_w,
        adj_out_hw,
        adj_implicit_m,
    };

    let base_of = |jumps: i32, stride: i32, pad: i32, ws: i32, kdil: i32, wgt_jump: i32| {
        (0..jumps)
            .map(|i| {
                let mut loop_pos = i * stride - pad;
                let mut base = 0;
                while base < ws && loop_pos % idil[0] != 0 {
                    base += 1;
                    loop_pos += kdil;
                }
                Conv2DGeneralBaseInfo {
                    weight_base: base,
                    weight_size: (ws - base + wgt_jump - 1) / wgt_jump,
                }
            })
            .collect::<Vec<_>>()
    };
    let base_h = base_of(f_out_jump_h, str[0], pad[0], w_s[0], kdil[0], f_wgt_jump_h);
    let base_w = base_of(f_out_jump_w, str[1], pad[1], w_s[1], kdil[1], f_wgt_jump_w);

    let bm = if adj_implicit_m >= 8192 && c >= 64 { 64 } else { 32 };
    let bn = if bm == 64 && implicit_n >= 64 { 64 } else { 32 };
    let bk = 16i32;
    let tn = ceil_div(implicit_n, bn);
    let tm = ceil_div(adj_implicit_m, bm);
    let swizzle_log = 0i32;
    let align_c = c % bk == 0;

    let ijw = conv_params.in_strides[2] as i32 * kdil[1];
    let ijh = conv_params.in_strides[1] as i32 * kdil[0];
    let gemm_params = ImplicitGemmConv2DParams {
        m: implicit_m,
        n: implicit_n,
        k: w_s[0] * w_s[1] * c,
        gemm_k_iterations: ceil_div(c, bk),
        inp_jump_w: ijw,
        inp_jump_h: ijh - (w_s[1] - 1) * ijw,
        inp_jump_c: bk - (w_s[0] - 1) * ijh - (w_s[1] - 1) * ijw,
        tiles_n: tn,
        tiles_m: tm,
        swizzle_log,
    };

    let name = format!("implicit_gemm_conv_2d_general_{tname}_bm{bm}_bn{bn}_bk{bk}_wm{wm}_wn{wn}");
    let constants = Some(ConstantValues::new(vec![(200, Value::Bool(align_c))]));
    let pipeline = stream.load_pipeline_with_constants(LibraryName::MlxConv, &name, constants)?;

    stream.retain_tensor(input);
    stream.retain_tensor(weights);
    stream.retain_tensor(output);

    let command_buffer = stream.command_buffer();
    command_buffer.encode(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_metal_tensor(0, input, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(1, weights, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(2, output, metal::MTLResourceUsage::Write);
        encoder.set_slice(3, std::slice::from_ref(&conv_params));
        encoder.set_slice(4, std::slice::from_ref(&gemm_params));
        encoder.set_slice(5, std::slice::from_ref(&jump_params));
        encoder.set_slice(6, &base_h);
        encoder.set_slice(7, &base_w);
        let tile = 1 << swizzle_log;
        let grid = MTLSize {
            width: (tn * tile) as _,
            height: ceil_div(tm, tile) as _,
            depth: (f_out_jump_h * f_out_jump_w) as _,
        };
        let group = MTLSize { width: 32, height: wn as _, depth: wm as _ };
        encoder.dispatch_thread_groups(grid, group);
    });
    Ok(())
}

/// Runtime counterpart of `mlx_conv_supported`, checked against the device
/// tensors actually being dispatched.
pub fn mlx_conv_dispatchable(op: &Conv, input: &DeviceTensor, weights: &DeviceTensor) -> bool {
    if op.group != 1 || op.q_params.is_some() || op.kernel_fmt != KernelFormat::OHWI {
        return false;
    }
    if !matches!(input.datum_type(), DatumType::F16 | DatumType::F32)
        || input.datum_type() != weights.datum_type()
    {
        return false;
    }
    if !op.pool_spec.data_format.c_is_last() || input.rank() != 4 || weights.rank() != 4 {
        return false;
    }
    let natural = |t: &DeviceTensor| {
        let mut s = 1isize;
        t.shape().iter().rev().zip(t.strides().iter().rev()).all(|(&d, &st)| {
            let ok = st == s;
            s *= d as isize;
            ok
        })
    };
    natural(input) && natural(weights)
}

/// Whether the ported depthwise kernel can take this convolution: one channel
/// per group in and out, kernel up to 7×7, stride up to 2, channels a multiple
/// of 16 — mlx's own gate, plus NHWC and a contiguous `OHWI` kernel.
pub fn mlx_depthwise_eligible(op: &Conv, in_facts: &[&TypedFact]) -> bool {
    if op.q_params.is_some() || op.group < 2 {
        return false;
    }
    let (c, o) = (op.pool_spec.input_channels, op.pool_spec.output_channels);
    if c != o || op.group != c || !c.is_multiple_of(16) {
        return false;
    }
    if !matches!(in_facts[0].datum_type, DatumType::F16 | DatumType::F32)
        || in_facts[0].datum_type != in_facts[1].datum_type
    {
        return false;
    }
    if !op.pool_spec.data_format.c_is_last() || in_facts[0].rank() != 4 {
        return false;
    }
    let k = op.pool_spec.kernel_shape.as_slice();
    if k.len() != 2 || k.iter().any(|&x| x > 7) {
        return false;
    }
    if op.pool_spec.dilations().iter().any(|&d| d != 1) {
        return false;
    }
    if op.pool_spec.strides().iter().any(|&s| s > 2) {
        return false;
    }
    in_facts.iter().all(|f| f.shape.as_concrete().is_some())
}

/// Runtime counterpart of `mlx_depthwise_eligible`.
pub fn mlx_depthwise_dispatchable(op: &Conv, input: &DeviceTensor, weights: &DeviceTensor) -> bool {
    let c = op.pool_spec.input_channels;
    op.q_params.is_none()
        && op.group == c
        && op.pool_spec.output_channels == c
        && c.is_multiple_of(16)
        && op.kernel_fmt == KernelFormat::OIHW
        && matches!(input.datum_type(), DatumType::F16 | DatumType::F32)
        && input.datum_type() == weights.datum_type()
        && op.pool_spec.data_format.c_is_last()
        && input.rank() == 4
        && weights.rank() == 4
        && op.pool_spec.kernel_shape.iter().all(|&k| k <= 7)
        && op.pool_spec.dilations().iter().all(|&d| d == 1)
        && op.pool_spec.strides().iter().all(|&s| s <= 2)
}

/// Depthwise `out[N, oH, oW, C] = conv(in[N, iH, iW, C], wt[C, kH, kW, 1])`,
/// mirroring mlx `depthwise_conv_2D_gpu`.
pub fn dispatch_mlx_depthwise_conv_2d(
    stream: &MetalStream,
    op: &Conv,
    input: &DeviceTensor,
    weights: &DeviceTensor,
    output: &DeviceTensor,
) -> TractResult<()> {
    let dt = input.datum_type();
    let tname = match dt {
        DatumType::F32 => "float32",
        DatumType::F16 => "float16",
        _ => bail!("MLX depthwise conv: F32/F16 only, got {dt:?}"),
    };
    let in_shape = op.pool_spec.data_format.shape(input.shape())?;
    let out_shape = op.pool_spec.data_format.shape(output.shape())?;
    let n = *in_shape.n().unwrap_or(&1) as i32;
    let c = *in_shape.c() as i32;
    let i_s = [in_shape.hw_dims()[0] as i32, in_shape.hw_dims()[1] as i32];
    let o_s = [out_shape.hw_dims()[0] as i32, out_shape.hw_dims()[1] as i32];
    // tract hands depthwise kernels over as OIHW [C, 1, kH, kW], which is the
    // same bytes as the [C, kH, kW, 1] the kernel indexes.
    let w_s = [weights.shape()[2] as i32, weights.shape()[3] as i32];
    let strides = op.pool_spec.strides();
    let str = [strides[0] as i32, strides[1] as i32];
    let padding = op.pool_spec.computed_padding(in_shape.hw_dims());
    let pad = [padding[0].pad_before as i32, padding[1].pad_before as i32];

    let to4 = |s: &[isize]| -> [i64; 4] { [s[0] as i64, s[1] as i64, s[2] as i64, s[3] as i64] };
    let wt_strides = [(w_s[0] * w_s[1]) as i64, w_s[1] as i64, 1, 1];
    let conv_params = MlxConvParams2D {
        n,
        c,
        o: c,
        i_s,
        w_s,
        o_s,
        str,
        pad,
        kdil: [1, 1],
        idil: [1, 1],
        in_strides: to4(input.strides()),
        wt_strides,
        out_strides: to4(output.strides()),
        groups: c,
        flip: false,
    };

    let (tc, tw, th) = (8i32, 8i32, 4i32);
    let constants = Some(ConstantValues::new(vec![
        (0, Value::I32(w_s[0])),   // ker_h
        (1, Value::I32(w_s[1])),   // ker_w
        (10, Value::I32(str[0])),  // str_h
        (11, Value::I32(str[1])),  // str_w
        (100, Value::I32(th)),     // tgp_h
        (101, Value::I32(tw)),     // tgp_w
        (200, Value::Bool(false)), // do_flip
    ]));
    let name = format!("depthwise_conv_2d_{tname}");
    let pipeline = stream.load_pipeline_with_constants(LibraryName::MlxConvDw, &name, constants)?;

    stream.retain_tensor(input);
    stream.retain_tensor(weights);
    stream.retain_tensor(output);

    let command_buffer = stream.command_buffer();
    command_buffer.encode(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_metal_tensor(0, input, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(1, weights, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(2, output, metal::MTLResourceUsage::Write);
        encoder.set_slice(3, std::slice::from_ref(&conv_params));
        let ceil_div = |a: i32, b: i32| (a + b - 1) / b;
        let grid = MTLSize {
            width: (c / tc) as _,
            height: ceil_div(o_s[1], tw) as _,
            depth: (ceil_div(o_s[0], th) * n) as _,
        };
        let group = MTLSize { width: tc as _, height: tw as _, depth: th as _ };
        encoder.dispatch_thread_groups(grid, group);
    });
    Ok(())
}

/// Mirrors mlx `implicit_gemm_conv_2D_gpu`: no per-output-position jump tables,
/// and the kernel is specialised on small channel counts and short filters.
pub fn dispatch_mlx_conv_2d_specialized(
    stream: &MetalStream,
    op: &Conv,
    input: &DeviceTensor,
    weights: &DeviceTensor,
    output: &DeviceTensor,
) -> TractResult<()> {
    let dt = input.datum_type();
    let tname = match dt {
        DatumType::F32 => "float32",
        DatumType::F16 => "float16",
        _ => bail!("MLX conv: F32/F16 only, got {dt:?}"),
    };
    let in_shape = op.pool_spec.data_format.shape(input.shape())?;
    let out_shape = op.pool_spec.data_format.shape(output.shape())?;
    let n = *in_shape.n().unwrap_or(&1) as i32;
    let c = *in_shape.c() as i32;
    let o = *out_shape.c() as i32;
    let i_s = [in_shape.hw_dims()[0] as i32, in_shape.hw_dims()[1] as i32];
    let w_s = [weights.shape()[1] as i32, weights.shape()[2] as i32];
    let o_s = [out_shape.hw_dims()[0] as i32, out_shape.hw_dims()[1] as i32];
    let strides = op.pool_spec.strides();
    let dilations = op.pool_spec.dilations();
    let str = [strides[0] as i32, strides[1] as i32];
    let kdil = [dilations[0] as i32, dilations[1] as i32];
    let padding = op.pool_spec.computed_padding(in_shape.hw_dims());
    let pad = [padding[0].pad_before as i32, padding[1].pad_before as i32];
    let ceil_div = |a: i32, b: i32| (a + b - 1) / b;
    let to4 = |s: &[isize]| -> [i64; 4] { [s[0] as i64, s[1] as i64, s[2] as i64, s[3] as i64] };

    let conv_params = MlxConvParams2D {
        n,
        c,
        o,
        i_s,
        w_s,
        o_s,
        str,
        pad,
        kdil,
        idil: [1, 1],
        in_strides: to4(input.strides()),
        wt_strides: to4(weights.strides()),
        out_strides: to4(output.strides()),
        groups: 1,
        flip: false,
    };

    let implicit_m = n * o_s[0] * o_s[1];
    let implicit_n = o;
    let implicit_k = w_s[0] * w_s[1] * c;
    let (mut wm, mut wn) = (2i32, 2i32);
    let bm = if implicit_m >= 8192 && c >= 64 { 64 } else { 32 };
    let mut bn = if bm == 64 || implicit_n >= 64 { 64 } else { 32 };
    let bk = 16i32;
    if implicit_n <= 16 {
        bn = 8;
        wm = 4;
        wn = 1;
    }

    let channel_k_iters = ceil_div(c, bk);
    let (gemm_k_iterations, channel_spec) = if c <= 2 {
        (ceil_div(implicit_k, bk), c)
    } else if c <= 4 {
        (ceil_div(w_s[0] * w_s[1] * 4, bk), c)
    } else {
        (w_s[0] * w_s[1] * channel_k_iters, 0)
    };
    let small_filter = channel_spec == 0 && w_s[0] <= 16 && w_s[1] <= 16;

    let ijw = conv_params.in_strides[2] as i32 * kdil[1];
    let ijh = conv_params.in_strides[1] as i32 * kdil[0];
    let gemm_params = ImplicitGemmConv2DParams {
        m: implicit_m,
        n: implicit_n,
        k: implicit_k,
        gemm_k_iterations,
        inp_jump_w: ijw,
        inp_jump_h: ijh - (w_s[1] - 1) * ijw,
        inp_jump_c: bk - (w_s[0] - 1) * ijh - (w_s[1] - 1) * ijw,
        tiles_n: ceil_div(implicit_n, bn),
        tiles_m: ceil_div(implicit_m, bm),
        swizzle_log: 0,
    };

    let channel = if channel_spec == 0 { "l".to_string() } else { channel_spec.to_string() };
    let filter = if small_filter { 's' } else { 'l' };
    let name = format!(
        "implicit_gemm_conv_2d_{tname}_bm{bm}_bn{bn}_bk{bk}_wm{wm}_wn{wn}_channel_{channel}_filter_{filter}"
    );
    let pipeline = stream.load_pipeline(LibraryName::MlxConvSpec, &name)?;

    stream.retain_tensor(input);
    stream.retain_tensor(weights);
    stream.retain_tensor(output);

    let command_buffer = stream.command_buffer();
    command_buffer.encode(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_metal_tensor(0, input, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(1, weights, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(2, output, metal::MTLResourceUsage::Write);
        encoder.set_slice(3, std::slice::from_ref(&conv_params));
        encoder.set_slice(4, std::slice::from_ref(&gemm_params));
        encoder.dispatch_thread_groups(
            MTLSize { width: gemm_params.tiles_n as _, height: gemm_params.tiles_m as _, depth: 1 },
            MTLSize { width: 32, height: wn as _, depth: wm as _ },
        );
    });
    Ok(())
}
