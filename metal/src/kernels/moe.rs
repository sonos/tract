use crate::encoder::EncoderExt;
use crate::kernels::matmul::dispatch_router_scores;
use crate::{LibraryName, MetalStream};
use anyhow::ensure;
use metal::{MTLSize, NSUInteger};
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;
use tract_transformers::ops::moe_ffn::GateMode;

fn gate_mode_code(gate: &GateMode) -> u32 {
    match gate {
        GateMode::SoftmaxTopk => 0,
        GateMode::SoftmaxAll => 1,
        GateMode::Sigmoid => 2,
        GateMode::Raw => 3,
    }
}

#[allow(clippy::too_many_arguments)]
pub fn dispatch_route_topk_f32(
    stream: &MetalStream,
    x: &DeviceTensor,
    wg: &DeviceTensor,
    wg_bias: Option<&DeviceTensor>,
    k: usize,
    gate: &GateMode,
    route_token_ids: &DeviceTensor,
    route_expert_ids: &DeviceTensor,
    route_weights: &DeviceTensor,
) -> TractResult<()> {
    stream.retain_tensor(x);
    stream.retain_tensor(wg);
    if let Some(wg_bias) = wg_bias {
        stream.retain_tensor(wg_bias);
    }
    stream.retain_tensor(route_token_ids);
    stream.retain_tensor(route_expert_ids);
    stream.retain_tensor(route_weights);

    ensure!(x.rank() == 2 || x.rank() == 3, "x must be rank 2 or 3");
    ensure!(wg.rank() == 2 || wg.rank() == 3, "wg must be rank 2 or 3");
    ensure!(matches!(x.datum_type(), DatumType::F32 | DatumType::F16));
    ensure!(wg.datum_type() == f32::datum_type());
    if let Some(wg_bias) = wg_bias {
        ensure!(wg_bias.rank() == 1, "wg_bias must be rank 1");
        ensure!(wg_bias.datum_type() == f32::datum_type());
    }
    ensure!(route_token_ids.datum_type() == i64::datum_type());
    ensure!(route_expert_ids.datum_type() == i64::datum_type());
    ensure!(route_weights.datum_type() == f32::datum_type());
    ensure!(k <= 16, "Metal RouteTopK supports k <= 16, got {k}");

    let d_model = *x.shape().last().context("x has no feature axis")?;
    ensure!(d_model > 0, "Metal RouteTopK requires a nonzero feature dimension");
    let token_count: usize = x.shape()[..x.rank() - 1].iter().product();
    let (num_experts, wg_d_model) = match wg.rank() {
        2 => (wg.shape()[0], wg.shape()[1]),
        3 => {
            ensure!(wg.shape()[0] == 1, "rank-3 wg must have leading dimension 1");
            (wg.shape()[1], wg.shape()[2])
        }
        _ => unreachable!(),
    };
    ensure!(wg_d_model == d_model);
    if let Some(wg_bias) = wg_bias {
        ensure!(
            wg_bias.shape() == [num_experts],
            "wg_bias shape {:?} does not match expert count {num_experts}",
            wg_bias.shape()
        );
    }
    ensure!(k <= num_experts, "top-k {k} exceeds expert count {num_experts}");
    ensure!(num_experts <= 256, "Metal RouteTopK supports at most 256 experts");

    let route_count = token_count * k;
    ensure!(route_token_ids.shape() == [route_count]);
    ensure!(route_expert_ids.shape() == [route_count]);
    ensure!(route_weights.shape() == [route_count]);

    let token_count = token_count as u32;
    let d_model = d_model as u32;
    let num_experts = num_experts as u32;
    let k = k as u32;
    let gate_mode = gate_mode_code(gate);
    let has_wg_bias = u32::from(wg_bias.is_some());
    let wg_bias = wg_bias.unwrap_or(wg);

    let x2 = x.reshaped(tvec![token_count as usize, d_model as usize])?;
    let wg2 = wg.reshaped(tvec![num_experts as usize, d_model as usize])?;
    let scores = DeviceTensor::uninitialized_dt(
        f32::datum_type(),
        &[token_count as usize, num_experts as usize],
    )?;
    dispatch_router_scores(stream, &x2, &wg2, &scores)?;

    // Phase 2: tiny per-token top-k over the score rows, one simdgroup per
    // token.
    let pipeline = stream.load_pipeline(LibraryName::MoeOps, "route_select_topk_f32")?;
    let command_buffer = stream.command_buffer();
    command_buffer.encode(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_metal_tensor(0, &scores, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(1, route_token_ids, metal::MTLResourceUsage::Write);
        encoder.set_metal_tensor(2, route_expert_ids, metal::MTLResourceUsage::Write);
        encoder.set_metal_tensor(3, route_weights, metal::MTLResourceUsage::Write);
        encoder.set_slice(4, &[token_count]);
        encoder.set_slice(5, &[num_experts]);
        encoder.set_slice(6, &[k]);
        encoder.set_slice(7, &[gate_mode]);
        encoder.set_metal_tensor(8, wg_bias, metal::MTLResourceUsage::Read);
        encoder.set_slice(9, &[has_wg_bias]);

        let grid_size = MTLSize { width: token_count.max(1) as NSUInteger, height: 1, depth: 1 };
        let group_size = MTLSize { width: 32, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(grid_size, group_size);
    });
    // Cadence submission can move the initial retention onto an earlier buffer.
    stream.retain_tensor(&scores);
    Ok(())
}

pub fn dispatch_clamped_swiglu_f32(
    stream: &MetalStream,
    gate: &DeviceTensor,
    up: &DeviceTensor,
    alpha: f32,
    limit: f32,
    output: &DeviceTensor,
) -> TractResult<()> {
    stream.retain_tensor(gate);
    stream.retain_tensor(up);
    stream.retain_tensor(output);

    ensure!(gate.datum_type() == f32::datum_type());
    ensure!(up.datum_type() == f32::datum_type());
    ensure!(output.datum_type() == f32::datum_type());
    ensure!(gate.shape() == up.shape());
    ensure!(gate.shape() == output.shape());

    let len = gate.len() as u32;
    let pipeline = stream.load_pipeline(LibraryName::MoeOps, "clamped_swiglu_f32")?;
    let group_width = pipeline.max_total_threads_per_threadgroup().min(256).min(len as u64).max(1);
    let grid_width = (len as u64).div_ceil(group_width);
    let command_buffer = stream.command_buffer();
    command_buffer.encode(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_metal_tensor(0, gate, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(1, up, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(2, output, metal::MTLResourceUsage::Write);
        encoder.set_slice(3, &[alpha]);
        encoder.set_slice(4, &[limit]);
        encoder.set_slice(5, &[len]);

        let grid_size = MTLSize { width: grid_width as NSUInteger, height: 1, depth: 1 };
        let group_size = MTLSize { width: group_width as NSUInteger, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(grid_size, group_size);
    });
    Ok(())
}

pub fn dispatch_routed_combine_f32(
    stream: &MetalStream,
    route_values: &DeviceTensor,
    route_token_ids: &DeviceTensor,
    route_weights: &DeviceTensor,
    output: &DeviceTensor,
    routes_per_token: usize,
) -> TractResult<()> {
    stream.retain_tensor(route_values);
    stream.retain_tensor(route_token_ids);
    stream.retain_tensor(route_weights);
    stream.retain_tensor(output);

    ensure!(route_values.rank() == 2, "route_values must be [routes, d_model]");
    ensure!(route_token_ids.rank() == 1, "route_token_ids must be [routes]");
    ensure!(route_weights.rank() == 1, "route_weights must be [routes]");
    ensure!(output.rank() == 2 || output.rank() == 3, "output must be rank 2 or 3");
    ensure!(route_values.datum_type() == f32::datum_type());
    ensure!(route_token_ids.datum_type() == i64::datum_type());
    ensure!(route_weights.datum_type() == f32::datum_type());
    ensure!(output.datum_type() == f32::datum_type());

    let route_count = route_token_ids.shape()[0];
    ensure!(route_values.shape()[0] == route_count);
    ensure!(route_weights.shape()[0] == route_count);

    let d_model = *output.shape().last().context("output has no feature axis")?;
    let token_count: usize = output.shape()[..output.rank() - 1].iter().product();
    ensure!(
        route_values.shape()[1] == d_model,
        "route value dim {} does not match output dim {d_model}",
        route_values.shape()[1]
    );

    // Only trust the token-major fast path when the counts line up.
    let rpt = if routes_per_token > 0 && token_count * routes_per_token == route_count {
        routes_per_token as u32
    } else {
        0
    };
    let route_count = route_count as u32;
    let token_count = token_count as u32;
    let d_model = d_model as u32;
    let pipeline = stream.load_pipeline(LibraryName::MoeOps, "routed_combine_f32")?;
    let total = output.len() as u64;
    let group_width = pipeline.max_total_threads_per_threadgroup().min(256).min(total).max(1);
    let grid_width = total.div_ceil(group_width);
    let command_buffer = stream.command_buffer();
    command_buffer.encode(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_metal_tensor(0, route_values, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(1, route_token_ids, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(2, route_weights, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(3, output, metal::MTLResourceUsage::Write);
        encoder.set_slice(4, &[route_count]);
        encoder.set_slice(5, &[token_count]);
        encoder.set_slice(6, &[d_model]);
        encoder.set_slice(7, &[rpt]);

        let grid_size = MTLSize { width: grid_width as NSUInteger, height: 1, depth: 1 };
        let group_size = MTLSize { width: group_width as NSUInteger, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(grid_size, group_size);
    });
    Ok(())
}

/// Fused per-route expert bias add (see routed_bias_add_f32).
pub fn dispatch_routed_bias_add_f32(
    stream: &MetalStream,
    value: &DeviceTensor,
    bias: &DeviceTensor,
    route_expert_ids: &DeviceTensor,
    out: &DeviceTensor,
) -> TractResult<()> {
    for t in [value, bias, route_expert_ids] {
        stream.retain_tensor(t);
    }
    stream.retain_tensor(out);
    ensure!(value.datum_type() == f32::datum_type());
    ensure!(bias.datum_type() == f32::datum_type());
    ensure!(route_expert_ids.datum_type() == i64::datum_type());
    let n = *value.shape().last().context("value rank 0")?;
    let route_count = value.len() / n;
    ensure!(bias.shape().last() == Some(&n));
    ensure!(route_expert_ids.len() == route_count);
    let pipeline = stream.load_pipeline(LibraryName::MoeOps, "routed_bias_add_f32")?;
    let total = (route_count * n) as u64;
    let command_buffer = stream.command_buffer();
    command_buffer.encode(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_metal_tensor(0, value, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(1, bias, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(2, route_expert_ids, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(3, out, metal::MTLResourceUsage::Write);
        encoder.set_slice(4, &[route_count as u32]);
        encoder.set_slice(5, &[n as u32]);
        let group_width = total.clamp(1, 256);
        let grid = MTLSize { width: total.div_ceil(group_width), height: 1, depth: 1 };
        let group = MTLSize { width: group_width, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(grid, group);
    });
    Ok(())
}
