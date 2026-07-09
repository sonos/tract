use crate::encoder::EncoderExt;
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
    ensure!(x.datum_type() == f32::datum_type());
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
    let token_count = x.len() / d_model;
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

    let pipeline = stream.load_pipeline(LibraryName::MoeOps, "route_topk_f32")?;
    let max_group_width = pipeline.max_total_threads_per_threadgroup() as u32;
    ensure!(
        num_experts <= max_group_width,
        "Metal RouteTopK requires at least one thread per expert: experts={num_experts}, max_threads={max_group_width}"
    );
    let group_width = num_experts.next_power_of_two().min(max_group_width).max(1);

    let command_buffer = stream.command_buffer();
    command_buffer.encode(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_metal_tensor(0, x, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(1, wg, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(2, route_token_ids, metal::MTLResourceUsage::Write);
        encoder.set_metal_tensor(3, route_expert_ids, metal::MTLResourceUsage::Write);
        encoder.set_metal_tensor(4, route_weights, metal::MTLResourceUsage::Write);
        encoder.set_slice(5, &[token_count]);
        encoder.set_slice(6, &[d_model]);
        encoder.set_slice(7, &[num_experts]);
        encoder.set_slice(8, &[k]);
        encoder.set_slice(9, &[gate_mode]);
        encoder.set_metal_tensor(10, wg_bias, metal::MTLResourceUsage::Read);
        encoder.set_slice(11, &[has_wg_bias]);

        let grid_size = MTLSize { width: token_count as NSUInteger, height: 1, depth: 1 };
        let group_size = MTLSize { width: group_width as NSUInteger, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(grid_size, group_size);
    });
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
    let group_width =
        (pipeline.max_total_threads_per_threadgroup() as u64).min(256).min(len as u64).max(1);
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
    let token_count = output.len() / d_model;
    ensure!(
        route_values.shape()[1] == d_model,
        "route value dim {} does not match output dim {d_model}",
        route_values.shape()[1]
    );

    let route_count = route_count as u32;
    let token_count = token_count as u32;
    let d_model = d_model as u32;
    let pipeline = stream.load_pipeline(LibraryName::MoeOps, "routed_combine_f32")?;
    let total = output.len() as u64;
    let group_width =
        (pipeline.max_total_threads_per_threadgroup() as u64).min(256).min(total).max(1);
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

        let grid_size = MTLSize { width: grid_width as NSUInteger, height: 1, depth: 1 };
        let group_size = MTLSize { width: group_width as NSUInteger, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(grid_size, group_size);
    });
    Ok(())
}
