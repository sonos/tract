use crate::encoder::EncoderExt;
use crate::kernels::matmul::dispatch_router_scores;
use crate::{LibraryName, MetalStream};
use anyhow::ensure;
use metal::{MTLSize, NSUInteger};
use tract_core::internal::*;
use tract_gpu::ops::routing::gate_code;
use tract_gpu::tensor::DeviceTensor;
use tract_transformers::ops::moe_ffn::GateMode;

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

    let x2 = x.reshaped(tvec![token_count, d_model])?;
    let wg2 = wg.reshaped(tvec![num_experts, d_model])?;
    let scores = DeviceTensor::uninitialized_dt(f32::datum_type(), &[token_count, num_experts])?;
    dispatch_router_scores(stream, &x2, &wg2, &scores)?;
    dispatch_route_select_f32(
        stream,
        &scores,
        wg_bias,
        k,
        gate_code(gate),
        route_token_ids,
        route_expert_ids,
        route_weights,
    )?;
    // Cadence submission can move the initial retention onto an earlier buffer.
    stream.retain_tensor(&scores);
    Ok(())
}

/// Top-k selection over f32 router `scores [T, E]` plus the optional f32 `bias [E]`, one
/// simdgroup per token, to token-major `token_ids`, `expert_ids` (i64) and `weights` (f32),
/// `[T * k]` each, gated by `gate` (see [`gate_code`]).
#[allow(clippy::too_many_arguments)]
pub fn dispatch_route_select_f32(
    stream: &MetalStream,
    scores: &DeviceTensor,
    bias: Option<&DeviceTensor>,
    k: usize,
    gate: u32,
    route_token_ids: &DeviceTensor,
    route_expert_ids: &DeviceTensor,
    route_weights: &DeviceTensor,
) -> TractResult<()> {
    let _retained = stream.retain_for_dispatch(&[
        scores,
        bias.unwrap_or(scores),
        route_token_ids,
        route_expert_ids,
        route_weights,
    ]);
    ensure!(scores.rank() == 2 && scores.datum_type() == f32::datum_type());
    let (token_count, num_experts) = (scores.shape()[0], scores.shape()[1]);
    ensure!((1..=256).contains(&num_experts), "Metal route select scores at most 256 experts");
    ensure!(k <= 16 && k <= num_experts, "Metal route select picks at most 16 routes");
    if let Some(bias) = bias {
        ensure!(bias.datum_type() == f32::datum_type() && bias.shape() == [num_experts]);
    }
    for ids in [route_token_ids, route_expert_ids] {
        ensure!(ids.datum_type() == i64::datum_type() && ids.shape() == [token_count * k]);
    }
    ensure!(
        route_weights.datum_type() == f32::datum_type()
            && route_weights.shape() == [token_count * k]
    );
    let has_bias = u32::from(bias.is_some());
    let bias = bias.unwrap_or(scores);
    let pipeline = stream.load_pipeline(LibraryName::MoeOps, "route_select_topk_f32")?;
    let command_buffer = stream.command_buffer();
    command_buffer.encode(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_metal_tensor(0, scores, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(1, route_token_ids, metal::MTLResourceUsage::Write);
        encoder.set_metal_tensor(2, route_expert_ids, metal::MTLResourceUsage::Write);
        encoder.set_metal_tensor(3, route_weights, metal::MTLResourceUsage::Write);
        encoder.set_slice(4, &[token_count as u32]);
        encoder.set_slice(5, &[num_experts as u32]);
        encoder.set_slice(6, &[k as u32]);
        encoder.set_slice(7, &[gate]);
        encoder.set_metal_tensor(8, bias, metal::MTLResourceUsage::Read);
        encoder.set_slice(9, &[has_bias]);
        let grid_size = MTLSize { width: token_count.max(1) as NSUInteger, height: 1, depth: 1 };
        let group_size = MTLSize { width: 32, height: 1, depth: 1 };
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

/// Weighted sum of token-major f32 `route_values [T * k, D]` by f32 `route_weights [T * k]`,
/// `k` routes per token, into an f32 or f16 `output [T, D]`.
pub fn dispatch_routed_combine(
    stream: &MetalStream,
    route_values: &DeviceTensor,
    route_weights: &DeviceTensor,
    k: usize,
    output: &DeviceTensor,
) -> TractResult<()> {
    let _retained = stream.retain_for_dispatch(&[route_values, route_weights, output]);
    ensure!(route_values.rank() == 2 && route_values.datum_type() == f32::datum_type());
    ensure!(route_weights.rank() == 1 && route_weights.datum_type() == f32::datum_type());
    ensure!(matches!(output.datum_type(), DatumType::F32 | DatumType::F16));
    let d_model = *output.shape().last().context("output has no feature axis")?;
    let token_count = output.shape()[..output.rank() - 1].iter().product::<usize>();
    ensure!(route_values.shape() == [token_count * k, d_model]);
    ensure!(route_weights.shape() == [token_count * k]);
    let kernel = if output.datum_type() == f32::datum_type() {
        "routed_combine_f32"
    } else {
        "routed_combine_f16"
    };
    let pipeline = stream.load_pipeline(LibraryName::MoeOps, kernel)?;
    let total = output.len() as u64;
    let command_buffer = stream.command_buffer();
    command_buffer.encode(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_metal_tensor(0, route_values, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(1, route_weights, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(2, output, metal::MTLResourceUsage::Write);
        encoder.set_slice(3, &[token_count as u32]);
        encoder.set_slice(4, &[d_model as u32]);
        encoder.set_slice(5, &[k as u32]);
        let grid_size = MTLSize { width: total.div_ceil(256), height: 1, depth: 1 };
        let group_size = MTLSize { width: 256, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(grid_size, group_size);
    });
    Ok(())
}

/// RoutedLinear postamble over f32 `[R, N]` projections: `gate`, then `up`, `bias` and
/// `bias_up` (`[E, N]`, read at each route's expert) when present, and the activation (see
/// [`tract_gpu::ops::routing::routed_activation_code`]) to f32 `out [R, N]`.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_routed_postamble_f32(
    stream: &MetalStream,
    gate: &DeviceTensor,
    up: Option<&DeviceTensor>,
    bias: Option<&DeviceTensor>,
    bias_up: Option<&DeviceTensor>,
    expert_ids: &DeviceTensor,
    (activation, alpha, limit): (u32, f32, f32),
    out: &DeviceTensor,
) -> TractResult<()> {
    let _retained = stream.retain_for_dispatch(&[
        gate,
        up.unwrap_or(gate),
        bias.unwrap_or(gate),
        bias_up.unwrap_or(gate),
        expert_ids,
        out,
    ]);
    ensure!(gate.rank() == 2 && gate.datum_type() == f32::datum_type());
    ensure!(out.shape() == gate.shape() && out.datum_type() == f32::datum_type());
    ensure!(up.is_none_or(|up| up.shape() == gate.shape()));
    let (routes, n) = (gate.shape()[0], gate.shape()[1]);
    ensure!(expert_ids.datum_type() == i64::datum_type() && expert_ids.shape() == [routes]);
    for b in bias.iter().chain(bias_up.iter()) {
        ensure!(b.datum_type() == f32::datum_type() && b.rank() == 2 && b.shape()[1] == n);
    }
    let flags = u32::from(up.is_some())
        | (u32::from(bias.is_some()) << 1)
        | (u32::from(bias_up.is_some()) << 2);
    let pipeline = stream.load_pipeline(LibraryName::MoeOps, "routed_postamble_f32")?;
    let total = (routes * n) as u64;
    let command_buffer = stream.command_buffer();
    command_buffer.encode(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_metal_tensor(0, gate, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(1, up.unwrap_or(gate), metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(2, bias.unwrap_or(gate), metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(3, bias_up.unwrap_or(gate), metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(4, expert_ids, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(5, out, metal::MTLResourceUsage::Write);
        encoder.set_slice(6, &[routes as u32]);
        encoder.set_slice(7, &[n as u32]);
        encoder.set_slice(8, &[activation]);
        encoder.set_slice(9, &[alpha]);
        encoder.set_slice(10, &[limit]);
        encoder.set_slice(11, &[flags]);
        let grid_size = MTLSize { width: total.div_ceil(256), height: 1, depth: 1 };
        let group_size = MTLSize { width: 256, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(grid_size, group_size);
    });
    Ok(())
}
