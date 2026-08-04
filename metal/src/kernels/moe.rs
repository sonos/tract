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
    // One simdgroup (32 lanes) per expert, capped by the device threadgroup
    // limit; the kernel strides experts when there are more experts than
    // simdgroups.
    let group_width = (num_experts * 32).min(max_group_width).max(32);

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

pub fn dispatch_gpt_oss_sinks_softmax_f16(
    stream: &MetalStream,
    scores: &DeviceTensor,
    mask: &DeviceTensor,
    sinks: &DeviceTensor,
    probs: &DeviceTensor,
    s_len: usize,
    scale: f32,
) -> TractResult<()> {
    stream.retain_tensor(scores);
    stream.retain_tensor(mask);
    stream.retain_tensor(sinks);
    stream.retain_tensor(probs);

    ensure!(scores.datum_type() == f16::datum_type());
    ensure!(probs.datum_type() == f16::datum_type());
    ensure!(mask.datum_type() == f32::datum_type());
    ensure!(sinks.datum_type() == f32::datum_type());
    let t_len = *scores.shape().last().context("scores rank 0")?;
    let rows = scores.len() / t_len;
    ensure!(probs.len() == scores.len());
    ensure!(rows % s_len == 0, "rows {rows} not a multiple of s_len {s_len}");
    ensure!(mask.len() >= s_len * t_len, "mask too small");

    let pipeline = stream.load_pipeline(LibraryName::MoeOps, "gpt_oss_sinks_softmax_f16")?;
    let group_width = (pipeline.max_total_threads_per_threadgroup() as u64).min(256);
    let command_buffer = stream.command_buffer();
    command_buffer.encode(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_metal_tensor(0, scores, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(1, mask, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(2, sinks, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(3, probs, metal::MTLResourceUsage::Write);
        encoder.set_slice(4, &[rows as u32]);
        encoder.set_slice(5, &[t_len as u32]);
        encoder.set_slice(6, &[s_len as u32]);
        encoder.set_slice(7, &[scale]);
        let grid = MTLSize { width: rows as NSUInteger, height: 1, depth: 1 };
        let group = MTLSize { width: group_width, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(grid, group);
    });
    Ok(())
}

#[cfg(test)]
mod sinks_softmax_tests {
    use super::*;
    use tract_gpu::tensor::IntoDevice;

    #[test]
    fn sinks_softmax_matches_cpu() -> TractResult<()> {
        crate::utils::with_borrowed_metal_stream(|stream| {
            let (hq, s_len, t_len) = (4usize, 3usize, 37usize);
            let rows = hq * s_len;
            let mut seed = 11u64;
            let mut next = || {
                seed = seed.wrapping_mul(6364136223846793005).wrapping_add(1);
                ((seed >> 33) as f32 / (1u64 << 31) as f32) - 1.0
            };
            let scores: Vec<f32> = (0..rows * t_len).map(|_| next() * 3.0).collect();
            let mask: Vec<f32> =
                (0..s_len * t_len).map(|i| if i % 7 == 0 { -1e30 } else { 0.0 }).collect();
            let sinks: Vec<f32> = (0..hq).map(|_| next()).collect();
            let scale = 0.37f32;

            let scores_f16 =
                Tensor::from_shape(&[rows, t_len], &scores)?.cast_to::<f16>()?.into_owned();
            let scores_dev = scores_f16.clone().into_device()?;
            let mask_dev = Tensor::from_shape(&[s_len, t_len], &mask)?.into_device()?;
            let sinks_dev = Tensor::from_shape(&[hq], &sinks)?.into_device()?;
            let probs_dev = DeviceTensor::uninitialized_dt(f16::datum_type(), &[rows, t_len])?;

            dispatch_gpt_oss_sinks_softmax_f16(
                stream, &scores_dev, &mask_dev, &sinks_dev, &probs_dev, s_len, scale,
            )?;
            stream.wait_until_completed()?;
            let got = probs_dev.to_host()?.into_tensor().cast_to::<f32>()?.into_owned();
            let got = got.try_as_plain()?.as_slice::<f32>()?;

            // CPU reference from the f16-rounded scores.
            let s16 = scores_f16.cast_to::<f32>()?.into_owned();
            let s16 = s16.try_as_plain()?.as_slice::<f32>()?;
            for r in 0..rows {
                let head = r / s_len;
                let mrow = r % s_len;
                let logits: Vec<f32> = (0..t_len)
                    .map(|j| s16[r * t_len + j] * scale + mask[mrow * t_len + j])
                    .collect();
                let m = logits.iter().copied().fold(sinks[head], f32::max);
                let den: f32 =
                    logits.iter().map(|l| (l - m).exp()).sum::<f32>() + (sinks[head] - m).exp();
                for j in 0..t_len {
                    let want = (logits[j] - m).exp() / den;
                    let g = got[r * t_len + j];
                    ensure!(
                        (want - g).abs() <= 1e-3 + want.abs() * 2e-2,
                        "row {r} col {j}: want {want} got {g}"
                    );
                }
            }
            Ok(())
        })
    }
}
