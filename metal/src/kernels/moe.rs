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

/// Quantize rows of an f16 device buffer into q8_0 blocks (KV-cache shadow
/// maintenance). Strides in elements / blocks; `valid` elements from row
/// start, blocks beyond it quantize to zero.
#[allow(clippy::too_many_arguments)]
pub fn dispatch_gpt_oss_kv_quantize_q8_0(
    stream: &MetalStream,
    src: &DeviceTensor,
    dst: &DeviceTensor,
    heads: usize,
    rows: usize,
    src_head_stride: usize,
    src_row_stride: usize,
    dst_head_stride_blocks: usize,
    dst_row_stride_blocks: usize,
    src_row_offset: usize,
    dst_row_offset: usize,
    b0: usize,
    n_blocks: usize,
    valid: usize,
) -> TractResult<()> {
    stream.retain_tensor(src);
    stream.retain_tensor(dst);
    ensure!(src.datum_type() == f16::datum_type());
    let pipeline = stream.load_pipeline(LibraryName::MoeOps, "gpt_oss_kv_quantize_q8_0")?;
    let command_buffer = stream.command_buffer();
    command_buffer.encode(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_metal_tensor(0, src, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(1, dst, metal::MTLResourceUsage::Write);
        encoder.set_slice(2, &[src_head_stride as u32]);
        encoder.set_slice(3, &[src_row_stride as u32]);
        encoder.set_slice(4, &[dst_head_stride_blocks as u32]);
        encoder.set_slice(5, &[dst_row_stride_blocks as u32]);
        encoder.set_slice(6, &[src_row_offset as u32]);
        encoder.set_slice(7, &[dst_row_offset as u32]);
        encoder.set_slice(8, &[b0 as u32]);
        encoder.set_slice(9, &[valid as u32]);
        let grid = MTLSize {
            width: heads as NSUInteger,
            height: rows as NSUInteger,
            depth: n_blocks as NSUInteger,
        };
        let group = MTLSize { width: 32, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(grid, group);
    });
    Ok(())
}

/// Fused flash-attention decode step: out = softmax(q.K^T*scale + mask).V
/// with the per-head sink logit in the denominator. Two dispatches: a
/// partial pass with one threadgroup per (kv head, key chunk, query pos)
/// sharing each K/V read across the whole GQA group, then a small merge.
/// `k`/`v` are seq-major capacity-buffer views (head strides passed
/// explicitly); `q`/`out` dense [1, Hq, S, D].
#[allow(clippy::too_many_arguments)]
/// Chunking rule shared with callers sizing the scratch buffer: enough
/// chunks to occupy the GPU (hkv * n_chunks * s_len threadgroups) without
/// shrinking chunks into pure per-dispatch overhead.
pub fn flash_attn_chunking(t_len: usize) -> (usize, usize) {
    let n_chunks = t_len.div_ceil(512).clamp(1, 16);
    // Chunk boundaries stay 32-aligned: the kernel's half4 loads require it,
    // and each simdgroup block covers 32 keys.
    let chunk = t_len.div_ceil(n_chunks).next_multiple_of(32);
    (t_len.div_ceil(chunk), chunk)
}

/// f32 elements of scratch the flash kernels need for a given geometry.
pub fn flash_attn_scratch_len(hq: usize, s_len: usize, t_len: usize, d: usize) -> usize {
    let (n_chunks, _) = flash_attn_chunking(t_len);
    hq * s_len * n_chunks * FLASH_SG * (2 + d)
}

const FLASH_SG: usize = 8;

pub fn dispatch_gpt_oss_flash_attn_f16(
    stream: &MetalStream,
    q: &DeviceTensor,
    k: &DeviceTensor,
    v: &DeviceTensor,
    mask: &DeviceTensor,
    sinks: &DeviceTensor,
    out: &DeviceTensor,
    scratch: &DeviceTensor,
    dims: GptOssFlashAttnDims,
    scale: f32,
) -> TractResult<()> {
    let GptOssFlashAttnDims {
        hq,
        s_len,
        t_len,
        d,
        group,
        k_head_stride,
        v_head_stride,
        v_seq_stride,
    } = dims;
    let hkv = hq / group;
    for t in [q, k, v, mask, sinks] {
        stream.retain_tensor(t);
    }
    stream.retain_tensor(out);

    ensure!(q.datum_type() == f16::datum_type());
    ensure!(k.datum_type() == f16::datum_type() && v.datum_type() == f16::datum_type());
    ensure!(mask.datum_type() == f32::datum_type());
    ensure!(sinks.datum_type() == f32::datum_type());
    ensure!(d <= 64, "flash attention kernel supports head dim <= 64, got {d}");
    ensure!(group <= 8, "flash attention kernel supports GQA group <= 8, got {group}");
    ensure!(mask.len() >= s_len * t_len, "mask too small");
    ensure!(sinks.len() == hq);

    let (n_chunks, chunk) = flash_attn_chunking(t_len);
    let rows = hq * s_len;
    let n_parts = n_chunks * FLASH_SG;
    ensure!(
        scratch.datum_type() == f32::datum_type() && scratch.len() >= rows * n_parts * (2 + d),
        "flash scratch too small: {} < {}",
        scratch.len(),
        rows * n_parts * (2 + d)
    );
    stream.retain_tensor(scratch);

    // group/dpl are function constants: compile-time loop bounds keep the
    // per-head register arrays out of stack memory.
    let fuse_merge = n_chunks == 1;
    let constants = crate::func_constants::ConstantValues::new(vec![
        (0, crate::func_constants::Value::USize(group)),
        (1, crate::func_constants::Value::USize(d.div_ceil(32))),
        (2, crate::func_constants::Value::Bool(fuse_merge)),
    ]);
    let part = stream.load_pipeline_with_constants(
        LibraryName::MoeOps,
        "gpt_oss_flash_attn_part_f16",
        Some(constants),
    )?;
    let command_buffer = stream.command_buffer();
    command_buffer.encode(|encoder| {
        encoder.set_compute_pipeline_state(&part);
        encoder.set_metal_tensor(0, q, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(1, k, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(2, v, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(3, mask, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(4, scratch, metal::MTLResourceUsage::Write);
        encoder.set_slice(5, &[s_len as u32]);
        encoder.set_slice(6, &[t_len as u32]);
        encoder.set_slice(7, &[d as u32]);
        encoder.set_slice(8, &[k_head_stride as u32]);
        encoder.set_slice(9, &[v_head_stride as u32]);
        encoder.set_slice(10, &[v_seq_stride as u32]);
        encoder.set_slice(11, &[chunk as u32]);
        encoder.set_slice(12, &[n_chunks as u32]);
        encoder.set_slice(13, &[scale]);
        encoder.set_metal_tensor(14, sinks, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(15, out, metal::MTLResourceUsage::Write);
        let grid = MTLSize {
            width: hkv as NSUInteger,
            height: n_chunks as NSUInteger,
            depth: s_len as NSUInteger,
        };
        let group_size = MTLSize { width: (FLASH_SG * 32) as NSUInteger, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(grid, group_size);
    });
    if fuse_merge {
        return Ok(());
    }
    let merge = stream.load_pipeline(LibraryName::MoeOps, "gpt_oss_flash_attn_merge_f16")?;
    let command_buffer = stream.command_buffer();
    command_buffer.encode(|encoder| {
        encoder.set_compute_pipeline_state(&merge);
        encoder.set_metal_tensor(0, scratch, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(1, sinks, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(2, out, metal::MTLResourceUsage::Write);
        encoder.set_slice(3, &[s_len as u32]);
        encoder.set_slice(4, &[d as u32]);
        encoder.set_slice(5, &[n_parts as u32]);
        encoder.set_slice(6, &[scale]);
        let grid = MTLSize { width: rows as NSUInteger, height: 1, depth: 1 };
        let group_size = MTLSize { width: 32, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(grid, group_size);
    });
    Ok(())
}

#[derive(Debug, Clone, Copy)]
pub struct GptOssFlashAttnDims {
    pub hq: usize,
    pub s_len: usize,
    pub t_len: usize,
    pub d: usize,
    pub group: usize,
    /// K is seq-major: elements between consecutive heads.
    pub k_head_stride: usize,
    /// V is transposed ([Hkv, D, cap]): elements between consecutive heads.
    pub v_head_stride: usize,
    /// V transposed: elements between consecutive dims of one head (= cap).
    pub v_seq_stride: usize,
}

pub fn dispatch_gpt_oss_sinks_softmax_f16(
    stream: &MetalStream,
    scores: &DeviceTensor,
    mask: &DeviceTensor,
    sinks: &DeviceTensor,
    probs: &DeviceTensor,
    s_len: usize,
    scale: f32,
    t_len: usize,
) -> TractResult<()> {
    stream.retain_tensor(scores);
    stream.retain_tensor(mask);
    stream.retain_tensor(sinks);
    stream.retain_tensor(probs);

    ensure!(scores.datum_type() == f16::datum_type());
    ensure!(probs.datum_type() == f16::datum_type());
    ensure!(mask.datum_type() == f32::datum_type());
    ensure!(sinks.datum_type() == f32::datum_type());
    // The physical row stride may exceed t_len (q8 block padding); the pad
    // columns of probs are zeroed by the kernel.
    let row_stride = *scores.shape().last().context("scores rank 0")?;
    ensure!(row_stride >= t_len);
    let rows = scores.len() / row_stride;
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
        encoder.set_slice(8, &[row_stride as u32]);
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
                stream, &scores_dev, &mask_dev, &sinks_dev, &probs_dev, s_len, scale, t_len,
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

#[cfg(test)]
mod flash_attn_bench {
    use super::*;
    use tract_gpu::tensor::IntoDevice;

    #[test]
    #[ignore]
    fn bench_flash_attn_gpt_oss_geometry() -> TractResult<()> {
        crate::utils::with_borrowed_metal_stream(|stream| {
            for t in [256usize, 1024, 2800, 8192] {
            let (hq, hkv, d) = (64usize, 8usize, 64usize);
            let group = hq / hkv;
            let cap = 8192;
            let q = Tensor::zero::<f16>(&[1, hq, 1, d])?.into_device()?;
            let k = Tensor::zero::<f16>(&[1, hkv, cap, d])?.into_device()?;
            let v = Tensor::zero::<f16>(&[1, hkv, d, cap])?.into_device()?;
            let mask = Tensor::zero::<f32>(&[1, t])?.into_device()?;
            let sinks = Tensor::zero::<f32>(&[hq])?.into_device()?;
            let out = Tensor::zero::<f16>(&[1, hq, 1, d])?.into_device()?;
            let scratch = unsafe {
                DeviceTensor::uninitialized_dt(
                    f32::datum_type(),
                    &[flash_attn_scratch_len(hq, 1, t, d)],
                )?
            };
            let dims = GptOssFlashAttnDims {
                hq,
                s_len: 1,
                t_len: t,
                d,
                group,
                k_head_stride: cap * d,
                v_head_stride: cap * d,
                v_seq_stride: cap,
            };
            // warmup
            for _ in 0..10 {
                dispatch_gpt_oss_flash_attn_f16(
                    stream, &q, &k, &v, &mask, &sinks, &out, &scratch, dims, 0.125,
                )?;
            }
            stream.wait_until_completed()?;
            let start = std::time::Instant::now();
            const N: usize = 200;
            for _ in 0..N {
                dispatch_gpt_oss_flash_attn_f16(
                    stream, &q, &k, &v, &mask, &sinks, &out, &scratch, dims, 0.125,
                )?;
            }
            stream.wait_until_completed()?;
            let per = start.elapsed().as_secs_f64() / N as f64;
            eprintln!("flash attn t={t}: {:.1} us/layer, {:.2} ms/token(24 layers)", per * 1e6, per * 24.0 * 1e3);
            }
            Ok(())
        })
    }
}
