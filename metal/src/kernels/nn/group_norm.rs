use crate::LibraryName;
use crate::encoder::EncoderExt;
use metal::MTLSize;
use tract_core::internal::*;
use tract_gpu::ops::group_norm::GroupNormView;
use tract_gpu::tensor::DeviceTensor;

const THREADS: usize = 256;
const MOMENTS_BYTES: u64 = 3 * 4;

pub fn is_supported_dt(dt: DatumType) -> bool {
    matches!(dt, DatumType::F32 | DatumType::F16)
}

pub fn kernel_names() -> Vec<String> {
    ["f32", "f16"]
        .into_iter()
        .flat_map(|t| {
            [
                format!("nn_ops::group_norm_stats_{t}"),
                format!("nn_ops::group_norm_stats_channel_last_{t}"),
                format!("nn_ops::group_norm_apply_{t}"),
            ]
        })
        .chain(["group_norm_finalize".to_string()])
        .collect()
}

fn size(width: usize, height: usize) -> MTLSize {
    MTLSize { width: width as _, height: height as _, depth: 1 }
}

pub fn metal_group_norm_stats_dispatch(
    input: &DeviceTensor,
    view: &GroupNormView,
    partials: &DeviceTensor,
) -> TractResult<()> {
    ensure!(is_supported_dt(input.datum_type()));
    crate::with_metal_stream(|stream| {
        stream.retain_tensor(input);
        stream.retain_tensor(partials);
        let tname = DeviceTensor::tname(input.datum_type())?;
        let command_buffer = stream.command_buffer();
        if view.channel_last {
            let pipeline = stream.load_pipeline(
                LibraryName::NNOps,
                &format!("nn_ops::group_norm_stats_channel_last_{tname}"),
            )?;
            let dims = [view.pre, view.channels, view.splits].map(|d| d as i32);
            let grid = size(view.outer * view.channels.div_ceil(THREADS), view.splits);
            command_buffer.encode(|encoder| {
                encoder.set_compute_pipeline_state(&pipeline);
                encoder.set_metal_tensor(0, input, metal::MTLResourceUsage::Read);
                encoder.set_metal_tensor(1, partials, metal::MTLResourceUsage::Write);
                encoder.set_slice(2, &dims);
                encoder.dispatch_thread_groups(grid, size(THREADS, 1));
            });
        } else {
            let pipeline = stream
                .load_pipeline(LibraryName::NNOps, &format!("nn_ops::group_norm_stats_{tname}"))?;
            let dims =
                [view.pre, view.channels, view.post, view.groups, view.splits].map(|d| d as i32);
            command_buffer.encode(|encoder| {
                encoder.set_compute_pipeline_state(&pipeline);
                encoder.set_metal_tensor(0, input, metal::MTLResourceUsage::Read);
                encoder.set_metal_tensor(1, partials, metal::MTLResourceUsage::Write);
                encoder.set_slice(2, &dims);
                encoder.set_threadgroup_memory_length(0, (THREADS as u64 / 32) * MOMENTS_BYTES);
                encoder.dispatch_thread_groups(
                    size(view.outer * view.groups, view.splits),
                    size(THREADS, 1),
                );
            });
        }
        Ok(())
    })
}

pub fn metal_group_norm_apply_dispatch(
    input: &DeviceTensor,
    partials: &DeviceTensor,
    view: &GroupNormView,
    eps: f32,
    output: &DeviceTensor,
) -> TractResult<()> {
    ensure!(is_supported_dt(input.datum_type()));
    crate::with_metal_stream(|stream| {
        stream.retain_tensor(input);
        stream.retain_tensor(partials);
        stream.retain_tensor(output);
        let finalize = stream.load_pipeline(LibraryName::NNOps, "group_norm_finalize")?;
        let tname = DeviceTensor::tname(input.datum_type())?;
        let apply = stream
            .load_pipeline(LibraryName::NNOps, &format!("nn_ops::group_norm_apply_{tname}"))?;
        let entries = view.entries as i32;
        let dims = [input.len(), view.pre, view.channels, view.post, view.groups, view.entries]
            .map(|d| d as i32);
        let command_buffer = stream.command_buffer();
        command_buffer.encode(|encoder| {
            encoder.set_compute_pipeline_state(&finalize);
            encoder.set_metal_tensor(0, partials, metal::MTLResourceUsage::Write);
            encoder.set_slice(1, &[entries]);
            encoder.set_slice(2, &[eps]);
            encoder.set_threadgroup_memory_length(0, (THREADS as u64 / 32) * MOMENTS_BYTES);
            encoder.dispatch_thread_groups(size(view.outer * view.groups, 1), size(THREADS, 1));
        });
        command_buffer.encode(|encoder| {
            encoder.set_compute_pipeline_state(&apply);
            encoder.set_metal_tensor(0, input, metal::MTLResourceUsage::Read);
            encoder.set_metal_tensor(1, partials, metal::MTLResourceUsage::Read);
            encoder.set_metal_tensor(2, output, metal::MTLResourceUsage::Write);
            encoder.set_slice(3, &dims);
            encoder
                .dispatch_thread_groups(size(input.len().div_ceil(THREADS), 1), size(THREADS, 1));
        });
        Ok(())
    })
}
