use crate::LibraryName;
use crate::encoder::EncoderExt;
use metal::MTLSize;
use tract_core::internal::*;
use tract_gpu::ops::group_norm::GroupNormView;
use tract_gpu::tensor::DeviceTensor;

const THREADS: u64 = 256;

pub fn is_supported_dt(dt: DatumType) -> bool {
    matches!(dt, DatumType::F32 | DatumType::F16)
}

pub fn kernel_names() -> Vec<String> {
    ["f32", "f16"]
        .into_iter()
        .flat_map(|t| {
            [format!("nn_ops::group_norm_stats_{t}"), format!("nn_ops::group_norm_apply_{t}")]
        })
        .collect()
}

fn dims(view: &GroupNormView) -> [i32; 5] {
    [view.pre, view.channels, view.post, view.groups, view.splits].map(|d| d as i32)
}

fn grid(view: &GroupNormView) -> MTLSize {
    MTLSize { width: (view.outer * view.groups) as _, height: view.splits as _, depth: 1 }
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
        let pipeline = stream
            .load_pipeline(LibraryName::NNOps, &format!("nn_ops::group_norm_stats_{tname}"))?;
        let command_buffer = stream.command_buffer();
        command_buffer.encode(|encoder| {
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_metal_tensor(0, input, metal::MTLResourceUsage::Read);
            encoder.set_metal_tensor(1, partials, metal::MTLResourceUsage::Write);
            encoder.set_slice(2, &dims(view));
            encoder.set_threadgroup_memory_length(0, (THREADS / 32) * 3 * 4);
            let group_size = MTLSize { width: THREADS, height: 1, depth: 1 };
            encoder.dispatch_thread_groups(grid(view), group_size);
        });
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
        let tname = DeviceTensor::tname(input.datum_type())?;
        let pipeline = stream
            .load_pipeline(LibraryName::NNOps, &format!("nn_ops::group_norm_apply_{tname}"))?;
        let command_buffer = stream.command_buffer();
        command_buffer.encode(|encoder| {
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_metal_tensor(0, input, metal::MTLResourceUsage::Read);
            encoder.set_metal_tensor(1, partials, metal::MTLResourceUsage::Read);
            encoder.set_metal_tensor(2, output, metal::MTLResourceUsage::Write);
            encoder.set_slice(3, &dims(view));
            encoder.set_slice(4, &[eps]);
            let group_size = MTLSize { width: THREADS, height: 1, depth: 1 };
            encoder.dispatch_thread_groups(grid(view), group_size);
        });
        Ok(())
    })
}
