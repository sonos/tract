use crate::encoder::EncoderExt;
use crate::{LibraryName, MetalStream};
use metal::{MTLResourceUsage, MTLSize, NSUInteger};
use tract_core::internal::*;
use tract_core::ops::array::Topk;
use tract_gpu::ops::topk::GpuTopK;
use tract_gpu::ops::topk::TOPK_CHUNK;
use tract_gpu::ops::topk::TopKKernels;
use tract_gpu::tensor::DeviceTensor;

pub fn kernel_names() -> Vec<String> {
    let mut names = vec!["array_ops::topk_chunk_keys".to_string()];
    for dt in ["f32", "f16"] {
        names.push(format!("array_ops::topk_chunk_values_{dt}"));
        names.push(format!("array_ops::topk_finish_{dt}"));
    }
    names
}

fn encode_chunks(
    stream: &MetalStream,
    kernel: &str,
    input: &DeviceTensor,
    output: &DeviceTensor,
    rows: usize,
    chunks: usize,
    params: &[i32],
) -> TractResult<()> {
    stream.retain_tensor(input);
    stream.retain_tensor(output);
    let pipeline = stream.load_pipeline(LibraryName::ArrayOps, kernel)?;
    let width = ((TOPK_CHUNK / 2) as NSUInteger).min(pipeline.max_total_threads_per_threadgroup());
    let command_buffer = stream.command_buffer();
    command_buffer.encode(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_metal_tensor(0, input, MTLResourceUsage::Read);
        encoder.set_metal_tensor(1, output, MTLResourceUsage::Write);
        encoder.set_slice(2, params);
        let grid_size = MTLSize { width: (rows * chunks) as NSUInteger, height: 1, depth: 1 };
        let group_size = MTLSize { width, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(grid_size, group_size);
    });
    Ok(())
}

fn chunk_values(
    stream: &MetalStream,
    largest: bool,
    values: &DeviceTensor,
    rows: usize,
    n: usize,
    k: usize,
    keys: &DeviceTensor,
) -> TractResult<()> {
    ensure!(n <= i32::MAX as usize);
    let chunks = n.div_ceil(TOPK_CHUNK);
    let kernel =
        format!("array_ops::topk_chunk_values_{}", DeviceTensor::tname(values.datum_type())?);
    let params = [n as i32, k as i32, largest as i32, chunks as i32, (chunks * k) as i32];
    encode_chunks(stream, &kernel, values, keys, rows, chunks, &params)
}

fn chunk_keys(
    stream: &MetalStream,
    keys: &DeviceTensor,
    rows: usize,
    len: usize,
    k: usize,
    out: &DeviceTensor,
) -> TractResult<()> {
    ensure!(len <= i32::MAX as usize);
    let chunks = len.div_ceil(TOPK_CHUNK);
    let params = [len as i32, k as i32, chunks as i32, (chunks * k) as i32];
    encode_chunks(stream, "array_ops::topk_chunk_keys", keys, out, rows, chunks, &params)
}

#[allow(clippy::too_many_arguments)]
fn finish(
    stream: &MetalStream,
    keys: &DeviceTensor,
    values: &DeviceTensor,
    rows: usize,
    n: usize,
    k: usize,
    out_values: &DeviceTensor,
    out_indices: &DeviceTensor,
) -> TractResult<()> {
    stream.retain_tensor(keys);
    stream.retain_tensor(values);
    stream.retain_tensor(out_values);
    stream.retain_tensor(out_indices);
    let kernel = format!("array_ops::topk_finish_{}", DeviceTensor::tname(values.datum_type())?);
    let pipeline = stream.load_pipeline(LibraryName::ArrayOps, &kernel)?;
    let width = (k as NSUInteger).min(256);
    let params = [n as i32, k as i32];
    let command_buffer = stream.command_buffer();
    command_buffer.encode(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_metal_tensor(0, keys, MTLResourceUsage::Read);
        encoder.set_metal_tensor(1, values, MTLResourceUsage::Read);
        encoder.set_metal_tensor(2, out_values, MTLResourceUsage::Write);
        encoder.set_metal_tensor(3, out_indices, MTLResourceUsage::Write);
        encoder.set_slice(4, &params);
        let grid_size =
            MTLSize { width: (k as NSUInteger).div_ceil(width), height: rows as _, depth: 1 };
        let group_size = MTLSize { width, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(grid_size, group_size);
    });
    Ok(())
}

pub const METAL_TOPK_KERNELS: TopKKernels = TopKKernels {
    chunk_values: |largest, values, rows, n, k, keys| {
        crate::with_metal_stream(|s| chunk_values(s, largest, values, rows, n, k, keys))
    },
    chunk_keys: |keys, rows, len, k, out| {
        crate::with_metal_stream(|s| chunk_keys(s, keys, rows, len, k, out))
    },
    finish: |keys, values, rows, n, k, out_values, out_indices| {
        crate::with_metal_stream(|s| finish(s, keys, values, rows, n, k, out_values, out_indices))
    },
};

crate::register_metal_op!(Topk, |source, node, op| {
    Ok(GpuTopK::from_tract_core(source, node, op, "Metal", METAL_TOPK_KERNELS)?
        .map(|op| Box::new(op) as _))
});
