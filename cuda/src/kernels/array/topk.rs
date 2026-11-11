use crate::context::TractCudaStream;
use crate::context::cuda_context;
use crate::kernels::LibraryName;
use crate::kernels::get_cuda_view;
use crate::kernels::launch_args::TractLaunchArgs;
use cudarc::driver::LaunchConfig;
use tract_core::internal::*;
use tract_core::ops::array::Topk;
use tract_gpu::ops::topk::GpuTopK;
use tract_gpu::ops::topk::TOPK_CHUNK;
use tract_gpu::ops::topk::TopKKernels;
use tract_gpu::tensor::DeviceTensor;

const THREADS: usize = TOPK_CHUNK / 2;

pub fn kernel_names() -> Vec<String> {
    let mut names = vec!["topk_chunk_keys".to_string()];
    for dt in ["f32", "f16"] {
        names.push(format!("topk_chunk_values_{dt}"));
        names.push(format!("topk_finish_{dt}"));
    }
    names
}

fn chunk_launch(rows: usize, len: usize) -> TractResult<(usize, LaunchConfig)> {
    ensure!(len <= i32::MAX as usize);
    let chunks = len.div_ceil(TOPK_CHUNK);
    let blocks = rows * chunks;
    ensure!(blocks <= i32::MAX as usize);
    let cfg = LaunchConfig {
        grid_dim: (blocks as _, 1, 1),
        block_dim: (THREADS as _, 1, 1),
        shared_mem_bytes: 0,
    };
    Ok((chunks, cfg))
}

fn chunk_values(
    stream: &TractCudaStream,
    largest: bool,
    values: &DeviceTensor,
    rows: usize,
    n: usize,
    k: usize,
    keys: &DeviceTensor,
) -> TractResult<()> {
    let (chunks, cfg) = chunk_launch(rows, n)?;
    let name = format!("topk_chunk_values_{}", DeviceTensor::tname(values.datum_type())?);
    let func = cuda_context().load_pipeline(LibraryName::Array, name)?;
    let v_view = get_cuda_view(values);
    let k_view = get_cuda_view(keys);
    let mut args = TractLaunchArgs::new(stream, &func);
    args.push_view(&v_view);
    args.push_view(&k_view);
    args.push_i32(n);
    args.push_i32(k);
    args.push_i32(largest as usize);
    args.push_i32(chunks);
    args.push_i32(chunks * k);
    args.launch(cfg)
}

fn chunk_keys(
    stream: &TractCudaStream,
    keys: &DeviceTensor,
    rows: usize,
    len: usize,
    k: usize,
    out: &DeviceTensor,
) -> TractResult<()> {
    let (chunks, cfg) = chunk_launch(rows, len)?;
    let func = cuda_context().load_pipeline(LibraryName::Array, "topk_chunk_keys".to_string())?;
    let i_view = get_cuda_view(keys);
    let o_view = get_cuda_view(out);
    let mut args = TractLaunchArgs::new(stream, &func);
    args.push_view(&i_view);
    args.push_view(&o_view);
    args.push_i32(len);
    args.push_i32(k);
    args.push_i32(chunks);
    args.push_i32(chunks * k);
    args.launch(cfg)
}

#[allow(clippy::too_many_arguments)]
fn finish(
    stream: &TractCudaStream,
    keys: &DeviceTensor,
    values: &DeviceTensor,
    rows: usize,
    n: usize,
    k: usize,
    out_values: &DeviceTensor,
    out_indices: &DeviceTensor,
) -> TractResult<()> {
    ensure!(rows <= 65535, "CUDA TopK supports up to 65535 rows");
    let name = format!("topk_finish_{}", DeviceTensor::tname(values.datum_type())?);
    let func = cuda_context().load_pipeline(LibraryName::Array, name)?;
    let k_view = get_cuda_view(keys);
    let v_view = get_cuda_view(values);
    let ov_view = get_cuda_view(out_values);
    let oi_view = get_cuda_view(out_indices);
    let mut args = TractLaunchArgs::new(stream, &func);
    args.push_view(&k_view);
    args.push_view(&v_view);
    args.push_view(&ov_view);
    args.push_view(&oi_view);
    args.push_i32(n);
    args.push_i32(k);
    let block = k.min(256);
    let cfg = LaunchConfig {
        grid_dim: (k.div_ceil(block) as _, rows as _, 1),
        block_dim: (block as _, 1, 1),
        shared_mem_bytes: 0,
    };
    args.launch(cfg)
}

pub const CUDA_TOPK_KERNELS: TopKKernels = TopKKernels {
    chunk_values: |largest, values, rows, n, k, keys| {
        crate::with_cuda_stream(|s| chunk_values(s, largest, values, rows, n, k, keys))
    },
    chunk_keys: |keys, rows, len, k, out| {
        crate::with_cuda_stream(|s| chunk_keys(s, keys, rows, len, k, out))
    },
    finish: |keys, values, rows, n, k, out_values, out_indices| {
        crate::with_cuda_stream(|s| finish(s, keys, values, rows, n, k, out_values, out_indices))
    },
};

crate::register_cuda_op!(Topk, |source, node, op| {
    Ok(GpuTopK::from_tract_core(source, node, op, "Cuda", CUDA_TOPK_KERNELS)?
        .map(|op| Box::new(op) as _))
});
