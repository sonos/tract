use cudarc::driver::LaunchConfig;
use tract_core::internal::*;
use tract_gpu::ops::routing::GpuRouteSelect;
use tract_gpu::ops::routing::GpuRoutedCombine;
use tract_gpu::ops::routing::RoutingKernels;
use tract_gpu::tensor::DeviceTensor;
use tract_transformers::ops::moe_ffn::RouteSelect;
use tract_transformers::ops::moe_ffn::RoutedCombine;

use crate::context::{TractCudaStream, cuda_context};
use crate::kernels::launch_args::TractLaunchArgs;
use crate::kernels::{LibraryName, get_cuda_view};

#[allow(clippy::too_many_arguments)]
fn select(
    stream: &TractCudaStream,
    scores: &DeviceTensor,
    bias: Option<&DeviceTensor>,
    k: usize,
    gate: u32,
    token_ids: &DeviceTensor,
    expert_ids: &DeviceTensor,
    weights: &DeviceTensor,
) -> TractResult<()> {
    let (tokens, experts) = (scores.shape()[0], scores.shape()[1]);
    let func = cuda_context().load_pipeline(LibraryName::Routing, "route_select_f32".into())?;
    let scores = get_cuda_view(scores);
    let bias = bias.map(get_cuda_view);
    let views = [token_ids, expert_ids, weights].map(get_cuda_view);
    let mut args = TractLaunchArgs::new(stream, &func);
    args.push_view(&scores);
    match &bias {
        Some(bias) => args.push_view(bias),
        None => args.push_null_ptr(),
    }
    for view in &views {
        args.push_view(view);
    }
    args.push_i32(tokens);
    args.push_i32(experts);
    args.push_i32(k);
    args.push_i32(gate);
    args.push_i32(bias.is_some() as usize);
    args.launch(LaunchConfig {
        grid_dim: (tokens as u32, 1, 1),
        block_dim: (32, 1, 1),
        shared_mem_bytes: 0,
    })
}

fn combine(
    stream: &TractCudaStream,
    values: &DeviceTensor,
    weights: &DeviceTensor,
    k: usize,
    out: &DeviceTensor,
) -> TractResult<()> {
    let d = *out.shape().last().context("combine output has no feature axis")?;
    let tokens = out.len() / d;
    let name = format!("routed_combine_{}", DeviceTensor::tname(out.datum_type())?);
    let func = cuda_context().load_pipeline(LibraryName::Routing, name)?;
    let views = [values, weights, out].map(get_cuda_view);
    let mut args = TractLaunchArgs::new(stream, &func);
    for view in &views {
        args.push_view(view);
    }
    args.push_i32(tokens);
    args.push_i32(d);
    args.push_i32(k);
    args.launch(LaunchConfig {
        grid_dim: (out.len().div_ceil(256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    })
}

const CUDA_ROUTING_KERNELS: RoutingKernels = RoutingKernels {
    select: |scores, bias, k, gate, token_ids, expert_ids, weights| {
        crate::with_cuda_stream(|s| {
            select(s, scores, bias, k, gate, token_ids, expert_ids, weights)
        })
    },
    combine: |values, weights, k, out| {
        crate::with_cuda_stream(|s| combine(s, values, weights, k, out))
    },
};

crate::register_cuda_op!(RouteSelect, |source, node, op| {
    Ok(GpuRouteSelect::from_tract_core(source, node, op, "Cuda", CUDA_ROUTING_KERNELS)?
        .map(|op| Box::new(op) as _))
});

crate::register_cuda_op!(RoutedCombine, |source, node, op| {
    Ok(GpuRoutedCombine::from_tract_core(source, node, op, "Cuda", CUDA_ROUTING_KERNELS)?
        .map(|op| Box::new(op) as _))
});
