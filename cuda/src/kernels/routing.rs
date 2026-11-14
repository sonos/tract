use cudarc::driver::LaunchConfig;
use tract_core::internal::*;
use tract_core::tract_linalg::block_quant::BlockQuant;
use tract_core::tract_linalg::block_quant::BlockQuantFact;
use tract_core::tract_linalg::block_quant::Q2_0_T;
use tract_core::tract_linalg::block_quant::Q4_0;
use tract_gpu::ops::routing::GpuRouteSelect;
use tract_gpu::ops::routing::GpuRoutedCombine;
use tract_gpu::ops::routing::GpuRoutedLinear;
use tract_gpu::ops::routing::RoutedLinearKernels;
use tract_gpu::ops::routing::RoutingKernels;
use tract_gpu::tensor::DeviceTensor;
use tract_transformers::ops::moe_ffn::MoeFfn;
use tract_transformers::ops::moe_ffn::RouteSelect;
use tract_transformers::ops::moe_ffn::RoutedCombine;
use tract_transformers::ops::moe_ffn::RoutedLinear;

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

const ROUTED_WARPS: u32 = 4;
const ROUTED_ROWS: usize = 4;

/// The block-quantized formats the routed kernels read, with the name their kernels carry.
/// A format here needs a `w_<name>` struct in `routing.cu`, instantiated with
/// `INSTANTIATE_ROUTED_BLOCK_QUANT`.
fn block_quant_kernel_name(format: &dyn BlockQuant) -> Option<&'static str> {
    let formats: [(&dyn BlockQuant, &'static str); 2] = [(&Q4_0, "q40"), (&Q2_0_T, "q20t")];
    formats.into_iter().find(|(f, _)| f.dyn_eq(format)).map(|(_, name)| name)
}

/// Plain f32 or f16 expert weights, or block-quantized ones in a format the routed kernels
/// read, with K in whole blocks.
fn supports(weights: &TypedFact) -> bool {
    if let Some(bqf) = weights.exotic_fact().and_then(|f| f.downcast_ref::<BlockQuantFact>()) {
        block_quant_kernel_name(&*bqf.format).is_some() && bqf.k() % bqf.format.block_len() == 0
    } else {
        weights.is_plain() && matches!(weights.datum_type, DatumType::F32 | DatumType::F16)
    }
}

/// The routed kernels' name for these weights, and whether they are block-quantized.
fn weights_kernel_name(weights: &DeviceTensor) -> TractResult<(&'static str, bool)> {
    let exotic = match weights {
        DeviceTensor::Owned(t) => t.exotic_fact(),
        DeviceTensor::View(v) => v.exotic_fact(),
    };
    if let Some(bqf) = exotic.and_then(|f| f.downcast_ref::<BlockQuantFact>()) {
        let name = block_quant_kernel_name(&*bqf.format)
            .with_context(|| format!("No CUDA routed kernel for {:?}", bqf.format))?;
        Ok((name, true))
    } else {
        Ok((DeviceTensor::tname(weights.datum_type())?, false))
    }
}

fn project(
    stream: &TractCudaStream,
    input: &DeviceTensor,
    weights: &DeviceTensor,
    rows: Option<&DeviceTensor>,
    expert_ids: &DeviceTensor,
    out: &DeviceTensor,
) -> TractResult<()> {
    let (routes, n, k) = (expert_ids.shape()[0], weights.shape()[1], input.shape()[1]);
    let experts = weights.shape()[0];
    let (name, block_quant) = weights_kernel_name(weights)?;
    let per_expert = routes / experts.max(1);
    if experts <= RMM_MAX_EXPERTS && k.is_multiple_of(RMM_K) {
        if per_expert >= if block_quant { 14 } else { 20 } {
            return project_sorted(stream, name, RMM_M, input, weights, rows, expert_ids, out);
        }
        if !block_quant && per_expert >= 3 {
            return project_sorted(stream, name, RMV_ROUTES, input, weights, rows, expert_ids, out);
        }
    }
    ensure!(routes <= 65535, "CUDA RoutedLinear takes up to 65535 routes");
    let rows_per_warp = if block_quant { ROUTED_ROWS } else { 1 };
    ensure!(
        rows_per_warp == 1 || input.buffer_offset::<usize>().is_multiple_of(16),
        "the block-quant routed GEMV reads its input as float4"
    );
    let func =
        cuda_context().load_pipeline(LibraryName::Routing, format!("routed_mul_vec_{name}"))?;
    let w = get_cuda_view(weights);
    let x = get_cuda_view(input);
    let o = get_cuda_view(out);
    let rows = rows.map(get_cuda_view);
    let ids = get_cuda_view(expert_ids);
    let mut args = TractLaunchArgs::new(stream, &func);
    args.push_view(&w);
    args.push_view(&x);
    args.push_view(&o);
    match &rows {
        Some(rows) => args.push_view(rows),
        None => args.push_null_ptr(),
    }
    args.push_view(&ids);
    args.push_i32(k);
    args.push_i32(n);
    args.launch(LaunchConfig {
        grid_dim: (n.div_ceil(ROUTED_WARPS as usize * rows_per_warp) as u32, routes as u32, 1),
        block_dim: (32, ROUTED_WARPS, 1),
        shared_mem_bytes: 0,
    })
}

const RMM_M: usize = 64;
const RMM_N: usize = 64;
const RMV_ROUTES: usize = 8;
const RMM_K: usize = 32;
const RMM_MAX_EXPERTS: usize = 256;

/// The routes sorted by expert in chunks of `chunk`, then the grouped GEMV (chunks of
/// [`RMV_ROUTES`]) or the tiled matmul over every chunk.
#[allow(clippy::too_many_arguments)]
fn project_sorted(
    stream: &TractCudaStream,
    name: &str,
    chunk: usize,
    input: &DeviceTensor,
    weights: &DeviceTensor,
    rows: Option<&DeviceTensor>,
    expert_ids: &DeviceTensor,
    out: &DeviceTensor,
) -> TractResult<()> {
    let (routes, experts) = (expert_ids.shape()[0], weights.shape()[0]);
    let (n, k) = (weights.shape()[1], input.shape()[1]);
    ensure!(
        chunk != RMV_ROUTES || input.buffer_offset::<usize>().is_multiple_of(16),
        "the grouped routed GEMV reads its input as float4"
    );
    let max_chunks = routes.div_ceil(chunk) + experts;
    ensure!(max_chunks <= 65535, "CUDA RoutedLinear takes up to {} routes", 65535 * chunk);
    let sorted = DeviceTensor::uninitialized_dt(i32::datum_type(), &[routes])?;
    let chunks = DeviceTensor::uninitialized_dt(i32::datum_type(), &[max_chunks * 3])?;
    let ids = get_cuda_view(expert_ids);
    let (sorted_view, chunks_view) = (get_cuda_view(&sorted), get_cuda_view(&chunks));
    let plan = cuda_context().load_pipeline(LibraryName::Routing, "route_plan".into())?;
    let mut args = TractLaunchArgs::new(stream, &plan);
    args.push_view(&ids);
    args.push_i32(routes);
    args.push_i32(experts);
    args.push_view(&sorted_view);
    args.push_view(&chunks_view);
    args.push_i32(max_chunks);
    args.push_i32(chunk);
    args.launch(LaunchConfig { grid_dim: (1, 1, 1), block_dim: (256, 1, 1), shared_mem_bytes: 0 })?;

    let grouped = chunk == RMV_ROUTES;
    let kernel =
        if grouped { format!("routed_mv_grouped_{name}") } else { format!("routed_mm_{name}") };
    let func = cuda_context().load_pipeline(LibraryName::Routing, kernel)?;
    let (w, x, o) = (get_cuda_view(weights), get_cuda_view(input), get_cuda_view(out));
    let rows = rows.map(get_cuda_view);
    let mut args = TractLaunchArgs::new(stream, &func);
    args.push_view(&w);
    args.push_view(&x);
    args.push_view(&o);
    match &rows {
        Some(rows) => args.push_view(rows),
        None => args.push_null_ptr(),
    }
    args.push_view(&sorted_view);
    args.push_view(&chunks_view);
    args.push_i32(k);
    args.push_i32(n);
    let cfg = if grouped {
        LaunchConfig {
            grid_dim: (
                n.div_ceil(ROUTED_WARPS as usize * ROUTED_ROWS) as u32,
                max_chunks as u32,
                1,
            ),
            block_dim: (32, ROUTED_WARPS, 1),
            shared_mem_bytes: 0,
        }
    } else {
        LaunchConfig {
            grid_dim: (n.div_ceil(RMM_N) as u32, max_chunks as u32, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        }
    };
    args.launch(cfg)
}

#[allow(clippy::too_many_arguments)]
fn postamble(
    stream: &TractCudaStream,
    gate: &DeviceTensor,
    up: Option<&DeviceTensor>,
    bias: Option<&DeviceTensor>,
    bias_up: Option<&DeviceTensor>,
    expert_ids: &DeviceTensor,
    (activation, alpha, limit): (u32, f32, f32),
    out: &DeviceTensor,
) -> TractResult<()> {
    let (routes, n) = (gate.shape()[0], gate.shape()[1]);
    let func = cuda_context().load_pipeline(LibraryName::Routing, "routed_postamble".into())?;
    let optional = [up, bias, bias_up].map(|t| t.map(get_cuda_view));
    let (g, ids, o) = (get_cuda_view(gate), get_cuda_view(expert_ids), get_cuda_view(out));
    let mut args = TractLaunchArgs::new(stream, &func);
    args.push_view(&g);
    for view in &optional {
        match view {
            Some(view) => args.push_view(view),
            None => args.push_null_ptr(),
        }
    }
    args.push_view(&ids);
    args.push_view(&o);
    args.push_i32(routes);
    args.push_i32(n);
    args.push_i32(activation);
    args.push::<f32>(alpha);
    args.push::<f32>(limit);
    args.launch(LaunchConfig {
        grid_dim: ((routes * n).div_ceil(256) as u32, 1, 1),
        block_dim: (256, 1, 1),
        shared_mem_bytes: 0,
    })
}

const CUDA_ROUTED_LINEAR_KERNELS: RoutedLinearKernels = RoutedLinearKernels {
    supports,
    project: |input, weights, rows, expert_ids, out| {
        crate::with_cuda_stream(|s| project(s, input, weights, rows, expert_ids, out))
    },
    postamble: |gate, up, bias, bias_up, expert_ids, activation, out| {
        crate::with_cuda_stream(|s| {
            postamble(s, gate, up, bias, bias_up, expert_ids, activation, out)
        })
    },
    fused_pair: None,
};

crate::register_cuda_op!(RoutedLinear, |source, node, op| {
    Ok(GpuRoutedLinear::from_tract_core(source, node, op, "Cuda", CUDA_ROUTED_LINEAR_KERNELS)?
        .map(|op| Box::new(op) as _))
});

/// [`tract_gpu::ops::routing::lower_moe_ffn`] for the weights CUDA projects.
pub fn lower_moe_ffn(
    _ctx: &(),
    model: &TypedModel,
    node: &TypedNode,
    _name: &str,
    op: &MoeFfn,
) -> TractResult<Option<TypedModelPatch>> {
    tract_gpu::ops::routing::lower_moe_ffn(model, node, op, supports)
}
