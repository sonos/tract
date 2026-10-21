use crate::context::cuda_context;
use crate::kernels::launch_args::TractLaunchArgs;
use crate::kernels::{LibraryName, get_cuda_view};
use cudarc::driver::LaunchConfig;
use tract_core::internal::*;
use tract_core::ops::nn::resize::Resize as CoreResize;
use tract_gpu::ops::resize::GpuResize;
use tract_gpu::tensor::DeviceTensor;

const BLOCK: usize = 256;

pub fn is_supported_dt(dt: DatumType) -> bool {
    matches!(dt, DatumType::F32 | DatumType::F16)
}

pub fn kernel_name(dt: DatumType) -> TractResult<String> {
    ensure!(is_supported_dt(dt), "Unsupported dt {dt:?} for cuda resize");
    Ok(format!("resize_axis_{}", DeviceTensor::tname(dt)?))
}

/// One axis of a [`GpuResize`]: `output[.., x, ..] = sum_k weights[x, k] * input[.., indices[x, k], ..]`.
pub fn cuda_resize_axis_dispatch(
    input: &DeviceTensor,
    axis: usize,
    indices: &DeviceTensor,
    weights: &DeviceTensor,
    window: usize,
    output: &DeviceTensor,
) -> TractResult<()> {
    ensure!(input.rank() > axis);
    ensure!(output.datum_type() == input.datum_type());
    ensure!(indices.datum_type() == i32::datum_type());
    ensure!(weights.datum_type() == f32::datum_type());
    let len_in = input.shape()[axis];
    let len_out = output.shape()[axis];
    let inner: usize = input.shape()[axis + 1..].iter().product();
    let outer: usize = input.shape()[..axis].iter().product();
    ensure!(indices.len() == len_out * window && weights.len() == len_out * window);
    crate::with_cuda_stream(|stream| {
        let func =
            cuda_context().load_pipeline(LibraryName::Array, kernel_name(input.datum_type())?)?;
        let input = get_cuda_view(input);
        let output = get_cuda_view(output);
        let indices = get_cuda_view(indices);
        let weights = get_cuda_view(weights);
        let mut launcher = TractLaunchArgs::new(stream, &func);
        launcher.push_view(&input);
        launcher.push_view(&output);
        launcher.push_view(&indices);
        launcher.push_view(&weights);
        for arg in [outer, len_in, len_out, inner, window] {
            launcher.push_i32(arg as i32);
        }
        launcher.launch(LaunchConfig {
            grid_dim: ((outer * len_out * inner).div_ceil(BLOCK) as u32, 1, 1),
            block_dim: (BLOCK as u32, 1, 1),
            shared_mem_bytes: 0,
        })
    })
}

/// Translates a core Resize whose plan can be baked (see [`GpuResize::from_core`]).
/// Wired by `transform.rs` rather than `register_cuda_op!` because the resulting op drops
/// the scales/sizes input, whose `TDim` datum type has no device equivalent.
pub fn cuda_resize(source: &TypedModel, node: &TypedNode) -> TractResult<Option<Box<dyn TypedOp>>> {
    let Some(op) = node.op_as::<CoreResize>() else { return Ok(None) };
    let facts = source.node_input_facts(node.id)?;
    rule_if!(facts[0].is_plain());
    rule_if!(is_supported_dt(facts[0].datum_type));
    Ok(GpuResize::from_core(source, node, op, "Cuda", cuda_resize_axis_dispatch)?
        .map(|op| Box::new(op) as Box<dyn TypedOp>))
}
