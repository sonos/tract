use crate::context::TractCudaStream;
use crate::context::cuda_context;
use crate::kernels::LibraryName;
use crate::kernels::MAX_THREADS;
use crate::kernels::get_cuda_view;
use crate::kernels::launch_args::TractLaunchArgs;
use cudarc::driver::LaunchConfig;
use tract_core::internal::*;
use tract_core::ops::array::CumSum;
use tract_gpu::ops::cumsum::GpuCumSum;
use tract_gpu::tensor::DeviceTensor;

pub fn kernel_name(dt: DatumType) -> TractResult<String> {
    ensure!(GpuCumSum::is_supported_dt(dt), "Unsupported dt {dt:?} for cuda cumsum");
    Ok(format!("cumsum_{}", DeviceTensor::tname(dt)?))
}

fn dispatch_eval(
    stream: &TractCudaStream,
    axis: usize,
    exclusive: bool,
    reverse: bool,
    input: &DeviceTensor,
    output: &DeviceTensor,
) -> TractResult<()> {
    ensure!(input.shape() == output.shape() && input.datum_type() == output.datum_type());
    ensure!(input.strides() == &*Tensor::natural_strides(input.shape()));
    let shape = input.shape();
    let outer: usize = shape[..axis].iter().product();
    let len = shape[axis];
    let inner: usize = shape[axis + 1..].iter().product();
    let lanes = outer * inner;
    if lanes == 0 || len == 0 {
        return Ok(());
    }

    let func =
        cuda_context().load_pipeline(LibraryName::Array, kernel_name(input.datum_type())?)?;
    let i_view = get_cuda_view(input);
    let o_view = get_cuda_view(output);
    let mut args = TractLaunchArgs::new(stream, &func);
    args.push_view(&i_view);
    args.push_view(&o_view);
    args.push_i32(outer);
    args.push_i32(len);
    args.push_i32(inner);
    args.push_i32(exclusive as usize);
    args.push_i32(reverse as usize);

    let block = lanes.min(MAX_THREADS);
    let cfg = LaunchConfig {
        grid_dim: (lanes.div_ceil(block) as _, 1, 1),
        block_dim: (block as _, 1, 1),
        shared_mem_bytes: 0,
    };
    args.launch(cfg)
}

pub fn cuda_cumsum_dispatch(
    axis: usize,
    exclusive: bool,
    reverse: bool,
    input: &DeviceTensor,
    output: &DeviceTensor,
) -> TractResult<()> {
    crate::with_cuda_stream(|stream| dispatch_eval(stream, axis, exclusive, reverse, input, output))
}

crate::register_cuda_op!(CumSum, |source, node, op| {
    rule_if!(GpuCumSum::is_supported_dt(source.node_input_facts(node.id)?[0].datum_type));
    Ok(Some(Box::new(GpuCumSum::from_tract_core(op, "Cuda", cuda_cumsum_dispatch))))
});
