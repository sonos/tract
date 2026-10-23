use crate::encoder::EncoderExt;
use crate::{LibraryName, MetalStream};
use metal::{MTLSize, NSUInteger};
use tract_core::internal::*;
use tract_core::ops::array::CumSum;
use tract_gpu::ops::cumsum::GpuCumSum;
use tract_gpu::tensor::DeviceTensor;

pub fn kernel_name(dt: DatumType) -> TractResult<String> {
    ensure!(GpuCumSum::is_supported_dt(dt), "Unsupported dt {dt:?} for metal cumsum");
    Ok(format!("array_ops::cumsum_{}", DeviceTensor::tname(dt)?))
}

fn dispatch_eval(
    stream: &MetalStream,
    axis: usize,
    exclusive: bool,
    reverse: bool,
    input: &DeviceTensor,
    output: &DeviceTensor,
) -> TractResult<()> {
    stream.retain_tensor(input);
    stream.retain_tensor(output);

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
    let params = [outer as i32, len as i32, inner as i32, exclusive as i32, reverse as i32];

    let pipeline =
        stream.load_pipeline(LibraryName::ArrayOps, &kernel_name(input.datum_type())?)?;
    let width = (lanes as NSUInteger).min(pipeline.max_total_threads_per_threadgroup());
    let command_buffer = stream.command_buffer();
    command_buffer.encode(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_metal_tensor(0, input, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(1, output, metal::MTLResourceUsage::Write);
        encoder.set_slice(2, &params);
        let grid_size =
            MTLSize { width: (lanes as NSUInteger).div_ceil(width), height: 1, depth: 1 };
        let group_size = MTLSize { width, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(grid_size, group_size);
    });
    Ok(())
}

pub fn metal_cumsum_dispatch(
    axis: usize,
    exclusive: bool,
    reverse: bool,
    input: &DeviceTensor,
    output: &DeviceTensor,
) -> TractResult<()> {
    crate::with_metal_stream(|stream| {
        dispatch_eval(stream, axis, exclusive, reverse, input, output)
    })
}

crate::register_metal_op!(CumSum, |source, node, op| {
    rule_if!(GpuCumSum::is_supported_dt(source.node_input_facts(node.id)?[0].datum_type));
    Ok(Some(Box::new(GpuCumSum::from_tract_core(op, "Metal", metal_cumsum_dispatch))))
});
