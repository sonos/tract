use crate::encoder::EncoderExt;
use crate::{LibraryName, MetalStream};
use metal::{MTLSize, NSUInteger};
use tract_core::internal::*;
use tract_core::ops::array::GatherNd;
use tract_gpu::ops::gather_nd::{GpuGatherNd, MAX_COORDS};
use tract_gpu::tensor::DeviceTensor;

pub fn is_supported_dt(dt: DatumType) -> bool {
    matches!(dt, DatumType::F32 | DatumType::F16 | DatumType::I64)
}

pub fn kernel_name(dt: DatumType) -> TractResult<String> {
    ensure!(is_supported_dt(dt), "Unsupported dt {dt:?} for metal gather_nd");
    Ok(format!("array_ops::gather_nd_{}", DeviceTensor::tname(dt)?))
}

fn dispatch_eval(
    stream: &MetalStream,
    batch_dims: usize,
    data: &DeviceTensor,
    indices: &DeviceTensor,
    output: &DeviceTensor,
) -> TractResult<()> {
    stream.retain_tensor(data);
    stream.retain_tensor(indices);
    stream.retain_tensor(output);

    ensure!(indices.datum_type() == f32::datum_type());
    ensure!(output.datum_type() == data.datum_type());
    let n = *indices.shape().last().context("GatherNd indices of rank 0")?;
    ensure!(0 < n && n <= MAX_COORDS && batch_dims + n <= data.rank());
    let shape = data.shape();
    let batch: usize = shape[..batch_dims].iter().product();
    let len = output.len();
    let mut params = [0i64; 5 + 2 * MAX_COORDS];
    params[0] = len as i64;
    params[1] = shape[batch_dims + n..].iter().product::<usize>() as i64;
    params[2] = (indices.len() / n / batch.max(1)) as i64;
    params[3] = n as i64;
    params[4] = shape[batch_dims..].iter().product::<usize>() as i64;
    for k in 0..n {
        params[5 + k] = shape[batch_dims + k] as i64;
        params[5 + MAX_COORDS + k] = shape[batch_dims + k + 1..].iter().product::<usize>() as i64;
    }

    let pipeline = stream.load_pipeline(LibraryName::ArrayOps, &kernel_name(data.datum_type())?)?;
    let width = (len as NSUInteger).min(pipeline.max_total_threads_per_threadgroup());
    let command_buffer = stream.command_buffer();
    command_buffer.encode(|encoder| {
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_metal_tensor(0, data, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(1, indices, metal::MTLResourceUsage::Read);
        encoder.set_metal_tensor(2, output, metal::MTLResourceUsage::Write);
        encoder.set_slice(3, &params);
        let grid_size = MTLSize { width: (len as NSUInteger).div_ceil(width), height: 1, depth: 1 };
        let group_size = MTLSize { width, height: 1, depth: 1 };
        encoder.dispatch_thread_groups(grid_size, group_size);
    });
    Ok(())
}

pub fn metal_gather_nd_dispatch(
    batch_dims: usize,
    data: &DeviceTensor,
    indices: &DeviceTensor,
    output: &DeviceTensor,
) -> TractResult<()> {
    crate::with_metal_stream(|stream| dispatch_eval(stream, batch_dims, data, indices, output))
}

crate::register_metal_op!(GatherNd, |source, node, op| {
    let facts = source.node_input_facts(node.id)?;
    rule_if!(facts[0].is_plain() && is_supported_dt(facts[0].datum_type));
    rule_if!(GpuGatherNd::is_supported(&facts));
    Ok(Some(Box::new(GpuGatherNd::new(op.batch_dims, "Metal", metal_gather_nd_dispatch))))
});

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::with_borrowed_metal_stream;
    use tract_gpu::tensor::IntoDevice;

    fn run_against_cpu(
        batch_dims: usize,
        data_shape: &[usize],
        indices_shape: &[usize],
        indices_data: &[f32],
    ) -> TractResult<()> {
        with_borrowed_metal_stream(|stream| {
            let len: usize = data_shape.iter().product();
            let data =
                Tensor::from_shape(data_shape, &(0..len).map(|i| i as f32).collect::<Vec<_>>())?;
            let indices = Tensor::from_shape(indices_shape, indices_data)?;
            let cpu = GatherNd::new(batch_dims)
                .eval(
                    &EvalContext::out_of_plan(),
                    tvec!(data.clone().into_tvalue(), indices.clone().into_tvalue()),
                )?
                .remove(0);
            let metal_data = data.into_device()?;
            let metal_indices = indices.into_device()?;
            let output = unsafe { DeviceTensor::uninitialized_dt(f32::datum_type(), cpu.shape())? };
            dispatch_eval(stream, batch_dims, &metal_data, &metal_indices, &output)?;
            stream.wait_until_completed()?;
            cpu.close_enough(&output.to_host()?.into_tensor(), Approximation::Exact)
        })
    }

    #[test]
    fn test_gather_nd_points() -> TractResult<()> {
        run_against_cpu(0, &[4, 5], &[3, 2], &[0., 0., 3., 4., 2., 1.])
    }

    #[test]
    fn test_gather_nd_slices() -> TractResult<()> {
        run_against_cpu(0, &[4, 5, 3], &[2, 1], &[3., 1.])
    }

    #[test]
    fn test_gather_nd_batch_dims() -> TractResult<()> {
        run_against_cpu(
            1,
            &[2, 6, 5, 3],
            &[2, 4, 2],
            &[0., 0., 5., 4., 1., 2., 3., 3., 2., 2., 0., 4., 5., 0., 4., 1.],
        )
    }
}
