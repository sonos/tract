use crate::context::{TractCudaStream, cuda_context};
use crate::kernels::launch_args::TractLaunchArgs;
use crate::kernels::{LibraryName, MAX_THREADS, get_cuda_view};
use cudarc::driver::LaunchConfig;
use tract_core::internal::*;
use tract_core::ops::array::GatherNd;
use tract_gpu::ops::gather_nd::{GpuGatherNd, MAX_COORDS};
use tract_gpu::tensor::DeviceTensor;

pub fn is_supported_dt(dt: DatumType) -> bool {
    matches!(dt, DatumType::F32 | DatumType::F16 | DatumType::I64)
}

pub fn kernel_name(dt: DatumType) -> TractResult<String> {
    ensure!(is_supported_dt(dt), "Unsupported dt {dt:?} for cuda gather_nd");
    Ok(format!("gather_nd_{}", DeviceTensor::tname(dt)?))
}

fn dispatch_eval(
    stream: &TractCudaStream,
    batch_dims: usize,
    data: &DeviceTensor,
    indices: &DeviceTensor,
    output: &DeviceTensor,
) -> TractResult<()> {
    ensure!(indices.datum_type() == f32::datum_type());
    ensure!(output.datum_type() == data.datum_type());
    let n = *indices.shape().last().context("GatherNd indices of rank 0")?;
    ensure!(0 < n && n <= MAX_COORDS && batch_dims + n <= data.rank());
    let shape = data.shape();
    let batch_stride: usize = shape[batch_dims..].iter().product();
    let inner: usize = shape[batch_dims + n..].iter().product();
    let mut dims = [0usize; MAX_COORDS];
    let mut strides = [0usize; MAX_COORDS];
    for k in 0..n {
        dims[k] = shape[batch_dims + k];
        strides[k] = shape[batch_dims + k + 1..].iter().product();
    }
    let batch: usize = shape[..batch_dims].iter().product();
    let rows = indices.len() / n / batch.max(1);
    let len = output.len();

    let func = cuda_context().load_pipeline(LibraryName::Array, kernel_name(data.datum_type())?)?;
    let d_view = get_cuda_view(data);
    let i_view = get_cuda_view(indices);
    let o_view = get_cuda_view(output);
    let mut args = TractLaunchArgs::new(stream, &func);
    args.push_view(&d_view);
    args.push_view(&i_view);
    args.push_view(&o_view);
    args.push::<i64>(len);
    args.push::<i64>(inner);
    args.push::<i64>(rows);
    args.push_i32(n);
    args.push::<i64>(batch_stride);
    args.push_slice_i32(&dims);
    args.push_slice::<i64>(&strides);

    let block = len.min(MAX_THREADS);
    let cfg = LaunchConfig {
        grid_dim: (len.div_ceil(block) as _, 1, 1),
        block_dim: (block as _, 1, 1),
        shared_mem_bytes: 0,
    };
    args.launch(cfg)
}

pub fn cuda_gather_nd_dispatch(
    batch_dims: usize,
    data: &DeviceTensor,
    indices: &DeviceTensor,
    output: &DeviceTensor,
) -> TractResult<()> {
    crate::with_cuda_stream(|stream| dispatch_eval(stream, batch_dims, data, indices, output))
}

crate::register_cuda_op!(GatherNd, |source, node, op| {
    let facts = source.node_input_facts(node.id)?;
    rule_if!(facts[0].is_plain() && is_supported_dt(facts[0].datum_type));
    rule_if!(GpuGatherNd::is_supported(&facts));
    Ok(Some(Box::new(GpuGatherNd::new(op.batch_dims, "Cuda", cuda_gather_nd_dispatch))))
});

#[cfg(test)]
mod tests {
    use super::*;
    use tract_gpu::tensor::IntoDevice;

    fn run_against_cpu(
        batch_dims: usize,
        data_shape: &[usize],
        indices_shape: &[usize],
        indices_data: &[f32],
    ) -> TractResult<()> {
        crate::with_cuda_stream(|stream| {
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
            let cuda_data = data.into_device()?;
            let cuda_indices = indices.into_device()?;
            let output = unsafe { DeviceTensor::uninitialized_dt(f32::datum_type(), cpu.shape())? };
            dispatch_eval(stream, batch_dims, &cuda_data, &cuda_indices, &output)?;
            stream.synchronize()?;
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
