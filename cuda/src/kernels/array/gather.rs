use crate::context::{TractCudaStream, cuda_context};
use crate::kernels::launch_args::TractLaunchArgs;
use crate::kernels::{LibraryName, MAX_THREADS, get_cuda_view};
use anyhow::ensure;
use cudarc::driver::{CudaStream, LaunchConfig, PushKernelArg};
use std::fmt;
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Gather;

impl fmt::Display for Gather {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{self:?}")
    }
}

impl Gather {
    pub fn is_supported_dt(dt: DatumType) -> bool {
        matches!(dt, DatumType::F32 | DatumType::F16)
    }

    pub fn kernel_name(&self, dt: DatumType) -> TractResult<String> {
        ensure!(Self::is_supported_dt(dt), "Unsupported dt {:?} for cuda gather op", dt);
        let tname = DeviceTensor::tname(dt)?;
        Ok(format!("gather_{tname}"))
    }

    pub fn eval(
        &self,
        stream: &TractCudaStream,
        data: &DeviceTensor,
        indices: &DeviceTensor,
        axis: usize,
    ) -> TractResult<DeviceTensor> {
        ensure!(data.rank() > axis);
        let mut out_shape: TVec<usize> = data.shape()[..axis].into();
        out_shape.extend(indices.shape().iter().copied());
        out_shape.extend(data.shape()[axis + 1..].iter().copied());
        let output = unsafe { DeviceTensor::uninitialized_dt(data.datum_type(), &out_shape)? };
        self.dispatch_eval(stream, data, indices, axis, &output)?;
        stream.synchronize()?;
        Ok(output)
    }

    pub fn dispatch_eval(
        &self,
        stream: &TractCudaStream,
        data: &DeviceTensor,
        indices: &DeviceTensor,
        axis: usize,
        output: &DeviceTensor,
    ) -> TractResult<()> {
        ensure!(data.rank() > axis);
        ensure!(indices.datum_type() == i64::datum_type());
        ensure!(output.datum_type() == data.datum_type());

        let data_shape = data.shape();
        let pre: usize = data_shape[..axis].iter().product();
        let a_size: usize = data_shape[axis];
        let post: usize = data_shape[axis + 1..].iter().product();
        let n_indices: usize = indices.shape().iter().product();

        // Output volume must match (pre, n_indices, post) under natural strides;
        // the GpuGather::output_facts code computes this same shape.
        let expected: usize = pre * n_indices * post;
        ensure!(
            output.shape().iter().product::<usize>() == expected,
            "Gather output shape mismatch: data={:?} axis={} indices={:?} output={:?}",
            data_shape,
            axis,
            indices.shape(),
            output.shape()
        );

        let d_view = get_cuda_view(data);
        let i_view = get_cuda_view(indices);
        let o_view = get_cuda_view(output);

        let func = cuda_context()
            .load_pipeline(LibraryName::Array, self.kernel_name(data.datum_type())?)?;

        let mut launch_args = TractLaunchArgs::new(stream, &func);
        launch_args.push_view(&d_view);
        launch_args.push_view(&i_view);
        launch_args.push_view(&o_view);
        launch_args.push::<i32>(pre as i32);
        launch_args.push::<i32>(a_size as i32);
        launch_args.push::<i32>(post as i32);
        launch_args.push::<i32>(n_indices as i32);

        let block_x = post.clamp(32, MAX_THREADS);
        let grid_x = post.div_ceil(block_x);
        let cfg = LaunchConfig {
            grid_dim: (grid_x as _, n_indices as _, pre as _),
            block_dim: (block_x as _, 1, 1),
            shared_mem_bytes: 0,
        };
        launch_args.launch(cfg)
    }
}

pub fn cuda_gather_dispatch(
    data: &DeviceTensor,
    indices: &DeviceTensor,
    axis: usize,
    output: &DeviceTensor,
) -> TractResult<()> {
    crate::with_cuda_stream(|stream| Gather.dispatch_eval(stream, data, indices, axis, output))
}

crate::register_cuda_op!(tract_core::ops::array::Gather, |source, node, op| {
    let facts = source.node_input_facts(node.id)?;
    // Plain-tensor path only.  The CPU op also handles block-quant and packed
    // matrix storage; those decompose into a dequantization step that the GPU
    // path doesn't (yet) cover.
    rule_if!(facts[0].is_plain());
    rule_if!(Gather::is_supported_dt(facts[0].datum_type));
    rule_if!(facts[1].datum_type == i64::datum_type());
    rule_if!(op.output_type.is_none() || op.output_type == Some(facts[0].datum_type));
    Ok(Some(Box::new(tract_gpu::ops::gather::GpuGather::new(
        op.axis,
        "Cuda",
        cuda_gather_dispatch,
    ))))
});

#[cfg(test)]
mod tests {
    use super::*;
    use tract_core::internal::Tensor;
    use tract_core::ops::array::Gather as CpuGather;
    use tract_gpu::tensor::IntoDevice;

    fn run_against_cpu(
        data_shape: &[usize],
        indices_shape: &[usize],
        indices_data: &[i64],
        axis: usize,
    ) -> TractResult<()> {
        crate::with_cuda_stream(|stream| {
            let n: usize = data_shape.iter().product();
            let data = Tensor::from_shape(
                data_shape,
                &(0..n).map(|i| i as f32 / 10.0).collect::<Vec<_>>(),
            )?;
            let indices = Tensor::from_shape(indices_shape, indices_data)?;
            let cuda_data = data.clone().into_device()?;
            let cuda_indices = indices.clone().into_device()?;

            let cpu_op = CpuGather::new(axis);
            let cpu_out = cpu_op.eval(tvec![data.into_tvalue(), indices.into_tvalue()])?[0]
                .clone()
                .into_tensor();
            let cuda_out = Gather.eval(stream, &cuda_data, &cuda_indices, axis)?;
            cpu_out
                .close_enough(&cuda_out.to_host()?.into_tensor(), Approximation::Exact)
                .with_context(|| {
                    format!(
                        "data={data_shape:?} indices={indices_shape:?} axis={axis} \
                         indices_data={indices_data:?}"
                    )
                })
        })
    }

    /// Embedding lookup: rank-2 table, rank-2 index — the nemotron decoder shape.
    #[test]
    fn test_gather_embedding() -> TractResult<()> {
        run_against_cpu(&[1025, 640], &[1, 1], &[42], 0)
    }

    /// Multi-batch embedding lookup.
    #[test]
    fn test_gather_embedding_multi() -> TractResult<()> {
        run_against_cpu(&[100, 16], &[2, 3], &[0, 1, 99, 50, 25, 7], 0)
    }

    /// Non-zero axis: pre-batch axes flatten correctly.
    #[test]
    fn test_gather_axis_1() -> TractResult<()> {
        run_against_cpu(&[3, 10, 4], &[2], &[0, 9], 1)
    }

    /// Negative indices wrap (axis size = 100, so -1 → 99, -100 → 0).
    #[test]
    fn test_gather_negative_indices() -> TractResult<()> {
        run_against_cpu(&[100, 4], &[3], &[-1, -100, -50], 0)
    }

    /// Scalar index input (rank-0).
    #[test]
    fn test_gather_scalar_index() -> TractResult<()> {
        run_against_cpu(&[5, 8], &[], &[3], 0)
    }
}
