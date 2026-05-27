use crate::encoder::EncoderExt;
use crate::{LibraryName, MetalStream};
use anyhow::ensure;
use metal::MTLSize;
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
        ensure!(Self::is_supported_dt(dt), "Unsupported dt {:?} for metal gather op", dt);
        let tname = DeviceTensor::tname(dt)?;
        Ok(format!("array_ops::gather_{tname}"))
    }

    pub fn eval(
        &self,
        stream: &MetalStream,
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
        stream.wait_until_completed()?;
        Ok(output)
    }

    pub fn dispatch_eval(
        &self,
        stream: &MetalStream,
        data: &DeviceTensor,
        indices: &DeviceTensor,
        axis: usize,
        output: &DeviceTensor,
    ) -> TractResult<()> {
        stream.retain_tensor(data);
        stream.retain_tensor(indices);
        stream.retain_tensor(output);

        ensure!(data.rank() > axis);
        ensure!(indices.datum_type() == i64::datum_type());
        ensure!(output.datum_type() == data.datum_type());

        let data_shape = data.shape();
        let pre: usize = data_shape[..axis].iter().product();
        let a_size: usize = data_shape[axis];
        let post: usize = data_shape[axis + 1..].iter().product();
        let n_indices: usize = indices.shape().iter().product();

        let expected: usize = pre * n_indices * post;
        ensure!(
            output.shape().iter().product::<usize>() == expected,
            "Gather output shape mismatch: data={:?} axis={} indices={:?} output={:?}",
            data_shape,
            axis,
            indices.shape(),
            output.shape()
        );

        let params: [i32; 4] = [pre as i32, a_size as i32, post as i32, n_indices as i32];

        let pipeline =
            stream.load_pipeline(LibraryName::ArrayOps, &self.kernel_name(data.datum_type())?)?;
        let command_buffer = stream.command_buffer();
        command_buffer.encode(|encoder| {
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_metal_tensor(0, data, metal::MTLResourceUsage::Read);
            encoder.set_metal_tensor(1, indices, metal::MTLResourceUsage::Read);
            encoder.set_metal_tensor(2, output, metal::MTLResourceUsage::Write);
            encoder.set_slice(3, &params);
            let grid_size = MTLSize { width: post as _, height: n_indices as _, depth: pre as _ };
            let group_size = MTLSize { width: 1, height: 1, depth: 1 };
            encoder.dispatch_thread_groups(grid_size, group_size);
        });
        Ok(())
    }
}

pub fn metal_gather_dispatch(
    data: &DeviceTensor,
    indices: &DeviceTensor,
    axis: usize,
    output: &DeviceTensor,
) -> TractResult<()> {
    crate::with_metal_stream(|stream| Gather.dispatch_eval(stream, data, indices, axis, output))
}

crate::register_metal_op!(tract_core::ops::array::Gather, |source, node, op| {
    let facts = source.node_input_facts(node.id)?;
    rule_if!(facts[0].is_plain());
    rule_if!(Gather::is_supported_dt(facts[0].datum_type));
    rule_if!(facts[1].datum_type == i64::datum_type());
    rule_if!(op.output_type.is_none() || op.output_type == Some(facts[0].datum_type));
    Ok(Some(Box::new(tract_gpu::ops::gather::GpuGather::new(
        op.axis,
        "Metal",
        metal_gather_dispatch,
    ))))
});

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::with_borrowed_metal_stream;
    use tract_core::internal::Tensor;
    use tract_core::ops::array::Gather as CpuGather;
    use tract_gpu::tensor::IntoDevice;

    fn run_against_cpu(
        data_shape: &[usize],
        indices_shape: &[usize],
        indices_data: &[i64],
        axis: usize,
    ) -> TractResult<()> {
        with_borrowed_metal_stream(|stream| {
            let n: usize = data_shape.iter().product();
            let data = Tensor::from_shape(
                data_shape,
                &(0..n).map(|i| i as f32 / 10.0).collect::<Vec<_>>(),
            )?;
            let indices = Tensor::from_shape(indices_shape, indices_data)?;
            let metal_data = data.clone().into_device()?;
            let metal_indices = indices.clone().into_device()?;

            let cpu_op = CpuGather::new(axis);
            let cpu_out = cpu_op.eval(tvec![data.into_tvalue(), indices.into_tvalue()])?[0]
                .clone()
                .into_tensor();
            let metal_out = Gather.eval(stream, &metal_data, &metal_indices, axis)?;
            cpu_out
                .close_enough(&metal_out.to_host()?.into_tensor(), Approximation::Exact)
                .with_context(|| {
                    format!(
                        "data={data_shape:?} indices={indices_shape:?} axis={axis} \
                         indices_data={indices_data:?}"
                    )
                })
        })
    }

    #[test]
    fn test_gather_embedding() -> TractResult<()> {
        run_against_cpu(&[1025, 640], &[1, 1], &[42], 0)
    }

    #[test]
    fn test_gather_embedding_multi() -> TractResult<()> {
        run_against_cpu(&[100, 16], &[2, 3], &[0, 1, 99, 50, 25, 7], 0)
    }

    #[test]
    fn test_gather_axis_1() -> TractResult<()> {
        run_against_cpu(&[3, 10, 4], &[2], &[0, 9], 1)
    }

    #[test]
    fn test_gather_negative_indices() -> TractResult<()> {
        run_against_cpu(&[100, 4], &[3], &[-1, -100, -50], 0)
    }

    #[test]
    fn test_gather_scalar_index() -> TractResult<()> {
        run_against_cpu(&[5, 8], &[], &[3], 0)
    }
}
