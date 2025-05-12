use crate::encoder::EncoderExt;
use crate::kernels::{utils, BroadcastKind};
use crate::{LibraryName, MetalStream};
use std::fmt;
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct PermuteAxes;

impl fmt::Display for PermuteAxes {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl PermuteAxes {
    pub fn is_supported_dt(dt: DatumType) -> bool {
        matches!(
            dt,
            DatumType::F32
                | DatumType::F16
                | DatumType::U8
                | DatumType::U16
                | DatumType::U32
                | DatumType::U64
                | DatumType::I8
                | DatumType::I16
                | DatumType::I32
                | DatumType::I64
                | DatumType::Bool
        )
    }

    pub fn kernel_name(&self, dt: DatumType, broadcast_kind: BroadcastKind) -> TractResult<String> {
        ensure!(Self::is_supported_dt(dt), "Unsupport dt {:?} for metal permute axes  op", dt);
        let tname = DeviceTensor::tname(dt)?;
        let broadcast_name = broadcast_kind.name();
        Ok(format!("array_ops::copy_{broadcast_name}_{tname}"))
    }

    pub fn output_shape<D: DimLike>(shape: &[D], axes: &[usize]) -> TractResult<TVec<D>> {
        let rank = shape.len();
        let mut new_shape = tvec![0.into(); rank];

        for (new_axis, &axis) in axes.iter().enumerate() {
            new_shape[new_axis] = shape[axis].clone();
        }
        Ok(new_shape)
    }

    pub fn eval(
        &self,
        stream: &MetalStream,
        input: &DeviceTensor,
        axes: &[usize],
    ) -> TractResult<DeviceTensor> {
        let output = unsafe {
            DeviceTensor::uninitialized_dt(
                input.datum_type(),
                &Self::output_shape(input.shape(), axes)?,
            )?
        };
        self.dispatch_eval(stream, input, axes, &output)?;
        stream.wait_until_completed()?;
        Ok(output)
    }

    pub fn dispatch_eval(
        &self,
        stream: &MetalStream,
        input: &DeviceTensor,
        axes: &[usize],
        output: &DeviceTensor,
    ) -> TractResult<()> {
        stream.retain_tensor(input);
        stream.retain_tensor(output);

        // Validate give axes permutation
        let mut usage_counts = vec![0; input.rank()];
        for axis in axes {
            usage_counts[*axis] += 1;
        }
        for count in usage_counts {
            assert_eq!(count, 1, "each axis must be listed exactly once");
        }

        let shape = input.shape();
        let strides = input.strides();
        let mut new_strides = vec![0; input.rank()];
        let mut new_shape = vec![0; input.rank()];

        for (new_axis, &axis) in axes.iter().enumerate() {
            new_shape[new_axis] = shape[axis];
            new_strides[new_axis] = strides[axis];
        }

        ensure!(
            output.shape() == new_shape,
            "Mismatch between expected new shape {:?} and output shape {:?}",
            new_shape,
            output.shape()
        );

        let broadcast_kind = BroadcastKind::from_rank(input.rank())
            .with_context(|| format!("Unsupported rank {:?} for PermuteAxes", input.rank()))?;

        let kernel_name = self.kernel_name(input.datum_type(), broadcast_kind)?;

        let out_shape = output.shape();
        let pipeline = stream.load_pipeline(LibraryName::ArrayOps, &kernel_name)?;
        let command_buffer = stream.command_buffer();
        command_buffer.encode(|encoder| {
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_metal_tensor(0, input, metal::MTLResourceUsage::Read);
            encoder.set_slice(1, &new_strides);
            encoder.set_metal_tensor(2, output, metal::MTLResourceUsage::Write);
            encoder.set_slice(3, output.shape());
            encoder.set_slice(4, output.strides());

            let (grid_size, group_size) = utils::build_metal_grid_and_groups_for_el_wise_op(
                out_shape,
                pipeline.max_total_threads_per_threadgroup() as _,
            );

            encoder.dispatch_thread_groups(grid_size, group_size);
        });
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::utils::with_borrowed_metal_stream;

    use super::*;

    use num_traits::Zero;

    use tract_core::internal::Tensor;
    use tract_gpu::tensor::IntoDevice;

    fn run_test_case<F: Datum + Zero + Copy>(shape: &[usize], axes: &[usize]) -> TractResult<()> {
        with_borrowed_metal_stream(|stream| {
            let a_len = shape.iter().product::<usize>();
            let a_data = (0..a_len).map(|f| f as f32).collect::<Vec<_>>();

            let a = Tensor::from_shape(shape, &a_data)?.into_device()?;

            let output = PermuteAxes.eval(stream, &a, axes)?;
            let ref_output = a.to_host()?.into_tensor().permute_axes(axes)?;
            assert_eq!(ref_output, output.to_host()?.into_tensor());
            Ok(())
        })
    }

    #[test]
    fn test_permute_axes() -> TractResult<()> {
        run_test_case::<f32>(&[3, 4], &[1, 0])?;
        run_test_case::<f32>(&[1, 2, 3, 4, 5], &[4, 1, 2, 3, 0])?;
        run_test_case::<f32>(&[1, 1, 2, 2, 3, 2], &[0, 3, 1, 2, 4, 5])?;
        Ok(())
    }
}
