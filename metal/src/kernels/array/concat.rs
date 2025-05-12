use crate::encoder::EncoderExt;
use crate::kernels::{utils, BroadcastKind};
use crate::{LibraryName, MetalStream};
use anyhow::ensure;
use std::fmt;
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Concat {
    pub axis: usize,
}

impl fmt::Display for Concat {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{:?}", self)
    }
}

impl Concat {
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
        )
    }

    pub fn kernel_name(&self, dt: DatumType, broadcast_kind: BroadcastKind) -> TractResult<String> {
        ensure!(Self::is_supported_dt(dt), "Unsupport dt {:?} for metal concat  op", dt);
        let tname = DeviceTensor::tname(dt)?;
        let broadcast_name = broadcast_kind.name();
        Ok(format!("array_ops::copy_{broadcast_name}_{tname}"))
    }

    pub fn eval(
        &self,
        stream: &MetalStream,
        inputs: &[&DeviceTensor],
    ) -> TractResult<DeviceTensor> {
        ensure!(!inputs.is_empty());
        let mut output_shape = inputs[0].shape().to_vec();
        output_shape[self.axis] = inputs.iter().map(|it| it.shape()[self.axis]).sum();
        let output =
            unsafe { DeviceTensor::uninitialized_dt(inputs[0].datum_type(), &output_shape)? };

        self.dispatch_eval(stream, inputs, &output)?;
        stream.wait_until_completed()?;
        Ok(output)
    }

    pub fn dispatch_eval(
        &self,
        stream: &MetalStream,
        inputs: &[&DeviceTensor],
        output: &DeviceTensor,
    ) -> TractResult<()> {
        ensure!(!inputs.is_empty());

        stream.retain_tensor(output);

        let output_shape = output.shape();
        let output_strides = output.strides();

        let mut offsets = tvec![0; inputs.len()];
        let mut cursor = 0;

        for (i_idx, input) in inputs.iter().enumerate() {
            let i_shape = input.shape();
            ensure!(i_shape[..self.axis] == output_shape[..self.axis]);
            ensure!(i_shape[self.axis + 1..] == output_shape[self.axis + 1..]);
            offsets[i_idx] =
                cursor * (output_strides[self.axis] as usize) * output.datum_type().size_of();
            cursor += i_shape[self.axis];
        }

        let broadcast_kind = BroadcastKind::from_rank(output.rank()).with_context(|| {
            format!(
                "Unsupported broadcast for broadcast op: (in: {:?}, out: {:?})",
                inputs[0].shape(),
                output_shape
            )
        })?;

        let kernel_name = self.kernel_name(output.datum_type(), broadcast_kind)?;
        let pipeline = stream.load_pipeline(LibraryName::ArrayOps, &kernel_name)?;
        let command_buffer = stream.command_buffer();

        for (input, offset) in inputs.iter().zip(offsets.into_iter()) {
            if input.len() == 0 {
                continue;
            }
            stream.retain_tensor(input);

            let i_strides = input.strides();
            let i_shape = input.shape();

            command_buffer.encode(|encoder| {
                encoder.set_compute_pipeline_state(&pipeline);
                encoder.set_metal_tensor(0, input, metal::MTLResourceUsage::Read);
                encoder.set_slice(1, i_strides);
                encoder.set_metal_tensor_with_offset(
                    2,
                    output,
                    offset as _,
                    metal::MTLResourceUsage::Write,
                );
                encoder.set_slice(3, i_shape);
                encoder.set_slice(4, output_strides);

                let (grid_size, group_size) = utils::build_metal_grid_and_groups_for_el_wise_op(
                    i_shape,
                    pipeline.max_total_threads_per_threadgroup() as _,
                );
                encoder.dispatch_thread_groups(grid_size, group_size);
            });
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::utils::with_borrowed_metal_stream;

    use super::*;
    use tract_gpu::tensor::IntoDevice;
    use tract_itertools::Itertools;

    use num_traits::Zero;

    use tract_core::internal::Tensor;

    fn run_test_case<F: Datum + Zero + Copy>(shapes: &[&[usize]], axis: usize) -> TractResult<()> {
        with_borrowed_metal_stream(|stream| {
            let mut inputs = tvec![];
            for shape in shapes {
                let len = shape.iter().product::<usize>();
                let data = (0..len).map(|f| f as f32).collect::<Vec<_>>();
                inputs.push(Tensor::from_shape(shape, &data)?.into_device()?);
            }

            let output = Concat { axis }.eval(stream, &inputs.iter().collect_vec())?;
            let ref_output = Tensor::stack_tensors(
                axis,
                &inputs.iter().map(|it| it.to_host()).collect::<TractResult<Vec<_>>>()?,
            )?;
            assert_eq!(output.to_host()?.into_tensor(), ref_output);
            Ok(())
        })
    }

    #[test]
    fn test_concat() -> TractResult<()> {
        run_test_case::<f32>(&[&[3, 4], &[3, 4]], 0)?;
        run_test_case::<f32>(&[&[3, 4], &[3, 4]], 1)?;
        run_test_case::<f32>(&[&[1, 5, 4], &[2, 5, 4]], 0)?;
        run_test_case::<f32>(&[&[3, 1, 8], &[3, 2, 8]], 1)?;
        run_test_case::<f32>(&[&[3, 5, 1], &[3, 5, 2]], 2)?;
        run_test_case::<f32>(&[&[1, 5, 4, 10], &[2, 5, 4, 10]], 0)?;
        run_test_case::<f32>(&[&[3, 1, 8, 10], &[3, 2, 8, 10]], 1)?;
        run_test_case::<f32>(&[&[3, 5, 1, 10], &[3, 5, 2, 10]], 2)?;
        run_test_case::<f32>(&[&[3, 5, 1, 10], &[3, 5, 0, 10]], 2)?;
        run_test_case::<f32>(&[&[3, 5, 1, 2], &[3, 5, 1, 8]], 3)?;
        Ok(())
    }
}
