use crate::kernels::{utils, BroadcastKind};
use crate::MetalTensor;
use crate::{LibraryName, MetalContext};
use anyhow::{ensure, Result};
use std::fmt;
use tract_core::internal::*;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Concat;

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

    pub fn kernel_name(&self, dt: DatumType, broadcast_kind: BroadcastKind) -> Result<String> {
        ensure!(Self::is_supported_dt(dt), "Unsupport dt {:?} for metal concat  op", dt);
        let tname = MetalTensor::tname(dt)?;
        let broadcast_name = broadcast_kind.to_func_part();
        Ok(format!("array_ops::copy_{broadcast_name}_{tname}"))
    }

    pub fn eval(
        &self,
        context: &MetalContext,
        inputs: &[&MetalTensor],
        axis: usize,
    ) -> Result<MetalTensor> {
        let output = self.dispatch_eval(context, inputs, axis)?;
        context.wait_until_completed()?;
        Ok(output)
    }

    pub fn dispatch_eval(
        &self,
        context: &MetalContext,
        inputs: &[&MetalTensor],
        axis: usize,
    ) -> Result<MetalTensor> {
        ensure!(!inputs.is_empty());

        let output_dt = inputs[0].datum_type();
        let mut output_shape = inputs[0].shape().to_vec();
        output_shape[axis] = inputs.iter().map(|it| it.shape()[axis]).sum();
        let output_strides = Tensor::natural_strides(&output_shape);

        let mut offsets = tvec![0; inputs.len()];
        let mut cursor = 0;

        for (i_idx, input) in inputs.iter().enumerate() {
            let i_shape = input.shape();
            ensure!(i_shape[..axis] == output_shape[..axis]);
            ensure!(i_shape[axis + 1..] == output_shape[axis + 1..]);
            offsets[i_idx] = cursor * (output_strides[axis] as usize) * output_dt.size_of();
            cursor += i_shape[axis];
        }

        let output = unsafe { MetalTensor::uninitialized_dt(output_dt, &output_shape)? };
        output.retain_until_completion();

        let broadcast_kind = BroadcastKind::from_rank(output.rank()).with_context(|| {
            anyhow!(
                "Unsupported broadcast for broadcast op: (in: {:?}, out: {:?})",
                inputs[0].shape(),
                output_shape
            )
        })?;

        let kernel_name = self.kernel_name(output_dt, broadcast_kind)?;
        let pipeline =
            context.shared_context().load_pipeline(LibraryName::ArrayOps, &kernel_name)?;
        let command_buffer = context.command_buffer();

        for (input, offset) in inputs.iter().zip(offsets.into_iter()) {
            if input.len() == 0 {
                continue;
            }
            input.retain_until_completion();
            let i_strides = input.strides();
            let i_shape = input.shape();

            let encoder = command_buffer.new_compute_command_encoder();

            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_buffer(0, Some(input.metal()), 0);
            encoder.set_bytes(
                1,
                (i_strides.len() * std::mem::size_of::<usize>()) as _,
                i_strides.as_ptr() as *const _,
            );
            encoder.set_buffer(2, Some(output.metal()), offset as _);
            encoder.set_bytes(3, std::mem::size_of_val(i_shape) as _, i_shape.as_ptr() as *const _);
            encoder.set_bytes(
                4,
                (output_strides.len() * std::mem::size_of::<usize>()) as _,
                output_strides.as_ptr() as *const _,
            );

            let grid_size = utils::build_metal_size_for_shape(i_shape);
            let group_size = utils::build_metal_size_with_ones();

            encoder.use_resource(input.metal(), metal::MTLResourceUsage::Read);
            encoder.use_resource(output.metal(), metal::MTLResourceUsage::Write);
            encoder.dispatch_thread_groups(grid_size, group_size);
            encoder.end_encoding();
        }

        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::IntoMetal;
    use tract_itertools::Itertools;

    use num_traits::Zero;

    use tract_core::internal::Tensor;

    fn run_test_case<F: Datum + Zero + Copy>(shapes: &[&[usize]], axis: usize) -> Result<()> {
        objc::rc::autoreleasepool(|| {
            crate::METAL_CONTEXT.with_borrow(|context| {
                let mut inputs = tvec![];
                for shape in shapes {
                    let len = shape.iter().product::<usize>();
                    let data = (0..len).map(|f| f as f32).collect::<Vec<_>>();
                    inputs.push(Tensor::from_shape(shape, &data)?.into_metal()?);
                }

                let output = Concat.eval(context, &inputs.iter().collect_vec(), axis)?;
                let ref_output = Tensor::stack_tensors(
                    axis,
                    &inputs.iter().map(|it| it.to_cpu()).collect_vec(),
                )?;
                assert_eq!(ref_output, output.to_cpu());
                Ok(())
            })
        })
    }

    #[test]
    fn test_concat() -> Result<()> {
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
