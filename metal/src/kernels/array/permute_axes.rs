use crate::kernels::{utils, BroadcastKind};
use crate::MetalTensor;
use crate::{LibraryName, MetalContext};
use anyhow::Result;
use std::fmt;
use tract_core::internal::*;

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

    pub fn kernel_name(&self, dt: DatumType, broadcast_kind: BroadcastKind) -> Result<String> {
        ensure!(Self::is_supported_dt(dt), "Unsupport dt {:?} for metal permute axes  op", dt);
        let tname = MetalTensor::tname(dt)?;
        let broadcast_name = broadcast_kind.to_func_part();
        Ok(format!("array_ops::copy_{broadcast_name}_{tname}"))
    }

    pub fn eval(
        &self,
        context: &MetalContext,
        input: &MetalTensor,
        axes: &[usize],
    ) -> Result<MetalTensor> {
        let output = self.dispatch_eval(context, input, axes)?;
        context.wait_until_completed()?;
        Ok(output)
    }

    pub fn dispatch_eval(
        &self,
        context: &MetalContext,
        input: &MetalTensor,
        axes: &[usize],
    ) -> Result<MetalTensor> {
        input.retain_until_completion();
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

        let mut new_shape = vec![0; input.rank()];
        let mut new_strides = vec![0; input.rank()];

        for (new_axis, &axis) in axes.iter().enumerate() {
            new_shape[new_axis] = shape[axis];
            new_strides[new_axis] = strides[axis];
        }

        let output = unsafe { MetalTensor::uninitialized_dt(input.datum_type(), &new_shape)? };
        output.retained_until_completion();

        let broadcast_kind = BroadcastKind::from_rank(input.rank())
            .with_context(|| anyhow!("Unsupported rank {:?} for PermuteAxes", input.rank()))?;

        let kernel_name = self.kernel_name(input.datum_type(), broadcast_kind)?;

        let pipeline =
            context.shared_context().load_pipeline(LibraryName::ArrayOps, &kernel_name)?;
        let command_buffer = context.command_buffer();
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(input.metal()), 0);
        encoder.set_bytes(
            1,
            (new_strides.len() * std::mem::size_of::<usize>()) as _,
            new_strides.as_ptr() as *const _,
        );
        encoder.set_buffer(2, Some(output.metal()), 0);
        encoder.set_bytes(
            3,
            std::mem::size_of_val(output.shape()) as _,
            output.shape().as_ptr() as *const _,
        );
        encoder.set_bytes(
            4,
            (output.strides().len() * std::mem::size_of::<usize>()) as _,
            output.strides().as_ptr() as *const _,
        );

        let grid_size = utils::build_metal_size_for_shape(output.shape());
        let group_size = utils::build_metal_size_with_ones();

        encoder.use_resource(input.metal(), metal::MTLResourceUsage::Read);
        encoder.use_resource(output.metal(), metal::MTLResourceUsage::Write);
        encoder.dispatch_thread_groups(grid_size, group_size);
        encoder.end_encoding();
        Ok(output)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::IntoMetal;

    use num_traits::Zero;

    use tract_core::internal::Tensor;

    fn run_test_case<F: Datum + Zero + Copy>(shape: &[usize], axes: &[usize]) -> Result<()> {
        objc::rc::autoreleasepool(|| {
            crate::METAL_CONTEXT.with_borrow(|context| {
                let a_len = shape.iter().product::<usize>();
                let a_data = (0..a_len).map(|f| f as f32).collect::<Vec<_>>();

                let a = Tensor::from_shape(shape, &a_data)?.into_metal()?;

                let output = PermuteAxes.eval(context, &a, axes)?;
                let ref_output = a.to_cpu().permute_axes(axes)?;
                assert_eq!(ref_output, output.to_cpu());
                Ok(())
            })
        })
    }

    #[test]
    fn test_permute_axes() -> Result<()> {
        run_test_case::<f32>(&[3, 4], &[1, 0])?;
        run_test_case::<f32>(&[1, 2, 3, 4, 5], &[4, 1, 2, 3, 0])?;
        run_test_case::<f32>(&[1, 1, 2, 2, 3, 2], &[0, 3, 1, 2, 4, 5])?;
        Ok(())
    }
}
