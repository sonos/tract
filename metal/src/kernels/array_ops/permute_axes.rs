use crate::kernels::BroadcastKind;
use crate::MetalTensor;
use crate::{LibraryName, MetalContext};
use anyhow::{bail, Result};
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
        Self::tname(dt).is_ok()
    }

    pub fn tname(dt: DatumType) -> Result<&'static str> {
        let tname = match dt {
            DatumType::F32 => "f32",
            DatumType::F16 => "f16",
            DatumType::U8 => "u8",
            DatumType::U16 => "u16",
            DatumType::U32 => "u32",
            DatumType::U64 => "u64",
            DatumType::I8 => "i8",
            DatumType::I16 => "i16",
            DatumType::I32 => "i32",
            DatumType::I64 => "i64",
            _ => bail!("Unsupport dt {:?} for metal memory ops", dt),
        };
        Ok(tname)
    }

    pub fn kernel_name(&self, dt: DatumType, broadcast_kind: BroadcastKind) -> Result<String> {
        let tname = Self::tname(dt)?;

        let broadcast_name = match broadcast_kind {
            BroadcastKind::Nd1 => "nd1",
            BroadcastKind::Nd2 => "nd2",
            BroadcastKind::Nd3 => "nd3",
            BroadcastKind::Nd4 => "nd4",
            BroadcastKind::Nd5 => "nd5",
            BroadcastKind::Nd6 => "nd6",
            _ => bail!("Unsupported broadcast kind {:?} for array ops", broadcast_kind),
        };

        Ok(format!("array_ops::broadcast_{broadcast_name}_{tname}"))
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
        let mut new_strides = vec![0u32; input.rank()];

        for (new_axis, &axis) in axes.iter().enumerate() {
            new_shape[new_axis] = shape[axis];
            new_strides[new_axis] = strides[axis] as u32;
        }

        let output = unsafe { MetalTensor::uninitialized_dt(input.datum_type(), &new_shape)? };

        let broadcast_kind = BroadcastKind::from_rank(input.rank())
            .with_context(|| anyhow!("Unsupported rank {:?} for PermuteAxes", input.rank()))?;

        let kernel_name = self.kernel_name(input.datum_type(), broadcast_kind)?;

        let output_shape = output.shape().iter().map(|d| *d as u32).collect::<Vec<_>>();

        let pipeline =
            context.shared_context().load_pipeline(LibraryName::ArrayOps, &kernel_name)?;
        let command_buffer = context.command_buffer()?;
        let encoder = command_buffer.new_compute_command_encoder();
        encoder.set_compute_pipeline_state(&pipeline);
        encoder.set_buffer(0, Some(input.metal()), 0);
        encoder.set_bytes(
            1,
            (new_strides.len() * std::mem::size_of::<u32>()) as _,
            new_strides.as_ptr() as *const _,
        );
        encoder.set_buffer(2, Some(output.metal()), 0);
        encoder.set_bytes(
            3,
            (output_shape.len() * std::mem::size_of::<u32>()) as _,
            output_shape.as_ptr() as *const _,
        );

        let grid_size = super::build_metal_size_for_shape(output.shape());
        let group_size = super::build_metal_size_with_ones();

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
                let ref_output = a.tensor().clone().permute_axes(axes)?;
                assert_eq!(&ref_output, output.tensor());
                Ok(())
            })
        })
    }

    #[test]
    fn test_permute_axes() -> Result<()> {
        run_test_case::<f32>(&[3, 4], &[1, 0])?;
        run_test_case::<f32>(&[1, 2, 3, 4, 5], &[4, 1, 2, 3, 0])?;
        Ok(())
    }
}
