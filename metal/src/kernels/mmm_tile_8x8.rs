use crate::func_constants::{ConstantValues, Value};
use crate::MetalTensor;
use crate::{LibraryName, MetalContext};
use anyhow::{ensure, Result};
use metal::{Buffer, MTLSize, NSUInteger};
use tract_core::internal::DatumType;

pub fn mmm_tile_8x8(
    context: &MetalContext,
    lhs: &MetalTensor,
    rhs: &MetalTensor,
) -> Result<MetalTensor> {
    ensure!(lhs.rank() == 2 && rhs.rank() == 2);
    ensure!(lhs.datum_type() == rhs.datum_type());
    ensure!(lhs.datum_type() == DatumType::F32);

    let m = lhs.shape()[0];
    let n = rhs.shape()[1];
    let k = lhs.shape()[1];

    ensure!(m == n && m == k);

    let o_dt = lhs.datum_type();
    let o_shape = &[m, m];

    let output = MetalTensor::zero_dt(o_dt, o_shape)?;

    crate::kernels::metal_mmm_tile_8x8(context, m, &lhs.metal(), &rhs.metal(), output.metal())?;
    context.wait_until_completed()?;

    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub fn metal_mmm_tile_8x8(
    context: &MetalContext,
    dim: usize,
    lhs_buffer: &Buffer,
    rhs_buffer: &Buffer,
    output: &Buffer,
) -> Result<()> {
    ensure!(dim % 8 == 0, "Dim must be a multiple of 8");

    let constants = Some(ConstantValues::new(vec![(0, Value::USize(2)), (1, Value::USize(dim))]));
    let pipeline = context.shared_context().load_pipeline_with_constants(
        LibraryName::MmmTile8x8,
        "mmm_tile_8x8",
        constants,
    )?;

    let command_buffer = context.command_buffer()?;
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(output), 0);
    encoder.set_buffer(1, Some(lhs_buffer), 0);
    encoder.set_buffer(2, Some(rhs_buffer), 0);

    let grid_size = MTLSize {
        width: crate::utils::div_ceil(dim, 8 * 4),
        height: crate::utils::div_ceil(dim, 8 * 4 * 2),
        depth: 1 as NSUInteger,
    };
    let group_size = MTLSize { width: 32, height: 2, depth: 1 };
    encoder.use_resource(lhs_buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(rhs_buffer, metal::MTLResourceUsage::Read);
    encoder.use_resource(output, metal::MTLResourceUsage::Write);
    encoder.dispatch_thread_groups(grid_size, group_size);
    encoder.end_encoding();

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::IntoMetal;
    use tract_core::internal::Tensor;

    #[test]
    fn test_mmm_tile_8x8() -> Result<()> {
        objc::rc::autoreleasepool(|| {
            crate::METAL_CONTEXT.with_borrow(|context| {
                let n = 512;

                let constants =
                    Some(ConstantValues::new(vec![(0, Value::USize(2)), (1, Value::USize(n))]));

                context.shared_context().load_pipeline_with_constants(
                    LibraryName::MmmTile8x8,
                    "mmm_tile_8x8",
                    constants,
                )?;
                context.wait_until_completed()?;

                let mut cpu_start = 0;
                let mut gpu_start = 0;
                context.device().sample_timestamps(&mut cpu_start, &mut gpu_start);

                let a = Tensor::from_shape(
                    &[n, n],
                    &(0..n * n).map(|_f| 1 as f32).collect::<Vec<_>>(),
                )?
                .into_metal()?;
                let b = Tensor::from_shape(
                    &[n, n],
                    &(0..n * n).map(|_f| 1 as f32).collect::<Vec<_>>(),
                )?
                .into_metal()?;
                let start = std::time::Instant::now();
                let num_iter = 100;
                for _ in 0..num_iter {
                    let _c = mmm_tile_8x8(&context, &a, &b)?;
                }

                let mut cpu_end = 0;
                let mut gpu_end = 0;
                context.device().sample_timestamps(&mut cpu_end, &mut gpu_end);

                dbg!(start.elapsed().as_secs_f32() / num_iter as f32);
                println!(
                    "{:3?} GOP/s",
                    (n * n * n * 2 * num_iter) as f32 / start.elapsed().as_secs_f32() / 10.0e9
                );
                Ok(())
            })
        })
    }
}
