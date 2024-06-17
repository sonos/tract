use crate::MetalTensor;
use crate::{LibraryName, MetalContext};
use anyhow::bail;
use anyhow::{ensure, Result};
use metal::{Buffer, MTLSize, NSUInteger};
use num_traits::Float;
use tract_core::internal::*;

pub fn mat_vec_with_slice<T: Datum + Float>(
    context: &MetalContext,
    (m, k, n): (usize, usize, usize),
    lhs: &[T],
    rhs: &[T],
    output: &mut [T],
) -> Result<()> {
    ensure!(T::datum_type() == DatumType::F32);

    ensure!(n == 1 || m == 1);

    let lhs_buff = context.buffer_from_slice(lhs);
    let rhs_buff = context.buffer_from_slice(rhs);
    let out_buff = context.buffer_from_slice_mut(output);

    if n == 1 {
        metal_mat_vec(context, m, k, &lhs_buff, &rhs_buff, &out_buff)?;
    } else if m == 1 {
        metal_mat_vec(context, n, k, &lhs_buff, &rhs_buff, &out_buff)?;
    } else {
        bail!("Incompatible shape for MatVec {:?}, {:?}", &[m, k], &[k, n]);
    }
    context.wait_until_completed()?;
    Ok(())
}

pub fn mat_vec(
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

    ensure!(n == 1 || m == 1);

    let o_dt = lhs.datum_type();
    let o_shape = &[m, n];

    let output = MetalTensor::zero_dt(o_dt, o_shape)?;

    if n == 1 {
        metal_mat_vec(context, m, k, lhs.metal(), rhs.metal(), output.metal())?;
    } else if m == 1 {
        metal_mat_vec(context, n, k, rhs.metal(), lhs.metal(), output.metal())?;
    } else {
        bail!("Incompatible shape for MatVec {:?}, {:?}", lhs.shape(), rhs.shape());
    }
    context.wait_until_completed()?;

    Ok(output)
}

#[allow(clippy::too_many_arguments)]
pub fn metal_mat_vec(
    context: &MetalContext,
    nrows: usize,
    ncols: usize,
    lhs_buffer: &Buffer,
    rhs_buffer: &Buffer,
    output: &Buffer,
) -> Result<()> {
    let pipeline =
        context.shared_context().load_pipeline(LibraryName::MulMatVec, "op_mat_vec_f32")?;

    let command_buffer = context.command_buffer()?;
    let encoder = command_buffer.new_compute_command_encoder();
    encoder.set_compute_pipeline_state(&pipeline);
    encoder.set_buffer(0, Some(lhs_buffer), 0);
    encoder.set_buffer(1, Some(rhs_buffer), 0);
    encoder.set_buffer(2, Some(output), 0);
    encoder.set_bytes(3, 4, &(nrows as i32) as *const i32 as *const _);
    encoder.set_bytes(4, 4, &(ncols as i32) as *const i32 as *const _);

    let grid_size =
        MTLSize { width: 1, height: crate::utils::div_ceil(nrows, 4), depth: 1 as NSUInteger };
    let group_size = MTLSize { width: 32, height: 1, depth: 1 };
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
    fn test_mat_vec() -> Result<()> {
        objc::rc::autoreleasepool(|| {
            crate::METAL_CONTEXT.with_borrow(|context| {
                let m = 4;
                let k = 4;

                let a =
                    Tensor::from_shape(&[m, k], &(0..m * k).map(|_f| 1_f32).collect::<Vec<_>>())?
                        .into_metal()?;
                let b = Tensor::from_shape(&[k, 1], &(0..k).map(|_f| 1_f32).collect::<Vec<_>>())?
                    .into_metal()?;
                dbg!(mat_vec(context, &a, &b)?);
                Ok(())
            })
        })
    }

    #[test]
    fn test_mat_vec_v2() -> Result<()> {
        objc::rc::autoreleasepool(|| {
            crate::METAL_CONTEXT.with_borrow(|context| {
                let m = 4;
                let k = 100;

                let a =
                    Tensor::from_shape(&[m, k], &(0..m * k).map(|_f| 2_f32).collect::<Vec<_>>())?
                        .into_metal()?;
                let b = Tensor::from_shape(&[k, 1], &(0..k).map(|_f| 1_f32).collect::<Vec<_>>())?
                    .into_metal()?;
                dbg!(mat_vec(context, &a, &b)?);
                Ok(())
            })
        })
    }
}
