use crate::context::cuda_context;
use crate::kernels::launch_args::TractLaunchArgs;
use crate::kernels::utils::cuda_launch_cfg_for_cpy;
use crate::kernels::{BroadcastKind, LibraryName, get_sliced_cuda_view};
use cudarc::driver::PushKernelArg;
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

/// Single dispatch function for all copy_nd kernel launches.
/// Used by GpuMultiBroadcastTo, GpuSlice, GpuConcat, and GpuAxisOp.
pub fn cuda_copy_nd_dispatch(
    input: &DeviceTensor,
    input_offset: usize,
    input_strides: &[isize],
    output: &DeviceTensor,
    output_offset: usize,
    output_shape: &[usize],
    output_strides: &[isize],
) -> TractResult<()> {
    if output_shape.contains(&0) {
        return Ok(());
    }
    crate::with_cuda_stream(|stream| {
        let kernel_name = BroadcastKind::from_rank(output_shape.len())?
            .copy_kernel_name(input.datum_type(), "")?;
        let func = cuda_context().load_pipeline(LibraryName::Array, kernel_name)?;

        let i_view = get_sliced_cuda_view(
            input,
            input_offset,
            input.len() * input.datum_type().size_of() - input_offset,
        )?;
        let o_view = get_sliced_cuda_view(
            output,
            output_offset,
            output.len() * output.datum_type().size_of() - output_offset,
        )?;

        let mut launch_args = TractLaunchArgs::new(stream, &func);
        launch_args.push_view(&i_view);
        launch_args.push_view(&o_view);
        launch_args.push_slice_i32(input_strides);
        launch_args.push_slice_i32(output_shape);
        launch_args.push_slice_i32(output_strides);

        let cfg = cuda_launch_cfg_for_cpy(output_shape);
        launch_args.launch(cfg)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use tract_gpu::tensor::IntoDevice;

    /// Copy a zone of `shape` out of a `src` tensor into a `dst` tensor, both
    /// held in their natural layout, and compare against the same copy done on
    /// host. The zone starts at the origin of both, so a shorter zone than the
    /// tensor leaves the copy strided.
    fn run_copy_case(src_shape: &[usize], dst_shape: &[usize], zone: &[usize]) -> TractResult<()> {
        let src = Tensor::from_shape(
            src_shape,
            &(0..src_shape.iter().product::<usize>()).map(|i| i as f32).collect::<TVec<_>>(),
        )?;
        let mut expected = Tensor::zero::<f32>(dst_shape)?;
        let mut dst_view = expected.to_plain_array_view_mut::<f32>()?;
        let src_view = src.to_plain_array_view::<f32>()?;
        let zone_ix: TVec<usize> = zone.into();
        dst_view
            .slice_each_axis_mut(|a| (0..zone_ix[a.axis.index()] as isize).into())
            .assign(&src_view.slice_each_axis(|a| (0..zone_ix[a.axis.index()] as isize).into()));

        let found = crate::with_cuda_stream(|stream| {
            let src = src.into_device()?;
            let dst = Tensor::zero::<f32>(dst_shape)?.into_device()?;
            cuda_copy_nd_dispatch(&src, 0, src.strides(), &dst, 0, zone, dst.strides())?;
            stream.synchronize()?;
            Ok(dst.to_host()?.into_tensor())
        })?;
        found
            .close_enough(&expected, false)
            .with_context(|| format!("src {src_shape:?} dst {dst_shape:?} zone {zone:?}"))
    }

    #[test]
    fn test_copy_nd() -> TractResult<()> {
        // A row wider than a block, a row per block, and the short rows a
        // pulsed window retires a frame through.
        run_copy_case(&[3000], &[3000], &[3000])?;
        run_copy_case(&[5, 7], &[5, 9], &[5, 7])?;
        run_copy_case(&[4, 1025], &[4, 1025], &[4, 1025])?;
        run_copy_case(&[2, 1024], &[2, 1024], &[2, 1024])?;
        run_copy_case(&[6, 33, 11], &[6, 33, 12], &[6, 33, 11])?;
        run_copy_case(&[43, 1024, 11], &[43, 1024, 12], &[43, 1024, 11])?;
        run_copy_case(&[3, 5, 65, 16], &[3, 5, 65, 17], &[3, 5, 65, 16])?;
        run_copy_case(&[2, 3, 4, 8, 128], &[2, 3, 4, 8, 128], &[2, 3, 4, 8, 128])?;
        run_copy_case(&[2, 3, 4, 5, 6, 7], &[2, 3, 4, 5, 6, 9], &[2, 3, 4, 5, 6, 7])?;
        run_copy_case(&[2, 3, 4, 5, 6, 7], &[2, 3, 4, 5, 6, 7], &[1, 2, 3, 4, 5, 6])?;
        run_copy_case(&[4, 5], &[4, 5], &[0, 5])?;
        // More rows than a grid dimension other than x holds.
        run_copy_case(&[70000, 300], &[70000, 300], &[70000, 300])?;
        run_copy_case(&[70000, 8], &[70000, 9], &[70000, 8])?;
        Ok(())
    }
}
