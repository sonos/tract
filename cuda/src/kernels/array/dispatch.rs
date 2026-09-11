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
        // More of the axes beyond the innermost two than one such dimension
        // holds, so they take two, the second of which runs past them.
        run_copy_case(&[70000, 3, 5], &[70000, 3, 5], &[70000, 3, 5])?;
        run_copy_case(&[260, 256, 65, 17], &[260, 256, 65, 17], &[260, 256, 65, 16])?;
        run_copy_case(&[70001, 2, 3, 4], &[70001, 2, 3, 5], &[70001, 2, 3, 4])?;
        // The same, with a row a block does not hold, which addresses its rows
        // through x rather than packing them with the threads.
        run_copy_case(&[70000, 2, 300], &[70000, 2, 300], &[70000, 2, 300])?;
        Ok(())
    }

    /// Every shape a copy can be asked for has to land inside what the device
    /// takes, and a launch the driver refuses is the whole failure mode here, so
    /// this walks the shapes rather than naming them.
    #[test]
    fn test_copy_launch_cfg_within_device_limits() -> TractResult<()> {
        let props = cuda_context().properties();
        let max_grid = props.maxGridSize;
        let max_threads = props.maxThreadsPerBlock as usize;
        // Extents on both sides of every threshold the helper and the driver
        // carry: a block's threads, a warp, and a grid dimension's 65535.
        const EXTENTS: [usize; 13] =
            [1, 2, 3, 17, 31, 32, 255, 256, 257, 1024, 65535, 65536, 70000];
        // Past this the tensor is larger than any device holds, and the helper
        // is never handed one. The axes beyond the innermost two are bounded
        // separately, the helper refusing what two grid dimensions cannot hold.
        const MAX_ELEMS: usize = 1 << 33;
        const MAX_OUTER: usize = 65535 * 65535;
        let mut checked = 0usize;
        let mut shape: TVec<usize> = tvec!();
        let mut walk = |shape: &TVec<usize>, checked: &mut usize| -> TractResult<()> {
            let rank = shape.len();
            if rank > 2 && shape[..rank - 2].iter().product::<usize>() > MAX_OUTER {
                return Ok(());
            }
            let cfg = cuda_launch_cfg_for_cpy(shape);
            let grid = [cfg.grid_dim.0, cfg.grid_dim.1, cfg.grid_dim.2];
            for (axis, (asked, allowed)) in grid.iter().zip(max_grid.iter()).enumerate() {
                ensure!(
                    *asked >= 1 && (*asked as i64) <= *allowed as i64,
                    "{shape:?} asks for {asked} blocks on the grid's axis {axis}, of {allowed}"
                );
            }
            let block =
                cfg.block_dim.0 as usize * cfg.block_dim.1 as usize * cfg.block_dim.2 as usize;
            ensure!(
                block >= 1 && block <= max_threads,
                "{shape:?} asks for {block} threads a block, of {max_threads}"
            );
            // Every element of the copy has to be reachable: a block covers
            // whole rows of the innermost axis, so the grid has to span the rest.
            let covered = if rank == 1 {
                cfg.grid_dim.0 as usize * cfg.block_dim.0 as usize
            } else {
                let inner = shape[rank - 1];
                let rows = (cfg.block_dim.0 as usize / inner).max(1);
                cfg.grid_dim.0 as usize * rows
            };
            ensure!(
                covered >= if rank == 1 { shape[0] } else { shape[rank - 2] },
                "{shape:?} leaves rows past the {covered} the grid covers"
            );
            if rank > 2 {
                let outer: usize = shape[..rank - 2].iter().product();
                let packed = cfg.grid_dim.1 as usize * cfg.grid_dim.2 as usize;
                ensure!(packed >= outer, "{shape:?} leaves {outer} axes over {packed} blocks");
            }
            *checked += 1;
            Ok(())
        };
        fn extend(
            shape: &mut TVec<usize>,
            elems: usize,
            checked: &mut usize,
            walk: &mut impl FnMut(&TVec<usize>, &mut usize) -> TractResult<()>,
        ) -> TractResult<()> {
            walk(shape, checked)?;
            if shape.len() == 6 {
                return Ok(());
            }
            for extent in EXTENTS {
                let Some(elems) = elems.checked_mul(extent).filter(|e| *e <= MAX_ELEMS) else {
                    continue;
                };
                shape.push(extent);
                extend(shape, elems, checked, walk)?;
                shape.pop();
            }
            Ok(())
        }
        for extent in EXTENTS {
            shape.push(extent);
            extend(&mut shape, extent, &mut checked, &mut walk)?;
            shape.pop();
        }
        assert!(checked > 10_000, "only {checked} shapes walked");
        Ok(())
    }
}
