use crate::encoder::EncoderExt;
use crate::kernels::utils::build_metal_grid_and_groups_for_el_wise_op;
use crate::{LibraryName, MetalStream};
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;
use tract_gpu::utils::BroadcastKind;

/// Single dispatch function for all copy_nd kernel launches.
/// Used by GpuMultiBroadcastTo, GpuSlice, GpuConcat, and GpuAxisOp.
pub fn metal_copy_nd_dispatch(
    input: &DeviceTensor,
    input_offset: usize,
    input_strides: &[isize],
    output: &DeviceTensor,
    output_offset: usize,
    output_shape: &[usize],
    output_strides: &[isize],
) -> TractResult<()> {
    // The copy_nd kernels index the output innermost axis directly
    // (`output[out_idx + i]`): only the input side is fully strided. Callers
    // writing to a transposed/strided layout must permute the copy so the
    // output's contiguous axis comes last.
    ensure!(
        output_shape.last().is_none_or(|d| *d <= 1) || output_strides.last() == Some(&1),
        "copy_nd requires a contiguous innermost output axis, got strides {output_strides:?}"
    );
    if std::env::var_os("TRACT_METAL_LOG_COPY_ND").is_some() {
        eprintln!(
            "copy-nd shape={output_shape:?} in_strides={input_strides:?} out_strides={output_strides:?}"
        );
    }
    if let Some(plan) =
        transpose2d_plan(output_shape, input_strides, output_strides)
    {
        return dispatch_copy_transpose2d(input, input_offset, output, output_offset, &plan);
    }

    crate::with_metal_stream(|stream| {
        stream.retain_tensor(input);
        stream.retain_tensor(output);

        let kernel_name = BroadcastKind::from_rank(output_shape.len())?
            .copy_kernel_name(input.datum_type(), "array_ops::")?;

        let pipeline = stream.load_pipeline(LibraryName::ArrayOps, &kernel_name)?;
        let command_buffer = stream.command_buffer();

        // Convert isize strides to usize for Metal buffers
        let input_strides_usize: TVec<usize> = input_strides.iter().map(|&s| s as usize).collect();
        let output_strides_usize: TVec<usize> =
            output_strides.iter().map(|&s| s as usize).collect();

        command_buffer.encode(|encoder| {
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_metal_tensor_with_offset(
                0,
                input,
                input_offset as _,
                metal::MTLResourceUsage::Read,
            );
            encoder.set_slice(1, &input_strides_usize);
            encoder.set_metal_tensor_with_offset(
                2,
                output,
                output_offset as _,
                metal::MTLResourceUsage::Write,
            );
            encoder.set_slice(3, output_shape);
            encoder.set_slice(4, &output_strides_usize);

            let (grid_size, group_size) = build_metal_grid_and_groups_for_el_wise_op(
                output_shape,
                pipeline.max_total_threads_per_threadgroup() as _,
            );
            encoder.dispatch_thread_groups(grid_size, group_size);
        });
        Ok(())
    })
}

/// Kernel arguments for the tiled 2D-transpose copy fast path. See
/// `copy_transpose2d` in array_ops.metal.
struct Transpose2dPlan {
    /// [m, n, in_stride_n, out_stride_m, n_batch0, in_b0, out_b0, in_b1, out_b1]
    args: [usize; 9],
    batch: usize,
}

/// Detects the copies the generic copy_nd kernels handle pathologically: a
/// dense row-major output whose innermost axis is strided on the input while
/// another axis is input-contiguous (a 2D transpose, possibly batched). Both
/// axes must be large enough for the 32x32 tiling to win.
fn transpose2d_plan(
    output_shape: &[usize],
    input_strides: &[isize],
    output_strides: &[isize],
) -> Option<Transpose2dPlan> {
    let rank = output_shape.len();
    if rank < 2 {
        return None;
    }
    // Dense row-major output.
    let mut expected: isize = 1;
    for i in (0..rank).rev() {
        if output_shape[i] > 1 && output_strides[i] != expected {
            return None;
        }
        expected *= output_shape[i] as isize;
    }
    let n = output_shape[rank - 1];
    let in_stride_n = input_strides[rank - 1];
    if in_stride_n <= 1 || n < 16 {
        return None;
    }
    // The input-contiguous axis (innermost such wins).
    let a = (0..rank - 1)
        .rev()
        .find(|&i| output_shape[i] > 1 && input_strides[i] == 1)?;
    let m = output_shape[a];
    if m < 16 {
        return None;
    }
    // Everything else is batch: at most two non-unit axes.
    let mut batches: Vec<(usize, isize, isize)> = Vec::new();
    for i in 0..rank - 1 {
        if i == a || output_shape[i] == 1 {
            continue;
        }
        if input_strides[i] < 0 || output_strides[i] < 0 {
            return None;
        }
        batches.push((output_shape[i], input_strides[i], output_strides[i]));
        if batches.len() > 2 {
            return None;
        }
    }
    let (b0, b1) = match batches.len() {
        0 => ((1usize, 0isize, 0isize), (1usize, 0isize, 0isize)),
        1 => (batches[0], (1usize, 0isize, 0isize)),
        _ => (batches[0], batches[1]),
    };
    Some(Transpose2dPlan {
        args: [
            m,
            n,
            in_stride_n as usize,
            output_strides[a] as usize,
            b0.0,
            b0.1 as usize,
            b0.2 as usize,
            b1.1 as usize,
            b1.2 as usize,
        ],
        batch: b0.0 * b1.0,
    })
}

pub(crate) fn dispatch_copy_transpose2d(
    input: &DeviceTensor,
    input_offset: usize,
    output: &DeviceTensor,
    output_offset: usize,
    plan: &Transpose2dPlan,
) -> TractResult<()> {
    crate::with_metal_stream(|stream| {
        stream.retain_tensor(input);
        stream.retain_tensor(output);

        let tname = BroadcastKind::copy_tname(input.datum_type());
        let kernel_name = format!("array_ops::copy_transpose2d_{tname}");
        let pipeline = stream.load_pipeline(LibraryName::ArrayOps, &kernel_name)?;
        let command_buffer = stream.command_buffer();
        command_buffer.encode(|encoder| {
            encoder.set_compute_pipeline_state(&pipeline);
            encoder.set_metal_tensor_with_offset(
                0,
                input,
                input_offset as _,
                metal::MTLResourceUsage::Read,
            );
            encoder.set_slice(1, &plan.args);
            encoder.set_metal_tensor_with_offset(
                2,
                output,
                output_offset as _,
                metal::MTLResourceUsage::Write,
            );
            let grid = metal::MTLSize {
                width: plan.args[0].div_ceil(32) as u64,
                height: plan.args[1].div_ceil(32) as u64,
                depth: plan.batch as u64,
            };
            let group = metal::MTLSize { width: 32, height: 8, depth: 1 };
            encoder.dispatch_thread_groups(grid, group);
        });
        Ok(())
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::utils::with_borrowed_metal_stream;
    use tract_gpu::tensor::IntoDevice;

    /// The transpose fast path must produce exactly what the generic
    /// strided copy produces: exercise [B, C, S] -> [B, S, C]-style copies
    /// (batched, ragged tiles, with and without a channel slice).
    fn run_transpose_case(batch: usize, c: usize, s: usize) -> TractResult<()> {
        with_borrowed_metal_stream(|_stream| {
            let len = batch * c * s;
            let input =
                Tensor::from_shape(&[batch, c, s], &(0..len).map(|x| x as f32).collect::<Vec<_>>())?;
            let device_in = input.clone().into_device()?;

            // CPU reference: permute axes (0, 2, 1).
            let reference = input.clone().permute_axes(&[0, 2, 1])?;

            let out_shape = [batch, s, c];
            let output = unsafe {
                tract_gpu::tensor::DeviceTensor::uninitialized_dt(f32::datum_type(), &out_shape)?
            };
            // input strides for reading in output order [b, s, c]:
            let in_strides: [isize; 3] = [(c * s) as isize, 1, s as isize];
            let out_strides: [isize; 3] = [(s * c) as isize, c as isize, 1];
            let plan = transpose2d_plan(&out_shape, &in_strides, &out_strides)
                .context("expected the transpose fast path to match")?;
            dispatch_copy_transpose2d(&device_in, 0, &output, 0, &plan)?;
            let result = output.to_host()?.into_tensor();
            result.close_enough(&reference, Approximation::Exact)?;
            Ok(())
        })
    }

    #[test]
    fn test_copy_transpose2d_matches_generic() -> TractResult<()> {
        run_transpose_case(1, 64, 64)?;
        run_transpose_case(1, 2048, 512)?;
        run_transpose_case(3, 100, 37)?;
        run_transpose_case(2, 33, 65)?;
        Ok(())
    }

    /// Small or degenerate copies must NOT take the fast path.
    #[test]
    fn test_transpose2d_plan_detection() {
        // Canonical transpose: matches.
        assert!(transpose2d_plan(&[1, 512, 2048], &[0, 1, 512], &[0, 2048, 1]).is_some());
        // Contiguous innermost input: generic path is already coalesced.
        assert!(transpose2d_plan(&[1, 512, 2048], &[0, 2048, 1], &[0, 2048, 1]).is_none());
        // Too small to tile.
        assert!(transpose2d_plan(&[1, 8, 8], &[0, 1, 8], &[0, 8, 1]).is_none());
        // Non-dense output.
        assert!(transpose2d_plan(&[1, 512, 2048], &[0, 1, 512], &[0, 4096, 1]).is_none());
    }
}

#[cfg(test)]
mod bench_tests {
    use super::*;
    use crate::utils::with_borrowed_metal_stream;
    use tract_gpu::tensor::IntoDevice;

    #[test]
    #[ignore]
    fn bench_transpose_kernels() -> TractResult<()> {
        with_borrowed_metal_stream(|stream| {
            let (b, c, s) = (16usize, 2048usize, 512usize);
            let len = b * c * s;
            let input = Tensor::from_shape(
                &[b, c, s],
                &(0..len).map(|x| x as f32 as f32).collect::<Vec<_>>(),
            )?
            .cast_to::<f16>()?
            .into_owned();
            let device_in = input.clone().into_device()?;
            let out_shape = [b, s, c];
            let output = unsafe {
                tract_gpu::tensor::DeviceTensor::uninitialized_dt(f16::datum_type(), &out_shape)?
            };
            let in_strides: [isize; 3] = [(c * s) as isize, 1, s as isize];
            let out_strides: [isize; 3] = [(s * c) as isize, c as isize, 1];
            // warm
            for _ in 0..3 {
                metal_copy_nd_dispatch(&device_in, 0, &in_strides, &output, 0, &out_shape, &out_strides)?;
            }
            stream.wait_until_completed()?;
            let t = std::time::Instant::now();
            for _ in 0..200 {
                metal_copy_nd_dispatch(&device_in, 0, &in_strides, &output, 0, &out_shape, &out_strides)?;
            }
            stream.wait_until_completed()?;
            let dt = t.elapsed().as_secs_f64() / 200.0;
            eprintln!(
                "fast path: {:.3} ms -> {:.1} GB/s",
                dt * 1000.0,
                (len * 2 * 2) as f64 / dt / 1e9
            );
            // Generic strided kernel on the same copy, for the record.
            let generic = |input: &DeviceTensor, output: &DeviceTensor| -> TractResult<()> {
                crate::with_metal_stream(|stream| {
                    let kernel_name = BroadcastKind::from_rank(3)?
                        .copy_kernel_name(input.datum_type(), "array_ops::")?;
                    let pipeline = stream.load_pipeline(LibraryName::ArrayOps, &kernel_name)?;
                    let command_buffer = stream.command_buffer();
                    let in_s: TVec<usize> = in_strides.iter().map(|&s| s as usize).collect();
                    let out_s: TVec<usize> = out_strides.iter().map(|&s| s as usize).collect();
                    command_buffer.encode(|encoder| {
                        encoder.set_compute_pipeline_state(&pipeline);
                        encoder.set_metal_tensor_with_offset(0, input, 0, metal::MTLResourceUsage::Read);
                        encoder.set_slice(1, &in_s);
                        encoder.set_metal_tensor_with_offset(2, output, 0, metal::MTLResourceUsage::Write);
                        encoder.set_slice(3, &out_shape);
                        encoder.set_slice(4, &out_s);
                        let (grid_size, group_size) = build_metal_grid_and_groups_for_el_wise_op(
                            &out_shape,
                            pipeline.max_total_threads_per_threadgroup() as _,
                        );
                        encoder.dispatch_thread_groups(grid_size, group_size);
                    });
                    Ok(())
                })
            };
            for _ in 0..3 { generic(&device_in, &output)?; }
            stream.wait_until_completed()?;
            let t = std::time::Instant::now();
            for _ in 0..200 { generic(&device_in, &output)?; }
            stream.wait_until_completed()?;
            let dt = t.elapsed().as_secs_f64() / 200.0;
            eprintln!(
                "generic  : {:.3} ms -> {:.1} GB/s",
                dt * 1000.0,
                (len * 2 * 2) as f64 / dt / 1e9
            );
            // force generic: bump n below the 16 threshold? instead call generic by using plan=None path:
            // temporarily emulate by calling with a non-dense output? Just measure the old kernel via
            // direct pipeline name.
            Ok(())
        })
    }
}
