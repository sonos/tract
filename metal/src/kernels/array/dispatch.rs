use crate::encoder::EncoderExt;
use crate::{LibraryName, MetalStream};
use metal::MTLSize;
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;
use tract_gpu::utils::BroadcastKind;
use tract_gpu::utils::COPY_THREADS;
use tract_gpu::utils::TRANSPOSE_ROWS;
use tract_gpu::utils::TRANSPOSE_TILE;
use tract_gpu::utils::copy_launch;
use tract_gpu::utils::merge_copy_axes;
use tract_gpu::utils::transpose_launch;

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
    if output_shape.contains(&0) {
        return Ok(());
    }
    let geo = merge_copy_axes(input_strides, output_shape, output_strides);
    let (output_shape, input_strides, output_strides) =
        (&geo.shape[..], &geo.input_strides[..], &geo.output_strides[..]);
    crate::with_metal_stream(|stream| {
        stream.retain_tensor(input);
        stream.retain_tensor(output);

        if let Some(t) = transpose_launch(input_strides, output_shape, output_strides) {
            let pipeline = stream.load_pipeline(
                LibraryName::ArrayOps,
                &BroadcastKind::transpose_kernel_name(input.datum_type(), "array_ops::"),
            )?;
            let params = [
                t.in_batch_stride,
                t.in_col_stride,
                t.out_batch_stride,
                t.out_row_stride,
                t.rows as isize,
                t.cols as isize,
            ]
            .map(|x| x as i32);
            stream.command_buffer().encode(|encoder| {
                encoder.set_compute_pipeline_state(&pipeline);
                encoder.set_metal_tensor_with_offset(
                    0,
                    input,
                    input_offset as _,
                    metal::MTLResourceUsage::Read,
                );
                encoder.set_metal_tensor_with_offset(
                    1,
                    output,
                    output_offset as _,
                    metal::MTLResourceUsage::Write,
                );
                encoder.set_slice(2, &params);
                let [width, height, depth] = t.groups.map(|g| g as u64);
                encoder.dispatch_thread_groups(
                    MTLSize { width, height, depth },
                    MTLSize { width: TRANSPOSE_TILE as _, height: TRANSPOSE_ROWS as _, depth: 1 },
                );
            });
            return Ok(());
        }

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

            let launch = copy_launch(
                output_shape,
                COPY_THREADS.min(pipeline.max_total_threads_per_threadgroup() as usize),
            );
            let [width, height, depth] = launch.groups.map(|g| g as u64);
            encoder.dispatch_thread_groups(
                MTLSize { width, height, depth },
                MTLSize { width: launch.threads as _, height: 1, depth: 1 },
            );
        });
        Ok(())
    })
}
