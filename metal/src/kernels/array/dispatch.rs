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
    crate::with_metal_stream(|stream| {
        stream.retain_tensor(input);
        stream.retain_tensor(output);

        let broadcast_kind = BroadcastKind::from_rank(output_shape.len())?;
        let tname = DeviceTensor::tname(input.datum_type())?;
        let kernel_name = format!("array_ops::copy_{}_{tname}", broadcast_kind.name());

        let pipeline = stream.load_pipeline(LibraryName::ArrayOps, &kernel_name)?;
        let command_buffer = stream.command_buffer();

        // Convert isize strides to usize for Metal buffers
        let input_strides_usize: TVec<usize> =
            input_strides.iter().map(|&s| s as usize).collect();
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
