use crate::encoder::EncoderExt;
use crate::kernels::utils::build_metal_grid_and_groups_for_el_wise_op;
use crate::{LibraryName, MetalStream};
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;
use tract_gpu::utils::BroadcastKind;

// --- Inventory registrations for all ops that use metal_copy_nd_dispatch ---

crate::register_metal_op!(tract_core::ops::array::MultiBroadcastTo, |_source, _node, op| {
    Ok(Some(Box::new(tract_gpu::ops::broadcast::GpuMultiBroadcastTo::new(
        op.shape.clone(),
        "Metal",
        metal_copy_nd_dispatch,
    ))))
});

crate::register_metal_op!(AxisOp, |source, node, op| {
    let in_fact = source.node_input_facts(node.id)?[0];
    Ok(Some(Box::new(tract_gpu::ops::change_axes::GpuAxisOp::from_tract_core_with_fact(
        op.clone(),
        in_fact,
        "Metal",
        metal_copy_nd_dispatch,
    ))))
});

crate::register_metal_op!(tract_core::ops::array::Slice, |_source, _node, op| {
    Ok(Some(Box::new(tract_gpu::ops::slice::GpuSlice::new(
        op.clone(),
        "Metal",
        metal_copy_nd_dispatch,
    ))))
});

crate::register_metal_op!(tract_core::ops::array::TypedConcat, |_source, _node, op| {
    Ok(Some(Box::new(tract_gpu::ops::concat::GpuConcat::new(
        op.axis,
        "Metal",
        metal_copy_nd_dispatch,
    ))))
});

crate::register_metal_op!(
    tract_transformers::ops::dyn_kv_cache::DynKeyValueCache,
    |_source, _node, op| {
        Ok(Some(Box::new(tract_gpu::ops::dyn_kv_cache::GpuDynKVCache::from_tract_transformers(
            op,
            "Metal",
            metal_copy_nd_dispatch,
        ))))
    }
);

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
