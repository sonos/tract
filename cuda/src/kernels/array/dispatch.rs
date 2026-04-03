use crate::context::cuda_context;
use crate::kernels::launch_args::TractLaunchArgs;
use crate::kernels::utils::cuda_launch_cfg_for_cpy;
use crate::kernels::{BroadcastKind, LibraryName, get_sliced_cuda_view};
use cudarc::driver::PushKernelArg;
use tract_core::internal::*;
use tract_gpu::tensor::DeviceTensor;

// --- Inventory registrations for all ops that use cuda_copy_nd_dispatch ---

crate::register_cuda_op!(tract_core::ops::array::MultiBroadcastTo, |_source, _node, op| {
    Ok(Some(Box::new(tract_gpu::ops::broadcast::GpuMultiBroadcastTo::new(
        op.shape.clone(),
        "Cuda",
        cuda_copy_nd_dispatch,
    ))))
});

crate::register_cuda_op!(AxisOp, |source, node, op| {
    let in_fact = source.node_input_facts(node.id)?[0];
    Ok(Some(Box::new(tract_gpu::ops::change_axes::GpuAxisOp::from_tract_core_with_fact(
        op.clone(),
        in_fact,
        "Cuda",
        cuda_copy_nd_dispatch,
    ))))
});

crate::register_cuda_op!(tract_core::ops::array::Slice, |_source, _node, op| {
    Ok(Some(Box::new(tract_gpu::ops::slice::GpuSlice::new(
        op.clone(),
        "Cuda",
        cuda_copy_nd_dispatch,
    ))))
});

crate::register_cuda_op!(tract_core::ops::array::TypedConcat, |_source, _node, op| {
    Ok(Some(Box::new(tract_gpu::ops::concat::GpuConcat::new(
        op.axis,
        "Cuda",
        cuda_copy_nd_dispatch,
    ))))
});

crate::register_cuda_op!(
    tract_transformers::ops::dyn_kv_cache::DynKeyValueCache,
    |_source, _node, op| {
        Ok(Some(Box::new(tract_gpu::ops::dyn_kv_cache::GpuDynKVCache::from_tract_transformers(
            op,
            "Cuda",
            cuda_copy_nd_dispatch,
        ))))
    }
);

crate::register_cuda_op!(tract_pulse_opl::ops::Delay, |_source, _node, op| {
    Ok(Some(Box::new(tract_gpu::ops::pulse::GpuDelay::new(op, "Cuda", cuda_copy_nd_dispatch))))
});

crate::register_cuda_op!(tract_pulse_opl::ops::PulsePad, |_source, _node, op| {
    Ok(Some(Box::new(tract_gpu::ops::pulse::GpuPulsePad::new(op, "Cuda", cuda_copy_nd_dispatch)?)))
});

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
