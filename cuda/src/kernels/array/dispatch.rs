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
