use crate::context::CUDA_STREAM;
use crate::kernels::nn::CudaReducer;
use tract_core::internal::*;
use tract_gpu::ops::reduce::{DispatchReduceFn, GpuReduce, GpuStream, Reducer};
use tract_gpu::tensor::{DeviceTensor, DeviceTensorExt};

// impl GpuStream for crate::context::TractCudaStream {}

// fn with_stream(
//     dispatch: DispatchReduceFn,
//     reducer: &Reducer,
//     axes: &[usize],
//     node_id: usize,
//     session: &TurnState,
//     inputs: TVec<TValue>,
// ) -> TractResult<TVec<TValue>> {
//     CUDA_STREAM.with(|stream| {
//         let input_value = args_1!(inputs);
//         let input = input_value.to_device_tensor()?;
//         let mut output_shape = input.shape().to_vec();
//         output_shape[axes[0]] = 1;
//         let output = tract_gpu::session_handler::make_tensor_for_node(
//             session,
//             node_id,
//             input.datum_type(),
//             &output_shape,
//         )?;
//         dispatch(stream as &dyn GpuStream, reducer, input, axes[0], &output)?;
//         Ok(tvec!(output.into_tensor().into_tvalue()))
//     })
// }

// fn dispatch(
//     stream: &dyn GpuStream,
//     reducer: &Reducer,
//     input: &DeviceTensor,
//     axis: usize,
//     output: &DeviceTensor,
// ) -> TractResult<()> {
//     let stream = stream.downcast_ref::<crate::context::TractCudaStream>().unwrap();
//     CudaReducer(*reducer).dispatch_eval(stream, input, axis, output).context("In cuda reducer")
// }

// pub type CudaReduce = GpuReduce;

pub fn cuda_reduce(core_reduce: &tract_core::ops::nn::Reduce) -> TractResult<CudaReduce> {
    GpuReduce::from_tract_core(core_reduce, "Cuda", cuda_reduce_launch)
}
