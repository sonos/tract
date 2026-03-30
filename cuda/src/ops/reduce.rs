use crate::context::CUDA_STREAM;
use crate::kernels::nn::CudaReducer;
use tract_core::internal::*;
use tract_gpu::ops::reduce::{GpuReduce, GpuReduceBackend, Reducer};
use tract_gpu::tensor::DeviceTensorExt;

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct CudaReduceBackend;

impl GpuReduceBackend for CudaReduceBackend {
    fn name() -> &'static str {
        "Cuda"
    }

    fn eval(
        reducer: &Reducer,
        axes: &[usize],
        node_id: usize,
        session: &TurnState,
        inputs: TVec<TValue>,
    ) -> TractResult<TVec<TValue>> {
        CUDA_STREAM.with(|stream| {
            let input_value = args_1!(inputs);
            let input = input_value.to_device_tensor()?;
            let mut output_shape = input.shape().to_vec();
            output_shape[axes[0]] = 1;
            let output = tract_gpu::session_handler::make_tensor_for_node(
                session,
                node_id,
                input.datum_type(),
                &output_shape,
            )?;

            CudaReducer(*reducer)
                .dispatch_eval(stream, input, axes[0], &output)
                .context("In cuda reducer")?;

            Ok(tvec!(output.into_tensor().into_tvalue()))
        })
    }
}

pub type CudaReduce = GpuReduce<CudaReduceBackend>;
