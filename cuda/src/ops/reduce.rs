use crate::context::{CUDA_STREAM, TractCudaStream};
use crate::kernels::nn::CudaReducer;
use tract_core::internal::*;
use tract_gpu::ops::reduce::{GpuReduce, GpuReduceBackend, Reducer};
use tract_gpu::tensor::DeviceTensor;

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct CudaReduceBackend;

impl GpuReduceBackend for CudaReduceBackend {
    type Stream = TractCudaStream;

    fn name() -> &'static str {
        "Cuda"
    }

    fn with_stream<R>(f: impl FnOnce(&Self::Stream) -> TractResult<R>) -> TractResult<R> {
        CUDA_STREAM.with(f)
    }

    fn dispatch_reduce(
        stream: &Self::Stream,
        reducer: &Reducer,
        input: &DeviceTensor,
        axis: usize,
        output: &DeviceTensor,
    ) -> TractResult<()> {
        CudaReducer(*reducer).dispatch_eval(stream, input, axis, output).context("In cuda reducer")
    }
}

pub type CudaReduce = GpuReduce<CudaReduceBackend>;
