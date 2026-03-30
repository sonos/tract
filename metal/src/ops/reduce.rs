use crate::MetalStream;
use crate::kernels::nn::MetalReducer;
use crate::utils::with_borrowed_metal_stream;
use tract_core::internal::*;
use tract_gpu::ops::reduce::{GpuReduce, GpuReduceBackend, Reducer};
use tract_gpu::tensor::DeviceTensor;

#[derive(Clone, Debug, Hash, PartialEq, Eq)]
pub struct MetalReduceBackend;

impl GpuReduceBackend for MetalReduceBackend {
    type Stream = MetalStream;

    fn name() -> &'static str {
        "Metal"
    }

    fn with_stream<R>(f: impl FnOnce(&Self::Stream) -> TractResult<R>) -> TractResult<R> {
        with_borrowed_metal_stream(f)
    }

    fn dispatch_reduce(
        stream: &Self::Stream,
        reducer: &Reducer,
        input: &DeviceTensor,
        axis: usize,
        output: &DeviceTensor,
    ) -> TractResult<()> {
        MetalReducer(*reducer).dispatch_eval(stream, input, axis, output)
    }
}

pub type MetalReduce = GpuReduce<MetalReduceBackend>;
