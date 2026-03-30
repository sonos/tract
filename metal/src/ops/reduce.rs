use crate::kernels::nn::MetalReducer;
use crate::utils::with_borrowed_metal_stream;
use tract_core::internal::*;
use tract_gpu::ops::reduce::{GpuReduce, GpuStream, Reducer};
use tract_gpu::tensor::DeviceTensor;

impl GpuStream for crate::MetalStream {}

fn with_stream(
    f: Box<dyn FnOnce(&dyn GpuStream) -> TractResult<TVec<TValue>>>,
) -> TractResult<TVec<TValue>> {
    with_borrowed_metal_stream(|s| f(s as &dyn GpuStream))
}

fn dispatch(
    stream: &dyn GpuStream,
    reducer: &Reducer,
    input: &DeviceTensor,
    axis: usize,
    output: &DeviceTensor,
) -> TractResult<()> {
    let stream = stream.downcast_ref::<crate::MetalStream>().unwrap();
    MetalReducer(*reducer).dispatch_eval(stream, input, axis, output)
}

pub type MetalReduce = GpuReduce;

pub fn metal_reduce(core_reduce: &tract_core::ops::nn::Reduce) -> TractResult<MetalReduce> {
    GpuReduce::from_tract_core(core_reduce, "Metal", with_stream, dispatch)
}
