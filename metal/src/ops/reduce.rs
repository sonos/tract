use crate::kernels::nn::MetalReducer;
use tract_core::internal::*;
use tract_gpu::GpuStream;
use tract_gpu::ops::reduce::GpuReduce;
use tract_gpu::tensor::DeviceTensor;

pub type MetalReduce = GpuReduce;

fn dispatch(
    stream: &dyn GpuStream,
    reducer: &tract_gpu::ops::reduce::Reducer,
    input: &DeviceTensor,
    axis: usize,
    output: &DeviceTensor,
) -> TractResult<()> {
    let stream = stream.downcast_ref::<crate::MetalStream>().unwrap();
    MetalReducer(*reducer).dispatch_eval(stream, input, axis, output)
}

pub fn metal_reduce(core_reduce: &tract_core::ops::nn::Reduce) -> TractResult<MetalReduce> {
    GpuReduce::from_tract_core(core_reduce, "Metal", dispatch)
}
