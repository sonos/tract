use crate::kernels::nn::cuda_reduce_launch;
use tract_core::internal::*;
use tract_gpu::GpuStream;
use tract_gpu::ops::reduce::GpuReduce;
use tract_gpu::tensor::DeviceTensor;

pub type CudaReduce = GpuReduce;

fn dispatch(
    stream: &dyn GpuStream,
    reducer: &tract_gpu::ops::reduce::Reducer,
    input: &DeviceTensor,
    axis: usize,
    output: &DeviceTensor,
) -> TractResult<()> {
    let stream = stream.downcast_ref::<crate::context::TractCudaStream>().unwrap();
    cuda_reduce_launch(stream, reducer, input, axis, output).context("In cuda reducer")
}

pub fn cuda_reduce(core_reduce: &tract_core::ops::nn::Reduce) -> TractResult<CudaReduce> {
    GpuReduce::from_tract_core(core_reduce, "Cuda", dispatch)
}
