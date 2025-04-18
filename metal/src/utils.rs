use crate::context::MetalBuffer;
use metal::Buffer;
use tract_gpu::device::DeviceBuffer;

#[macro_export]
macro_rules! impl_eval_op_for_metal_op {
    ($op:ty) => {
        impl tract_core::internal::EvalOp for $op {
            fn is_stateless(&self) -> bool {
                false
            }

            #[allow(unused_variables)]
            fn state(
                &self,
                session: &mut tract_core::internal::SessionState,
                node_id: usize,
            ) -> TractResult<Option<Box<dyn OpState>>> {
                Ok(Some(Box::new($crate::ops::MetalOpState::new(node_id, self.clone()))))
            }
        }
    };
}

pub fn as_metal_buffer(device_buffer: &Box<dyn DeviceBuffer>) -> &Buffer {
    if let Some(metal_buffer) = device_buffer.downcast_ref::<MetalBuffer>() {
        &metal_buffer.inner
    } else {
        panic!("Non-Metal Buffer accessed during Metal execution")
    }
}

pub fn rescale_gpu_duration(
    pass_duration: u64,
    cpu_start: u64,
    cpu_end: u64,
    gpu_start: u64,
    gpu_end: u64,
) -> u64 {
    let cpu_time_span = cpu_end - cpu_start;
    let gpu_time_span = gpu_end - gpu_start;

    pass_duration * cpu_time_span / gpu_time_span
}
