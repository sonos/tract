use tract_gpu::tensor::DeviceTensor;

use crate::context::WgpuBuffer;

pub fn get_wgpu_buffer(tensor: &DeviceTensor) -> &WgpuBuffer {
    tensor
        .device_buffer()
        .downcast_ref::<WgpuBuffer>()
        .expect("Non-wgpu buffer accessed during wgpu execution")
}

/// Element offset of this tensor inside its storage buffer, plus an extra byte offset.
pub fn element_offset(tensor: &DeviceTensor, extra_bytes: usize) -> usize {
    let dt = tensor.datum_type().size_of();
    (tensor.buffer_offset::<usize>() + extra_bytes) / dt
}
