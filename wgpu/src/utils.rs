use tract_core::internal::{DatumType, TypedFact};
use tract_gpu::tensor::DeviceTensor;

use crate::context::WgpuBuffer;

/// The datum types this backend has kernels for.
///
/// `DeviceTensor::is_supported_dt` is wider — it covers everything a device
/// tensor can hold, which suits the backends whose kernels are generated per
/// type. Every WGSL kernel here is f32 or f16, so a node carrying anything else
/// has to stay on the CPU: taking it and failing at dispatch is worse than not
/// taking it.
pub fn is_supported_dt(dt: DatumType) -> bool {
    matches!(dt, DatumType::F32 | DatumType::F16)
}

/// Whether a generic kernel can take this input. Packed weights stay off this
/// path: `translate_gemm` accepts `Q4_0` and `Q2_0_T` on its B operand.
pub fn is_supported_fact(f: &TypedFact) -> bool {
    is_supported_dt(f.datum_type) && f.exotic_fact().is_none()
}

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
