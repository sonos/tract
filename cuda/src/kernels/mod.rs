mod silu;
use cust::memory::DevicePointer;
pub use silu::Silu;
use tract_gpu::tensor::DeviceTensor;

use crate::tensor::CudaBuffer;

const UNARY_OPS: &str = include_str!("unary.ptx");

pub fn get_cuda_ptr(t: &DeviceTensor) -> DevicePointer<u8> {
    let ptr = t.device_buffer().downcast_ref::<CudaBuffer>().unwrap().as_device_ptr();
    let offset = t.buffer_offset::<isize>() * t.datum_type().size_of() as isize;
    unsafe { ptr.offset(offset) }
}
