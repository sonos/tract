mod silu;
use cudarc::driver::CudaView;
pub use silu::Silu;
use tract_gpu::tensor::DeviceTensor;

use crate::tensor::CudaBuffer;

const UNARY_OPS: &str = include_str!("ptx/unary.ptx");

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LibraryName {
    UnaryOps,
}

impl LibraryName {
    pub fn content(&self) -> &str {
        match self {
            Self::UnaryOps => UNARY_OPS,
        }
    }
}

pub fn get_cuda_view(t: &DeviceTensor) -> CudaView<'_, u8> {
    let buffer = t.device_buffer().downcast_ref::<CudaBuffer>().unwrap();
    let offset = t.buffer_offset::<usize>() * t.datum_type().size_of();
    let size = t.len() * t.datum_type().size_of();
    buffer.inner.slice(offset..(offset + size))
}
