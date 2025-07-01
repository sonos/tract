#![allow(unused)]

mod binary;
mod unary;
mod utils;
mod launch_args;

pub use binary::BinOps;
use cudarc::driver::CudaView;
use tract_gpu::tensor::DeviceTensor;
pub use unary::UnaryOps;

use crate::tensor::CudaBuffer;

const UNARY_OPS: &str = include_str!("ptx/unary.ptx");
const BINARY_OPS: &str = include_str!("ptx/binary.ptx");

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LibraryName {
    UnaryOps,
    BinaryOps,
}

impl LibraryName {
    pub fn content(&self) -> &str {
        match self {
            Self::UnaryOps => UNARY_OPS,
            Self::BinaryOps => BINARY_OPS,
        }
    }
}

pub fn get_cuda_view(t: &DeviceTensor) -> CudaView<'_, u8> {
    let buffer = t.device_buffer().downcast_ref::<CudaBuffer>().unwrap();
    let offset = t.buffer_offset::<usize>() * t.datum_type().size_of();
    let size = t.len() * t.datum_type().size_of();
    buffer.inner.slice(offset..(offset + size))
}
