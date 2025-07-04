#![allow(unused)]

pub mod array;
mod binary;
mod launch_args;
mod unary;
mod utils;

use anyhow::{bail, ensure};
pub use binary::BinOps;
use cudarc::driver::CudaView;
use tract_core::prelude::TractResult;
use tract_gpu::tensor::DeviceTensor;
pub use unary::UnaryOps;

use crate::tensor::CudaBuffer;

const UNARY_OPS: &str = include_str!("ptx/unary.ptx");
const BINARY_OPS: &str = include_str!("ptx/binary.ptx");
const ARRAY_OPS: &str = include_str!("ptx/array.ptx");

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LibraryName {
    UnaryOps,
    BinaryOps,
    ArrayOps,
}

impl LibraryName {
    pub fn content(&self) -> &str {
        match self {
            Self::UnaryOps => UNARY_OPS,
            Self::BinaryOps => BINARY_OPS,
            Self::ArrayOps => ARRAY_OPS,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum BroadcastKind {
    Unicast,
    ByScalarLeft,
    ByScalarRight,
    Nd1,
    Nd2,
    Nd3,
    Nd4,
    Nd5,
    Nd6,
}

impl BroadcastKind {
    pub fn from_rank(rank: usize) -> TractResult<Self> {
        match rank {
            1 => Ok(Self::Nd1),
            2 => Ok(Self::Nd2),
            3 => Ok(Self::Nd3),
            4 => Ok(Self::Nd4),
            5 => Ok(Self::Nd5),
            6 => Ok(Self::Nd6),
            _ => bail!("Unsupported rank {rank} for broadcasting"),
        }
    }

    pub fn name(&self) -> &'static str {
        match self {
            BroadcastKind::Unicast => "unicast",
            BroadcastKind::ByScalarLeft => "by_scalar_lhs",
            BroadcastKind::ByScalarRight => "by_scalar_rhs",
            BroadcastKind::Nd1 => "nd1",
            BroadcastKind::Nd2 => "nd2",
            BroadcastKind::Nd3 => "nd3",
            BroadcastKind::Nd4 => "nd4",
            BroadcastKind::Nd5 => "nd5",
            BroadcastKind::Nd6 => "nd6",
        }
    }
}

impl BroadcastKind {
    const ALL: [BroadcastKind; 8] = [
        Self::Unicast,
        Self::ByScalarLeft,
        Self::ByScalarRight,
        Self::Nd1,
        Self::Nd2,
        Self::Nd3,
        Self::Nd4,
        Self::Nd5,
    ];
}

pub fn get_cuda_view(t: &DeviceTensor) -> CudaView<'_, u8> {
    let size = t.len() * t.datum_type().size_of();
    get_sliced_cuda_view(t, 0, size).unwrap()
}

// NOTE: offset and len are in bytes
pub fn get_sliced_cuda_view(
    t: &DeviceTensor,
    offset: usize,
    len: usize,
) -> TractResult<CudaView<'_, u8>> {
    ensure!(offset + len <= t.len() * t.datum_type().size_of());
    let buffer = t.device_buffer().downcast_ref::<CudaBuffer>().unwrap();
    let offset = t.buffer_offset::<usize>() + offset;
    Ok(buffer.slice(offset..(offset + len)))
}
