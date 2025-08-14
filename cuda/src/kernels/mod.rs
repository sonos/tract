#![allow(unused)]

pub mod array;
mod binary;
mod launch_args;
mod mm_mv;
pub mod nn;
mod unary;
mod utils;

use crate::tensor::{CudaBuffer, CudaTensor};
use anyhow::{bail, ensure};
pub use binary::BinOps;
use cudarc::driver::{CudaView, CudaViewMut};
pub use mm_mv::Matmul;
use tract_core::prelude::TractResult;
use tract_core::tract_linalg::block_quant::{BlockQuant, Q4_0};
use tract_gpu::tensor::{DeviceTensor, OwnedDeviceTensor};
use tract_gpu::utils::as_q40_tensor;
pub use unary::UnaryOps;

const MAX_THREADS: usize = 1024;

const UNARY_OPS: &str = include_str!("ptx/unary.ptx");
const BINARY_OPS: &str = include_str!("ptx/binary.ptx");
const ARRAY_OPS: &str = include_str!("ptx/array.ptx");
const NN_OPS: &str = include_str!("ptx/nn.ptx");
const GGML_MM_MV: &str = include_str!("ptx/mm_mv.ptx");
const GGML_MM_MV_Q: &str = include_str!("ptx/mm_mv_q.ptx");

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LibraryName {
    Unary,
    Binary,
    Array,
    NN,
    Ggml,
    GgmlQ,
}

impl LibraryName {
    pub fn content(&self) -> &str {
        match self {
            Self::Unary => UNARY_OPS,
            Self::Binary => BINARY_OPS,
            Self::Array => ARRAY_OPS,
            Self::NN => NN_OPS,
            Self::Ggml => GGML_MM_MV,
            Self::GgmlQ => GGML_MM_MV_Q,
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

fn tensor_size(t: &DeviceTensor) -> usize {
    if let DeviceTensor::Owned(ot) = t {
        let cuda_tensor =
            ot.downcast_ref::<CudaTensor>().expect("Non Cuda-Tensor in a Cuda Context");

        if let Some(bqf) = cuda_tensor.block_quant_fact() {
            return bqf.shape().iter().product::<usize>() * Q4_0.block_bytes() / Q4_0.block_len();
        }
    }

    t.len() * t.datum_type().size_of()
}

pub fn get_cuda_view(t: &DeviceTensor) -> CudaView<'_, u8> {
    let size = tensor_size(t);
    get_sliced_cuda_view(t, 0, size).unwrap()
}

// NOTE: offset and len are in bytes
pub fn get_sliced_cuda_view(
    t: &DeviceTensor,
    offset: usize,
    len: usize,
) -> TractResult<CudaView<'_, u8>> {
    ensure!(offset + len <= tensor_size(t));
    let buffer = t.device_buffer().downcast_ref::<CudaBuffer>().unwrap();
    let offset = t.buffer_offset::<usize>() + offset;
    Ok(buffer.slice(offset..(offset + len)))
}

pub fn get_cuda_view_mut(t: &DeviceTensor) -> CudaViewMut<'_, u8> {
    let size = t.len() * t.datum_type().size_of();
    get_sliced_cuda_view_mut(t, 0, size).unwrap()
}

// NOTE: offset and len are in bytes
pub fn get_sliced_cuda_view_mut(
    t: &DeviceTensor,
    offset: usize,
    len: usize,
) -> TractResult<CudaViewMut<'_, u8>> {
    ensure!(offset + len <= t.len() * t.datum_type().size_of());
    let mut buffer = t.device_buffer().downcast_ref::<CudaBuffer>().unwrap();
    let offset = t.buffer_offset::<usize>() + offset;
    Ok(buffer.as_view_mut().slice_mut(offset..(offset + len)))
}
