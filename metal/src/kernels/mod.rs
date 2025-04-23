#![allow(unused)]

pub mod array;
mod bin_ops;
mod element_wise;
pub mod matmul;
pub mod nn;
mod utils;

pub use bin_ops::BinOps;
pub use element_wise::ElementWiseOps;

use tract_core::internal::*;

#[cfg(target_os = "ios")]
const METAL_FLASH_ATTENTION_LIB: &[u8] =
    include_bytes!("matmul/mfa/libMetalFlashAttention-ios.metallib");
#[cfg(target_os = "macos")]
const METAL_FLASH_ATTENTION_LIB: &[u8] =
    include_bytes!("matmul/mfa/libMetalFlashAttention-macos.metallib");

const MLX_GEMM: &str = include_str!("matmul/mlx_gemm/mlx_gemm.metal");
const MLX_GEMV: &str = include_str!("matmul/mlx_gemm/mlx_gemv.metal");
const GGML: &str = include_str!("matmul/ggml_gemm/ggml_mm_mv.metal");
const BASIC_MAT_MUL: &str = include_str!("matmul/basic/basic_mat_mul.metal");
const ARRAY_OPS: &str = include_str!("array/array_ops.metal");
const BIN_OPS: &str = include_str!("bin_ops.metal");
const NN_OPS: &str = include_str!("nn/nn_ops.metal");
const ELEMENT_WISE_OPS: &str = include_str!("element_wise.metal");

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub enum LibraryContent<'a> {
    Data(&'a [u8]),
    Source(&'a str),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LibraryName {
    MlxGemm,
    MlxGemv,
    MfaLib,
    BasicMatMul,
    BinOps,
    ArrayOps,
    NNOps,
    ElementWiseOps,
    Ggml,
}

impl LibraryName {
    pub fn content(&self) -> LibraryContent<'static> {
        match self {
            Self::MfaLib => LibraryContent::Data(METAL_FLASH_ATTENTION_LIB),
            Self::BasicMatMul => LibraryContent::Source(BASIC_MAT_MUL),
            Self::ArrayOps => LibraryContent::Source(ARRAY_OPS),
            Self::BinOps => LibraryContent::Source(BIN_OPS),
            Self::NNOps => LibraryContent::Source(NN_OPS),
            Self::ElementWiseOps => LibraryContent::Source(ELEMENT_WISE_OPS),
            Self::MlxGemm => LibraryContent::Source(MLX_GEMM),
            Self::MlxGemv => LibraryContent::Source(MLX_GEMV),
            Self::Ggml => LibraryContent::Source(GGML),
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
