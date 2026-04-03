#![allow(unused)]

pub mod array;
pub mod bin_ops;
pub mod conv;
pub mod element_wise;
pub mod matmul;
pub mod nn;
mod utils;

use tract_core::internal::*;

#[cfg(target_os = "ios")]
const METAL_FLASH_ATTENTION_LIB: &[u8] =
    include_bytes!("matmul/mfa/libMetalFlashAttention-ios.metallib");

#[cfg(target_os = "macos")]
const METAL_FLASH_ATTENTION_LIB: &[u8] =
    include_bytes!("matmul/mfa/libMetalFlashAttention-macos.metallib");

#[cfg(not(any(target_os = "ios", target_os = "macos")))]
const METAL_FLASH_ATTENTION_LIB: &[u8] = &[];

const MLX_GEMM: &str = include_str!("matmul/mlx_gemm/mlx_gemm.metal");
const MLX_GEMV: &str = include_str!("matmul/mlx_gemm/mlx_gemv.metal");
const GGML: &str = include_str!("matmul/ggml_gemm/ggml_mm_mv.metal");
const BASIC_MAT_MUL: &str = include_str!("matmul/basic/basic_mat_mul.metal");
const ARRAY_OPS: &str = include_str!("array/array_ops.metal");
const BIN_OPS: &str = include_str!("bin_ops.metal");
const NN_OPS: &str = include_str!("nn/nn_ops.metal");
const CONV_OPS: &str = include_str!("conv.metal");
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
    ConvOps,
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
            Self::ConvOps => LibraryContent::Source(CONV_OPS),
            Self::NNOps => LibraryContent::Source(NN_OPS),
            Self::ElementWiseOps => LibraryContent::Source(ELEMENT_WISE_OPS),
            Self::MlxGemm => LibraryContent::Source(MLX_GEMM),
            Self::MlxGemv => LibraryContent::Source(MLX_GEMV),
            Self::Ggml => LibraryContent::Source(GGML),
        }
    }
}

pub use tract_gpu::utils::BroadcastKind;
