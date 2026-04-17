#![allow(unused)]

pub mod array;
pub mod binary;
pub mod conv;
pub mod conv_cudnn;
pub mod element_wise;
pub mod flash_attn;
pub mod ggml_flash_attn;
mod iff;
pub(crate) mod launch_args;
pub mod matmul;
pub mod nn;
pub(crate) mod utils;

use std::env;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

use crate::ops::GgmlQuantQ81Fact;
use crate::tensor::{CudaBuffer, CudaTensor};
use anyhow::{bail, ensure};
use cudarc::driver::{CudaView, CudaViewMut};
pub use iff::Iff;
use tract_core::internal::ExoticFact;
use tract_core::prelude::{TDim, TractResult};
use tract_core::tract_linalg::block_quant::{BlockQuant, BlockQuantFact, Q4_0, Q8_1};
use tract_gpu::tensor::{DeviceTensor, OwnedDeviceTensor};
use tract_gpu::utils::as_q40_tensor;

const MAX_THREADS: usize = 1024;
const WARP_SIZE: usize = 32;

static CUBIN_FOLDER: OnceLock<PathBuf> = OnceLock::new();

pub fn cubin_dir() -> &'static Path {
    CUBIN_FOLDER
        .get_or_init(|| {
            dirs::cache_dir()
                .unwrap_or_else(|| ".cache".into())
                .join("tract")
                .join(env!("CARGO_PKG_VERSION"))
                .join("cuda")
                .join(crate::utils::REQUIRED_CUDA_API.to_string())
                .join("cubins")
        })
        .as_path()
}

const ELEMENT_WISE_OPS: &str = include_str!("cu/element_wise.cu");
const BINARY_OPS: &str = include_str!("cu/binary.cu");
const ARRAY_OPS: &str = include_str!("cu/array.cu");
const NN_OPS: &str = include_str!("cu/nn.cu");
const CNN_OPS: &str = include_str!("cu/cnn.cu");
const GGML_MM_MV: &str = include_str!("cu/mm_mv.cu");
const GGML_MM_MV_Q: &str = include_str!("cu/mm_mv_q.cu");
const GGML_QUANTIZE: &str = include_str!("cu/quantize.cu");
const GGML_FLASH_ATTN: &str = include_str!("cu/ggml_flash_attn.cu");
const FLASH_ATTN: &str = include_str!("cu/flash_attn.cu");
pub const COMMON_H: &str = include_str!("cu/common.cuh");

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum LibraryName {
    ElementWise,
    Binary,
    Array,
    NN,
    Cnn,
    Ggml,
    GgmlQ,
    Quant,
    GgmlFlashAttn,
    FlashAttn,
}

fn fnv1a64(text: &str) -> u64 {
    const FNV_OFFSET_BASIS: u64 = 0xcbf29ce484222325;
    const FNV_PRIME: u64 = 0x00000100000001B3;

    let mut hash = FNV_OFFSET_BASIS;
    for b in text.as_bytes() {
        hash ^= *b as u64;
        hash = hash.wrapping_mul(FNV_PRIME);
    }
    hash
}

impl LibraryName {
    pub const ALL: [LibraryName; 10] = [
        Self::FlashAttn,
        Self::GgmlFlashAttn,
        Self::ElementWise,
        Self::Binary,
        Self::Array,
        Self::NN,
        Self::Cnn,
        Self::Ggml,
        Self::GgmlQ,
        Self::Quant,
    ];

    pub fn content(&self) -> &str {
        match self {
            Self::ElementWise => ELEMENT_WISE_OPS,
            Self::Binary => BINARY_OPS,
            Self::Array => ARRAY_OPS,
            Self::NN => NN_OPS,
            Self::Cnn => CNN_OPS,
            Self::Ggml => GGML_MM_MV,
            Self::GgmlQ => GGML_MM_MV_Q,
            Self::Quant => GGML_QUANTIZE,
            Self::GgmlFlashAttn => GGML_FLASH_ATTN,
            Self::FlashAttn => FLASH_ATTN,
        }
    }

    pub fn cubin_path(&self) -> PathBuf {
        let basename = match self {
            Self::ElementWise => "element_wise",
            Self::Binary => "binary",
            Self::Array => "array",
            Self::NN => "nn",
            Self::Cnn => "cnn",
            Self::Ggml => "mm_mv",
            Self::GgmlQ => "mm_mv_q",
            Self::Quant => "quantize",
            Self::GgmlFlashAttn => "flash_attn",
            Self::FlashAttn => "minimal_flash_attn",
        };
        let hash = fnv1a64(self.content());
        cubin_dir().join(format!("{}_{}.cubin", basename, hash))
    }
}

pub use tract_gpu::utils::BroadcastKind;

fn tensor_size(t: &DeviceTensor) -> usize {
    let exotic_fact: Option<&dyn ExoticFact> = match t {
        DeviceTensor::Owned(ot) => {
            let cuda_tensor =
                ot.downcast_ref::<CudaTensor>().expect("Non Cuda-Tensor in a Cuda Context");
            cuda_tensor.exotic_fact()
        }
        DeviceTensor::ArenaView(av) => av.exotic_fact(),
    };

    if let Some(of) = exotic_fact {
        of.buffer_sizes()
            .iter()
            .sum::<TDim>()
            .as_i64()
            .expect("Symbols should be resolved at this point") as usize
    } else {
        t.len() * t.datum_type().size_of()
    }
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
    let buffer: &CudaBuffer = t.device_buffer().downcast_ref::<CudaBuffer>().unwrap();
    let offset = t.buffer_offset::<usize>() + offset;
    let ptr: *const CudaBuffer = buffer;
    let mut_buffer: &mut CudaBuffer = unsafe { (ptr as *mut CudaBuffer).as_mut().unwrap() };
    Ok(mut_buffer.inner.slice_mut(offset..(offset + len)))
}
