use tract_data::half::f16;

mod by_scalar;
mod leaky_relu;
mod max;
pub use by_scalar::*;
pub use leaky_relu::*;
pub use max::*;

use crate::frame::mmm::*;
MMMKernel!(f16, arm64fp16_mmm_f16_16x8_gen; 16, 8; 16, 16; 1, 1; no_prefetch, crate::arm64::has_fp16());
MMMKernel!(f16, arm64fp16_mmm_f16_16x8_a55; 16, 8; 16, 16; 1, 1; no_prefetch, crate::arm64::has_fp16());
MMMKernel!(f16, arm64fp16_mmm_f16_32x4_gen; 32, 4; 16, 16; 1, 1; no_prefetch, crate::arm64::has_fp16());
MMMKernel!(f16, arm64fp16_mmm_f16_32x4_a55; 32, 4; 16, 16; 1, 1; no_prefetch, crate::arm64::has_fp16());
MMMKernel!(f16, arm64fp16_mmm_f16_128x1_gen; 128, 1; 16, 16; 1, 1; no_prefetch, crate::arm64::has_fp16());
MMMKernel!(f16, arm64fp16_mmm_f16_128x1_a55; 128, 1; 16, 16; 1, 1; no_prefetch, crate::arm64::has_fp16());

tanh_impl!(f16, arm64fp16_tanh_f16_8n, 8, 8, crate::arm64::has_fp16());
sigmoid_impl!(f16, arm64fp16_sigmoid_f16_8n, 8, 8, crate::arm64::has_fp16());
