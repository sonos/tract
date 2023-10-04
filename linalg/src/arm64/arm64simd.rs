use crate::frame::mmm::*;
#[cfg(not(feature = "no_fp16"))]
use tract_data::half::f16;

MMMKernel!(f32, arm64simd_mmm_f32_8x8_a55; 8, 8; 16, 16; 1, 1; no_prefetch, true);
MMMKernel!(f32, arm64simd_mmm_f32_12x8_a55; 12, 8; 16, 16; 1, 1; no_prefetch, true);
MMMKernel!(f32, arm64simd_mmm_f32_16x4_a55; 16, 4; 16, 16; 1, 1; no_prefetch, true);
MMMKernel!(f32, arm64simd_mmm_f32_24x4_a55; 24, 4; 16, 16; 1, 1; no_prefetch, true);
MMMKernel!(f32, arm64simd_mmm_f32_64x1_a55; 64, 1; 16, 16; 1, 1; no_prefetch, true);

MMMKernel!(f32, arm64simd_mmm_f32_16x4_a53; 16, 4; 16, 16; 1, 1; no_prefetch, true);
MMMKernel!(f32, arm64simd_mmm_f32_24x4_a53; 24, 4; 16, 16; 1, 1; no_prefetch, true);
MMMKernel!(f32, arm64simd_mmm_f32_8x8_a53; 8, 8; 16, 16; 1, 1; no_prefetch, true);
MMMKernel!(f32, arm64simd_mmm_f32_12x8_a53; 12, 8; 16, 16; 1, 1; no_prefetch, true);
MMMKernel!(f32, arm64simd_mmm_f32_64x1_a53; 64, 1; 16, 16; 1, 1; no_prefetch, true);

MMMKernel!(f32, arm64simd_mmm_f32_16x4_gen; 16, 4; 16, 16; 1, 1; no_prefetch, true);
MMMKernel!(f32, arm64simd_mmm_f32_24x4_gen; 24, 4; 16, 16; 1, 1; no_prefetch, true);
MMMKernel!(f32, arm64simd_mmm_f32_8x8_gen; 8, 8; 16, 16; 1, 1; no_prefetch, true);
MMMKernel!(f32, arm64simd_mmm_f32_12x8_gen; 12, 8; 16, 16; 1, 1; no_prefetch, true);
MMMKernel!(f32, arm64simd_mmm_f32_64x1_gen; 64, 1; 16, 16; 1, 1; no_prefetch, true);

MMMKernel!(i32, arm64simd_mmm_i32_8x8; 8, 8; 16, 16; 0,0; no_prefetch, true);
MMMKernel!(i32, arm64simd_mmm_i32_64x1; 64, 1; 16, 1; 0,0; no_prefetch, true);

#[cfg(not(feature = "no_fp16"))]
MMMKernel!(f16, arm64fp16_mmm_f16_16x8_gen; 16, 8; 16, 16; 1, 1; no_prefetch, crate::arm64::has_fp16());
#[cfg(not(feature = "no_fp16"))]
MMMKernel!(f16, arm64fp16_mmm_f16_16x8_a55; 16, 8; 16, 16; 1, 1; no_prefetch, crate::arm64::has_fp16());
#[cfg(not(feature = "no_fp16"))]
MMMKernel!(f16, arm64fp16_mmm_f16_32x4_gen; 32, 4; 16, 16; 1, 1; no_prefetch, crate::arm64::has_fp16());
#[cfg(not(feature = "no_fp16"))]
MMMKernel!(f16, arm64fp16_mmm_f16_32x4_a55; 32, 4; 16, 16; 1, 1; no_prefetch, crate::arm64::has_fp16());
#[cfg(not(feature = "no_fp16"))]
MMMKernel!(f16, arm64fp16_mmm_f16_128x1_gen; 128, 1; 16, 16; 1, 1; no_prefetch, crate::arm64::has_fp16());
#[cfg(not(feature = "no_fp16"))]
MMMKernel!(f16, arm64fp16_mmm_f16_128x1_a55; 128, 1; 16, 16; 1, 1; no_prefetch, crate::arm64::has_fp16());

tanh_impl!(f32, arm64simd_tanh_f32_4n, 4, 4, true);
sigmoid_impl!(f32, arm64simd_sigmoid_f32_4n, 4, 4, true);

#[cfg(not(feature = "no_fp16"))]
tanh_impl!(f16, arm64fp16_tanh_f16_8n, 8, 8, crate::arm64::has_fp16());
#[cfg(not(feature = "no_fp16"))]
sigmoid_impl!(f16, arm64fp16_sigmoid_f16_8n, 8, 8, crate::arm64::has_fp16());
