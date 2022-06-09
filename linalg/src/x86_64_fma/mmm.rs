use crate::frame::mmm::*;

MMMKernel!(f32, fma_mmm_f32_8x8; 8, 8; 32, 4; 0, 0; no_prefetch, is_x86_feature_detected!("fma"));
MMMKernel!(f32, fma_mmm_f32_16x6; 16, 6; 32, 4; 0, 0; no_prefetch, is_x86_feature_detected!("fma"));
MMMKernel!(f32, fma_mmm_f32_16x5; 16, 5; 32, 4; 0, 0; no_prefetch, is_x86_feature_detected!("fma"));
MMMKernel!(f32, fma_mmm_f32_24x4; 24, 4; 32, 4; 0, 0; no_prefetch, is_x86_feature_detected!("fma"));
MMMKernel!(f32, fma_mmm_f32_32x3; 32, 3; 32, 4; 0, 0; no_prefetch, is_x86_feature_detected!("fma"));
MMMKernel!(f32, fma_mmm_f32_40x2; 40, 2; 32, 4; 0, 0; no_prefetch, is_x86_feature_detected!("fma"));
MMMKernel!(f32, fma_mmm_f32_64x1; 64, 1; 32, 4; 0, 0; no_prefetch, is_x86_feature_detected!("fma"));

MMMKernel!(i32, avx2_mmm_i32_8x8; 8, 8; 32, 4; 0, 0; no_prefetch, is_x86_feature_detected!("avx2"));
