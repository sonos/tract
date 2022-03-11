use crate::frame::element_wise::ElementWiseKer;
use crate::frame::mmm::*;
use tract_data::half::f16;

extern_kernel!(fn arm64simd_sigmoid_f32_4n(ptr: *mut f32, count: usize) -> ());
extern_kernel!(fn arm64simd_tanh_f32_4n(ptr: *mut f32, count: usize) -> ());

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

MMMKernel!(f16, arm64fp16_mmm_f16_16x8_gen; 16, 8; 16, 16; 1, 1; no_prefetch, crate::arm64::has_fp16());
MMMKernel!(f16, arm64fp16_mmm_f16_16x8_a55; 16, 8; 16, 16; 1, 1; no_prefetch, crate::arm64::has_fp16());
MMMKernel!(f16, arm64fp16_mmm_f16_32x4_gen; 32, 4; 16, 16; 1, 1; no_prefetch, crate::arm64::has_fp16());
MMMKernel!(f16, arm64fp16_mmm_f16_32x4_a55; 32, 4; 16, 16; 1, 1; no_prefetch, crate::arm64::has_fp16());

#[derive(Copy, Clone, Debug)]
pub struct SigmoidF32x4n;

impl ElementWiseKer<f32> for SigmoidF32x4n {
    #[inline(always)]
    fn name() -> &'static str {
        "arm64simd"
    }
    #[inline(always)]
    fn nr() -> usize {
        4
    }
    #[inline(always)]
    fn alignment_items() -> usize {
        4
    }
    #[inline(always)]
    fn alignment_bytes() -> usize {
        16
    }
    #[inline(never)]
    fn run(buf: &mut [f32]) {
        unsafe { arm64simd_sigmoid_f32_4n(buf.as_mut_ptr(), buf.len()) }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct TanhF32x4n;

impl ElementWiseKer<f32> for TanhF32x4n {
    #[inline(always)]
    fn name() -> &'static str {
        "arm64simd"
    }
    #[inline(always)]
    fn nr() -> usize {
        4
    }
    #[inline(always)]
    fn alignment_items() -> usize {
        4
    }
    #[inline(always)]
    fn alignment_bytes() -> usize {
        16
    }
    #[inline(never)]
    fn run(buf: &mut [f32]) {
        unsafe { arm64simd_tanh_f32_4n(buf.as_mut_ptr(), buf.len()) }
    }
}

#[cfg(test)]
mod test_simd {
    sigmoid_frame_tests!(true, crate::arm64::arm64simd::SigmoidF32x4n);
    tanh_frame_tests!(true, crate::arm64::arm64simd::TanhF32x4n);
}
