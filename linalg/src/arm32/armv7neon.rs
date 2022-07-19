use crate::frame::element_wise::*;
use crate::frame::mmm::*;

extern_kernel!(fn armv7neon_sigmoid_f32_4n(ptr: *mut f32, count: usize) -> ());
extern_kernel!(fn armv7neon_tanh_f32_4n(ptr: *mut f32, count: usize) -> ());
extern_kernel!(fn armv7neon_prefetch(start: *const u8, end: *const u8) -> ());

#[inline(always)]
pub fn prefetch(start: *const u8, len: usize) {
    unsafe { armv7neon_prefetch(start, start.offset(len as isize)) }
}

MMMKernel!(i32, armv7neon_mmm_i32_8x4; 8, 4; 32, 4; 0, 0; prefetch, crate::arm32::has_neon());
MMMKernel!(i32, armv7neon_mmm_i32_32x1; 32,1 ; 32, 4; 0, 0; prefetch, crate::arm32::has_neon());
MMMKernel!(f32, armv7neon_mmm_f32_8x4_cortexa7; 8, 4; 4, 4; 0, 0; prefetch, crate::arm32::has_neon());
MMMKernel!(f32, armv7neon_mmm_f32_8x4_cortexa9; 8, 4; 4, 4; 0, 0; prefetch, crate::arm32::has_neon());
MMMKernel!(f32, armv7neon_mmm_f32_8x4_generic; 8, 4; 4, 4; 0, 0; prefetch, crate::arm32::has_neon());
MMMKernel!(f32, armv7neon_mmm_f32_8x6_cortexa7; 8, 6; 4, 4; 0, 0; prefetch, crate::arm32::has_neon());
MMMKernel!(f32, armv7neon_mmm_f32_8x6_cortexa9; 8, 6; 4, 4; 0, 0; prefetch, crate::arm32::has_neon());
MMMKernel!(f32, armv7neon_mmm_f32_8x6_generic; 8, 6; 4, 4; 0, 0; prefetch, crate::arm32::has_neon());
MMMKernel!(f32, armv7neon_mmm_f32_32x1_cortexa7; 32, 1; 4, 4; 0, 0; prefetch, crate::arm32::has_neon());
MMMKernel!(f32, armv7neon_mmm_f32_32x1_cortexa9; 32, 1; 4, 4; 0, 0; prefetch, crate::arm32::has_neon());
MMMKernel!(f32, armv7neon_mmm_f32_32x1_generic; 32, 1; 4, 4; 0, 0; prefetch, crate::arm32::has_neon());

#[derive(Copy, Clone, Debug)]
pub struct SigmoidF32x4n;

impl ElementWiseKer<f32> for SigmoidF32x4n {
    #[inline(always)]
    fn name() -> &'static str {
        "neon"
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
        unsafe { armv7neon_sigmoid_f32_4n(buf.as_mut_ptr(), buf.len()) }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct TanhF32x4n;

impl ElementWiseKer<f32> for TanhF32x4n {
    #[inline(always)]
    fn name() -> &'static str {
        "neon"
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
        unsafe { armv7neon_tanh_f32_4n(buf.as_mut_ptr(), buf.len()) }
    }
}

#[cfg(test)]
mod test_neon_fn {
    sigmoid_frame_tests!(crate::arm32::has_neon(), f32, crate::arm32::armv7neon::SigmoidF32x4n);
    tanh_frame_tests!(crate::arm32::has_neon(), f32, crate::arm32::armv7neon::TanhF32x4n);
}
