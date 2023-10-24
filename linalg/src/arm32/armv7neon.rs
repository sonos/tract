use crate::frame::mmm::*;

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

sigmoid_impl!(f32, armv7neon_sigmoid_f32_4n, 4, 4, crate::arm32::has_neon());
tanh_impl!(f32, armv7neon_tanh_f32_4n, 4, 4, crate::arm32::has_neon());

