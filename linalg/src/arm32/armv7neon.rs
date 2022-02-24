use crate::frame::element_wise::*;
use crate::frame::mmm::*;

extern_kernel!(fn armv7neon_mmm_i32_8x4(op: *const FusedKerSpec<i32>) -> isize);
extern_kernel!(fn armv7neon_mmm_i32_32x1(op: *const FusedKerSpec<i32>) -> isize);
extern_kernel!(fn armv7neon_mmm_f32_8x4_cortexa7(op: *const FusedKerSpec<f32>) -> isize);
extern_kernel!(fn armv7neon_mmm_f32_8x4_cortexa9(op: *const FusedKerSpec<f32>) -> isize);
extern_kernel!(fn armv7neon_mmm_f32_8x4_generic(op: *const FusedKerSpec<f32>) -> isize);
extern_kernel!(fn armv7neon_mmm_f32_8x6_cortexa7(op: *const FusedKerSpec<f32>) -> isize);
extern_kernel!(fn armv7neon_mmm_f32_8x6_cortexa9(op: *const FusedKerSpec<f32>) -> isize);
extern_kernel!(fn armv7neon_mmm_f32_8x6_generic(op: *const FusedKerSpec<f32>) -> isize);
extern_kernel!(fn armv7neon_mmm_f32_32x1_cortexa7(op: *const FusedKerSpec<f32>) -> isize);
extern_kernel!(fn armv7neon_mmm_f32_32x1_cortexa9(op: *const FusedKerSpec<f32>) -> isize);
extern_kernel!(fn armv7neon_mmm_f32_32x1_generic(op: *const FusedKerSpec<f32>) -> isize);
extern_kernel!(fn armv7neon_sigmoid_f32_4n(ptr: *mut f32, count: usize) -> ());
extern_kernel!(fn armv7neon_tanh_f32_4n(ptr: *mut f32, count: usize) -> ());
extern_kernel!(fn armv7neon_prefetch(start: *const u8, end: *const u8) -> ());

#[inline(always)]
pub fn prefetch(start: *const u8, len: usize) {
    unsafe { armv7neon_prefetch(start, start.offset(len as isize)) }
}

MMMKernel!(MatMatMulI32x8x4<i32>, armv7neon_mmm_i32_8x4; 8, 4; 32, 4; 0, 0, prefetch);
MMMKernel!(MatMatMulI32x32x1<i32>, armv7neon_mmm_i32_32x1; 32,1 ; 32, 4; 0, 0, prefetch);
MMMKernel!(MatMatMulF32x8x4CortexA7<f32>, armv7neon_mmm_f32_8x4_cortexa7; 8, 4; 4, 4; 0, 0, prefetch);
MMMKernel!(MatMatMulF32x8x4CortexA9<f32>, armv7neon_mmm_f32_8x4_cortexa9; 8, 4; 4, 4; 0, 0, prefetch);
MMMKernel!(MatMatMulF32x8x4Generic<f32>, armv7neon_mmm_f32_8x4_generic; 8, 4; 4, 4; 0, 0, prefetch);
MMMKernel!(MatMatMulF32x8x6CortexA7<f32>, armv7neon_mmm_f32_8x6_cortexa7; 8, 6; 4, 4; 0, 0, prefetch);
MMMKernel!(MatMatMulF32x8x6CortexA9<f32>, armv7neon_mmm_f32_8x6_cortexa9; 8, 6; 4, 4; 0, 0, prefetch);
MMMKernel!(MatMatMulF32x8x6Generic<f32>, armv7neon_mmm_f32_8x6_generic; 8, 6; 4, 4; 0, 0, prefetch);
MMMKernel!(MatMatMulF32x32x1CortexA7<f32>, armv7neon_mmm_f32_32x1_cortexa7; 32, 1; 4, 4; 0, 0, prefetch);
MMMKernel!(MatMatMulF32x32x1CortexA9<f32>, armv7neon_mmm_f32_32x1_cortexa9; 32, 1; 4, 4; 0, 0, prefetch);
MMMKernel!(MatMatMulF32x32x1Generic<f32>, armv7neon_mmm_f32_32x1_generic; 32, 1; 4, 4; 0, 0, prefetch);

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

test_mmm_kernel_f32!(MatMatMulF32x8x4Generic, crate::arm32::has_neon());
test_mmm_kernel_f32!(MatMatMulF32x8x4CortexA7, crate::arm32::has_neon());
test_mmm_kernel_f32!(MatMatMulF32x8x4CortexA9, crate::arm32::has_neon());

test_mmm_kernel_f32!(MatMatMulF32x8x6Generic, crate::arm32::has_neon());
test_mmm_kernel_f32!(MatMatMulF32x8x6CortexA7, crate::arm32::has_neon());
test_mmm_kernel_f32!(MatMatMulF32x8x6CortexA9, crate::arm32::has_neon());

test_mmm_kernel_f32!(MatMatMulF32x32x1Generic, crate::arm32::has_neon());
test_mmm_kernel_f32!(MatMatMulF32x32x1CortexA7, crate::arm32::has_neon());
test_mmm_kernel_f32!(MatMatMulF32x32x1CortexA9, crate::arm32::has_neon());

test_mmm_kernel_i32!(MatMatMulI32x8x4, crate::arm32::has_neon());
test_mmm_kernel_i32!(MatMatMulI32x32x1, crate::arm32::has_neon());

#[cfg(test)]
mod test_neon_fn {
    sigmoid_frame_tests!(crate::arm32::has_neon(), crate::arm32::armv7neon::SigmoidF32x4n);
    tanh_frame_tests!(crate::arm32::has_neon(), crate::arm32::armv7neon::TanhF32x4n);
}
