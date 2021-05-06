use crate::frame::element_wise::*;
use crate::frame::mmm::*;

extern_kernel!(fn armv7neon_mmm_i8_8x4(op: *const MatMatMulKerSpec<i32>) -> isize);
extern_kernel!(fn armv7neon_mmm_i8_32x1(op: *const MatMatMulKerSpec<i32>) -> isize);
extern_kernel!(fn armv7neon_mmm_f32_8x4(op: *const MatMatMulKerSpec<f32>) -> isize);
extern_kernel!(fn armv7neon_mmm_f32_32x1(op: *const MatMatMulKerSpec<f32>) -> isize);
extern_kernel!(fn armv7neon_sigmoid_f32_4n(ptr: *mut f32, count: usize) -> ());
extern_kernel!(fn armv7neon_tanh_f32_4n(ptr: *mut f32, count: usize) -> ());
extern_kernel!(fn armv7neon_prefetch(start: *const u8, end: *const u8) -> ());

pub fn prefetch(start: *const u8, len: usize) {
    unsafe { armv7neon_prefetch(start, start.offset(len as isize)) }
}

MMMKernel!(MatMatMulI8x8x4<i32>, "neon", armv7neon_mmm_i8_8x4; 8, 4; 32, 4; 0, 0);
MMMKernel!(MatMatMulI8x32x1<i32>, "neon", armv7neon_mmm_i8_32x1; 32,1 ; 32, 4; 0, 0);
MMMKernel!(MatMatMulI8xI32x8x4<i32>, "neon", armv7neon_mmm_i8_8x4; 8, 4; 4, 4; 0, 0);
MMMKernel!(MatMatMulI8xI32x32x1<i32>, "neon", armv7neon_mmm_i8_32x1; 32, 1; 4, 4; 0, 0);
MMMKernel!(MatMatMulF32x8x4<f32>, "neon", armv7neon_mmm_f32_8x4; 8, 4; 4, 4; 0, 0);
MMMKernel!(MatMatMulF32x32x1<f32>, "neon", armv7neon_mmm_f32_32x1; 32, 1; 4, 4; 0, 0);

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

test_mmm_kernel_f32!(
    crate::arm32::armv7neon::MatMatMulF32x8x4,
    test_MatMatMulF32x8x4,
    crate::arm32::has_neon()
);

test_mmm_kernel_f32!(
    crate::arm32::armv7neon::MatMatMulF32x32x1,
    test_MatMatMulF32x32x1,
    crate::arm32::has_neon()
);

test_mmm_kernel_i8!(
    crate::arm32::armv7neon::MatMatMulI8x8x4,
    test_MatMatMulI8x8x4,
    crate::arm32::has_neon()
);
test_mmm_kernel_i8_i32!(
    crate::arm32::armv7neon::MatMatMulI8xI32x8x4,
    test_MatMatMulI8xI32x8x4,
    crate::arm32::has_neon()
);

test_mmm_kernel_i8!(
    crate::arm32::armv7neon::MatMatMulI8x32x1,
    test_MatMatMulI8x32x1,
    crate::arm32::has_neon()
);
test_mmm_kernel_i8_i32!(
    crate::arm32::armv7neon::MatMatMulI8xI32x32x1,
    test_MatMatMulI8xI32x32x1,
    crate::arm32::has_neon()
);

#[cfg(test)]
mod test_neon_fn {
    sigmoid_frame_tests!(crate::arm32::has_neon(), crate::arm32::armv7neon::SigmoidF32x4n);
    tanh_frame_tests!(crate::arm32::has_neon(), crate::arm32::armv7neon::TanhF32x4n);
}
