use crate::frame::element_wise::*;
use crate::frame::mmm::*;

extern_kernel!(fn armv7neon_mmm_i8_8x4(op: *const FusedKerSpec<i32>) -> isize);
extern_kernel!(fn armv7neon_mmm_i8_32x1(op: *const FusedKerSpec<i32>) -> isize);
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

MMMKernel!(MatMatMulI8x8x4<i32>, "neon", armv7neon_mmm_i8_8x4; 8, 4; 32, 4; 0, 0, prefetch);
MMMKernel!(MatMatMulI8x32x1<i32>, "neon", armv7neon_mmm_i8_32x1; 32,1 ; 32, 4; 0, 0, prefetch);
MMMKernel!(MatMatMulI8xI32x8x4<i32>, "neon", armv7neon_mmm_i8_8x4; 8, 4; 4, 4; 0, 0, prefetch);
MMMKernel!(MatMatMulI8xI32x32x1<i32>, "neon", armv7neon_mmm_i8_32x1; 32, 1; 4, 4; 0, 0, prefetch);
MMMKernel!(MatMatMulF32x8x4CortexA7<f32>, "neon/cortex-a7", armv7neon_mmm_f32_8x4_cortexa7; 8, 4; 4, 4; 0, 0, prefetch);
MMMKernel!(MatMatMulF32x8x4CortexA9<f32>, "neon/cortex-a9", armv7neon_mmm_f32_8x4_cortexa9; 8, 4; 4, 4; 0, 0, prefetch);
MMMKernel!(MatMatMulF32x8x4Generic<f32>, "neon/generic", armv7neon_mmm_f32_8x4_generic; 8, 4; 4, 4; 0, 0, prefetch);
MMMKernel!(MatMatMulF32x8x6CortexA7<f32>, "neon/cortex-a7", armv7neon_mmm_f32_8x6_cortexa7; 8, 6; 4, 4; 0, 0, prefetch);
MMMKernel!(MatMatMulF32x8x6CortexA9<f32>, "neon/cortex-a9", armv7neon_mmm_f32_8x6_cortexa9; 8, 6; 4, 4; 0, 0, prefetch);
MMMKernel!(MatMatMulF32x8x6Generic<f32>, "neon/generic", armv7neon_mmm_f32_8x6_generic; 8, 6; 4, 4; 0, 0, prefetch);
MMMKernel!(MatMatMulF32x32x1CortexA7<f32>, "neon/cortex-a7", armv7neon_mmm_f32_32x1_cortexa7; 32, 1; 4, 4; 0, 0, prefetch);
MMMKernel!(MatMatMulF32x32x1CortexA9<f32>, "neon/cortex-a9", armv7neon_mmm_f32_32x1_cortexa9; 32, 1; 4, 4; 0, 0, prefetch);
MMMKernel!(MatMatMulF32x32x1Generic<f32>, "neon/generic", armv7neon_mmm_f32_32x1_generic; 32, 1; 4, 4; 0, 0, prefetch);

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
    crate::arm32::armv7neon::MatMatMulF32x8x4Generic,
    test_MatMatMulF32x8x4Generic,
    crate::arm32::has_neon()
);

test_mmm_kernel_f32!(
    crate::arm32::armv7neon::MatMatMulF32x8x4CortexA7,
    test_MatMatMulF32x8x4CortextA7,
    crate::arm32::has_neon()
);

test_mmm_kernel_f32!(
    crate::arm32::armv7neon::MatMatMulF32x8x4CortexA9,
    test_MatMatMulF32x8x4CortextA9,
    crate::arm32::has_neon()
);

test_mmm_kernel_f32!(
    crate::arm32::armv7neon::MatMatMulF32x8x6Generic,
    test_MatMatMulF32x8x6Generic,
    crate::arm32::has_neon()
);

test_mmm_kernel_f32!(
    crate::arm32::armv7neon::MatMatMulF32x8x6CortexA7,
    test_MatMatMulF32x8x6CortexA7,
    crate::arm32::has_neon()
);

test_mmm_kernel_f32!(
    crate::arm32::armv7neon::MatMatMulF32x8x6CortexA9,
    test_MatMatMulF32x8x6CortexA9,
    crate::arm32::has_neon()
);

test_mmm_kernel_f32!(
    crate::arm32::armv7neon::MatMatMulF32x32x1Generic,
    test_MatMatMulF32x32x1Generic,
    crate::arm32::has_neon()
);

test_mmm_kernel_f32!(
    crate::arm32::armv7neon::MatMatMulF32x32x1CortexA7,
    test_MatMatMulF32x32x1CortexA7,
    crate::arm32::has_neon()
);

test_mmm_kernel_f32!(
    crate::arm32::armv7neon::MatMatMulF32x32x1CortexA9,
    test_MatMatMulF32x32x1CortexA9,
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
