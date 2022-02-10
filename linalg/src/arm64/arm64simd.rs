use crate::frame::element_wise::ElementWiseKer;
use crate::frame::mmm::*;

extern_kernel!(fn arm64simd_mmm_f32_16x4_a55(op: *const FusedKerSpec<f32>) -> isize);
extern_kernel!(fn arm64simd_mmm_f32_16x4_a53(op: *const FusedKerSpec<f32>) -> isize);
extern_kernel!(fn arm64simd_mmm_f32_16x4_gen(op: *const FusedKerSpec<f32>) -> isize);
extern_kernel!(fn arm64simd_mmm_f32_24x4_a53(op: *const FusedKerSpec<f32>) -> isize);
extern_kernel!(fn arm64simd_mmm_f32_24x4_gen(op: *const FusedKerSpec<f32>) -> isize);
extern_kernel!(fn arm64simd_mmm_f32_8x8_a53(op: *const FusedKerSpec<f32>) -> isize);
extern_kernel!(fn arm64simd_mmm_f32_8x8_a55(op: *const FusedKerSpec<f32>) -> isize);
extern_kernel!(fn arm64simd_mmm_f32_8x8_gen(op: *const FusedKerSpec<f32>) -> isize);
extern_kernel!(fn arm64simd_mmm_f32_12x8_a53(op: *const FusedKerSpec<f32>) -> isize);
extern_kernel!(fn arm64simd_mmm_f32_12x8_a55(op: *const FusedKerSpec<f32>) -> isize);
extern_kernel!(fn arm64simd_mmm_f32_12x8_gen(op: *const FusedKerSpec<f32>) -> isize);
extern_kernel!(fn arm64simd_mmm_f32_64x1_a55(op: *const FusedKerSpec<f32>) -> isize);
extern_kernel!(fn arm64simd_mmm_f32_64x1_a53(op: *const FusedKerSpec<f32>) -> isize);
extern_kernel!(fn arm64simd_mmm_f32_64x1_gen(op: *const FusedKerSpec<f32>) -> isize); // todo

extern_kernel!(fn arm64simd_mmm_i32_8x8(op: *const FusedKerSpec<i32>) -> isize);
extern_kernel!(fn arm64simd_mmm_i32_64x1(op: *const FusedKerSpec<i32>) -> isize);
extern_kernel!(fn arm64simd_sigmoid_f32_4n(ptr: *mut f32, count: usize) -> ());
extern_kernel!(fn arm64simd_tanh_f32_4n(ptr: *mut f32, count: usize) -> ());

MMMKernel!(MatMatMulF32x8x8A55<f32>, arm64simd_mmm_f32_8x8_a55; 8, 8; 16, 16; 1, 1);
MMMKernel!(MatMatMulF32x12x8A55<f32>, arm64simd_mmm_f32_12x8_a55; 12, 8; 16, 16; 1, 1);
MMMKernel!(MatMatMulF32x16x4A55<f32>, arm64simd_mmm_f32_16x4_a55; 16, 4; 16, 16; 1, 1);
MMMKernel!(MatMatMulF32x64x1A55<f32>, arm64simd_mmm_f32_64x1_a55; 64, 1; 16, 16; 1, 1);

MMMKernel!(MatMatMulF32x16x4A53<f32>,  arm64simd_mmm_f32_16x4_a53; 16, 4; 16, 16; 1, 1);
MMMKernel!(MatMatMulF32x24x4A53<f32>,  arm64simd_mmm_f32_24x4_a53; 24, 4; 16, 16; 1, 1);
MMMKernel!(MatMatMulF32x8x8A53<f32>, arm64simd_mmm_f32_8x8_a53; 8, 8; 16, 16; 1, 1);
MMMKernel!(MatMatMulF32x12x8A53<f32>, arm64simd_mmm_f32_12x8_a53; 12, 8; 16, 16; 1, 1);
MMMKernel!(MatMatMulF32x64x1A53<f32>, arm64simd_mmm_f32_64x1_a53; 64, 1; 16, 16; 1, 1);

MMMKernel!(MatMatMulF32x16x4<f32>, arm64simd_mmm_f32_16x4_gen; 16, 4; 16, 16; 1, 1);
MMMKernel!(MatMatMulF32x24x4<f32>, arm64simd_mmm_f32_24x4_gen; 24, 4; 16, 16; 1, 1);
MMMKernel!(MatMatMulF32x8x8<f32>, arm64simd_mmm_f32_8x8_gen; 8, 8; 16, 16; 1, 1);
MMMKernel!(MatMatMulF32x12x8<f32>, arm64simd_mmm_f32_12x8_gen; 12, 8; 16, 16; 1, 1);
MMMKernel!(MatMatMulF32x64x1<f32>, arm64simd_mmm_f32_64x1_gen; 64, 1; 16, 16; 1, 1);

MMMKernel!(MatMatMulI32x8x8<i32>, arm64simd_mmm_i32_8x8; 8, 8; 16, 16; 0,0);
MMMKernel!(MatMatMulI32x64x1<i32>, arm64simd_mmm_i32_64x1; 64, 1; 16, 1; 0,0);

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

test_mmm_kernel_f32!(crate::arm64::arm64simd::MatMatMulF32x8x8A55, test_MatMatMulF32x8x8a55, true);
test_mmm_kernel_f32!(crate::arm64::arm64simd::MatMatMulF32x12x8A55, test_MatMatMulF32x12x8a55, true);
test_mmm_kernel_f32!(crate::arm64::arm64simd::MatMatMulF32x16x4A55, test_MatMatMulF32x16x4a55, true);

test_mmm_kernel_f32!(
    crate::arm64::arm64simd::MatMatMulF32x16x4A53,
    test_MatMatMulF32x16x4a53,
    true
);

test_mmm_kernel_f32!(
    crate::arm64::arm64simd::MatMatMulF32x24x4A53,
    test_MatMatMulF32x24x4a53,
    true
);

test_mmm_kernel_f32!(crate::arm64::arm64simd::MatMatMulF32x8x8A53, test_MatMatMulF32x8x8a53, true);
test_mmm_kernel_f32!(crate::arm64::arm64simd::MatMatMulF32x16x4, test_MatMatMulF32x16x4, true);
test_mmm_kernel_f32!(crate::arm64::arm64simd::MatMatMulF32x24x4, test_MatMatMulF32x24x4, true);
test_mmm_kernel_f32!(crate::arm64::arm64simd::MatMatMulF32x8x8, test_MatMatMulF32x8x8, true);
test_mmm_kernel_f32!(
    crate::arm64::arm64simd::MatMatMulF32x12x8A53,
    test_MatMatMulF32x12x8a53,
    true
);
test_mmm_kernel_f32!(crate::arm64::arm64simd::MatMatMulF32x12x8, test_MatMatMulF32x12x8, true);
test_mmm_kernel_f32!(
    crate::arm64::arm64simd::MatMatMulF32x64x1A53,
    test_MatMatMulF32x64x1a53,
    true
);
test_mmm_kernel_f32!(
    crate::arm64::arm64simd::MatMatMulF32x64x1A55,
    test_MatMatMulF32x64x1a55,
    true
);
test_mmm_kernel_f32!(crate::arm64::arm64simd::MatMatMulF32x64x1, test_MatMatMulF32x64x1, true);
test_mmm_kernel_i32!(crate::arm64::arm64simd::MatMatMulI32x8x8, test_MatMatMulI32x8x8, true);
test_mmm_kernel_i32!(crate::arm64::arm64simd::MatMatMulI32x64x1, test_MatMatMulI32x64x1, true);

#[cfg(test)]
mod test_simd {
    sigmoid_frame_tests!(true, crate::arm64::arm64simd::SigmoidF32x4n);
    tanh_frame_tests!(true, crate::arm64::arm64simd::TanhF32x4n);
}
