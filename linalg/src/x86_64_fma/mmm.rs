use crate::frame::mmm::*;

extern_kernel!(fn fma_mmm_f32_16x6(op: *const FusedKerSpec<f32>) -> isize);
extern_kernel!(fn fma_mmm_f32_64x1(op: *const FusedKerSpec<f32>) -> isize);
extern_kernel!(fn fma_mmm_i8_8x8(op: *const FusedKerSpec<i32>) -> isize);

MMMKernel!(MatMatMulF32x16x6<f32>, "fma", fma_mmm_f32_16x6; 16, 6; 32, 4; 0, 0);
MMMKernel!(MatMatMulF32x64x1<f32>, "fma", fma_mmm_f32_64x1; 64, 1; 32, 4; 0, 0);
MMMKernel!(MatMatMulI8x8x8<i32>, "avx2", fma_mmm_i8_8x8; 8, 8; 32, 4; 0, 0);
MMMKernel!(MatMatMulI8xI32x8x8<i32>, "avx2", fma_mmm_i8_8x8; 8, 8; 32, 4; 0, 0);

test_mmm_kernel_f32!(
    crate::x86_64_fma::mmm::MatMatMulF32x16x6,
    test_MatMatMulF32x16x6,
    is_x86_feature_detected!("fma")
);

test_mmm_kernel_f32!(
    crate::x86_64_fma::mmm::MatMatMulF32x64x1,
    test_MatMatMulF32x64x1,
    is_x86_feature_detected!("fma")
);

test_mmm_kernel_i8!(
    crate::x86_64_fma::mmm::MatMatMulI8x8x8,
    test_MatMatMulI8x8x8,
    is_x86_feature_detected!("fma")
);

test_mmm_kernel_i8_i32!(
    crate::x86_64_fma::mmm::MatMatMulI8xI32x8x8,
    test_MatMatMulI8xI32x8x8,
    is_x86_feature_detected!("avx2")
);
