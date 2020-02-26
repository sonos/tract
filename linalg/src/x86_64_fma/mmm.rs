use crate::frame::mmm::*;

extern "C" {
    #[no_mangle]
    fn fma_mmm_f32_16x6(op: *const MatMatMulKerSpec<f32, f32, f32, f32>) -> isize;
    #[no_mangle]
    fn fma_mmm_i8_8x8(op: *const MatMatMulKerSpec<i8, i8, i8, i32>) -> isize;
}

#[derive(Copy, Clone, Debug)]
pub struct MatMatMulF32x16x6;

impl MatMatMulKer<f32, f32, f32, f32> for MatMatMulF32x16x6 {
    #[inline(always)]
    fn name() -> &'static str {
        "fma"
    }
    #[inline(always)]
    fn mr() -> usize {
        16
    }
    #[inline(always)]
    fn nr() -> usize {
        6
    }
    fn alignment_bytes_packed_a() -> usize {
        32
    }
    fn alignment_bytes_packed_b() -> usize {
        4
    }
    #[inline(never)]
    fn kernel(spec: &MatMatMulKerSpec<f32, f32, f32, f32>) -> isize {
        unsafe { fma_mmm_f32_16x6(spec) }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct MatMatMulI8x8x8;

impl MatMatMulKer<i8, i8, i8, i32> for MatMatMulI8x8x8 {
    #[inline(always)]
    fn name() -> &'static str {
        "fma"
    }
    #[inline(always)]
    fn mr() -> usize {
        8
    }
    #[inline(always)]
    fn nr() -> usize {
        8
    }
    fn alignment_bytes_packed_a() -> usize {
        32
    }
    fn alignment_bytes_packed_b() -> usize {
        4
    }
    #[inline(never)]
    fn kernel(spec: &MatMatMulKerSpec<i8, i8, i8, i32>) -> isize {
        unsafe { fma_mmm_i8_8x8(spec) }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct MatMatMulI8xI32x8x8;

impl MatMatMulKer<i8, i8, i32, i32> for MatMatMulI8xI32x8x8 {
    #[inline(always)]
    fn name() -> &'static str {
        "fma"
    }
    #[inline(always)]
    fn mr() -> usize {
        8
    }
    #[inline(always)]
    fn nr() -> usize {
        8
    }
    fn alignment_bytes_packed_a() -> usize {
        32
    }
    fn alignment_bytes_packed_b() -> usize {
        4
    }
    #[inline(never)]
    fn kernel(spec: &MatMatMulKerSpec<i8, i8, i32, i32>) -> isize {
        unsafe { fma_mmm_i8_8x8(spec as *const _ as _) }
    }
}

test_mmm_kernel_f32!(
    crate::x86_64_fma::mmm::MatMatMulF32x16x6,
    test_MatMatMulF32x16x6,
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
