use crate::frame::mmm::*;
use crate::frame::sigmoid::*;
use crate::frame::tanh::*;

extern "C" {
    #[no_mangle]
    fn arm64simd_mmm_f32_8x8_a5x(op: *const MatMatMulKerSpec<f32, f32, f32, f32>) -> isize;
    #[no_mangle]
    fn arm64simd_mmm_f32_8x8_a7x(op: *const MatMatMulKerSpec<f32, f32, f32, f32>) -> isize;
    #[no_mangle]
    fn arm64simd_mmm_i8_8x8(op: *const MatMatMulKerSpec<i8, i8, i8, i32>) -> isize;
    #[no_mangle]
    fn arm64simd_sigmoid_f32_4n(ptr: *mut f32, count: usize);
    #[no_mangle]
    fn arm64simd_tanh_f32_4n(ptr: *mut f32, count: usize);
}

#[derive(Copy, Clone, Debug)]
pub struct MatMatMulF32x8x8A5x;

impl MatMatMulKer<f32, f32, f32, f32> for MatMatMulF32x8x8A5x {
    #[inline(always)]
    fn name() -> &'static str {
        "arm64simd"
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
        16
    }
    fn alignment_bytes_packed_b() -> usize {
        16
    }
    #[inline(never)]
    fn kernel(op: &MatMatMulKerSpec<f32, f32, f32, f32>) -> isize {
        unsafe { arm64simd_mmm_f32_8x8_a5x(op) }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct MatMatMulF32x8x8A7x;

impl MatMatMulKer<f32, f32, f32, f32> for MatMatMulF32x8x8A7x {
    #[inline(always)]
    fn name() -> &'static str {
        "arm64simd"
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
        16
    }
    fn alignment_bytes_packed_b() -> usize {
        16
    }
    #[inline(never)]
    fn kernel(op: &MatMatMulKerSpec<f32, f32, f32, f32>) -> isize {
        unsafe { arm64simd_mmm_f32_8x8_a7x(op) }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct MatMatMulI8x8x8;

impl MatMatMulKer<i8, i8, i8, i32> for MatMatMulI8x8x8 {
    #[inline(always)]
    fn name() -> &'static str {
        "arm64simd"
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
        16
    }
    fn alignment_bytes_packed_b() -> usize {
        16
    }
    #[inline(never)]
    fn kernel(op: &MatMatMulKerSpec<i8, i8, i8, i32>) -> isize {
        unsafe { arm64simd_mmm_i8_8x8(op) }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct MatMatMulI8xI32x8x8;

impl MatMatMulKer<i8, i8, i32, i32> for MatMatMulI8xI32x8x8 {
    #[inline(always)]
    fn name() -> &'static str {
        "arm64simd"
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
        16
    }
    fn alignment_bytes_packed_b() -> usize {
        16
    }
    #[inline(never)]
    fn kernel(op: &MatMatMulKerSpec<i8, i8, i32, i32>) -> isize {
        unsafe { arm64simd_mmm_i8_8x8(op as *const _ as _) }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct SigmoidF32x4n;

impl SigmoidKer<f32> for SigmoidF32x4n {
    #[inline(always)]
    fn name() -> &'static str {
        "arm64simd"
    }
    #[inline(always)]
    fn nr() -> usize {
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

impl TanhKer<f32> for TanhF32x4n {
    #[inline(always)]
    fn name() -> &'static str {
        "arm64simd"
    }
    #[inline(always)]
    fn nr() -> usize {
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

test_mmm_kernel_f32!(crate::arm64::arm64simd::MatMatMulF32x8x8A5x, test_MatMatMulF32x8x8a5x, true);
test_mmm_kernel_f32!(crate::arm64::arm64simd::MatMatMulF32x8x8A7x, test_MatMatMulF32x8x8a7x, true);
test_mmm_kernel_i8!(crate::arm64::arm64simd::MatMatMulI8x8x8, test_MatMatMulI8x8x8, true);
test_mmm_kernel_i8_i32!(
    crate::arm64::arm64simd::MatMatMulI8xI32x8x8,
    test_MatMatMulI8xI32x8x8,
    true
);

#[cfg(test)]
mod test_simd {
    sigmoid_frame_tests!(true, crate::arm64::arm64simd::SigmoidF32x4n);
    tanh_frame_tests!(true, crate::arm64::arm64simd::TanhF32x4n);
}
