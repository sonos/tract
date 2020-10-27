use crate::frame::mmm::*;

extern "C" {
    fn armvfpv2_mmm_f32_4x4(op: *const MatMatMulKerSpec<f32>) -> isize;
}

#[derive(Copy, Clone, Debug)]
pub struct MatMatMulF32x4x4;

impl MatMatMulKer<f32> for MatMatMulF32x4x4 {
    #[inline(always)]
    fn name() -> &'static str {
        "vfpv2"
    }
    #[inline(always)]
    fn mr() -> usize {
        4
    }
    #[inline(always)]
    fn nr() -> usize {
        4
    }
    fn alignment_bytes_packed_a() -> usize {
        4
    }
    fn alignment_bytes_packed_b() -> usize {
        4
    }
    #[inline(never)]
    fn kernel(spec: &MatMatMulKerSpec<f32>) -> isize {
        unsafe { armvfpv2_mmm_f32_4x4(spec) }
    }
}

test_mmm_kernel_f32!(crate::arm32::armvfpv2::MatMatMulF32x4x4, test_MatMatMulF32x4x4, true);
