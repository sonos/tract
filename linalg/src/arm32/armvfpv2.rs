use crate::frame::mmm::*;

extern_kernel!(fn armvfpv2_mmm_f32_4x4(op: *const FusedKerSpec<f32>) -> isize);

MMMKernel!(MatMatMulF32x4x4<f32>, "vfpv2", armvfpv2_mmm_f32_4x4; 4, 4; 4, 4; 0, 0);

test_mmm_kernel_f32!(crate::arm32::armvfpv2::MatMatMulF32x4x4, test_MatMatMulF32x4x4, true);
