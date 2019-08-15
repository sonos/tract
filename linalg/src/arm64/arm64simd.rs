use crate::frame::mmm::*;

extern "C" {
    #[no_mangle]
    fn arm64simd_smmm8x8(op: *const MatMatMulKerSpec<f32>) -> isize;
}

#[derive(Copy, Clone, Debug)]
pub struct SMatMatMul8x8;

impl MatMatMulKer<f32> for SMatMatMul8x8 {
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
    fn kernel(op: &MatMatMulKerSpec<f32>) -> isize {
        unsafe { arm64simd_smmm8x8(op) }
    }
}

#[cfg(test)]
mod test {
    mmm_kernel_tests!(true, crate::arm64::arm64simd::SMatMatMul8x8, f32);
    mmm_frame_tests!(true, crate::arm64::arm64simd::SMatMatMul8x8);
}
