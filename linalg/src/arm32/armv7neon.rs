use crate::frame::mmm::*;

extern "C" {
    #[no_mangle]
    fn armv7neon_smmm_8x4(op: *const MatMatMulKerSpec<f32>) -> isize;
}

#[derive(Copy, Clone, Debug)]
pub struct SMatMatMul8x4;

impl MatMatMulKer<f32> for SMatMatMul8x4 {
    #[inline(always)]
    fn name() -> &'static str {
        "neon"
    }
    #[inline(always)]
    fn mr() -> usize {
        8
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
        unsafe { armv7neon_smmm_8x4(spec) }
    }
}

#[cfg(test)]
mod test {
    mmm_kernel_tests!(crate::arm32::has_neon(), crate::arm32::armv7neon::SMatMatMul8x4, f32);
    mmm_frame_tests!(crate::arm32::has_neon(), crate::arm32::armv7neon::SMatMatMul8x4);
}
