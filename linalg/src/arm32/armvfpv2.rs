use crate::frame::mmm::*;

extern "C" {
    #[no_mangle]
    fn armvfpv2_smmm_4x4(op: *const MatMatMulKerSpec<f32>) -> isize;
}

#[derive(Copy, Clone, Debug)]
pub struct SMatMatMul4x4;

impl MatMatMulKer<f32> for SMatMatMul4x4 {
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
        unsafe { armvfpv2_smmm_4x4(spec) }
    }
}

#[cfg(test)]
mod test {
    mmm_kernel_tests!(crate::arm32::has_neon(), crate::arm32::armvfpv2::SMatMatMul4x4, f32);
    mmm_frame_tests!(crate::arm32::has_neon(), crate::arm32::armvfpv2::SMatMatMul4x4);
}
