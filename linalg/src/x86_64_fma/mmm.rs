use crate::frame::mmm::*;

extern "C" {
    #[no_mangle]
    fn fma_smmm16x6(op: *const MatMatMulKerSpec<f32, f32, f32, f32>) -> isize;
}

#[derive(Copy, Clone, Debug)]
pub struct SMatMatMul16x6;

impl MatMatMulKer<f32, f32, f32, f32> for SMatMatMul16x6 {
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
        unsafe { fma_smmm16x6(spec) }
    }
}

#[cfg(test)]
#[cfg(all(any(target_arch = "x86", target_arch = "x86_64"),))]
mod test {
    mmm_frame_tests!(
        is_x86_feature_detected!("fma"),
        crate::x86_64_fma::mmm::SMatMatMul16x6,
        f32,
        f32,
        f32,
        f32
    );
    mmm_kernel_tests!(
        is_x86_feature_detected!("fma"),
        crate::x86_64_fma::mmm::SMatMatMul16x6,
        f32,
        f32,
        f32,
        f32
    );
}
