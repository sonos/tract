use crate::frame::mmm::*;
use crate::frame::sigmoid::*;
use crate::frame::tanh::*;

extern "C" {
    #[no_mangle]
    fn arm64simd_smmm_8x8(op: *const MatMatMulKerSpec<f32, f32, f32, f32>) -> isize;
    #[no_mangle]
    fn arm64simd_ssigmoid_4n(ptr: *mut f32, count: usize);
    #[no_mangle]
    fn arm64simd_stanh_8n(ptr: *mut f32, count: usize);
}

#[derive(Copy, Clone, Debug)]
pub struct SMatMatMul8x8;

impl MatMatMulKer<f32, f32, f32, f32> for SMatMatMul8x8 {
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
        unsafe { arm64simd_smmm_8x8(op) }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct SSigmoid4;

impl SigmoidKer<f32> for SSigmoid4 {
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
        unsafe { arm64simd_ssigmoid_4n(buf.as_mut_ptr(), buf.len()) }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct STanh8;

impl TanhKer<f32> for STanh8 {
    #[inline(always)]
    fn name() -> &'static str {
        "arm64simd"
    }
    #[inline(always)]
    fn nr() -> usize {
        8
    }
    #[inline(always)]
    fn alignment_bytes() -> usize {
        32
    }
    #[inline(never)]
    fn run(buf: &mut [f32]) {
        unsafe { arm64simd_stanh_8n(buf.as_mut_ptr(), buf.len()) }
    }
}

#[cfg(test)]
mod tests {
    test_mmm_kernel_f32!(crate::arm64::arm64simd::SMatMatMul8x8, test_SMatMatMul8x8, true);
    sigmoid_frame_tests!(true, crate::arm64::arm64simd::SSigmoid4);
}
