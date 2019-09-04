use crate::frame::mmm::*;
use crate::frame::sigmoid::*;
use crate::frame::tanh::*;

extern "C" {
    #[no_mangle]
    fn armv7neon_smmm_8x4(op: *const MatMatMulKerSpec<f32>) -> isize;
    #[no_mangle]
    fn armv7neon_ssigmoid_4(ptr: *mut f32, count: usize);
    #[no_mangle]
    fn armv7neon_stanh_4(ptr: *mut f32, count: usize);
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

#[derive(Copy, Clone, Debug)]
pub struct SSigmoid4;

impl SigmoidKer<f32> for SSigmoid4 {
    #[inline(always)]
    fn name() -> &'static str {
        "neon"
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
        unsafe { armv7neon_ssigmoid_4(buf.as_mut_ptr(), buf.len()) }
    }
}

#[derive(Copy, Clone, Debug)]
pub struct STanh4;

impl TanhKer<f32> for STanh4 {
    #[inline(always)]
    fn name() -> &'static str {
        "neon"
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
        unsafe { armv7neon_stanh_4(buf.as_mut_ptr(), buf.len()) }
    }
}

#[cfg(test)]
mod test {
    mmm_kernel_tests!(crate::arm32::has_neon(), crate::arm32::armv7neon::SMatMatMul8x4, f32);
    mmm_frame_tests!(crate::arm32::has_neon(), crate::arm32::armv7neon::SMatMatMul8x4);
    sigmoid_frame_tests!(crate::arm32::has_neon(), crate::arm32::armv7neon::SSigmoid4);
    tanh_frame_tests!(crate::arm32::has_neon(), crate::arm32::armv7neon::STanh4);
}
