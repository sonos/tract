use crate::frame::mmm::*;
use crate::frame::sigmoid::*;

extern "C" {
    #[no_mangle]
    fn armv7neon_smmm_8x4(op: *const MatMatMulKerSpec<f32>) -> isize;
    #[no_mangle]
    fn armv7neon_ssigmoid_4(ptr: *mut f32, count: usize);
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


#[cfg(test)]
mod test {
    sigmoid_frame_tests!(crate::arm32::has_neon(), crate::arm32::armv7neon::SSigmoid4);
}
