use crate::frame::sigmoid::*;

extern "C" {
    fn fma_sigmoid_f32(ptr: *mut f32, count: usize);
}

#[derive(Copy, Clone, Debug)]
pub struct SigmoidF32;

impl SigmoidKer<f32> for SigmoidF32 {
    #[inline(always)]
    fn name() -> &'static str {
        "fma"
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
        unsafe { fma_sigmoid_f32(buf.as_mut_ptr(), buf.len()) }
    }
}

#[cfg(test)]
mod test_simd {
    sigmoid_frame_tests!(true, crate::x86_64_fma::sigmoid::SigmoidF32);
}
