use crate::frame::element_wise::ElementWiseKer;

extern_kernel!(fn fma_tanh_f32(ptr: *mut f32, count: usize) -> ());

#[derive(Copy, Clone, Debug)]
pub struct TanhF32;

impl ElementWiseKer<f32> for TanhF32 {
    #[inline(always)]
    fn name() -> &'static str {
        "fma"
    }
    #[inline(always)]
    fn nr() -> usize {
        8
    }
    #[inline(always)]
    fn alignment_items() -> usize {
        8
    }
    #[inline(always)]
    fn alignment_bytes() -> usize {
        32
    }
    #[inline(never)]
    fn run(buf: &mut [f32]) {
        unsafe { fma_tanh_f32(buf.as_mut_ptr(), buf.len()) }
    }
}

#[cfg(test)]
mod test_simd {
    tanh_frame_tests!(is_x86_feature_detected!("fma"), crate::x86_64_fma::tanh::TanhF32);
}
