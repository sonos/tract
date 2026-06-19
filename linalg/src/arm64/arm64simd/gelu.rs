// Tanh-form GELU (pow=3) matching tract's GeluApproximate:
//   gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
//
// Composed at the kernel level: save the original x, compute the tanh
// argument, call tract's NEON tanh kernel in place, then finish with the
// 0.5 * x * (1 + tanh) multiply. Chunked to keep the scratch buffer L1-resident.

ew_impl_wrap!(
    f32,
    arm64simd_gelu_f32_4n,
    4,
    4,
    (),
    #[inline(never)]
    fn run(buf: &mut [f32], _: ()) {
        // Keep the composed symbol but route to the single-pass fused kernel:
        // same approximation, less memory traffic (no scratch copy).
        super::arm64simd_gelu_f32_4n_fused::run(buf, ());
    }
);

#[cfg(test)]
pub mod test_arm64simd_gelu_f32_4n {
    use super::*;
    gelu_frame_tests!(true, f32, arm64simd_gelu_f32_4n);
}
