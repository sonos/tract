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
        const SQRT_2_OVER_PI: f32 = 0.7978845608028654;
        const COEF: f32 = 0.044715;
        const CHUNK: usize = 256;
        let mut scratch = [0f32; CHUNK];
        let mut start = 0;
        while start < buf.len() {
            let end = (start + CHUNK).min(buf.len());
            let chunk = &mut buf[start..end];
            let n = chunk.len();
            // Save original x and pre-compute the tanh argument in place.
            for i in 0..n {
                let x = chunk[i];
                scratch[i] = x;
                chunk[i] = SQRT_2_OVER_PI * (x + COEF * x * x * x);
            }
            super::arm64simd_tanh_f32_4n::run(chunk, ());
            // chunk now holds tanh(arg). Combine with saved x.
            for i in 0..n {
                chunk[i] = 0.5 * scratch[i] * (1.0 + chunk[i]);
            }
            start = end;
        }
    }
);

#[cfg(test)]
pub mod test_arm64simd_gelu_f32_4n {
    use super::*;
    gelu_frame_tests!(true, f32, arm64simd_gelu_f32_4n);
}
