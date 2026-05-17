ew_impl_wrap!(
    f32,
    arm64simd_silu_f32_4n,
    4,
    4,
    (),
    #[inline(never)]
    fn run(buf: &mut [f32], _: ()) {
        // SiLU(x) = x * sigmoid(x). Compose by saving the input chunk to a
        // stack scratch buffer, running tract's NEON sigmoid kernel in place,
        // then multiplying back by the saved original. Multiply loop
        // auto-vectorises on aarch64.
        const CHUNK: usize = 256;
        let mut scratch = [0f32; CHUNK];
        let mut start = 0;
        while start < buf.len() {
            let end = (start + CHUNK).min(buf.len());
            let chunk = &mut buf[start..end];
            let n = chunk.len();
            scratch[..n].copy_from_slice(chunk);
            super::arm64simd_sigmoid_f32_4n::run(chunk, ());
            for i in 0..n {
                chunk[i] *= scratch[i];
            }
            start = end;
        }
    }
);

#[cfg(test)]
pub mod test_arm64simd_silu_f32_4n {
    use super::*;
    silu_frame_tests!(true, f32, arm64simd_silu_f32_4n);
}
