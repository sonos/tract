ew_impl_wrap2!(aarch64;
    f32,
    arm64simd_silu_f32_4n,
    4,
    4,
    (),
    #[inline(never)]
    fn run(buf: &mut [f32], _: ()) {
        // Keep the composed symbol but route to the single-pass fused kernel:
        // same formula, less memory traffic (no scratch copy).
        super::arm64simd_silu_f32_4n_fused::run(buf, ());
    }
);

#[cfg(test)]
pub mod test_arm64simd_silu_f32_4n {
    use super::*;
    silu_frame_tests!(cfg!(target_arch = "aarch64"), f32, arm64simd_silu_f32_4n);
}
