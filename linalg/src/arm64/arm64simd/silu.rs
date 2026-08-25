routine_ew_rust!(aarch64;
    f32,
    arm64simd_silu_f32_4n,
    4,
    4,
    #[inline(never)]
    fn run(buf: &mut [f32], _: ()) {
        // Keep the composed symbol but route to the single-pass fused kernel:
        // same formula, less memory traffic (no scratch copy).
        super::arm64simd_silu_f32_4n_fused::run(buf, ());
    },
    func(Silu),
    boost(crate::isa::NEVER_PREFERRED)
);
