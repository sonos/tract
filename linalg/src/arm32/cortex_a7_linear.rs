use crate::frame::mmm::LinearCostModel;

pub fn linear_model() -> LinearCostModel<'static> {
    LinearCostModel {
        default_kernel: "armv7neon_mmm_f32_8x6_generic",
        kernels: &[
            "armv7neon_mmm_f32_8x4_cortexa7",
            "armv7neon_mmm_f32_8x4_cortexa9",
            "armv7neon_mmm_f32_8x4_generic",
            "armv7neon_mmm_f32_8x6_cortexa7",
            "armv7neon_mmm_f32_8x6_cortexa9",
            "armv7neon_mmm_f32_8x6_generic",
            "generic_f32_4x4",
        ],
        coeffs: &[
            [1.0656308e-9, 3.9614042e-7, 6.674147e-7],
            [1.0895281e-9, 3.9633096e-7, 6.7588525e-7],
            [1.0303457e-9, 4.0090302e-7, 5.7453e-7],
            [9.855535e-10, 4.264751e-7, 7.1043894e-7],
            [1.0017487e-9, 4.25672e-7, 6.832612e-7],
            [9.64845e-10, 4.1894916e-7, 6.325448e-7],
            [1.4521687e-9, 4.4130726e-7, 6.1335317e-7],
        ],
    }
}
