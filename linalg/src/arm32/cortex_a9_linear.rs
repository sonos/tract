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
            [1.1680975e-9, 4.8285926e-7, 6.584693e-7],
            [8.220334e-10, 4.7899834e-7, 6.2217066e-7],
            [8.228106e-10, 4.8049515e-7, 6.103068e-7],
            [1.0119088e-9, 5.308459e-7, 7.677767e-7],
            [7.400674e-10, 5.369956e-7, 7.79905e-7],
            [7.480408e-10, 5.3532716e-7, 6.5622993e-7],
            [1.5155719e-9, 4.7390833e-7, 6.594678e-7],
        ],
    }
}
