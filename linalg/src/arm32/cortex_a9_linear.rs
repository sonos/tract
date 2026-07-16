use crate::frame::mmm::LinearCostModel;

pub fn linear_model() -> LinearCostModel<'static> {
    LinearCostModel {
        default_kernel: "armv7neon_mmm_f32_8x6_generic",
        kernels: &[
            "armv7neon_mmm_f32_32x1_cortexa7",
            "armv7neon_mmm_f32_32x1_cortexa9",
            "armv7neon_mmm_f32_32x1_generic",
            "armv7neon_mmm_f32_8x1_generic",
            "armv7neon_mmm_f32_8x4_cortexa7",
            "armv7neon_mmm_f32_8x4_cortexa9",
            "armv7neon_mmm_f32_8x4_generic",
            "armv7neon_mmm_f32_8x6_cortexa7",
            "armv7neon_mmm_f32_8x6_cortexa9",
            "armv7neon_mmm_f32_8x6_generic",
        ],
        coeffs: &[
            [2.3819031e-9, 4.1261504e-7, 4.381363e-6],
            [1.494289e-9, 3.1414405e-7, 9.7689335e-6],
            [1.8809876e-9, 1.4818202e-7, 4.8090465e-6],
            [2.6423652e-9, 3.509898e-7, 5.5710548e-6],
            [1.18603e-9, 4.1431818e-7, 4.6552004e-6],
            [8.4071117e-10, 4.2596056e-7, 5.4189e-6],
            [9.700343e-10, 3.0704652e-7, 9.6184485e-6],
            [1.0288873e-9, 5.014579e-7, 3.1088707e-6],
            [7.597749e-10, 4.850977e-7, 3.774817e-6],
            [9.535872e-10, 2.4634946e-7, 6.4295177e-6],
        ],
    }
}
