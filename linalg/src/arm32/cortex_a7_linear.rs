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
            [1.502748e-9, 4.0264993e-7, 1.868938e-6],
            [1.6340124e-9, 4.138497e-7, 1.9341217e-6],
            [1.6498122e-9, 3.1035634e-7, 0e0],
            [1.8198322e-9, 3.3692038e-7, 9.7276896e-8],
            [1.0817346e-9, 4.0466142e-7, 2.5148788e-6],
            [1.1050909e-9, 4.0264007e-7, 2.5470092e-6],
            [1.1004504e-9, 3.5158246e-7, 0e0],
            [9.987375e-10, 4.4740625e-7, 1.4864303e-6],
            [1.014668e-9, 4.4817304e-7, 1.5594427e-6],
            [1.0369664e-9, 3.704516e-7, 0e0],
        ],
    }
}
