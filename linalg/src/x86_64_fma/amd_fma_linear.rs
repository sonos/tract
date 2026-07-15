use crate::frame::mmm::LinearCostModel;

pub fn linear_model() -> LinearCostModel<'static> {
    LinearCostModel {
        default_kernel: "fma_mmm_f32_32x3",
        kernels: &[
            "fma_mmm_f32_16x5",
            "fma_mmm_f32_16x6",
            "fma_mmm_f32_24x4",
            "fma_mmm_f32_32x1",
            "fma_mmm_f32_32x3",
            "fma_mmm_f32_40x2",
            "fma_mmm_f32_64x1",
            "fma_mmm_f32_8x8",
        ],
        coeffs: &[
            [1.553544e-11, 4.0281478e-8, 6.8212915e-7],
            [1.5294823e-11, 4.468052e-8, 7.499282e-7],
            [1.531176e-11, 4.3690513e-8, 2.631608e-7],
            [2.6644824e-11, 2.6694732e-7, 0e0],
            [1.5703909e-11, 4.4871936e-8, 7.7575277e-7],
            [1.6770259e-11, 4.234506e-8, 7.636308e-7],
            [1.2904569e-11, 4.6748923e-7, 0e0],
            [1.961479e-11, 3.4140914e-8, 6.5937513e-7],
        ],
    }
}
