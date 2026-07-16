use crate::frame::mmm::LinearCostModel;

pub fn linear_model() -> LinearCostModel<'static> {
    LinearCostModel {
        default_kernel: "avx512_mmm_f32_64x3",
        kernels: &[
            "avx512_mmm_f32_128x1",
            "avx512_mmm_f32_16x12",
            "avx512_mmm_f32_16x8",
            "avx512_mmm_f32_32x5",
            "avx512_mmm_f32_32x6",
            "avx512_mmm_f32_48x4",
            "avx512_mmm_f32_64x3",
            "avx512_mmm_f32_80x2",
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
            [9.382483e-11, 0e0, 0e0],
            [1.5992785e-11, 5.857444e-8, 0e0],
            [1.6915367e-11, 4.6767095e-8, 0e0],
            [1.9238229e-11, 3.677961e-8, 0e0],
            [1.820184e-11, 5.616792e-8, 0e0],
            [2.4005508e-11, 0e0, 0e0],
            [2.985042e-11, 0e0, 0e0],
            [4.497658e-11, 0e0, 0e0],
            [2.1900143e-11, 2.063884e-8, 0e0],
            [2.0232802e-11, 2.2378256e-8, 0e0],
            [2.4201151e-11, 9.501525e-9, 0e0],
            [7.916074e-11, 0e0, 0e0],
            [2.9781747e-11, 0e0, 0e0],
            [4.2898976e-11, 0e0, 0e0],
            [8.390146e-11, 0e0, 0e0],
            [2.0536774e-11, 3.9820076e-8, 0e0],
        ],
    }
}
