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
            [1.36334164e-11, 3.8181614e-8, 2.6931977e-7],
            [1.3394619e-11, 4.289206e-8, 5.7438604e-7],
            [1.3691508e-11, 4.6516156e-8, 0e0],
            [3.2835155e-11, 2.5162239e-8, 8.746347e-7],
            [1.5176855e-11, 4.8502574e-8, 0e0],
            [1.760099e-11, 4.215597e-8, 1.0434346e-6],
            [2.9756902e-11, 3.138118e-8, 1.0509837e-5],
            [1.6825891e-11, 3.156506e-8, 6.957331e-7],
        ],
    }
}
