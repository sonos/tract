use crate::frame::mmm::kernel::MatMatMulKer;
use crate::frame::ElementWiseImpl;
use crate::Ops;

pub mod mmm;
pub mod sigmoid;
pub mod tanh;

mod intel;

pub fn plug(ops: &mut Ops) {
    if is_x86_feature_detected!("fma") {
        ops.mmv_f32 = Box::new(|_, _| mmm::fma_mmm_f32_64x1::mmm());

        ops.mmm_f32 = Box::new(|_, _, n| {
            if n.is_none() {
                return mmm::fma_mmm_f32_16x6::mmm();
            }

            let n = n.unwrap();

            match n {
                1 => unreachable!("should've been mmv"),
                2 => return mmm::fma_mmm_f32_40x2::mmm(),
                3 => return mmm::fma_mmm_f32_32x3::mmm(),
                4 => return mmm::fma_mmm_f32_24x4::mmm(),
                5 => return mmm::fma_mmm_f32_16x5::mmm(),
                6 => return mmm::fma_mmm_f32_16x6::mmm(),
                8 => return mmm::fma_mmm_f32_8x8::mmm(),
                _ => {}
            };

            let scaling_baseline = 60.0;
            let kernel_normalized_perf = [
                43.92 / scaling_baseline, // 8x8
                54.47 / scaling_baseline, // 2x6
                53.37 / scaling_baseline, // 2x5
                57.86 / scaling_baseline, // 3x4
                57.66 / scaling_baseline, // 4x3
                53.93 / scaling_baseline, // 5x2
            ];
            let efficiencies = [
                n as f32 / ((n as f32 / 8.0).ceil() * 8.0) * kernel_normalized_perf[0],
                n as f32 / ((n as f32 / 6.0).ceil() * 6.0) * kernel_normalized_perf[1],
                n as f32 / ((n as f32 / 5.0).ceil() * 5.0) * kernel_normalized_perf[2],
                n as f32 / ((n as f32 / 4.0).ceil() * 4.0) * kernel_normalized_perf[3],
                n as f32 / ((n as f32 / 3.0).ceil() * 3.0) * kernel_normalized_perf[4],
                n as f32 / ((n as f32 / 2.0).ceil() * 2.0) * kernel_normalized_perf[5],
            ];

            let best_idx = efficiencies.iter().copied().enumerate().fold((0, 0.0), |max, val| {
                if max.1 < val.1 {
                    val
                } else {
                    max
                }
            });

            match best_idx.0 {
                0 => return mmm::fma_mmm_f32_8x8::mmm(),
                1 => return mmm::fma_mmm_f32_16x6::mmm(),
                2 => return mmm::fma_mmm_f32_16x5::mmm(),
                3 => return mmm::fma_mmm_f32_24x4::mmm(),
                4 => return mmm::fma_mmm_f32_32x3::mmm(),
                5 => return mmm::fma_mmm_f32_40x2::mmm(),
                _ => unreachable!("not a valid index"),
            }
        });
        ops.mmm_f32_impls.push(mmm::fma_mmm_f32_16x6::mmm());
        ops.mmm_f32_impls.push(mmm::fma_mmm_f32_16x5::mmm());
        ops.mmm_f32_impls.push(mmm::fma_mmm_f32_24x4::mmm());
        ops.mmm_f32_impls.push(mmm::fma_mmm_f32_32x3::mmm());
        ops.mmm_f32_impls.push(mmm::fma_mmm_f32_40x2::mmm());
        ops.mmm_f32_impls.push(mmm::fma_mmm_f32_8x8::mmm());

        ops.sigmoid_f32 = Box::new(|| Box::new(ElementWiseImpl::<sigmoid::SigmoidF32, f32>::new()));
        ops.tanh_f32 = Box::new(|| Box::new(ElementWiseImpl::<tanh::TanhF32, f32>::new()));
        log::info!("mmm_f32, sigmoid_f32, tanh_f32: x86_64/fma activated");
    }
    if is_x86_feature_detected!("avx2") {
        ops.qmmm_i32 = Box::new(|_, _, _| mmm::avx2_mmm_i32_8x8::mmm());
        log::info!("mmm_i8_i8 and mmm_i8_i32: x86_64/avx2 activated");
    }
}
