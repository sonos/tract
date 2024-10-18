use crate::frame::element_wise::ElementWiseKer;
use crate::frame::reduce::{MapReduceKer, ReduceKer};
use crate::x86_64_fma::softmax::x86_64_fma_softmax2_fastcompact_f32_32n;
use crate::Ops;

pub mod mmm;

pub mod by_scalar;
mod intel;
pub mod max;
pub mod panel_extract;
pub mod softmax;

const AVX2: fn() -> bool = || is_x86_feature_detected!("avx2");
const FMA: fn() -> bool = || is_x86_feature_detected!("fma");
const AVX512F: fn() -> bool = || is_x86_feature_detected!("avx512f");

tanh_impl!(f32, fma_tanh_f32, 8, 8, is_x86_feature_detected!("fma"));
sigmoid_impl!(f32, fma_sigmoid_f32, 8, 8, is_x86_feature_detected!("fma"));

fn plug_avx2(ops: &mut Ops) {
    ops.mmm_impls.push(mmm::avx2_mmm_i32_8x8.mmm());
    ops.qmmm_i32 = Box::new(|_, _, _| mmm::avx2_mmm_i32_8x8.mmm());
    log::info!("qmmm_i32: x86_64/avx2 activated");
}

fn plug_fma(ops: &mut Ops) {
    mmm::plug(ops);
    ops.mmm_impls.extend([
        mmm::fma_mmm_f32_8x8.mmm(),
        mmm::fma_mmm_f32_16x5.mmm(),
        mmm::fma_mmm_f32_16x6.mmm(),
        mmm::fma_mmm_f32_24x4.mmm(),
        mmm::fma_mmm_f32_32x3.mmm(),
        mmm::fma_mmm_f32_40x2.mmm(),
        mmm::fma_mmm_f32_64x1.mmm(),
    ]);
    panel_extract::plug(ops);

    ops.mmv_f32 = Box::new(|_, _| mmm::fma_mmm_f32_64x1.mmm());

    ops.mmm_f32 = Box::new(|_, _, n| {
        if n.is_none() {
            return mmm::fma_mmm_f32_16x6.mmm();
        }

        let n = n.unwrap();

        match n {
            1 => unreachable!("should've been mmv"),
            2 => return mmm::fma_mmm_f32_40x2.mmm(),
            3 => return mmm::fma_mmm_f32_32x3.mmm(),
            4 => return mmm::fma_mmm_f32_24x4.mmm(),
            5 => return mmm::fma_mmm_f32_16x5.mmm(),
            6 => return mmm::fma_mmm_f32_16x6.mmm(),
            8 => return mmm::fma_mmm_f32_8x8.mmm(),
            _ => {}
        };

        let scaling_baseline = 60.0;
        let kernel_normalized_perf = [
            44.0 / scaling_baseline, // 8x8
            54.0 / scaling_baseline, // 2x6
            54.0 / scaling_baseline, // 2x5
            54.0 / scaling_baseline, // 3x4
            54.0 / scaling_baseline, // 4x3
            54.0 / scaling_baseline, // 5x2
        ];

        fn compute_efficiency(n: usize, kernel_width: usize, scale: f32) -> f32 {
            let kernel_width = kernel_width as f32;
            let n = n as f32;
            let batch_count = (n / kernel_width).ceil();
            let actual_count = batch_count * kernel_width;
            let multi_batch_penalty = 1.0 - batch_count / 100.0;
            n / actual_count * scale * multi_batch_penalty
        }

        let efficiencies = [
            compute_efficiency(n, 8, kernel_normalized_perf[0]),
            compute_efficiency(n, 6, kernel_normalized_perf[1]),
            compute_efficiency(n, 5, kernel_normalized_perf[2]),
            compute_efficiency(n, 4, kernel_normalized_perf[3]),
            compute_efficiency(n, 3, kernel_normalized_perf[4]),
            compute_efficiency(n, 2, kernel_normalized_perf[5]),
        ];

        let best_idx = efficiencies.iter().copied().enumerate().fold((0, 0.0), |max, val| {
            if val.1 > max.1 {
                val
            } else {
                max
            }
        });

        match best_idx.0 {
            0 => mmm::fma_mmm_f32_8x8.mmm(),
            1 => mmm::fma_mmm_f32_16x6.mmm(),
            2 => mmm::fma_mmm_f32_16x5.mmm(),
            3 => mmm::fma_mmm_f32_24x4.mmm(),
            4 => mmm::fma_mmm_f32_32x3.mmm(),
            5 => mmm::fma_mmm_f32_40x2.mmm(),
            _ => unreachable!("not a valid index"),
        }
    });

    ops.sigmoid_f32 = Box::new(|| fma_sigmoid_f32::ew());
    ops.tanh_f32 = Box::new(|| fma_tanh_f32::ew());

    ops.mul_by_scalar_f32 = Box::new(|| by_scalar::x86_64_avx_f32_mul_by_scalar_32n::ew());
    ops.max_f32 = Box::new(|| max::x86_64_fma_max_f32_32n::red());
    ops.softmax2_fastcompact_f32 = Box::new(|| x86_64_fma_softmax2_fastcompact_f32_32n::red());

    if is_x86_feature_detected!("f16c") {
        ops.mmm_impls.push(mmm::fma_mmm_f32_32x1.mmm()); // q40f32 requires f16c
        log::info!("found f16c, added fake-f16 and q40-able kernels");
    }

    log::info!("mmm_f32, mmv_f32, sigmoid_f32, tanh_f32: x86_64/fma activated");
}

fn plug_avx512f(ops: &mut Ops) {
    ops.mmm_impls.push(mmm::avx512_mmm_f32_128x1.mmm());
    ops.mmm_impls.push(mmm::avx512_mmm_f32_80x2.mmm());
    ops.mmm_impls.push(mmm::avx512_mmm_f32_48x4.mmm());
    ops.mmm_impls.push(mmm::avx512_mmm_f32_64x3.mmm());
    ops.mmm_impls.push(mmm::avx512_mmm_f32_16x12.mmm());
    ops.mmv_f32 = Box::new(|m, _k| match m {
        Some(m) if m < 31 => mmm::avx512_mmm_f32_16x1.mmm(),
        _ => mmm::avx512_mmm_f32_128x1.mmm(),
    });

    ops.mmm_f32 = Box::new(|m, _, n| match (m, n) {
        (_, Some(1)) => unreachable!("should've been mmv"),
        (_, Some(2)) => mmm::avx512_mmm_f32_80x2.mmm(),
        (Some(m), _) if m <= 16 => mmm::avx512_mmm_f32_16x12.mmm(),
        (_, Some(n)) if n % 4 == 0 && n % 3 != 0 && n < 32 => mmm::avx512_mmm_f32_48x4.mmm(),
        (_, Some(n)) if n < 32 => mmm::avx512_mmm_f32_64x3.mmm(),
        _ => mmm::avx512_mmm_f32_16x12.mmm(),
    });
    log::info!("mmm_f32, mmv_f32: x86_64/avx512f activated");
}

pub fn plug(ops: &mut Ops) {
    if is_x86_feature_detected!("avx2") {
        plug_avx2(ops);
        if is_x86_feature_detected!("fma") {
            plug_fma(ops);
            if is_x86_feature_detected!("avx512f") {
                plug_avx512f(ops);
            }
        }
    }
}
