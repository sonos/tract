use crate::block_quant::*;
use crate::mmm::ImplementationQuality::ManuallyOptimized;
use crate::pack::PackedFormat;
use crate::Ops;
use tract_data::internal::*;
use DatumType::*;

use super::*;

MMMExternKernel!(fma_mmm_f32_8x8 <f32>(8, 8)@(32,4) where(FMA) quality(ManuallyOptimized));
MMMExternKernel!(fma_mmm_f32_16x6<f32>(16,6)@(32,4) where(FMA) quality(ManuallyOptimized));
MMMExternKernel!(fma_mmm_f32_16x5<f32>(16,5)@(32,4) where(FMA) quality(ManuallyOptimized));
MMMExternKernel!(fma_mmm_f32_24x4<f32>(24,4)@(32,4) where(FMA) quality(ManuallyOptimized));
MMMExternKernel!(fma_mmm_f32_40x2<f32>(40,2)@(32,4) where(FMA) quality(ManuallyOptimized));
MMMExternKernel!(fma_mmm_f32_64x1<f32>(64,1)@(32,4) where(FMA) quality(ManuallyOptimized));

pub fn pq40_r32() -> PackedBlockQuantFormat {
    PackedBlockQuantFormat::new(&Q4_0, 32, 16, false)
}
MMMExternKernel! {fma_mmm_f32_32x1<f32>(32,1)@(32,4) where(FMA)
    packing[1] = q40f32 => |k| k.with_packing_a(pq40_r32());
    packing[2] = q40f16 => |k| k.with_packing(pq40_r32(), PackedFormat::new(F16, 1, 2));
    packing[3] = f16f16 => |k| k.with_packing(PackedFormat::new(F16, 32, 32), PackedFormat::new(F16, 1, 2));
    quality(ManuallyOptimized)
    store(f16)
}
MMMExternKernel!(fma_mmm_f32_32x3<f32>(32,3)@(32,4) where(FMA)
 packing[1] = f32f16 => |k| k.with_packing(f32::packing(32).align(32), f16::packing(3));
 quality(ManuallyOptimized)
 store(f16)
);

MMMExternKernel!(avx512_mmm_f32_128x1<f32>(128, 1)@(64,4) where (AVX512F) quality(ManuallyOptimized));
MMMExternKernel!(avx512_mmm_f32_16x1 <f32>( 16, 1)@(64,4) where (AVX512F) quality(ManuallyOptimized));
MMMExternKernel!(avx512_mmm_f32_16x12<f32>( 16,12)@(64,4) where (AVX512F) quality(ManuallyOptimized));
MMMExternKernel!(avx512_mmm_f32_16x8 <f32>( 16, 8)@(64,4) where (AVX512F) quality(ManuallyOptimized));
MMMExternKernel!(avx512_mmm_f32_32x6 <f32>( 32, 6)@(64,4) where (AVX512F) quality(ManuallyOptimized));
MMMExternKernel!(avx512_mmm_f32_32x5 <f32>( 32, 5)@(64,4) where (AVX512F) quality(ManuallyOptimized));
MMMExternKernel!(avx512_mmm_f32_48x4 <f32>( 48, 4)@(64,4) where (AVX512F) quality(ManuallyOptimized));
MMMExternKernel!(avx512_mmm_f32_64x3 <f32>( 64, 3)@(64,4) where (AVX512F) quality(ManuallyOptimized));
MMMExternKernel!(avx512_mmm_f32_80x2 <f32>( 80, 2)@(64,4) where (AVX512F) quality(ManuallyOptimized));

MMMExternKernel! { avx2_mmm_i32_8x8<i32>(8,8)@(32,4) where(AVX2)
    packing[1] = i8i8 => |k| k.with_packing(PackedFormat::new(DatumType::I8, 8,32), PackedFormat::new(DatumType::I8, 8, 4));
    quality(ManuallyOptimized)
    store(i8)
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

pub fn plug_avx2(ops: &mut Ops) {
    ops.mmm_impls.push(mmm::avx2_mmm_i32_8x8.mmm());
    ops.qmmm_i32 = Box::new(|_, _, _| mmm::avx2_mmm_i32_8x8.mmm());
    log::info!("qmmm_i32: x86_64/avx2 activated");
}

pub fn plug_fma(ops: &mut Ops) {
    ops.mmm_impls.extend([
        fma_mmm_f32_8x8.mmm(),
        fma_mmm_f32_16x5.mmm(),
        fma_mmm_f32_16x6.mmm(),
        fma_mmm_f32_24x4.mmm(),
        fma_mmm_f32_32x3.mmm(),
        fma_mmm_f32_40x2.mmm(),
        fma_mmm_f32_64x1.mmm(),
    ]);

    ops.mmv_f32 = Box::new(|_, _| fma_mmm_f32_64x1.mmm());

    ops.mmm_f32 = Box::new(|_, _, n| {
        if n.is_none() {
            return fma_mmm_f32_16x6.mmm();
        }

        let n = n.unwrap();

        match n {
            1 => unreachable!("should've been mmv"),
            2 => return fma_mmm_f32_40x2.mmm(),
            3 => return fma_mmm_f32_32x3.mmm(),
            4 => return fma_mmm_f32_24x4.mmm(),
            5 => return fma_mmm_f32_16x5.mmm(),
            6 => return fma_mmm_f32_16x6.mmm(),
            8 => return fma_mmm_f32_8x8.mmm(),
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
            0 => fma_mmm_f32_8x8.mmm(),
            1 => fma_mmm_f32_16x6.mmm(),
            2 => fma_mmm_f32_16x5.mmm(),
            3 => fma_mmm_f32_24x4.mmm(),
            4 => fma_mmm_f32_32x3.mmm(),
            5 => fma_mmm_f32_40x2.mmm(),
            _ => unreachable!("not a valid index"),
        }
    });
    log::info!("mmm_f32, mmv_f32: x86_64/fma activated");

    if is_x86_feature_detected!("f16c") {
        ops.mmm_impls.push(mmm::fma_mmm_f32_32x1.mmm()); // q40f32 requires f16c
        log::info!("found f16c, added fake-f16 and q40-able kernels");
    }
}

pub fn plug_avx512f(ops: &mut Ops) {
    ops.mmm_impls.push(avx512_mmm_f32_128x1.mmm());
    ops.mmm_impls.push(avx512_mmm_f32_80x2.mmm());
    ops.mmm_impls.push(avx512_mmm_f32_48x4.mmm());
    ops.mmm_impls.push(avx512_mmm_f32_64x3.mmm());
    ops.mmm_impls.push(avx512_mmm_f32_16x12.mmm());
    ops.mmv_f32 = Box::new(|m, _k| match m {
        Some(m) if m < 31 => avx512_mmm_f32_16x1.mmm(),
        _ => avx512_mmm_f32_128x1.mmm(),
    });

    ops.mmm_f32 = Box::new(|m, _, n| match (m, n) {
        (_, Some(1)) => unreachable!("should've been mmv"),
        (_, Some(2)) => avx512_mmm_f32_80x2.mmm(),
        (Some(m), _) if m <= 16 => mmm::avx512_mmm_f32_16x12.mmm(),
        (_, Some(n)) if n % 4 == 0 && n % 3 != 0 && n < 32 => avx512_mmm_f32_48x4.mmm(),
        (_, Some(n)) if n < 32 => avx512_mmm_f32_64x3.mmm(),
        _ => avx512_mmm_f32_16x12.mmm(),
    });
    log::info!("mmm_f32, mmv_f32: x86_64/avx512f activated");
}
