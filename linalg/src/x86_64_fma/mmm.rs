use crate::Ops;
use crate::block_quant::*;
use crate::mmm::ImplementationQuality::ManuallyOptimized;
use crate::mmm::MatMatMul;
use crate::pack::PackedFormat;

use super::*;

/// One candidate kernel in a dispatcher's pool, with its tile geometry
/// and a relative-throughput scale (1.0 = baseline, used to break
/// near-ties between kernels with similar tile waste).
#[derive(Clone, Copy)]
struct KernelChoice {
    mr: usize,
    nr: usize,
    scale: f32,
    ctor: fn() -> Box<dyn MatMatMul>,
}

/// Fraction of the M-or-N axis covered by useful work after rounding up
/// to the kernel's tile size. 1.0 = exact fit; smaller is worse.
/// Empty axis (d == 0) is treated as "no waste" — no work to misallocate.
fn tile_util(d: usize, tile: usize) -> f32 {
    if d == 0 {
        return 1.0;
    }
    let batches = d.div_ceil(tile);
    d as f32 / (batches * tile) as f32
}

/// Pick the kernel that maximises `scale * m_util * n_util`. Ties are
/// broken first in favour of fewer total tile passes (less loop
/// overhead), then in favour of larger `nr` (more K-loop amortisation
/// per inner iteration). An unknown M or N is treated as
/// "large enough" — its utilisation contribution is 1.0.
fn pick_mmm(candidates: &[KernelChoice], m: Option<usize>, n: Option<usize>) -> Box<dyn MatMatMul> {
    let key = |c: &KernelChoice| -> (f32, i32, i32) {
        let m_u = m.map(|m| tile_util(m, c.mr)).unwrap_or(1.0);
        let n_u = n.map(|n| tile_util(n, c.nr)).unwrap_or(1.0);
        let m_b = m.map(|m| m.div_ceil(c.mr)).unwrap_or(1) as i32;
        let n_b = n.map(|n| n.div_ceil(c.nr)).unwrap_or(1) as i32;
        (c.scale * m_u * n_u, -(m_b * n_b), c.nr as i32)
    };
    let best = candidates
        .iter()
        .max_by(|a, b| key(a).partial_cmp(&key(b)).unwrap())
        .expect("non-empty kernel pool");
    (best.ctor)()
}

MMMExternKernel!(fma_mmm_f32_8x8 <f32>(8, 8)@(256,4) where(FMA) quality(ManuallyOptimized));
MMMExternKernel!(fma_mmm_f32_16x6<f32>(16,6)@(256,4) where(FMA) quality(ManuallyOptimized));
MMMExternKernel!(fma_mmm_f32_16x5<f32>(16,5)@(256,4) where(FMA) quality(ManuallyOptimized));
MMMExternKernel!(fma_mmm_f32_24x4<f32>(24,4)@(256,4) where(FMA) quality(ManuallyOptimized));
MMMExternKernel!(fma_mmm_f32_40x2<f32>(40,2)@(256,4) where(FMA) quality(ManuallyOptimized));
MMMExternKernel!(fma_mmm_f32_64x1<f32>(64,1)@(256,4) where(FMA) quality(ManuallyOptimized));

pub fn pq40_r32() -> PackedBlockQuantFormat {
    PackedBlockQuantFormat::new(&Q4_0, 32, 16, false)
}
MMMExternKernel! {fma_mmm_f32_32x1<f32>(32,1)@(256,4) where(FMA)
    packing[1] = q40f32 => |k| k.with_packing_a(pq40_r32());
    packing[2] = q40f16 => |k| k.with_packing(pq40_r32(), f16::packing(1));
    packing[3] = f16f16 => |k| k.with_packing(f16::packing(32), f16::packing(1));
    packing[4] = f16f32 => |k| k.with_packing(f16::packing(32), f32::packing(1));
    packing[5] = f32f16 => |k| k.with_packing(f32::packing(32), f16::packing(1));
    quality(ManuallyOptimized)
    store(f16)
}
MMMExternKernel!(fma_mmm_f32_32x3<f32>(32,3)@(256,4) where(FMA)
 packing[1] = f32f16 => |k| k.with_packing(f32::packing(32).align(256), f16::packing(3));
 packing[2] = f16f32 => |k| k.with_packing(f16::packing(32).align(256), f32::packing(3));
 packing[3] = f16f16 => |k| k.with_packing(f16::packing(32).align(256), f16::packing(3));
 quality(ManuallyOptimized)
 store(f16)
);

MMMExternKernel!(avx512_mmm_f32_128x1<f32>(128, 1)@(512,4) where (AVX512F) quality(ManuallyOptimized));
MMMExternKernel!(avx512_mmm_f32_16x1 <f32>( 16, 1)@(512,4) where (AVX512F) quality(ManuallyOptimized));
MMMExternKernel!(avx512_mmm_f32_16x12<f32>( 16,12)@(512,4) where (AVX512F) quality(ManuallyOptimized));
MMMExternKernel!(avx512_mmm_f32_16x8 <f32>( 16, 8)@(512,4) where (AVX512F) quality(ManuallyOptimized));
MMMExternKernel!(avx512_mmm_f32_32x6 <f32>( 32, 6)@(512,4) where (AVX512F) quality(ManuallyOptimized));
MMMExternKernel!(avx512_mmm_f32_32x5 <f32>( 32, 5)@(512,4) where (AVX512F) quality(ManuallyOptimized));
MMMExternKernel!(avx512_mmm_f32_48x4 <f32>( 48, 4)@(512,4) where (AVX512F) quality(ManuallyOptimized));
MMMExternKernel!(avx512_mmm_f32_64x3 <f32>( 64, 3)@(512,4) where (AVX512F) quality(ManuallyOptimized));
MMMExternKernel!(avx512_mmm_f32_80x2 <f32>( 80, 2)@(512,4) where (AVX512F) quality(ManuallyOptimized));

MMMExternKernel! { avx2_mmm_i32_8x8<i32>(8,8)@(256,4) where(AVX2)
    packing[1] = i8i8 => |k| k.with_packing(PackedFormat::new(DatumType::I8, 8, 256), PackedFormat::new(DatumType::I8, 8, 4));
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

    // Hand-tuned for low N; calibration came from past measurements.
    // For other N, fall back to a generic (M, N)-aware tile-utilisation
    // picker over the same kernel pool.
    const FMA_CHOICES: &[KernelChoice] = &[
        KernelChoice { mr: 8, nr: 8, scale: 44.0 / 60.0, ctor: || fma_mmm_f32_8x8.mmm() },
        KernelChoice { mr: 16, nr: 6, scale: 54.0 / 60.0, ctor: || fma_mmm_f32_16x6.mmm() },
        KernelChoice { mr: 16, nr: 5, scale: 54.0 / 60.0, ctor: || fma_mmm_f32_16x5.mmm() },
        KernelChoice { mr: 24, nr: 4, scale: 54.0 / 60.0, ctor: || fma_mmm_f32_24x4.mmm() },
        KernelChoice { mr: 32, nr: 3, scale: 54.0 / 60.0, ctor: || fma_mmm_f32_32x3.mmm() },
        KernelChoice { mr: 40, nr: 2, scale: 54.0 / 60.0, ctor: || fma_mmm_f32_40x2.mmm() },
    ];

    ops.mmm_f32 = Box::new(|m, _, n| match n {
        None => fma_mmm_f32_16x6.mmm(),
        Some(1) => unreachable!("should've been mmv"),
        Some(2) => fma_mmm_f32_40x2.mmm(),
        Some(3) => fma_mmm_f32_32x3.mmm(),
        Some(4) => fma_mmm_f32_24x4.mmm(),
        Some(5) => fma_mmm_f32_16x5.mmm(),
        Some(6) => fma_mmm_f32_16x6.mmm(),
        Some(8) => fma_mmm_f32_8x8.mmm(),
        Some(_) => pick_mmm(FMA_CHOICES, m, n),
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
    ops.mmm_impls.push(avx512_mmm_f32_32x6.mmm());
    ops.mmm_impls.push(avx512_mmm_f32_32x5.mmm());
    ops.mmm_impls.push(avx512_mmm_f32_16x12.mmm());
    ops.mmm_impls.push(avx512_mmm_f32_16x8.mmm());
    ops.mmv_f32 = Box::new(|m, _k| match m {
        Some(m) if m < 31 => avx512_mmm_f32_16x1.mmm(),
        _ => avx512_mmm_f32_128x1.mmm(),
    });

    // No measured per-kernel scaling on AVX-512 yet; all kernels start
    // at 1.0 and the picker decides on (M, N) tile waste alone.
    const AVX512_CHOICES: &[KernelChoice] = &[
        KernelChoice { mr: 16, nr: 8, scale: 1.0, ctor: || avx512_mmm_f32_16x8.mmm() },
        KernelChoice { mr: 16, nr: 12, scale: 1.0, ctor: || avx512_mmm_f32_16x12.mmm() },
        KernelChoice { mr: 32, nr: 5, scale: 1.0, ctor: || avx512_mmm_f32_32x5.mmm() },
        KernelChoice { mr: 32, nr: 6, scale: 1.0, ctor: || avx512_mmm_f32_32x6.mmm() },
        KernelChoice { mr: 48, nr: 4, scale: 1.0, ctor: || avx512_mmm_f32_48x4.mmm() },
        KernelChoice { mr: 64, nr: 3, scale: 1.0, ctor: || avx512_mmm_f32_64x3.mmm() },
        KernelChoice { mr: 80, nr: 2, scale: 1.0, ctor: || avx512_mmm_f32_80x2.mmm() },
        KernelChoice { mr: 128, nr: 1, scale: 1.0, ctor: || avx512_mmm_f32_128x1.mmm() },
    ];

    ops.mmm_f32 = Box::new(|m, _, n| {
        if let Some(1) = n {
            unreachable!("should've been mmv");
        }
        pick_mmm(AVX512_CHOICES, m, n)
    });
    log::info!("mmm_f32, mmv_f32: x86_64/avx512f activated");
}
