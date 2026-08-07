//! RVV 1.0 matmul kernels.
//!
//! Each kernel pins `vl` to its own `MR`, so it is correct only where
//! `VLMAX >= MR`. The `where(...)` predicates are that constraint, and they
//! read the same `(MR, LMUL)` the assembly was rendered from -- see
//! `RVV_F32_KERNELS` and `RVV_F16_KERNELS` in build.rs, which must stay in
//! step with them.
//!
//! `vlmax_f32` and `vlmax_f16` return 0 without RVV, so the predicates also
//! subsume the `has_rvv` check.

use crate::Ops;
use crate::frame::mmm::ImplementationQuality::ManuallyOptimized;
use crate::pack::PackedFormat;

use super::{vlmax_f16, vlmax_f32};

const VLMAX_F32_M2_GE_8: fn() -> bool = || vlmax_f32(2) >= 8;
const VLMAX_F32_M2_GE_16: fn() -> bool = || vlmax_f32(2) >= 16;
const VLMAX_F32_M4_GE_16: fn() -> bool = || vlmax_f32(4) >= 16;
const VLMAX_F32_M4_GE_32: fn() -> bool = || vlmax_f32(4) >= 32;
const VLMAX_F32_M8_GE_32: fn() -> bool = || vlmax_f32(8) >= 32;
const VLMAX_F32_M8_GE_64: fn() -> bool = || vlmax_f32(8) >= 64;

MMMExternKernel!(rvv_mmm_f32_8x8  <f32>( 8, 8)@(16, 16) where(VLMAX_F32_M2_GE_8)  quality(ManuallyOptimized));
MMMExternKernel!(rvv_mmm_f32_16x8 <f32>(16, 8)@(16, 16) where(VLMAX_F32_M2_GE_16) quality(ManuallyOptimized));
MMMExternKernel!(rvv_mmm_f32_32x1 <f32>(32, 1)@(16, 16) where(VLMAX_F32_M8_GE_32) quality(ManuallyOptimized));
MMMExternKernel!(rvv_mmm_f32_64x1 <f32>(64, 1)@(16, 16) where(VLMAX_F32_M8_GE_64) quality(ManuallyOptimized));

MMMExternKernel!(rvv_mmm_i32_8x8<i32>(8, 8)@(16, 16)
    where(VLMAX_F32_M2_GE_8)
    packing[1] = i8i8 => |k| k.with_packing(PackedFormat::new(DatumType::I8, 8, 16), PackedFormat::new(DatumType::I8, 8, 16));
    quality(ManuallyOptimized)
    store(i8)
);
MMMExternKernel!(rvv_mmm_i32_16x8<i32>(16, 8)@(16, 16)
    where(VLMAX_F32_M2_GE_16)
    packing[1] = i8i8 => |k| k.with_packing(PackedFormat::new(DatumType::I8, 16, 16), PackedFormat::new(DatumType::I8, 8, 16));
    quality(ManuallyOptimized)
    store(i8)
);
MMMExternKernel!(rvv_mmm_i32_16x1<i32>(16, 1)@(16, 1)
    where(VLMAX_F32_M4_GE_16)
    packing[1] = i8i8 => |k| k.with_packing(PackedFormat::new(DatumType::I8, 16, 16), PackedFormat::new(DatumType::I8, 1, 1));
    quality(ManuallyOptimized)
    store(i8)
);
MMMExternKernel!(rvv_mmm_i32_32x1<i32>(32, 1)@(16, 1)
    where(VLMAX_F32_M4_GE_32)
    packing[1] = i8i8 => |k| k.with_packing(PackedFormat::new(DatumType::I8, 32, 16), PackedFormat::new(DatumType::I8, 1, 1));
    quality(ManuallyOptimized)
    store(i8)
);

#[cfg(tract_rvv_zvfh)]
mod zvfh {
    use super::*;
    use crate::f16;

    /// f16 arithmetic needs Zvfh on top of a wide enough vector unit; the
    /// profile does not imply it, and Zvfhmin alone cannot hold an f16
    /// accumulator.
    const VLMAX_F16_M2_GE_16: fn() -> bool = || super::super::has_zvfh() && vlmax_f16(2) >= 16;
    const VLMAX_F16_M2_GE_32: fn() -> bool = || super::super::has_zvfh() && vlmax_f16(2) >= 32;
    const VLMAX_F16_M8_GE_64: fn() -> bool = || super::super::has_zvfh() && vlmax_f16(8) >= 64;
    const VLMAX_F16_M8_GE_128: fn() -> bool = || super::super::has_zvfh() && vlmax_f16(8) >= 128;

    MMMExternKernel!(rvv_mmm_f16_16x8 <f16>( 16, 8)@(16, 16) where(VLMAX_F16_M2_GE_16)  quality(ManuallyOptimized));
    MMMExternKernel!(rvv_mmm_f16_32x8 <f16>( 32, 8)@(16, 16) where(VLMAX_F16_M2_GE_32)  quality(ManuallyOptimized));
    MMMExternKernel!(rvv_mmm_f16_64x1 <f16>( 64, 1)@(16, 16) where(VLMAX_F16_M8_GE_64)  quality(ManuallyOptimized));
    MMMExternKernel!(rvv_mmm_f16_128x1<f16>(128, 1)@(16, 16) where(VLMAX_F16_M8_GE_128) quality(ManuallyOptimized));
}

/// `(name, MR, LMUL, element size)` mirroring the build.rs kernel tables. The
/// i32 entries carry the accumulator LMUL, twice the one their table lists,
/// because that is what their dispatch predicate is written against.
#[cfg(test)]
const GEOMETRIES: &[(&str, usize, usize, usize)] = &[
    ("f32 8x8", 8, 2, 4),
    ("f32 16x8", 16, 2, 4),
    ("f32 32x1", 32, 8, 4),
    ("f32 64x1", 64, 8, 4),
    ("i32 8x8", 8, 2, 4),
    ("i32 16x8", 16, 2, 4),
    ("i32 16x1", 16, 4, 4),
    ("i32 32x1", 32, 4, 4),
    #[cfg(tract_rvv_zvfh)]
    ("f16 16x8", 16, 2, 2),
    #[cfg(tract_rvv_zvfh)]
    ("f16 32x8", 32, 2, 2),
    #[cfg(tract_rvv_zvfh)]
    ("f16 64x1", 64, 8, 2),
    #[cfg(tract_rvv_zvfh)]
    ("f16 128x1", 128, 8, 2),
];

pub fn plug(ops: &mut Ops) {
    ops.mmm_impls.extend_from_slice(&[
        rvv_mmm_f32_8x8.mmm(),
        rvv_mmm_f32_16x8.mmm(),
        rvv_mmm_f32_32x1.mmm(),
        rvv_mmm_f32_64x1.mmm(),
        rvv_mmm_i32_8x8.mmm(),
        rvv_mmm_i32_16x8.mmm(),
        rvv_mmm_i32_16x1.mmm(),
        rvv_mmm_i32_32x1.mmm(),
    ]);
    #[cfg(tract_rvv_zvfh)]
    ops.mmm_impls.extend_from_slice(&[
        zvfh::rvv_mmm_f16_16x8.mmm(),
        zvfh::rvv_mmm_f16_32x8.mmm(),
        zvfh::rvv_mmm_f16_64x1.mmm(),
        zvfh::rvv_mmm_f16_128x1.mmm(),
    ]);
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::frame::mmm::{FusedKerSpec, MatMatMulKer};

    fn supported() -> Vec<bool> {
        #[allow(unused_mut)]
        let mut v = vec![
            rvv_mmm_f32_8x8.is_supported_here(),
            rvv_mmm_f32_16x8.is_supported_here(),
            rvv_mmm_f32_32x1.is_supported_here(),
            rvv_mmm_f32_64x1.is_supported_here(),
            rvv_mmm_i32_8x8.is_supported_here(),
            rvv_mmm_i32_16x8.is_supported_here(),
            rvv_mmm_i32_16x1.is_supported_here(),
            rvv_mmm_i32_32x1.is_supported_here(),
        ];
        #[cfg(tract_rvv_zvfh)]
        v.extend([
            zvfh::rvv_mmm_f16_16x8.is_supported_here(),
            zvfh::rvv_mmm_f16_32x8.is_supported_here(),
            zvfh::rvv_mmm_f16_64x1.is_supported_here(),
            zvfh::rvv_mmm_f16_128x1.is_supported_here(),
        ]);
        v
    }

    /// The generated kernel suites early-return on an unsupported kernel and
    /// count as passes, so a green run says nothing about whether the VLEN
    /// predicates are right. This asserts the dispatch set directly.
    ///
    /// The permissive direction is the dangerous one: a kernel whose MR exceeds
    /// VLMAX computes a short tile, and short is not the same as failing.
    #[test]
    fn dispatch_matches_vlen() {
        let vlenb = super::super::vlenb();
        for ((name, mr, lmul, esize), got) in GEOMETRIES.iter().zip(supported()) {
            let mut want = vlenb * lmul / esize >= *mr;
            if *esize == 2 {
                want &= super::super::has_zvfh();
            }
            eprintln!(
                "VLEN={} zvfh={} {name}: {got} (want {want})",
                vlenb * 8,
                super::super::has_zvfh()
            );
            assert_eq!(got, want, "{name} dispatch disagrees with this hart");
        }
    }

    /// The `vsetvli` guard heading every kernel backstops the predicates above.
    /// Vacuous on a hart wide enough for all of them, hence no assertion on
    /// finding a candidate.
    ///
    /// Meaningful only where V is present: the guard is itself a vector
    /// instruction, so it covers "unit too narrow for this tile" but not "no
    /// unit", where calling at all is a SIGILL rather than a return code.
    #[test]
    fn oversized_tile_refuses_to_run() {
        if !super::super::has_rvv() {
            return;
        }
        #[allow(unused_mut)]
        let mut runners: Vec<Box<dyn Fn() -> isize>> = vec![
            Box::new(|| rvv_mmm_f32_8x8.kernel(&[FusedKerSpec::Done])),
            Box::new(|| rvv_mmm_f32_16x8.kernel(&[FusedKerSpec::Done])),
            Box::new(|| rvv_mmm_f32_32x1.kernel(&[FusedKerSpec::Done])),
            Box::new(|| rvv_mmm_f32_64x1.kernel(&[FusedKerSpec::Done])),
            Box::new(|| rvv_mmm_i32_8x8.kernel(&[FusedKerSpec::Done])),
            Box::new(|| rvv_mmm_i32_16x8.kernel(&[FusedKerSpec::Done])),
            Box::new(|| rvv_mmm_i32_16x1.kernel(&[FusedKerSpec::Done])),
            Box::new(|| rvv_mmm_i32_32x1.kernel(&[FusedKerSpec::Done])),
        ];
        #[cfg(tract_rvv_zvfh)]
        if super::super::has_zvfh() {
            runners.extend::<Vec<Box<dyn Fn() -> isize>>>(vec![
                Box::new(|| zvfh::rvv_mmm_f16_16x8.kernel(&[FusedKerSpec::Done])),
                Box::new(|| zvfh::rvv_mmm_f16_32x8.kernel(&[FusedKerSpec::Done])),
                Box::new(|| zvfh::rvv_mmm_f16_64x1.kernel(&[FusedKerSpec::Done])),
                Box::new(|| zvfh::rvv_mmm_f16_128x1.kernel(&[FusedKerSpec::Done])),
            ]);
        }
        for (((name, ..), ok), run) in GEOMETRIES.iter().zip(supported()).zip(runners) {
            if !ok {
                assert_eq!(run(), 1, "{name} ran on a hart whose VLMAX is below its MR");
                eprintln!("{name}: correctly refused");
            }
        }
    }
}
