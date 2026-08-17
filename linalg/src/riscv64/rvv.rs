//! RVV 1.0 matmul kernels.
//!
//! Each kernel pins `vl` to its own `MR`, so it is correct only where
//! `VLMAX >= MR`. The `where(...)` predicates are that constraint, and they
//! read the same `(MR, LMUL)` the assembly was rendered from -- see
//! `RVV_F32_KERNELS` in build.rs, which must stay in step with them.
//!
//! `vlmax_f32` returns 0 without RVV, so the predicates also subsume the
//! `has_rvv` check.

use crate::Ops;
use crate::frame::mmm::ImplementationQuality::ManuallyOptimized;

use super::vlmax_f32;

const VLMAX_M2_GE_8: fn() -> bool = || vlmax_f32(2) >= 8;
const VLMAX_M2_GE_16: fn() -> bool = || vlmax_f32(2) >= 16;
const VLMAX_M8_GE_32: fn() -> bool = || vlmax_f32(8) >= 32;
const VLMAX_M8_GE_64: fn() -> bool = || vlmax_f32(8) >= 64;

MMMExternKernel!(rvv_mmm_f32_8x8  <f32>( 8, 8)@(16, 16) where(VLMAX_M2_GE_8)  quality(ManuallyOptimized));
MMMExternKernel!(rvv_mmm_f32_16x8 <f32>(16, 8)@(16, 16) where(VLMAX_M2_GE_16) quality(ManuallyOptimized));
MMMExternKernel!(rvv_mmm_f32_32x1 <f32>(32, 1)@(16, 16) where(VLMAX_M8_GE_32) quality(ManuallyOptimized));
MMMExternKernel!(rvv_mmm_f32_64x1 <f32>(64, 1)@(16, 16) where(VLMAX_M8_GE_64) quality(ManuallyOptimized));

/// `(name, MR, LMUL)` mirroring the build.rs kernel table.
#[cfg(test)]
const GEOMETRIES: &[(&str, usize, usize)] =
    &[("8x8", 8, 2), ("16x8", 16, 2), ("32x1", 32, 8), ("64x1", 64, 8)];

pub fn plug(ops: &mut Ops) {
    ops.mmm_impls.extend_from_slice(&[
        rvv_mmm_f32_8x8.mmm(),
        rvv_mmm_f32_16x8.mmm(),
        rvv_mmm_f32_32x1.mmm(),
        rvv_mmm_f32_64x1.mmm(),
    ]);
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::frame::mmm::{FusedKerSpec, MatMatMulKer};

    fn supported() -> [bool; 4] {
        [
            rvv_mmm_f32_8x8.is_supported_here(),
            rvv_mmm_f32_16x8.is_supported_here(),
            rvv_mmm_f32_32x1.is_supported_here(),
            rvv_mmm_f32_64x1.is_supported_here(),
        ]
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
        for ((name, mr, lmul), got) in GEOMETRIES.iter().zip(supported()) {
            let want = vlenb * lmul / std::mem::size_of::<f32>() >= *mr;
            eprintln!("VLEN={} {name}: {got} (want {want})", vlenb * 8);
            assert_eq!(got, want, "{name} dispatch disagrees with VLEN={}", vlenb * 8);
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
        let runners: [&dyn Fn() -> isize; 4] = [
            &|| rvv_mmm_f32_8x8.kernel(&[FusedKerSpec::Done]),
            &|| rvv_mmm_f32_16x8.kernel(&[FusedKerSpec::Done]),
            &|| rvv_mmm_f32_32x1.kernel(&[FusedKerSpec::Done]),
            &|| rvv_mmm_f32_64x1.kernel(&[FusedKerSpec::Done]),
        ];
        for (((name, ..), ok), run) in GEOMETRIES.iter().zip(supported()).zip(runners) {
            if !ok {
                assert_eq!(run(), 1, "{name} ran on a hart whose VLMAX is below its MR");
                eprintln!("{name}: correctly refused");
            }
        }
    }
}
