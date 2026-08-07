//! RVV 1.0 matmul kernels.
//!
//! Each kernel pins `vl` to its own `MR`, so it is correct only where
//! `VLMAX >= MR`. The declared instruction set is that constraint: the
//! `(MR, LMUL)` pairs build.rs renders the assembly from split into tiles every
//! RVV 1.0 hart can reach and tiles needing `VLEN >= 256`, and each kernel
//! re-checks the granted `vl` on entry, so a predicate the hart disagrees with
//! is a clean refusal rather than a short tile.

use crate::DatumType;
use crate::isa::{Isa, IsaSet};
use crate::mmm::{Query, Suitable};

MMMExternKernel!(riscv64; rvv_mmm_f32_8x8  <f32>( 8, 8)@(16, 16) isa(RiscV64V));
MMMExternKernel!(riscv64; rvv_mmm_f32_16x8 <f32>(16, 8)@(16, 16) isa(RiscV64Vlen256));
MMMExternKernel!(riscv64; rvv_mmm_f32_32x1 <f32>(32, 1)@(16, 16) isa(RiscV64V));
MMMExternKernel!(riscv64; rvv_mmm_f32_64x1 <f32>(64, 1)@(16, 16) isa(RiscV64Vlen256));

/// The widest tile the hart can reach, for each of the two shapes. A vector
/// unit is the only thing these kernels are ranked on -- there is one RVV
/// kernel set, not a per-chip family -- so the choice is `n == 1` or not, and
/// then the width.
fn preferred(
    isa: &IsaSet,
    dt: DatumType,
    query: &Query,
    _suitable: &[Suitable],
) -> Option<&'static str> {
    if dt != DatumType::F32 {
        return None;
    }
    let wide = isa.has(Isa::RiscV64Vlen256);
    Some(match query.n {
        Some(1) if wide => rvv_mmm_f32_64x1.name.as_str(),
        Some(1) => rvv_mmm_f32_32x1.name.as_str(),
        _ if wide => rvv_mmm_f32_16x8.name.as_str(),
        _ => rvv_mmm_f32_8x8.name.as_str(),
    })
}

inventory::submit! {
    crate::mmm_tiers::MmmTier {
        arch: Some(crate::isa::Arch::RiscV64),
        precedence: 1,
        name: "rvv",
        applies: |isa| isa.has(Isa::RiscV64V),
        preferred,
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::frame::mmm::{FusedKerSpec, MatMatMulKer};

    /// `(name, MR, LMUL)` mirroring the build.rs kernel table.
    const GEOMETRIES: &[(&str, usize, usize)] =
        &[("8x8", 8, 2), ("16x8", 16, 2), ("32x1", 32, 8), ("64x1", 64, 8)];

    fn runnable() -> [bool; 4] {
        [
            rvv_mmm_f32_8x8.runnable(),
            rvv_mmm_f32_16x8.runnable(),
            rvv_mmm_f32_32x1.runnable(),
            rvv_mmm_f32_64x1.runnable(),
        ]
    }

    /// The generated kernel suites early-return on an unrunnable kernel and
    /// count as passes, so a green run says nothing about whether the declared
    /// instruction sets are right. This asserts the dispatch set directly.
    ///
    /// The permissive direction is the dangerous one: a kernel whose MR exceeds
    /// VLMAX computes a short tile, and short is not the same as failing.
    #[test]
    fn dispatch_matches_vlen() {
        let vlenb = super::super::vlenb();
        for ((name, mr, lmul), got) in GEOMETRIES.iter().zip(runnable()) {
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
        for (((name, ..), ok), run) in GEOMETRIES.iter().zip(runnable()).zip(runners) {
            if !ok {
                assert_eq!(run(), 1, "{name} ran on a hart whose VLMAX is below its MR");
                eprintln!("{name}: correctly refused");
            }
        }
    }
}
