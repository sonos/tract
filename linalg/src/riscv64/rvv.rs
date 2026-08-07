//! RVV 1.0 matmul kernels.
//!
//! Each kernel pins `vl` to its own `MR`, so it is correct only where
//! `VLMAX >= MR`. The declared instruction set is that constraint: the
//! `(MR, LMUL)` pairs build.rs renders the assembly from split into tiles every
//! RVV 1.0 hart can reach and tiles needing `VLEN >= 256`, and each kernel
//! re-checks the granted `vl` on entry, so a predicate the hart disagrees with
//! is a clean refusal rather than a short tile.
//!
//! The f16 set is the same four shapes at `SEW=16`, where `VLMAX` doubles and
//! so does every tile height for a given `LMUL` -- which is why the two tables
//! declare the same pair of widths for tiles twice as tall. They need Zvfh on
//! top, and the assembler needs to know it: `tract_rvv_zvfh` is the build-time
//! half of that, and without it f16 stays on the generic kernels.
//!
//! The i32 set runs its i8 inner loop at half the accumulator `LMUL`, so the
//! `(MR, LMUL)` in its build.rs table is not the pair its width follows from --
//! the accumulators are, and they sit one `LMUL` step above. That lands the four
//! tiles on the same two widths again.

use crate::DatumType;
use crate::isa::{Isa, IsaSet};
use crate::mmm::{Query, Suitable};
use crate::pack::PackedFormat;

MMMExternKernel!(riscv64; rvv_mmm_f32_8x8  <f32>( 8, 8)@(16, 16) isa(RiscV64V));
MMMExternKernel!(riscv64; rvv_mmm_f32_16x8 <f32>(16, 8)@(16, 16) isa(RiscV64Vlen256));
MMMExternKernel!(riscv64; rvv_mmm_f32_32x1 <f32>(32, 1)@(16, 16) isa(RiscV64V));
MMMExternKernel!(riscv64; rvv_mmm_f32_64x1 <f32>(64, 1)@(16, 16) isa(RiscV64Vlen256));

MMMExternKernel!(riscv64; rvv_mmm_i32_8x8 <i32>( 8, 8)@(16, 16) isa(RiscV64V)
    packing[1] = i8i8 => |k| k.with_packing(PackedFormat::new(DatumType::I8, 8, 16), PackedFormat::new(DatumType::I8, 8, 16));
    store(i8)
);
MMMExternKernel!(riscv64; rvv_mmm_i32_16x8<i32>(16, 8)@(16, 16) isa(RiscV64Vlen256)
    packing[1] = i8i8 => |k| k.with_packing(PackedFormat::new(DatumType::I8, 16, 16), PackedFormat::new(DatumType::I8, 8, 16));
    store(i8)
);
/// The wide step's matvec is `32x1`, a packing group away from its `16x8`, so on a wide hart
/// this kernel is the only GEMV sharing the `MR = 16` packing and dropping it leaves that group
/// usable for one role out of two. There it is a peer of the step above. On a narrower hart it
/// already sits beside `8x8` on its own rung, and lifting it there would drop that matrix kernel
/// instead — the same hole, at the other end of the ladder.
const I32_16X1_PEER: fn() -> isize = || {
    if crate::isa::native().has(Isa::RiscV64Vlen256) {
        crate::isa::peer_of(Isa::RiscV64V, Isa::RiscV64Vlen256)
    } else {
        0
    }
};

MMMExternKernel!(riscv64; rvv_mmm_i32_16x1<i32>(16, 1)@(16, 1) isa(RiscV64V)
    packing[1] = i8i8 => |k| k.with_packing(PackedFormat::new(DatumType::I8, 16, 16), PackedFormat::new(DatumType::I8, 1, 1));
    boost(I32_16X1_PEER)
    store(i8)
);
MMMExternKernel!(riscv64; rvv_mmm_i32_32x1<i32>(32, 1)@(16, 1) isa(RiscV64Vlen256)
    packing[1] = i8i8 => |k| k.with_packing(PackedFormat::new(DatumType::I8, 32, 16), PackedFormat::new(DatumType::I8, 1, 1));
    store(i8)
);

#[cfg(tract_rvv_zvfh)]
mod zvfh {
    use crate::f16;

    MMMExternKernel!(riscv64; rvv_mmm_f16_16x8 <f16>( 16, 8)@(16, 16) isa(RiscV64V, RiscV64Zvfh));
    MMMExternKernel!(riscv64; rvv_mmm_f16_32x8 <f16>( 32, 8)@(16, 16) isa(RiscV64Vlen256, RiscV64Zvfh));
    MMMExternKernel!(riscv64; rvv_mmm_f16_64x1 <f16>( 64, 1)@(16, 16) isa(RiscV64V, RiscV64Zvfh));
    MMMExternKernel!(riscv64; rvv_mmm_f16_128x1<f16>(128, 1)@(16, 16) isa(RiscV64Vlen256, RiscV64Zvfh));
}

/// The widest tile the hart can reach, for each of the two shapes. A vector
/// unit is the only thing these kernels are ranked on -- there is one RVV
/// kernel set, not a per-chip family -- so the choice is `n == 1` or not, and
/// then the width.
///
/// f16 without Zvfh is left to the tier below, which reaches the f32 kernels
/// through a round trip: an f16 accumulator needs arithmetic Zvfhmin does not
/// have.
fn preferred(
    isa: &IsaSet,
    dt: DatumType,
    query: &Query,
    _suitable: &[Suitable],
) -> Option<&'static str> {
    let wide = isa.has(Isa::RiscV64Vlen256);
    let gemv = query.n == Some(1);
    match dt {
        DatumType::F32 => Some(match (gemv, wide) {
            (true, true) => rvv_mmm_f32_64x1.name.as_str(),
            (true, false) => rvv_mmm_f32_32x1.name.as_str(),
            (false, true) => rvv_mmm_f32_16x8.name.as_str(),
            (false, false) => rvv_mmm_f32_8x8.name.as_str(),
        }),
        #[cfg(tract_rvv_zvfh)]
        DatumType::F16 if isa.has(Isa::RiscV64Zvfh) => Some(match (gemv, wide) {
            (true, true) => zvfh::rvv_mmm_f16_128x1.name.as_str(),
            (true, false) => zvfh::rvv_mmm_f16_64x1.name.as_str(),
            (false, true) => zvfh::rvv_mmm_f16_32x8.name.as_str(),
            (false, false) => zvfh::rvv_mmm_f16_16x8.name.as_str(),
        }),
        DatumType::I32 => Some(match (gemv, wide) {
            (true, true) => rvv_mmm_i32_32x1.name.as_str(),
            (true, false) => rvv_mmm_i32_16x1.name.as_str(),
            (false, true) => rvv_mmm_i32_16x8.name.as_str(),
            (false, false) => rvv_mmm_i32_8x8.name.as_str(),
        }),
        _ => None,
    }
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

    /// `(name, MR, LMUL, SEW in bytes)` mirroring the build.rs kernel tables. The
    /// i32 entries carry the accumulator `LMUL`, twice the one their table lists,
    /// since that is the state whose `VLMAX` has to reach `MR`.
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

    fn runnable() -> Vec<bool> {
        #[allow(unused_mut)]
        let mut runnable = vec![
            rvv_mmm_f32_8x8.runnable(),
            rvv_mmm_f32_16x8.runnable(),
            rvv_mmm_f32_32x1.runnable(),
            rvv_mmm_f32_64x1.runnable(),
            rvv_mmm_i32_8x8.runnable(),
            rvv_mmm_i32_16x8.runnable(),
            rvv_mmm_i32_16x1.runnable(),
            rvv_mmm_i32_32x1.runnable(),
        ];
        #[cfg(tract_rvv_zvfh)]
        runnable.extend([
            zvfh::rvv_mmm_f16_16x8.runnable(),
            zvfh::rvv_mmm_f16_32x8.runnable(),
            zvfh::rvv_mmm_f16_64x1.runnable(),
            zvfh::rvv_mmm_f16_128x1.runnable(),
        ]);
        runnable
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
        let zvfh = super::super::has_zvfh();
        for ((name, mr, lmul, sew), got) in GEOMETRIES.iter().zip(runnable()) {
            let want = vlenb * lmul / sew >= *mr && (*sew == 4 || zvfh);
            eprintln!("VLEN={} zvfh={zvfh} {name}: {got} (want {want})", vlenb * 8);
            assert_eq!(got, want, "{name} dispatch disagrees with this hart");
        }
    }

    /// The `vsetvli` guard heading every kernel backstops the predicates above.
    /// Vacuous on a hart wide enough for all of them, hence no assertion on
    /// finding a candidate.
    ///
    /// Meaningful only where V is present, and for the f16 tiles only where
    /// Zvfh is: the guard is itself an instruction of the extension it stands
    /// for, so it covers "unit too narrow for this tile" but not "no unit",
    /// where calling at all is a SIGILL rather than a return code.
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
        runners.extend::<Vec<Box<dyn Fn() -> isize>>>(vec![
            Box::new(|| zvfh::rvv_mmm_f16_16x8.kernel(&[FusedKerSpec::Done])),
            Box::new(|| zvfh::rvv_mmm_f16_32x8.kernel(&[FusedKerSpec::Done])),
            Box::new(|| zvfh::rvv_mmm_f16_64x1.kernel(&[FusedKerSpec::Done])),
            Box::new(|| zvfh::rvv_mmm_f16_128x1.kernel(&[FusedKerSpec::Done])),
        ]);
        for (((name, .., sew), ok), run) in GEOMETRIES.iter().zip(runnable()).zip(runners) {
            if *sew == 2 && !super::super::has_zvfh() {
                continue;
            }
            if !ok {
                assert_eq!(run(), 1, "{name} ran on a hart whose VLMAX is below its MR");
                eprintln!("{name}: correctly refused");
            }
        }
    }
}
