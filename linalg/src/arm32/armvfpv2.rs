use crate::DatumType;
use crate::frame::mmm::ImplementationQuality::ManuallyOptimized;
use crate::frame::mmm::*;
use crate::isa::IsaSet;
use crate::mmm::{Query, Suitable};
use crate::mmm_tiers::MmmTier;

// Baseline VFP asm, so it runs on every armv7 core and needs no `isa`. The NEON kernels are
// better wherever they run, which is what the instruction-set tier says.
MMMExternKernel!(arm; armvfpv2_mmm_f32_4x4<f32>(4, 4)@(4, 4) quality(ManuallyOptimized));

/// Baseline armv7: the VFP kernel runs on every core, so this tier always applies and the NEON
/// tier above it answers first wherever NEON is present.
fn preferred(
    _isa: &IsaSet,
    dt: DatumType,
    _query: &Query,
    _suitable: &[Suitable],
) -> Option<&'static str> {
    match dt {
        DatumType::F32 => Some(armvfpv2_mmm_f32_4x4.name.as_str()),
        _ => None,
    }
}

inventory::submit! {
    MmmTier {
        arch: Some(crate::isa::Arch::Arm),
        precedence: 1,
        name: "armvfpv2",
        applies: |_| true,
        preferred,
    }
}
