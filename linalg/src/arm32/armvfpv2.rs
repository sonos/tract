use crate::frame::mmm::ImplementationQuality::ManuallyOptimized;
use crate::frame::mmm::*;
use crate::{DatumType, Ops};

/// Baseline VFP asm, so it runs on every armv7 core, NEON or not — hence no `where`. The NEON
/// kernels cover every core that has NEON and are better there, so this one must lose every tie
/// against them.
const NO_NEON_TIER: fn() -> isize = || if crate::arm32::has_neon() { -1 } else { 0 };

MMMExternKernel!(arm; armvfpv2_mmm_f32_4x4<f32>(4, 4)@(4, 4) quality(ManuallyOptimized) boost(NO_NEON_TIER));

pub fn plug(ops: &mut Ops) {
    log::info!("armvfpv2 activated for smmm");
    ops.overlay_mmm_policy(|prev, dt, m, k, n| match dt {
        DatumType::F32 => Some(armvfpv2_mmm_f32_4x4.mmm()),
        _ => prev(dt, m, k, n),
    });
}
