use crate::Ops;
use crate::frame::mmm::ImplementationQuality::ManuallyOptimized;
use crate::frame::mmm::*;

// The NEON kernels cover every core that has NEON; this one is what is left for the others.
const NO_NEON: fn() -> bool = || !crate::arm32::has_neon();

MMMExternKernel!(arm; armvfpv2_mmm_f32_4x4<f32>(4, 4)@(4, 4) where(NO_NEON) quality(ManuallyOptimized));

pub fn plug(ops: &mut Ops) {
    log::info!("armvfpv2 activated for smmm");
    ops.mmm_f32 = Box::new(|_, _, _| armvfpv2_mmm_f32_4x4.mmm());
}
