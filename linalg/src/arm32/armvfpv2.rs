use crate::frame::mmm::ImplementationQuality::ManuallyOptimized;
use crate::frame::mmm::*;
use crate::{DatumType, Ops};

// Baseline VFP asm, so it runs on every armv7 core and needs no `isa`. The NEON kernels are
// better wherever they run, which is what the instruction-set tier says.
MMMExternKernel!(arm; armvfpv2_mmm_f32_4x4<f32>(4, 4)@(4, 4) quality(ManuallyOptimized));

pub fn plug(ops: &mut Ops) {
    log::info!("armvfpv2 activated for smmm");
    ops.overlay_mmm_policy(|prev, dt, query, suitable| match dt {
        DatumType::F32 => suitable_named(suitable, &armvfpv2_mmm_f32_4x4.name),
        _ => prev(dt, query, suitable),
    });
}
