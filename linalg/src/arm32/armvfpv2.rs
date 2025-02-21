use crate::frame::mmm::ImplementationQuality::ManuallyOptimized;
use crate::frame::mmm::*;
use crate::Ops;

MMMExternKernel!(armvfpv2_mmm_f32_4x4<f32>(4, 4)@(4, 4) quality(ManuallyOptimized));

pub fn plug(ops: &mut Ops) {
    log::info!("armvfpv2 activated for smmm");
    ops.mmm_f32 = Box::new(|_, _, _| armvfpv2_mmm_f32_4x4.mmm());
    ops.mmm_impls.push(armvfpv2_mmm_f32_4x4.mmm());
}
