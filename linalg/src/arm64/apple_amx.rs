use crate::frame::mmm::ImplementationQuality::ManuallyOptimized;
use crate::mmm::*;
use crate::Ops;
use tract_data::prelude::*;

use super::has_amx;

const AMX: fn() -> bool = crate::arm64::has_amx;
const CAN_FUSE: fn(&FusedSpec) -> bool = |f| !matches!(f, &FusedSpec::LeakyRelu(_));

MMMExternKernel!(apple_amx_mmm_f32_32x32<f32>(32, 32)@(128, 128) where(AMX) can_fuse(CAN_FUSE) quality(ManuallyOptimized));
MMMExternKernel!(apple_amx_mmm_f32_32x1<f32>(32, 1)@(128, 128) where(AMX) can_fuse(CAN_FUSE) quality(ManuallyOptimized));
MMMExternKernel!(apple_amx_mmm_f16_64x32<f16>(64, 32)@(128, 128) where(AMX) can_fuse(CAN_FUSE) quality(ManuallyOptimized));
MMMExternKernel!(apple_amx_mmm_f16_64x1<f16>(64, 1)@(128, 128) where(AMX) can_fuse(CAN_FUSE) quality(ManuallyOptimized));

pub fn plug(ops: &mut Ops) {
    if has_amx() {
        log::info!("AMX optimisation activated");
        ops.mmm_f16 = Box::new(|_, _, _| apple_amx_mmm_f16_64x32.mmm());
        ops.mmm_f32 = Box::new(|_, _, _| apple_amx_mmm_f32_32x32.mmm());
        ops.mmv_f16 = Box::new(|_, _| apple_amx_mmm_f16_64x1.mmm());
        ops.mmv_f32 = Box::new(|_, _| apple_amx_mmm_f32_32x1.mmm());
        ops.mmm_impls.extend_from_slice(&[
            apple_amx_mmm_f32_32x32.mmm(),
            apple_amx_mmm_f32_32x1.mmm(),
            apple_amx_mmm_f16_64x32.mmm(),
            apple_amx_mmm_f16_64x1.mmm(),
        ]);
    } else {
        log::info!("No AMX optimisation");
    }
}
