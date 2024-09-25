use crate::mmm::*;
use tract_data::prelude::*;

const AMX: fn() -> bool = crate::arm64::has_amx;
const CAN_FUSE: fn(&FusedSpec) -> bool = |f| !matches!(f, &FusedSpec::LeakyRelu(_));

MMMExternKernel!(apple_amx_mmm_f32_32x32<f32>(32, 32)@(128, 128) where(AMX) can_fuse(CAN_FUSE));
MMMExternKernel!(apple_amx_mmm_f32_32x1<f32>(32, 1)@(128, 128) where(AMX) can_fuse(CAN_FUSE));
MMMExternKernel!(apple_amx_mmm_f16_64x32<f16>(64, 32)@(128, 128) where(AMX) can_fuse(CAN_FUSE));
MMMExternKernel!(apple_amx_mmm_f16_64x1<f16>(64, 1)@(128, 128) where(AMX) can_fuse(CAN_FUSE));
