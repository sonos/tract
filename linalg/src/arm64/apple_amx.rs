use crate::mmm::*;
use tract_data::prelude::*;

use super::{arm64fp16_mmm_f16_16x8_gen, arm64simd_mmm_f32_8x8_gen, arm64simd_mmm_f32_64x1_gen};

const CAN_FUSE: fn(&FusedSpec) -> bool = |f| !matches!(f, &FusedSpec::LeakyRelu(_));

MMMExternKernel!(aarch64; apple_amx_mmm_f32_32x32<f32>(32, 32)@(128, 128) isa(Aarch64AppleAmx) can_fuse(CAN_FUSE) row_major_store(true));
MMMExternKernel!(aarch64; apple_amx_mmm_f32_32x1<f32>(32, 1)@(128, 128) isa(Aarch64AppleAmx) can_fuse(CAN_FUSE));
MMMExternKernel!(aarch64; apple_amx_mmm_f16_64x32<f16>(64, 32)@(128, 128) isa(Aarch64AppleAmx) can_fuse(CAN_FUSE) row_major_store(true));
MMMExternKernel!(aarch64; apple_amx_mmm_f16_64x1<f16>(64, 1)@(128, 128) isa(Aarch64AppleAmx) can_fuse(CAN_FUSE));

/// The AMX tile is 32x32, and it only pays once both M and N fill one: below that the tile
/// padding and the AMX dispatch cost more than the NEON kernel's whole call. The f16 side keeps
/// a low-M NEON route on the same reasoning, at a threshold its own kernels' 16-row tile sets.
fn preferred(
    _isa: &crate::isa::IsaSet,
    dt: crate::DatumType,
    query: &Query,
    _suitable: &[Suitable],
) -> Option<&'static str> {
    match (dt, query.n) {
        // The AMX 32x1 is dominated by the NEON 64x1 at every shape.
        (crate::DatumType::F32, Some(1)) => Some(arm64simd_mmm_f32_64x1_gen.name.as_str()),
        (crate::DatumType::F32, _) => {
            let big_enough = query.m.is_some_and(|m| m >= 32) && query.n.is_some_and(|n| n >= 32);
            Some(if big_enough {
                apple_amx_mmm_f32_32x32.name.as_str()
            } else {
                arm64simd_mmm_f32_8x8_gen.name.as_str()
            })
        }
        (crate::DatumType::F16, Some(1)) => Some(apple_amx_mmm_f16_64x1.name.as_str()),
        (crate::DatumType::F16, _) => Some(if query.m.is_some_and(|m| m <= 16) {
            arm64fp16_mmm_f16_16x8_gen.name.as_str()
        } else {
            apple_amx_mmm_f16_64x32.name.as_str()
        }),
        _ => None,
    }
}

inventory::submit! {
    crate::mmm_tiers::MmmTier {
        arch: Some(crate::isa::Arch::Aarch64),
        precedence: 3,
        name: "apple-amx",
        applies: |isa| isa.has(crate::isa::Isa::Aarch64AppleAmx),
        preferred,
    }
}
