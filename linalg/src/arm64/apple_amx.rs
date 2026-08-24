use crate::Ops;
use crate::frame::mmm::ImplementationQuality::ManuallyOptimized;
use crate::mmm::*;
use tract_data::prelude::*;

use super::{arm64fp16_mmm_f16_16x8_gen, arm64simd_mmm_f32_8x8_gen, arm64simd_mmm_f32_64x1_gen};

const CAN_FUSE: fn(&FusedSpec) -> bool = |f| !matches!(f, &FusedSpec::LeakyRelu(_));

MMMExternKernel!(aarch64; apple_amx_mmm_f32_32x32<f32>(32, 32)@(128, 128) isa(AppleAmx) can_fuse(CAN_FUSE) quality(ManuallyOptimized) row_major_store(true));
MMMExternKernel!(aarch64; apple_amx_mmm_f32_32x1<f32>(32, 1)@(128, 128) isa(AppleAmx) can_fuse(CAN_FUSE) quality(ManuallyOptimized));
MMMExternKernel!(aarch64; apple_amx_mmm_f16_64x32<f16>(64, 32)@(128, 128) isa(AppleAmx) can_fuse(CAN_FUSE) quality(ManuallyOptimized) row_major_store(true));
MMMExternKernel!(aarch64; apple_amx_mmm_f16_64x1<f16>(64, 1)@(128, 128) isa(AppleAmx) can_fuse(CAN_FUSE) quality(ManuallyOptimized));

pub fn plug(ops: &mut Ops) {
    if crate::isa::native().has(crate::isa::Isa::AppleAmx) {
        log::info!(
            "AMX optimisation activated (f32 mmm from M>=32 and N>=32; smaller f32 mmm and \
             every f32 mmv route to NEON)"
        );
        // The AMX tile is 32x32, and it only pays once both M and N fill one: below that the
        // tile padding and the AMX dispatch cost more than the NEON kernel's whole call. The
        // f16 side keeps a low-M NEON route on the same reasoning, at a threshold its own
        // kernels' 16-row tile sets.
        ops.overlay_mmm_policy(|prev, dt, query, candidates| match (dt, query.n) {
            // The AMX 32x1 is dominated by the NEON 64x1 at every shape.
            (crate::DatumType::F32, Some(1)) => {
                candidate_named(candidates, &arm64simd_mmm_f32_64x1_gen.name)
            }
            (crate::DatumType::F32, _) => {
                let big_enough =
                    query.m.is_some_and(|m| m >= 32) && query.n.is_some_and(|n| n >= 32);
                candidate_named(
                    candidates,
                    if big_enough {
                        &apple_amx_mmm_f32_32x32.name
                    } else {
                        &arm64simd_mmm_f32_8x8_gen.name
                    },
                )
            }
            (crate::DatumType::F16, Some(1)) => {
                candidate_named(candidates, &apple_amx_mmm_f16_64x1.name)
            }
            (crate::DatumType::F16, _) => candidate_named(
                candidates,
                if query.m.is_some_and(|m| m <= 16) {
                    &arm64fp16_mmm_f16_16x8_gen.name
                } else {
                    &apple_amx_mmm_f16_64x32.name
                },
            ),
            _ => prev(dt, query, candidates),
        });
    } else {
        log::info!("No AMX optimisation");
    }
}
