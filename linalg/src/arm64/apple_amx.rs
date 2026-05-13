use crate::Ops;
use crate::frame::mmm::ImplementationQuality::ManuallyOptimized;
use crate::mmm::*;
use tract_data::prelude::*;

use super::has_amx;
use super::{arm64fp16_mmm_f16_16x8_gen, arm64simd_mmm_f32_8x8_gen, arm64simd_mmm_f32_64x1_gen};

const AMX: fn() -> bool = crate::arm64::has_amx;
const CAN_FUSE: fn(&FusedSpec) -> bool = |f| !matches!(f, &FusedSpec::LeakyRelu(_));

MMMExternKernel!(apple_amx_mmm_f32_32x32<f32>(32, 32)@(128, 128) where(AMX) can_fuse(CAN_FUSE) quality(ManuallyOptimized));
MMMExternKernel!(apple_amx_mmm_f32_32x1<f32>(32, 1)@(128, 128) where(AMX) can_fuse(CAN_FUSE) quality(ManuallyOptimized));
MMMExternKernel!(apple_amx_mmm_f16_64x32<f16>(64, 32)@(128, 128) where(AMX) can_fuse(CAN_FUSE) quality(ManuallyOptimized));
MMMExternKernel!(apple_amx_mmm_f16_64x1<f16>(64, 1)@(128, 128) where(AMX) can_fuse(CAN_FUSE) quality(ManuallyOptimized));

pub fn plug(ops: &mut Ops) {
    if has_amx() {
        log::info!(
            "AMX optimisation activated (A7v2: AMX only for f32 mmm with M>=32 AND N>=32; \
             smaller shapes + all f32 mmv route to NEON kernels)"
        );
        // ----- A7v2 dispatch logic (data-driven) -----
        //
        // Empirical finding from /tmp/amx_vs_neon.md microbench (Apple M1 Pro):
        // the AMX 32x32 kernel beats NEON 8x8 only when BOTH M and N are at
        // least 32 — the AMX tile dimensions. At smaller shapes the per-tile
        // padding waste + AMX dispatch overhead make NEON faster.
        //
        // Predicate validation: 88.3% accuracy on 512-shape sweep.
        //
        // Canary impact (measured 2026-05-13, see notes/tract-amx-low-m-investigation.md):
        // turning AMX off entirely yielded:
        //   df_dec       1.55× faster      mobilenetv2  1.59× faster
        //   erb_dec      1.49×             squeezenet   1.22×
        //   enc          1.17×             yolov8n      1.15× SLOWER
        //   inception_v3 1.43× SLOWER      sam2_tiny    1.54× SLOWER
        // The shape-aware predicate keeps the AMX wins for the heavy models
        // (Inception, YOLO, SAM2) while routing small shapes to NEON.
        ops.mmm_f32 = Box::new(|m, _, n| {
            let big_enough = m.is_some_and(|m| m >= 32) && n.is_some_and(|n| n >= 32);
            if big_enough { apple_amx_mmm_f32_32x32.mmm() } else { arm64simd_mmm_f32_8x8_gen.mmm() }
        });
        // mmv (n=1) f32: AMX 32x1 is dominated by NEON 64x1 across the entire
        // shape sweep — confirmed by canary deltas on DFN3 (which is mmv-heavy).
        // Always use NEON.
        ops.mmv_f32 = Box::new(|_, _| arm64simd_mmm_f32_64x1_gen.mmm());

        // ----- f16 paths kept conservative for now -----
        //
        // We didn't run the f16 microbench yet, so retain the original logic
        // and the previous low-M-routes-to-NEON heuristic.
        ops.mmm_f16 = Box::new(|m, _, _| {
            if m.is_some_and(|m| m <= 16) {
                arm64fp16_mmm_f16_16x8_gen.mmm()
            } else {
                apple_amx_mmm_f16_64x32.mmm()
            }
        });
        ops.mmv_f16 = Box::new(|_, _| apple_amx_mmm_f16_64x1.mmm());
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
