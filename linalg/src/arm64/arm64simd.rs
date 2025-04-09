mod by_scalar;
mod leaky_relu;
mod max;
mod panel_extract;
mod softmax;
mod sum;
mod unicast;

pub use by_scalar::*;
pub use leaky_relu::arm64simd_leaky_relu_f32_8n;
pub use max::arm64simd_max_f32_16n;
pub use softmax::arm64simd_softmax2_fastcompact_f32_16n;
pub use sum::arm64simd_sum_f32_16n;
pub use unicast::*;

use crate::block_quant::{PackedBlockQuantFormat, Q4_0};
use crate::frame::mmm::ImplementationQuality::ManuallyOptimized;
use crate::pack::PackedFormat;
use crate::Ops;

MMMExternKernel!(arm64simd_mmm_f32_8x8_a55 <f32>(8,  8)@(16, 16) quality(ManuallyOptimized));
MMMExternKernel!(arm64simd_mmm_f32_12x8_a55<f32>(12, 8)@(16, 16) quality(ManuallyOptimized));
MMMExternKernel!(arm64simd_mmm_f32_16x4_a55<f32>(16, 4)@(16, 16) quality(ManuallyOptimized));
MMMExternKernel!(arm64simd_mmm_f32_24x4_a55<f32>(24, 4)@(16, 16) quality(ManuallyOptimized));
MMMExternKernel!(arm64simd_mmm_f32_64x1_a55<f32>(64, 1)@(16, 16) quality(ManuallyOptimized));

MMMExternKernel!(arm64simd_mmm_f32_16x4_a53<f32>(16, 4)@(16, 16) quality(ManuallyOptimized));
MMMExternKernel!(arm64simd_mmm_f32_24x4_a53<f32>(24, 4)@(16, 16) quality(ManuallyOptimized));
MMMExternKernel!(arm64simd_mmm_f32_8x8_a53 <f32>(8,  8)@(16, 16) quality(ManuallyOptimized));
MMMExternKernel!(arm64simd_mmm_f32_12x8_a53<f32>(12, 8)@(16, 16) quality(ManuallyOptimized));
MMMExternKernel!(arm64simd_mmm_f32_64x1_a53<f32>(64, 1)@(16, 16) quality(ManuallyOptimized));

MMMExternKernel!(arm64simd_mmm_f32_16x4_gen<f32>(16, 4)@(16, 16) quality(ManuallyOptimized));
MMMExternKernel!(arm64simd_mmm_f32_24x4_gen<f32>(24, 4)@(16, 16) quality(ManuallyOptimized));
MMMExternKernel!(arm64simd_mmm_f32_8x8_gen <f32>(8,  8)@(16, 16) quality(ManuallyOptimized));
MMMExternKernel!(arm64simd_mmm_f32_12x8_gen<f32>(12, 8)@(16, 16) quality(ManuallyOptimized));
MMMExternKernel!(arm64simd_mmm_f32_64x1_gen<f32>(64, 1)@(16, 16) quality(ManuallyOptimized));

fn q40p32z16se() -> PackedBlockQuantFormat {
    PackedBlockQuantFormat::new(&Q4_0, 32, 16, true)
}

MMMExternKernel!(arm64simd_mmm_f32_32x1_gen<f32>(32, 1)@(16, 16)
    packing[1] = q40f16 => |k| k.with_packing(q40p32z16se(), f16::packing(1));
    packing[2] = q40f32 => |k| k.with_packing(q40p32z16se(), f32::packing(1));
    packing[3] = f16f16 => |k| k.with_packing(f16::packing(32), f16::packing(1));
    packing[4] = f32f16 => |k| k.with_packing(f32::packing(32), f16::packing(1));
    packing[5] = f16f32 => |k| k.with_packing(f16::packing(32), f32::packing(1));
    quality(ManuallyOptimized)
    store(f16)
);

MMMExternKernel!(arm64simd_mmm_f32_32x3_gen<f32>(32, 3)@(16, 16)
    packing[1] = f32f16 => |k| k.with_packing(f32::packing(32), f16::packing(3));
    packing[2] = f16f32 => |k| k.with_packing(f16::packing(32), f32::packing(3));
    packing[3] = f16f16 => |k| k.with_packing(f16::packing(32), f16::packing(3));
    quality(ManuallyOptimized)
    store(f16)
);

MMMExternKernel!(arm64simd_mmm_i32_8x8<i32>(8, 8)@(16, 16)
   packing[1] = i8i8 => |k| k.with_packing(PackedFormat::new(DatumType::I8, 8, 16), PackedFormat::new(DatumType::I8, 8, 16));
   quality(ManuallyOptimized)
   store(i8)
);

MMMExternKernel!(arm64simd_mmm_i32_64x1<i32>(64, 1)@(16, 1)
   packing[1] = i8i8 => |k| k.with_packing(PackedFormat::new(DatumType::I8, 64,16), PackedFormat::new(DatumType::I8, 1, 1));
   quality(ManuallyOptimized)
   store(i8)
);

pub fn plug(ops: &mut Ops) {
    ops.mmm_impls.extend([
        arm64simd_mmm_f32_12x8_gen.mmm(),
        arm64simd_mmm_f32_12x8_a53.mmm(),
        arm64simd_mmm_f32_12x8_a55.mmm(),
        arm64simd_mmm_f32_8x8_gen.mmm(),
        arm64simd_mmm_f32_8x8_a53.mmm(),
        arm64simd_mmm_f32_8x8_a55.mmm(),
        arm64simd_mmm_f32_16x4_gen.mmm(),
        arm64simd_mmm_f32_16x4_a53.mmm(),
        arm64simd_mmm_f32_16x4_a55.mmm(),
        arm64simd_mmm_f32_24x4_gen.mmm(),
        arm64simd_mmm_f32_24x4_a53.mmm(),
        arm64simd_mmm_f32_24x4_a55.mmm(),
        arm64simd_mmm_f32_32x1_gen.mmm(),
        arm64simd_mmm_f32_32x3_gen.mmm(),
        arm64simd_mmm_f32_64x1_gen.mmm(),
        arm64simd_mmm_f32_64x1_a53.mmm(),
        arm64simd_mmm_i32_8x8.mmm(),
        arm64simd_mmm_i32_64x1.mmm(),
    ]);
    panel_extract::plug(ops);
}

tanh_impl!(f32, arm64simd_tanh_f32_4n, 4, 4, true);
sigmoid_impl!(f32, arm64simd_sigmoid_f32_4n, 4, 4, true);
