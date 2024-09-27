mod by_scalar;
mod leaky_relu;
mod max;
mod mul;
mod softmax;
mod sum;

use crate::frame::PackedFormat;

pub use by_scalar::arm64simd_mul_by_scalar_f32_16n;
pub use leaky_relu::arm64simd_leaky_relu_f32_8n;
pub use max::arm64simd_max_f32_16n;
pub use mul::arm64simd_unicast_mul_f32_16n;
pub use softmax::arm64simd_softmax2_fastcompact_f32_16n;
pub use sum::arm64simd_sum_f32_16n;

MMMExternKernel!(arm64simd_mmm_f32_8x8_a55 <f32>(8,  8)@(16, 16));
MMMExternKernel!(arm64simd_mmm_f32_12x8_a55<f32>(12, 8)@(16, 16));
MMMExternKernel!(arm64simd_mmm_f32_16x4_a55<f32>(16, 4)@(16, 16));
MMMExternKernel!(arm64simd_mmm_f32_24x4_a55<f32>(24, 4)@(16, 16));
MMMExternKernel!(arm64simd_mmm_f32_64x1_a55<f32>(64, 1)@(16, 16));

MMMExternKernel!(arm64simd_mmm_f32_16x4_a53<f32>(16, 4)@(16, 16));
MMMExternKernel!(arm64simd_mmm_f32_24x4_a53<f32>(24, 4)@(16, 16));
MMMExternKernel!(arm64simd_mmm_f32_8x8_a53 <f32>(8,  8)@(16, 16));
MMMExternKernel!(arm64simd_mmm_f32_12x8_a53<f32>(12, 8)@(16, 16));
MMMExternKernel!(arm64simd_mmm_f32_64x1_a53<f32>(64, 1)@(16, 16));

MMMExternKernel!(arm64simd_mmm_f32_16x4_gen<f32>(16, 4)@(16, 16));
MMMExternKernel!(arm64simd_mmm_f32_24x4_gen<f32>(24, 4)@(16, 16));
MMMExternKernel!(arm64simd_mmm_f32_8x8_gen <f32>(8,  8)@(16, 16));
MMMExternKernel!(arm64simd_mmm_f32_12x8_gen<f32>(12, 8)@(16, 16));
MMMExternKernel!(arm64simd_mmm_f32_64x1_gen<f32>(64, 1)@(16, 16));

MMMExternKernel!(arm64simd_mmm_i32_8x8<i32>(8, 8)@(16, 16)
   packing[1] = i8i8 => |k| k.with_packing(PackedFormat::new(DatumType::I8, 8, 16), PackedFormat::new(DatumType::I8, 8, 16));
   store(i8)
);

MMMExternKernel!(arm64simd_mmm_i32_64x1<i32>(64, 1)@(16, 1)
   packing[1] = i8i8 => |k| k.with_packing(PackedFormat::new(DatumType::I8, 64,16), PackedFormat::new(DatumType::I8, 1, 1));
   store(i8)
);

tanh_impl!(f32, arm64simd_tanh_f32_4n, 4, 4, true);
sigmoid_impl!(f32, arm64simd_sigmoid_f32_4n, 4, 4, true);
