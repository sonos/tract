mod by_scalar;
mod leaky_relu;
mod max;
mod softmax;
mod sum;

use crate::mmm::no_prefetch;

pub use by_scalar::arm64simd_mul_by_scalar_f32_16n;
pub use leaky_relu::arm64simd_leaky_relu_f32_8n;
pub use max::arm64simd_max_f32_16n;
pub use sum::arm64simd_sum_f32_16n;
pub use softmax::arm64simd_softmax2_fastcompact_f32_16n;

MMMExternKernel!(f32, arm64simd_mmm_f32_8x8_a55; 8, 8; 16, 16; 1, 1; no_prefetch, true);
MMMExternKernel!(f32, arm64simd_mmm_f32_12x8_a55; 12, 8; 16, 16; 1, 1; no_prefetch, true);
MMMExternKernel!(f32, arm64simd_mmm_f32_16x4_a55; 16, 4; 16, 16; 1, 1; no_prefetch, true);
MMMExternKernel!(f32, arm64simd_mmm_f32_24x4_a55; 24, 4; 16, 16; 1, 1; no_prefetch, true);
MMMExternKernel!(f32, arm64simd_mmm_f32_64x1_a55; 64, 1; 16, 16; 1, 1; no_prefetch, true);

MMMExternKernel!(f32, arm64simd_mmm_f32_16x4_a53; 16, 4; 16, 16; 1, 1; no_prefetch, true);
MMMExternKernel!(f32, arm64simd_mmm_f32_24x4_a53; 24, 4; 16, 16; 1, 1; no_prefetch, true);
MMMExternKernel!(f32, arm64simd_mmm_f32_8x8_a53; 8, 8; 16, 16; 1, 1; no_prefetch, true);
MMMExternKernel!(f32, arm64simd_mmm_f32_12x8_a53; 12, 8; 16, 16; 1, 1; no_prefetch, true);
MMMExternKernel!(f32, arm64simd_mmm_f32_64x1_a53; 64, 1; 16, 16; 1, 1; no_prefetch, true);

MMMExternKernel!(f32, arm64simd_mmm_f32_16x4_gen; 16, 4; 16, 16; 1, 1; no_prefetch, true);
MMMExternKernel!(f32, arm64simd_mmm_f32_24x4_gen; 24, 4; 16, 16; 1, 1; no_prefetch, true);
MMMExternKernel!(f32, arm64simd_mmm_f32_8x8_gen; 8, 8; 16, 16; 1, 1; no_prefetch, true);
MMMExternKernel!(f32, arm64simd_mmm_f32_12x8_gen; 12, 8; 16, 16; 1, 1; no_prefetch, true);
MMMExternKernel!(f32, arm64simd_mmm_f32_64x1_gen; 64, 1; 16, 16; 1, 1; no_prefetch, true);

MMMExternKernel!(i32, arm64simd_mmm_i32_8x8; 8, 8; 16, 16; 0,0; no_prefetch, true,
 packing_defs: {
     const I8_A: PackedFormat = PackedFormat::new(DatumType::I8, 8, 16, 0);
     const I8_B: PackedFormat = PackedFormat::new(DatumType::I8, 8, 16, 0);
     const I8_I8: (&dyn MMMInputFormat, &dyn MMMInputFormat) = (&I8_A, &I8_B);
 },
 packings: I8_I8,
 test: mmm_packed_packed_tests!{ true, arm64simd_mmm_i32_8x8, i8i8:1, i8, i8, i32, i32 }
);

MMMExternKernel!(i32, arm64simd_mmm_i32_64x1; 64, 1; 16, 1; 0,0; no_prefetch, true,
 packing_defs: {
     const I8_A: PackedFormat = PackedFormat::new(DatumType::I8, 64, 16, 0);
     const I8_B: PackedFormat = PackedFormat::new(DatumType::I8, 1, 1, 0);
     const I8_I8: (&dyn MMMInputFormat, &dyn MMMInputFormat) = (&I8_A, &I8_B);
 },
 packings: I8_I8,
 test: mmm_packed_packed_tests!{ true, arm64simd_mmm_i32_64x1, i8i8:1, i8, i8, i32, i32 }
);

tanh_impl!(f32, arm64simd_tanh_f32_4n, 4, 4, true);
sigmoid_impl!(f32, arm64simd_sigmoid_f32_4n, 4, 4, true);

