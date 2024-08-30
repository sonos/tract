//use crate::frame::block_quant::{PackedBlockQuantFormat, Q4_0};
use crate::mmm::no_prefetch;
use tract_data::prelude::f16;
use crate::frame::block_quant::*;

const PQ40_R32: PackedBlockQuantFormat = PackedBlockQuantFormat::new(&Q4_0, 32, 16, false);

MMMExternKernel!(f16, fma_mmm_f16_8x8; 8, 8; 32, 2; 0, 0; no_prefetch, is_x86_feature_detected!("fma") && is_x86_feature_detected!("f16c"));

MMMExternKernel!(f32, fma_mmm_f32_8x8; 8, 8; 32, 4; 0, 0; no_prefetch, is_x86_feature_detected!("fma"));
MMMExternKernel!(f32, fma_mmm_f32_16x6; 16, 6; 32, 4; 0, 0; no_prefetch, is_x86_feature_detected!("fma"));
MMMExternKernel!(f32, fma_mmm_f32_16x5; 16, 5; 32, 4; 0, 0; no_prefetch, is_x86_feature_detected!("fma"));
MMMExternKernel!(f32, fma_mmm_f32_24x4; 24, 4; 32, 4; 0, 0; no_prefetch, is_x86_feature_detected!("fma"));
MMMExternKernel!(f32, fma_mmm_f32_32x3; 32, 3; 32, 4; 0, 0; no_prefetch, is_x86_feature_detected!("fma"));
MMMExternKernel!(f32, fma_mmm_f32_40x2; 40, 2; 32, 4; 0, 0; no_prefetch, is_x86_feature_detected!("fma"));
MMMExternKernel!(f32, fma_mmm_f32_64x1; 64, 1; 32, 4; 0, 0; no_prefetch, is_x86_feature_detected!("fma"));

MMMExternKernel!(f32, fma_mmm_f32_32x1; 32, 1; 32, 4; 0, 0; no_prefetch, is_x86_feature_detected!("fma"),
     packing_defs: {
         const F32_B: PackedFormat = PackedFormat::new(DatumType::F32, 1, 4);
         const PQ40_F32: (&dyn MMMInputFormat, &dyn MMMInputFormat) = (&super::PQ40_R32, &F32_B);
     },
 packings: PQ40_F32,
 test: mmm_packed_packed_tests!{ is_x86_feature_detected!("fma"), fma_mmm_f32_32x1, q40f32:1 }
);

MMMExternKernel!(f32, avx512_mmm_f32_128x1; 128, 1; 64, 4; 0, 0; no_prefetch, is_x86_feature_detected!("avx512f"));
MMMExternKernel!(f32, avx512_mmm_f32_16x1; 16, 1; 64, 4; 0, 0; no_prefetch, is_x86_feature_detected!("avx512f"));
MMMExternKernel!(f32, avx512_mmm_f32_16x12; 16, 12; 64, 4; 0, 0; no_prefetch, is_x86_feature_detected!("avx512f"));
MMMExternKernel!(f32, avx512_mmm_f32_16x8; 16, 8; 64, 4; 0, 0; no_prefetch, is_x86_feature_detected!("avx512f"));
MMMExternKernel!(f32, avx512_mmm_f32_32x6; 32, 6; 64, 4; 0, 0; no_prefetch, is_x86_feature_detected!("avx512f"));
MMMExternKernel!(f32, avx512_mmm_f32_32x5; 32, 5; 64, 4; 0, 0; no_prefetch, is_x86_feature_detected!("avx512f"));
MMMExternKernel!(f32, avx512_mmm_f32_48x4; 48, 4; 64, 4; 0, 0; no_prefetch, is_x86_feature_detected!("avx512f"));
MMMExternKernel!(f32, avx512_mmm_f32_64x3; 64, 3; 64, 4; 0, 0; no_prefetch, is_x86_feature_detected!("avx512f"));
MMMExternKernel!(f32, avx512_mmm_f32_80x2; 80, 2; 64, 4; 0, 0; no_prefetch, is_x86_feature_detected!("avx512f"));

MMMExternKernel!(i32, avx2_mmm_i32_8x8; 8, 8; 32, 4; 0, 0; no_prefetch, is_x86_feature_detected!("avx2"),
 packing_defs: {
     const I8_A: PackedFormat = PackedFormat::new(DatumType::I8, 8, 32);
     const I8_B: PackedFormat = PackedFormat::new(DatumType::I8, 8, 4);
     const I8_I8: (&dyn MMMInputFormat, &dyn MMMInputFormat) = (&I8_A, &I8_B);
 },
 packings: I8_I8,
 test: mmm_packed_packed_tests!{ true, avx2_mmm_i32_8x8, i8i8:1 }
);
