//use crate::frame::block_quant::{PackedBlockQuantFormat, Q4_0};
use crate::frame::block_quant::*;
use crate::frame::PackedFormat;
use tract_data::prelude::*;

const AVX2: fn() -> bool = || is_x86_feature_detected!("avx2");
const FMA: fn() -> bool = || is_x86_feature_detected!("fma");
const FMA_F16C: fn() -> bool =
    || is_x86_feature_detected!("fma") && is_x86_feature_detected!("f16c");
const AVX512F: fn() -> bool = || is_x86_feature_detected!("avx512f");

MMMExternKernel!(fma_mmm_f16_8x8<f16>(8,8)@(32,2) where(FMA_F16C));

MMMExternKernel!(fma_mmm_f32_8x8 <f32>(8, 8)@(32,4) where(FMA));
MMMExternKernel!(fma_mmm_f32_16x6<f32>(16,6)@(32,4) where(FMA));
MMMExternKernel!(fma_mmm_f32_16x5<f32>(16,5)@(32,4) where(FMA));
MMMExternKernel!(fma_mmm_f32_24x4<f32>(24,4)@(32,4) where(FMA));
MMMExternKernel!(fma_mmm_f32_32x3<f32>(32,3)@(32,4) where(FMA));
MMMExternKernel!(fma_mmm_f32_40x2<f32>(40,2)@(32,4) where(FMA));
MMMExternKernel!(fma_mmm_f32_64x1<f32>(64,1)@(32,4) where(FMA));

const PQ40_R32: PackedBlockQuantFormat = PackedBlockQuantFormat::new(&Q4_0, 32, 16, false);
MMMExternKernel! {fma_mmm_f32_32x1<f32>(32,1)@(32,4) where(FMA)
    packing[1] = q40f32 => |k| k.with_packing_a(PQ40_R32);
}

MMMExternKernel!(avx512_mmm_f32_128x1<f32>(128, 1)@(64,4) where (AVX512F));
MMMExternKernel!(avx512_mmm_f32_16x1 <f32>( 16, 1)@(64,4) where (AVX512F));
MMMExternKernel!(avx512_mmm_f32_16x12<f32>( 16,12)@(64,4) where (AVX512F));
MMMExternKernel!(avx512_mmm_f32_16x8 <f32>( 16, 8)@(64,4) where (AVX512F));
MMMExternKernel!(avx512_mmm_f32_32x6 <f32>( 32, 6)@(64,4) where (AVX512F));
MMMExternKernel!(avx512_mmm_f32_32x5 <f32>( 32, 5)@(64,4) where (AVX512F));
MMMExternKernel!(avx512_mmm_f32_48x4 <f32>( 48, 4)@(64,4) where (AVX512F));
MMMExternKernel!(avx512_mmm_f32_64x3 <f32>( 64, 3)@(64,4) where (AVX512F));
MMMExternKernel!(avx512_mmm_f32_80x2 <f32>( 80, 2)@(64,4) where (AVX512F));

MMMExternKernel! { avx2_mmm_i32_8x8<i32>(8,8)@(32,4) where(AVX2)
    packing[1] = i8i8 => |k| k.with_packing(PackedFormat::new(DatumType::I8, 8,32), PackedFormat::new(DatumType::I8, 8, 4));
}
