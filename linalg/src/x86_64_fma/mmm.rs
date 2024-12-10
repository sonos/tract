use crate::frame::block_quant::*;
use crate::frame::PackedFormat;
use crate::mmm::MMMKit;
use crate::mmm::MatMatMulKer;
use crate::Ops;
use panel_extract::packed_32_f16_to_f32;
use panel_extract::packed_32_q40_to_f32;
use tract_data::internal::*;
use DatumType::*;

use super::*;

MMMExternKernel!(fma_mmm_f32_8x8 <f32>(8, 8)@(32,4) where(FMA));
MMMExternKernel!(fma_mmm_f32_16x6<f32>(16,6)@(32,4) where(FMA));
MMMExternKernel!(fma_mmm_f32_16x5<f32>(16,5)@(32,4) where(FMA));
MMMExternKernel!(fma_mmm_f32_24x4<f32>(24,4)@(32,4) where(FMA));
MMMExternKernel!(fma_mmm_f32_40x2<f32>(40,2)@(32,4) where(FMA));
MMMExternKernel!(fma_mmm_f32_64x1<f32>(64,1)@(32,4) where(FMA));

pub fn pq40_r32() -> PackedBlockQuantFormat {
    PackedBlockQuantFormat::new(&Q4_0, 32, 16, false)
}
MMMExternKernel! {fma_mmm_f32_32x1<f32>(32,1)@(32,4) where(FMA)
    packing[1] = q40f32 => |k| k.with_packing_a(pq40_r32());
    packing[2] = q40f16 => |k| k.with_packing(pq40_r32(), PackedFormat::new(F16, 1, 2));
    packing[3] = f16f16 => |k| k.with_packing(PackedFormat::new(F16, 32, 32), PackedFormat::new(F16, 1, 2));
    store(f16)
}
MMMExternKernel!(fma_mmm_f32_32x3<f32>(32,3)@(32,4) where(FMA)
 packing[1] = f32f16 => |k| k.with_packing(f32::packing(32).align(32), f16::packing(3));
 store(f16)
);

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
    store(i8)
}

pub fn plug(ops: &mut Ops) {
    if fma_mmm_f32_32x1.is_supported_here() {
        ops.mmm_kits.push(
            MMMKit::new_for_mmm(fma_mmm_f32_32x1.mmm(), 0).with_native(fma_mmm_f32_32x3.mmm(), 0),
        );
        ops.mmm_kits.push(MMMKit::new_for_mmm(fma_mmm_f32_32x1.mmm(), 1).with_extracting(
            fma_mmm_f32_32x3.mmm(),
            0,
            packed_32_q40_to_f32.clone(),
        ));
        ops.mmm_kits.push(MMMKit::new_for_mmm(fma_mmm_f32_32x1.mmm(), 2).with_extracting(
            fma_mmm_f32_32x3.mmm(),
            1,
            packed_32_q40_to_f32.clone(),
        ));
        ops.mmm_kits.push(
            MMMKit::new(F16, F32, F16, &PackedFormat::new(F16, 32, 32))
                .with_native(fma_mmm_f32_32x1.mmm(), 3)
                .with_extracting(fma_mmm_f32_32x3.mmm(), 1, packed_32_f16_to_f32.clone()),
        );
    }
}
