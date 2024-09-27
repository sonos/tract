use crate::frame::PackedFormat;

const NEON: fn() -> bool = || crate::arm32::has_neon();

MMMExternKernel!(armv7neon_mmm_f32_8x4_cortexa7 <f32>( 8, 4 )@(4, 4 ) where(NEON));
MMMExternKernel!(armv7neon_mmm_f32_8x4_cortexa9 <f32>( 8, 4 )@(4, 4 ) where(NEON));
MMMExternKernel!(armv7neon_mmm_f32_8x4_generic  <f32>( 8, 4 )@(4, 4 ) where(NEON));
MMMExternKernel!(armv7neon_mmm_f32_8x6_cortexa7 <f32>( 8, 6 )@(4, 4 ) where(NEON));
MMMExternKernel!(armv7neon_mmm_f32_8x6_cortexa9 <f32>( 8, 6 )@(4, 4 ) where(NEON));
MMMExternKernel!(armv7neon_mmm_f32_8x6_generic  <f32>( 8, 6 )@(4, 4 ) where(NEON));
MMMExternKernel!(armv7neon_mmm_f32_32x1_cortexa7<f32>( 32, 1)@( 4, 4) where(NEON));
MMMExternKernel!(armv7neon_mmm_f32_32x1_cortexa9<f32>( 32, 1)@( 4, 4) where(NEON));
MMMExternKernel!(armv7neon_mmm_f32_32x1_generic <f32>(32, 1 )@(4, 4 ) where(NEON));

MMMExternKernel!(armv7neon_mmm_i32_8x4<i32>(8, 4)@(32, 4) where(NEON)
  packing[1] = i8i8 => |k| k.with_packing(PackedFormat::new(DatumType::I8, 8, 32), PackedFormat::new(DatumType::I8, 4, 32));
  store(i8)
);

MMMExternKernel!(armv7neon_mmm_i32_32x1<i32>(32, 1)@(32, 4) where(NEON)
  packing[1] = i8i8 => |k| k.with_packing(PackedFormat::new(DatumType::I8, 32, 32), PackedFormat::new(DatumType::I8, 1, 4));
  store(i8)
);

sigmoid_impl!(f32, armv7neon_sigmoid_f32_4n, 4, 4, crate::arm32::has_neon());
tanh_impl!(f32, armv7neon_tanh_f32_4n, 4, 4, crate::arm32::has_neon());

