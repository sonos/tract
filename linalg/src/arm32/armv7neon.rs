use crate::pack::PackedFormat;

MMMExternKernel!(arm;armv7neon_mmm_f32_8x4_cortexa7 <f32>( 8, 4)@(16, 4) isa(ArmNeon));
MMMExternKernel!(arm;armv7neon_mmm_f32_8x4_cortexa9 <f32>( 8, 4)@(16, 4) isa(ArmNeon));
MMMExternKernel!(arm;armv7neon_mmm_f32_8x4_generic  <f32>( 8, 4)@(16, 4) isa(ArmNeon));
MMMExternKernel!(arm;armv7neon_mmm_f32_8x6_cortexa7 <f32>( 8, 6)@(16, 4) isa(ArmNeon));
MMMExternKernel!(arm;armv7neon_mmm_f32_8x6_cortexa9 <f32>( 8, 6)@(16, 4) isa(ArmNeon));
MMMExternKernel!(arm;armv7neon_mmm_f32_8x6_generic  <f32>( 8, 6)@(16, 4) isa(ArmNeon));
MMMExternKernel!(arm;armv7neon_mmm_f32_8x1_generic  <f32>( 8, 1)@(16, 4) isa(ArmNeon));
MMMExternKernel!(arm;armv7neon_mmm_f32_32x1_cortexa7<f32>(32, 1)@(16, 4) isa(ArmNeon));
MMMExternKernel!(arm;armv7neon_mmm_f32_32x1_cortexa9<f32>(32, 1)@(16, 4) isa(ArmNeon));
MMMExternKernel!(arm;armv7neon_mmm_f32_32x1_generic <f32>(32, 1)@(16, 4) isa(ArmNeon));

MMMExternKernel!(arm;armv7neon_mmm_i32_8x4<i32>(8, 4)@(32, 4) isa(ArmNeon)
  packing[1] = i8i8 => |k| k.with_packing(PackedFormat::new(DatumType::I8, 8, 32), PackedFormat::new(DatumType::I8, 4, 32));

  store(i8)
);

MMMExternKernel!(arm;armv7neon_mmm_i32_32x1<i32>(32, 1)@(32, 4) isa(ArmNeon)
  packing[1] = i8i8 => |k| k.with_packing(PackedFormat::new(DatumType::I8, 32, 32), PackedFormat::new(DatumType::I8, 1, 4));

  store(i8)
);

routine_ew_extern!(arm; Sigmoid, f32, armv7neon_sigmoid_f32_4n, 4, 4, isa(ArmNeon));
routine_ew_extern!(arm; Silu, f32, armv7neon_silu_f32_4n, 4, 4, isa(ArmNeon));
routine_ew_extern!(arm; Tanh, f32, armv7neon_tanh_f32_4n, 4, 4, isa(ArmNeon));
