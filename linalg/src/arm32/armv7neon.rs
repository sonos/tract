extern_kernel!(fn armv7neon_prefetch(start: *const u8, end: *const u8) -> ());

#[inline(always)]
pub fn prefetch(start: *const u8, len: usize) {
    unsafe { armv7neon_prefetch(start, start.offset(len as isize)) }
}

MMMExternKernel!(f32, armv7neon_mmm_f32_8x4_cortexa7; 8, 4; 4, 4; 0, 0; prefetch, crate::arm32::has_neon());
MMMExternKernel!(f32, armv7neon_mmm_f32_8x4_cortexa9; 8, 4; 4, 4; 0, 0; prefetch, crate::arm32::has_neon());
MMMExternKernel!(f32, armv7neon_mmm_f32_8x4_generic; 8, 4; 4, 4; 0, 0; prefetch, crate::arm32::has_neon());
MMMExternKernel!(f32, armv7neon_mmm_f32_8x6_cortexa7; 8, 6; 4, 4; 0, 0; prefetch, crate::arm32::has_neon());
MMMExternKernel!(f32, armv7neon_mmm_f32_8x6_cortexa9; 8, 6; 4, 4; 0, 0; prefetch, crate::arm32::has_neon());
MMMExternKernel!(f32, armv7neon_mmm_f32_8x6_generic; 8, 6; 4, 4; 0, 0; prefetch, crate::arm32::has_neon());
MMMExternKernel!(f32, armv7neon_mmm_f32_32x1_cortexa7; 32, 1; 4, 4; 0, 0; prefetch, crate::arm32::has_neon());
MMMExternKernel!(f32, armv7neon_mmm_f32_32x1_cortexa9; 32, 1; 4, 4; 0, 0; prefetch, crate::arm32::has_neon());
MMMExternKernel!(f32, armv7neon_mmm_f32_32x1_generic; 32, 1; 4, 4; 0, 0; prefetch, crate::arm32::has_neon());

MMMExternKernel!(i32, armv7neon_mmm_i32_8x4; 8, 4; 32, 4; 0, 0; prefetch, crate::arm32::has_neon(),
     packing_defs: {
         const I8_A: PackedFormat = PackedFormat::new(DatumType::I8, 8, 32, 0);
         const I8_B: PackedFormat = PackedFormat::new(DatumType::I8, 4, 4, 0);
         const I8_I8: (&dyn MMMInputFormat, &dyn MMMInputFormat) = (&I8_A, &I8_B);
     },
     packings: I8_I8,
     test: mmm_packed_packed_tests!{ true, armv7neon_mmm_i32_8x4, i8i8:1, i8, i8, i32, i32 }
);

MMMExternKernel!(i32, armv7neon_mmm_i32_32x1; 32,1 ; 32, 4; 0, 0; prefetch, crate::arm32::has_neon(),
     packing_defs: {
         const I8_A: PackedFormat = PackedFormat::new(DatumType::I8, 32, 32, 0);
         const I8_B: PackedFormat = PackedFormat::new(DatumType::I8, 1, 4, 0);
         const I8_I8: (&dyn MMMInputFormat, &dyn MMMInputFormat) = (&I8_A, &I8_B);
     },
     packings: I8_I8,
     test: mmm_packed_packed_tests!{ true, armv7neon_mmm_i32_32x1, i8i8:1, i8, i8, i32, i32 }
);

sigmoid_impl!(f32, armv7neon_sigmoid_f32_4n, 4, 4, crate::arm32::has_neon());
tanh_impl!(f32, armv7neon_tanh_f32_4n, 4, 4, crate::arm32::has_neon());

