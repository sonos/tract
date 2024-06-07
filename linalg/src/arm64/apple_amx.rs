MMMKernel!(f32, apple_amx_mmm_f32_32x32; 32, 32; 128, 128; 1, 1; no_prefetch, crate::arm64::has_amx(),
           can_fuse: |f| !matches!(f, &FusedSpec::LeakyRelu(_))
);

MMMKernel!(f32, apple_amx_mmm_f32_32x1; 32, 1; 128, 128; 1, 1; no_prefetch, crate::arm64::has_amx(),
           can_fuse: |f| !matches!(f, &FusedSpec::LeakyRelu(_))
);

MMMKernel!(f16, apple_amx_mmm_f16_64x32; 64, 32; 128, 128; 1, 1; no_prefetch, crate::arm64::has_amx(),
           can_fuse: |f| !matches!(f, &FusedSpec::LeakyRelu(_))
);

MMMKernel!(f16, apple_amx_mmm_f16_64x1; 64, 1; 128, 128; 1, 1; no_prefetch, crate::arm64::has_amx(),
           can_fuse: |f| !matches!(f, &FusedSpec::LeakyRelu(_))
);
