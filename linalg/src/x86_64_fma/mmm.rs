use crate::frame::mmm::*;

MMMKernel!(f32, fma_mmm_f32_8x8; 8, 8; 32, 4; 0, 0; no_prefetch, is_x86_feature_detected!("fma"));
MMMKernel!(f32, fma_mmm_f32_16x6; 16, 6; 32, 4; 0, 0; no_prefetch, is_x86_feature_detected!("fma"));
MMMKernel!(f32, fma_mmm_f32_16x5; 16, 5; 32, 4; 0, 0; no_prefetch, is_x86_feature_detected!("fma"));
MMMKernel!(f32, fma_mmm_f32_24x4; 24, 4; 32, 4; 0, 0; no_prefetch, is_x86_feature_detected!("fma"));
MMMKernel!(f32, fma_mmm_f32_32x3; 32, 3; 32, 4; 0, 0; no_prefetch, is_x86_feature_detected!("fma"));
MMMKernel!(f32, fma_mmm_f32_40x2; 40, 2; 32, 4; 0, 0; no_prefetch, is_x86_feature_detected!("fma"));
MMMKernel!(f32, fma_mmm_f32_64x1; 64, 1; 32, 4; 0, 0; no_prefetch, is_x86_feature_detected!("fma"));

MMMKernel!(i32, avx2_mmm_i32_8x8; 8, 8; 32, 4; 0, 0; no_prefetch, is_x86_feature_detected!("avx2"));

#[cfg(not(feature = "compile_all_kernels"))]
mod avx512_best {
    use super::*;
    MMMKernel!(f32, avx512_mmm_f32_16x1; 16, 1; 64, 4; 0, 0; no_prefetch, is_x86_feature_detected!("avx512f"));
    MMMKernel!(f32, avx512_mmm_f32_96x1; 96, 1; 64, 4; 0, 0; no_prefetch, is_x86_feature_detected!("avx512f"));
    MMMKernel!(f32, avx512_mmm_f32_96x2; 96, 2; 64, 4; 0, 0; no_prefetch, is_x86_feature_detected!("avx512f"));
    MMMKernel!(f32, avx512_mmm_f32_80x3; 80, 3; 64, 4; 0, 0; no_prefetch, is_x86_feature_detected!("avx512f"));
    MMMKernel!(f32, avx512_mmm_f32_64x4; 64, 4; 64, 4; 0, 0; no_prefetch, is_x86_feature_detected!("avx512f"));
    MMMKernel!(f32, avx512_mmm_f32_32x5; 32, 5; 64, 4; 0, 0; no_prefetch, is_x86_feature_detected!("avx512f"));
    MMMKernel!(f32, avx512_mmm_f32_32x6; 32, 6; 64, 4; 0, 0; no_prefetch, is_x86_feature_detected!("avx512f"));
    MMMKernel!(f32, avx512_mmm_f32_32x7; 32, 7; 64, 4; 0, 0; no_prefetch, is_x86_feature_detected!("avx512f"));
    MMMKernel!(f32, avx512_mmm_f32_32x8; 32, 8; 64, 4; 0, 0; no_prefetch, is_x86_feature_detected!("avx512f"));
    MMMKernel!(f32, avx512_mmm_f32_32x9; 32, 9; 64, 4; 0, 0; no_prefetch, is_x86_feature_detected!("avx512f"));
    MMMKernel!(f32, avx512_mmm_f32_32x10; 32, 10; 64, 4; 0, 0; no_prefetch, is_x86_feature_detected!("avx512f"));
    MMMKernel!(f32, avx512_mmm_f32_32x11; 32, 11; 64, 4; 0, 0; no_prefetch, is_x86_feature_detected!("avx512f"));
    MMMKernel!(f32, avx512_mmm_f32_32x12; 32, 12; 64, 4; 0, 0; no_prefetch, is_x86_feature_detected!("avx512f"));
    MMMKernel!(f32, avx512_mmm_f32_32x13; 32, 13; 64, 4; 0, 0; no_prefetch, is_x86_feature_detected!("avx512f"));
    MMMKernel!(f32, avx512_mmm_f32_32x14; 32, 14; 64, 4; 0, 0; no_prefetch, is_x86_feature_detected!("avx512f"));
}
#[cfg(not(feature = "compile_all_kernels"))]
pub use avx512_best::*;

#[cfg(feature = "compile_all_kernels")]
mod all_avx512 {
    use super::*;
    macro_rules! make_kernels_for_n {
        ($n:expr ; $m:expr) => (
            paste! {
                MMMKernel!(f32, [<avx512_mmm_f32_ $m x $n>]; $m, $n; 64, 4; 0, 0; no_prefetch, is_x86_feature_detected!("avx512f"));
            }
        );
        ($n:expr ; $m1:expr, $($y:expr),+) => (
            make_kernels_for_n!($n ; $m1);
            make_kernels_for_n!($n ; $($y),+);
        )
    }

    make_kernels_for_n!(1  ; 16, 32, 48, 64, 80, 96, 112, 128, 144, 160, 176, 192, 208, 224, 240);
    make_kernels_for_n!(2  ; 16, 32, 48, 64, 80, 96, 112, 128, 144, 160);
    make_kernels_for_n!(3  ; 16, 32, 48, 64, 80, 96, 112);
    make_kernels_for_n!(4  ; 16, 32, 48, 64, 80, 96);
    make_kernels_for_n!(5  ; 16, 32, 48, 64, 80);
    make_kernels_for_n!(6  ; 16, 32, 48, 64);
    make_kernels_for_n!(7  ; 16, 32, 48);
    make_kernels_for_n!(8  ; 16, 32, 48);
    make_kernels_for_n!(9  ; 16, 32, 48);
    make_kernels_for_n!(10 ; 16, 32);
    make_kernels_for_n!(11 ; 16, 32);
    make_kernels_for_n!(12 ; 16, 32);
    make_kernels_for_n!(13 ; 16, 32);
    make_kernels_for_n!(14 ; 16, 32);
    make_kernels_for_n!(15 ; 16);
    make_kernels_for_n!(16 ; 16);
    make_kernels_for_n!(17 ; 16);
    make_kernels_for_n!(18 ; 16);
    make_kernels_for_n!(19 ; 16);
    make_kernels_for_n!(20 ; 16);
    make_kernels_for_n!(21 ; 16);
    make_kernels_for_n!(22 ; 16);
    make_kernels_for_n!(23 ; 16);
    make_kernels_for_n!(24 ; 16);
    make_kernels_for_n!(25 ; 16);
    make_kernels_for_n!(26 ; 16);
    make_kernels_for_n!(27 ; 16);
    make_kernels_for_n!(28 ; 16);
    make_kernels_for_n!(29 ; 16);
}
#[cfg(feature = "compile_all_kernels")]
pub use all_avx512::*;
