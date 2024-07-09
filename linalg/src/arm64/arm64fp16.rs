use tract_data::half::f16;

mod by_scalar;
mod leaky_relu;
mod max;
mod sum;
pub use by_scalar::*;
pub use leaky_relu::*;
pub use max::*;
pub use sum::*;

use crate::mmm::no_prefetch;

MMMExternKernel!(f16, arm64fp16_mmm_f16_16x8_gen; 16, 8; 16, 16; 1, 1; no_prefetch, crate::arm64::has_fp16());
MMMExternKernel!(f16, arm64fp16_mmm_f16_16x8_a55; 16, 8; 16, 16; 1, 1; no_prefetch, crate::arm64::has_fp16());
MMMExternKernel!(f16, arm64fp16_mmm_f16_32x4_gen; 32, 4; 16, 16; 1, 1; no_prefetch, crate::arm64::has_fp16());
MMMExternKernel!(f16, arm64fp16_mmm_f16_32x4_a55; 32, 4; 16, 16; 1, 1; no_prefetch, crate::arm64::has_fp16());
MMMExternKernel!(f16, arm64fp16_mmm_f16_128x1_gen; 128, 1; 16, 16; 1, 1; no_prefetch, crate::arm64::has_fp16());
MMMExternKernel!(f16, arm64fp16_mmm_f16_128x1_a55; 128, 1; 16, 16; 1, 1; no_prefetch, crate::arm64::has_fp16());

MMMExternKernel!(f16, arm64fp16_mmm_f16_64x1_gen; 64, 1; 16, 16; 1, 1; no_prefetch, crate::arm64::has_fp16(),
     packing_defs: {
         use crate::frame::block_quant::*;
         const PQ40_R64: PackedBlockQuantFormat = PackedBlockQuantFormat::new(&Q4_0, 64);
         const F16_B: PackedFormat = PackedFormat::new(DatumType::F16, 1, 2, 0);
         const PQ40_F16: (&dyn MMMInputFormat, &dyn MMMInputFormat) = (&PQ40_R64, &F16_B);
     },
 packings: PQ40_F16,
 test: mmm_packed_packed_tests!{ crate::arm64::has_fp16(), arm64fp16_mmm_f16_64x1_gen, q40f16:1 }
);

tanh_impl!(f16, arm64fp16_tanh_f16_8n, 8, 8, crate::arm64::has_fp16());
sigmoid_impl!(f16, arm64fp16_sigmoid_f16_8n, 8, 8, crate::arm64::has_fp16());
