use tract_data::half::f16;

mod by_scalar;
mod leaky_relu;
mod max;
pub mod panel_extract;
mod sum;
mod unicast;
pub use by_scalar::*;
pub use leaky_relu::*;
pub use max::*;
pub use sum::*;
pub use unicast::*;

use crate::block_quant::PackedBlockQuantFormat;
use crate::block_quant::Q4_0;
use crate::frame::mmm::ImplementationQuality::ManuallyOptimized;
use crate::Ops;

const FP16: fn() -> bool = crate::arm64::has_fp16;

MMMExternKernel!(arm64fp16_mmm_f16_16x8_gen<f16>(16, 8)@(16, 16) where(FP16) quality(ManuallyOptimized));
MMMExternKernel!(arm64fp16_mmm_f16_16x8_a55<f16>(16, 8)@(16, 16) where(FP16) quality(ManuallyOptimized));
MMMExternKernel!(arm64fp16_mmm_f16_32x4_gen<f16>(32, 4)@(16, 16) where(FP16) quality(ManuallyOptimized));
MMMExternKernel!(arm64fp16_mmm_f16_32x4_a55<f16>(32, 4)@(16, 16) where(FP16) quality(ManuallyOptimized));
MMMExternKernel!(arm64fp16_mmm_f16_128x1_gen<f16>(128,1)@(16, 16) where(FP16) quality(ManuallyOptimized));
MMMExternKernel!(arm64fp16_mmm_f16_128x1_a55<f16>(128,1)@(16, 16) where(FP16) quality(ManuallyOptimized));

MMMExternKernel!(arm64fp16_mmm_f16_64x3_gen<f16>(64, 3)@(16, 16) where(FP16) quality(ManuallyOptimized));
MMMExternKernel!(arm64fp16_mmm_f16_32x6_gen<f16>(32, 6)@(16, 16) where(FP16) quality(ManuallyOptimized));

MMMExternKernel! { arm64fp16_mmm_f16_64x1_gen<f16>(64, 1)@(16, 16) where(FP16)
    packing[1] = q40f16z16se => |k| k.with_packing_a(PackedBlockQuantFormat::new(&Q4_0, 64, 16, true));
    packing[2] = q40f16z16 => |k| k.with_packing_a(PackedBlockQuantFormat::new(&Q4_0, 64, 16, false));
    quality(ManuallyOptimized)
}

pub fn plug(ops: &mut Ops) {
    panel_extract::plug(ops);
    ops.mmm_impls.extend_from_slice(&[
        arm64fp16_mmm_f16_16x8_a55.mmm(),
        arm64fp16_mmm_f16_16x8_gen.mmm(),
        arm64fp16_mmm_f16_32x4_a55.mmm(),
        arm64fp16_mmm_f16_16x8_gen.mmm(),
        arm64fp16_mmm_f16_128x1_a55.mmm(),
        arm64fp16_mmm_f16_128x1_gen.mmm(),
        arm64fp16_mmm_f16_64x1_gen.mmm(),
        arm64fp16_mmm_f16_32x6_gen.mmm(),
        arm64fp16_mmm_f16_64x1_gen.mmm(),
    ]);
}

tanh_impl!(f16, arm64fp16_tanh_f16_8n, 8, 8, crate::arm64::has_fp16());
sigmoid_impl!(f16, arm64fp16_sigmoid_f16_8n, 8, 8, crate::arm64::has_fp16());

#[cfg(test)]
mod test {

    #[test]
    fn kits() {
        let mut ops = crate::generic();
        super::plug(&mut ops);
    }
}
