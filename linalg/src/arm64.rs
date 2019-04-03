mod arm64simd;

use crate::frame::PackedConv;
use crate::frame::PackedMatMul;
use crate::Ops;

pub fn plug(ops: &mut Ops) {
    log::info!("arm64simd activated for smm");
    ops.smm =
        Box::new(|m, k, n| Box::new(PackedMatMul::<arm64simd::SMatMul8x8, f32>::new(m, k, n)));
    log::info!("arm64simd activated for sconv");
    ops.sconv = Box::new(|m, k, n| Box::new(PackedConv::<arm64simd::SConv8x8, f32>::new(m, k, n)));
}
