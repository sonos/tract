mod arm64simd;

use crate::Ops;
use crate::frame::PackedMatMul;

pub fn plug(ops: &mut Ops) {
    ops.smm = Box::new(|m, k, n| {
        log::info!("arm64simd activated for smm");
        Box::new(PackedMatMul::<arm64simd::SMatMul8x8, f32>::new(m, k, n))
    });
}
