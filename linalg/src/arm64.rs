mod arm64simd;

use crate::Ops;

use crate::frame::MatMatMulImpl;

pub fn plug(ops: &mut Ops) {
    log::info!("arm64simd activated for smmm");
    ops.smmm =
        Box::new(|m, k, n| Box::new(MatMatMulImpl::<arm64simd::SMatMatMul8x8, f32>::new(m, k, n)));
}
