mod arm64simd;

use crate::Ops;

use crate::frame::MatMatMulImpl;
use crate::frame::SigmoidImpl;
use crate::frame::TanhImpl;

pub fn plug(ops: &mut Ops) {
    log::info!("arm64simd activated for smmm");
    ops.smmm = Box::new(|m, k, n| {
        Box::new(MatMatMulImpl::<arm64simd::SMatMatMul8x8, f32, f32, f32, f32>::new(m, k, n))
    });
    ops.ssigmoid = Box::new(|| Box::new(SigmoidImpl::<arm64simd::SSigmoid4, f32>::new()));
    ops.stanh = Box::new(|| Box::new(TanhImpl::<arm64simd::STanh4, f32>::new()));
}
