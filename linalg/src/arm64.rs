mod arm64simd;

use crate::Ops;

use crate::frame::TileOp;

pub fn plug(ops: &mut Ops) {
    log::info!("arm64simd activated for stile");
    ops.stile = Box::new(|m, k, n| Box::new(TileOp::<arm64simd::STile8x8, f32>::new(m, k, n)));
}
