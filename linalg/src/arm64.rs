mod arm64simd;

use crate::Ops;

use crate::frame::MatMatMulImpl;
use crate::frame::QMatMatMulImpl;
use crate::frame::SigmoidImpl;
use crate::frame::TanhImpl;

pub fn plug(ops: &mut Ops) {
    log::info!("arm64simd activated for smmm");
    ops.mmm_f32 = Box::new(|m, k, n| {
        Box::new(MatMatMulImpl::<arm64simd::MatMatMulF32x8x8, f32, f32, f32, f32>::new(m, k, n))
    });
    ops.qmmm_i8_i8 = Box::new(|m, k, n| {
        Box::new(QMatMatMulImpl::from(
            MatMatMulImpl::<arm64simd::MatMatMulI8x8x8, i8, i8, i8, i32>::new(m, k, n),
        ))
    });
    ops.qmmm_i8_i32 = Box::new(|m, k, n| {
        Box::new(QMatMatMulImpl::from(MatMatMulImpl::<
            arm64simd::MatMatMulI8xI32x8x8,
            i8,
            i8,
            i32,
            i32,
        >::new(m, k, n)))
    });
    ops.sigmoid_f32 = Box::new(|| Box::new(SigmoidImpl::<arm64simd::SigmoidF32x4n, f32>::new()));
    ops.tanh_f32 = Box::new(|| Box::new(TanhImpl::<arm64simd::TanhF32x4n, f32>::new()));
}
