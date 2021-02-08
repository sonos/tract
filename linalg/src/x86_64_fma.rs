use crate::frame::MatMatMulImpl;
use crate::frame::SigmoidImpl;
use crate::Ops;

pub mod mmm;
pub mod sigmoid;

pub fn plug(ops: &mut Ops) {
    if is_x86_feature_detected!("fma") {
        ops.mmm_f32 = Box::new(|m, k, n| {
            Box::new(MatMatMulImpl::<mmm::MatMatMulF32x16x6, f32, f32, f32, f32>::new(m, k, n))
        });
        ops.sigmoid_f32 = Box::new(|| Box::new(SigmoidImpl::<sigmoid::SigmoidF32, f32>::new()));
        log::info!("mmm_f32, sigmoid_f32 x86_64/fma activated");
    }
    if is_x86_feature_detected!("avx2") {
        ops.qmmm_i8_i8 = Box::new(|m, k, n| {
            Box::new(MatMatMulImpl::<mmm::MatMatMulI8x8x8, i8, i8, i8, i32>::new(m, k, n))
        });
        ops.qmmm_i8_i32 = Box::new(|m, k, n| {
            Box::new(MatMatMulImpl::<mmm::MatMatMulI8xI32x8x8, i8, i8, i32, i32>::new(
                m, k, n,
            ))
        });
        log::info!("mmm_i8_i8 and mmm_i8_i32 x86_64/fma activated");
    }
}
