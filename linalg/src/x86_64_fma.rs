use crate::frame::ElementWiseImpl;
use crate::frame::MatMatMulImpl;
use crate::Ops;

pub mod mmm;
pub mod sigmoid;
pub mod tanh;

pub fn plug(ops: &mut Ops) {
    if is_x86_feature_detected!("fma") {
        ops.mmm_f32 =
            Box::new(|_, _, _| Box::new(MatMatMulImpl::<mmm::MatMatMulF32x16x6, f32>::new()));
        ops.mmv_f32 =
            Box::new(|_, _| Box::new(MatMatMulImpl::<mmm::MatMatMulF32x64x1, f32>::new()));
        ops.sigmoid_f32 = Box::new(|| Box::new(ElementWiseImpl::<sigmoid::SigmoidF32, f32>::new()));
        ops.tanh_f32 = Box::new(|| Box::new(ElementWiseImpl::<tanh::TanhF32, f32>::new()));
        log::info!("mmm_f32, sigmoid_f32, tanh_f32: x86_64/fma activated");
    }
    if is_x86_feature_detected!("avx2") {
        ops.qmmm_i32 =
            Box::new(|_, _, _| Box::new(MatMatMulImpl::<mmm::MatMatMulI8x8x8, i32>::new()));
        ops.qmmm_i32 =
            Box::new(|_, _, _| Box::new(MatMatMulImpl::<mmm::MatMatMulI8xI32x8x8, i32>::new()));
        log::info!("mmm_i8_i8 and mmm_i8_i32: x86_64/avx2 activated");
    }
}
