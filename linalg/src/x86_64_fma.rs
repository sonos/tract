use crate::frame::mmm::kernel::MatMatMulKer;
use crate::frame::ElementWiseImpl;
use crate::Ops;

pub mod mmm;
pub mod sigmoid;
pub mod tanh;

mod intel;

pub fn plug(ops: &mut Ops) {
    if is_x86_feature_detected!("fma") {
        ops.mmm_f32_impls.push((mmm::fma_mmm_f32_16x6::mmm(), None));
        ops.mmm_f32_impls.push((mmm::fma_mmm_f32_8x8::mmm(), None));
        ops.mmv_f32 = Box::new(|_, _| mmm::fma_mmm_f32_64x1::mmm());
        ops.sigmoid_f32 = Box::new(|| Box::new(ElementWiseImpl::<sigmoid::SigmoidF32, f32>::new()));
        ops.tanh_f32 = Box::new(|| Box::new(ElementWiseImpl::<tanh::TanhF32, f32>::new()));
        log::info!("mmm_f32, sigmoid_f32, tanh_f32: x86_64/fma activated");
    }
    if is_x86_feature_detected!("avx2") {
        ops.qmmm_i32 = Box::new(|_, _, _| mmm::avx2_mmm_i32_8x8::mmm());
        log::info!("mmm_i8_i8 and mmm_i8_i32: x86_64/avx2 activated");
    }
    ops.set_cost_models(intel::models());
}
