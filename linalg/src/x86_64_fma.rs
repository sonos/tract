use crate::Ops;
use crate::frame::element_wise::ElementWiseKer;
use crate::frame::reduce::{MapReduceKer, ReduceKer};

pub mod mmm;

pub mod by_scalar;
mod intel;
pub mod max;
pub mod panel_extract;
pub mod softmax;
pub mod sum;

const AVX2: fn() -> bool = || is_x86_feature_detected!("avx2");
const FMA: fn() -> bool = || is_x86_feature_detected!("fma");
const AVX512F: fn() -> bool = || is_x86_feature_detected!("avx512f");

tanh_impl!(f32, fma_tanh_f32, 8, 8, is_x86_feature_detected!("fma"));
sigmoid_impl!(f32, fma_sigmoid_f32, 8, 8, is_x86_feature_detected!("fma"));

fn plug_avx2(_ops: &mut Ops) {}

fn plug_fma(ops: &mut Ops) {
    panel_extract::plug_avx2(ops);

    ops.sigmoid_f32 = Box::new(|| fma_sigmoid_f32::ew());
    ops.tanh_f32 = Box::new(|| fma_tanh_f32::ew());

    ops.mul_by_scalar_f32 = Box::new(|| by_scalar::x86_64_avx_f32_mul_by_scalar_32n::ew());
    ops.max_f32 = Box::new(|| max::x86_64_fma_max_f32_32n::red());
    ops.sum_f32 = Box::new(|| sum::x86_64_fma_sum_f32_32n::red());
    ops.softmax2_fastcompact_f32 =
        Box::new(|| softmax::x86_64_fma_softmax2_fastcompact_f32_32n::red());
}

fn plug_avx512f(ops: &mut Ops) {
    panel_extract::plug_avx512f(ops);

    ops.mul_by_scalar_f32 = Box::new(|| by_scalar::x86_64_avx512_f32_mul_by_scalar_64n::ew());
    ops.max_f32 = Box::new(|| max::x86_64_avx512_max_f32_64n::red());
    ops.sum_f32 = Box::new(|| sum::x86_64_avx512_sum_f32_64n::red());
    ops.softmax2_fastcompact_f32 =
        Box::new(|| softmax::x86_64_avx512_softmax2_fastcompact_f32_64n::red());
}

pub fn plug(ops: &mut Ops) {
    mmm::plug(ops);
    if is_x86_feature_detected!("avx2") {
        plug_avx2(ops);
        if is_x86_feature_detected!("fma") {
            plug_fma(ops);
            if is_x86_feature_detected!("avx512f") {
                plug_avx512f(ops);
            }
        }
    }
}
