use crate::Ops;
use crate::frame::element_wise::ElementWiseKer;
use crate::frame::reduce::{MapReduceKer, ReduceKer};
use crate::x86_64_fma::softmax::x86_64_avx512_softmax2_fastcompact_f16_64n;
use crate::x86_64_fma::softmax::x86_64_fma_softmax2_fastcompact_f32_32n;

pub mod mmm;

pub mod act;
pub mod act_f16;
pub mod act_f16_fp16;
pub mod by_scalar;
pub mod erf;
mod intel;
pub mod max;
pub mod panel_extract;
pub mod rms_norm;
pub mod softmax;

const AVX2: fn() -> bool = || is_x86_feature_detected!("avx2");
const FMA: fn() -> bool = || is_x86_feature_detected!("fma");
const AVX512F: fn() -> bool = || is_x86_feature_detected!("avx512f");
#[cfg(tract_avx512vnni)]
const AVX512VNNI: fn() -> bool = || is_x86_feature_detected!("avx512vnni");

tanh_impl!(f32, fma_tanh_f32, 8, 8, is_x86_feature_detected!("fma"));
sigmoid_impl!(f32, fma_sigmoid_f32, 8, 8, is_x86_feature_detected!("fma"));

// AVX-512 (zmm, 16-wide) variants. The assembly lives in x86_64/avx512/; the
// main loop handles 64 lanes (4 zmm) per iteration with a 16-lane tail, so
// nr()=16 (any multiple of 16 is safe).
tanh_impl!(f32, avx512_tanh_f32, 16, 16, is_x86_feature_detected!("avx512f"));
sigmoid_impl!(f32, avx512_sigmoid_f32, 16, 16, is_x86_feature_detected!("avx512f"));

fn plug_avx2(_ops: &mut Ops) {}

fn plug_fma(ops: &mut Ops) {
    panel_extract::plug(ops);

    ops.sigmoid_f32 = Box::new(|| fma_sigmoid_f32::ew());
    ops.tanh_f32 = Box::new(|| fma_tanh_f32::ew());

    ops.mul_by_scalar_f32 = Box::new(|| by_scalar::x86_64_avx_f32_mul_by_scalar_32n::ew());
    ops.max_f32 = Box::new(|| max::x86_64_fma_max_f32_32n::red());
    ops.softmax2_fastcompact_f32 = Box::new(|| x86_64_fma_softmax2_fastcompact_f32_32n::red());

    log::info!("sigmoid_f32, tanh_f32: x86_64/fma activated");
}

/// On hosts that also support AVX-512_FP16 (Sapphire Rapids / Granite Rapids /
/// later, and recent Xeon-D / consumer parts), upgrade the f16 element-wise
/// kernels from the f32-roundtrip implementations in `act_f16.rs` to the
/// native f16 implementations in `act_f16_fp16.rs` where the native path is
/// actually faster on this uarch. We benched each op against its f32-roundtrip
/// equivalent on Sapphire Rapids and only plug in the ones that win:
///
///   hardswish_f16:  8.71 → 31.6 Gelem/s  (3.62× native) — plug in
///   leaky_relu_f16: 9.44 →  5.85 Gelem/s (0.62× native — regression) — keep
///                   the f32-roundtrip version from act_f16.rs. The native
///                   kernel exists in act_f16_fp16.rs for future revisits but
///                   is not wired here.
fn plug_avx512fp16(ops: &mut Ops) {
    ops.hardswish_f16 = Box::new(|| act_f16_fp16::x86_64_avx512fp16_hardswish_f16_128n::ew());

    log::info!("hardswish_f16: x86_64/avx512fp16 native activated");
}

fn plug_avx512f(ops: &mut Ops) {
    ops.sigmoid_f32 = Box::new(|| avx512_sigmoid_f32::ew());
    ops.tanh_f32 = Box::new(|| avx512_tanh_f32::ew());
    ops.hardswish_f32 = Box::new(|| act::x86_64_avx512_hardswish_f32_64n::ew());
    ops.leaky_relu_f32 = Box::new(|| act::x86_64_avx512_leaky_relu_f32_64n::ew());
    ops.silu_f32 = Box::new(|| act::x86_64_avx512_silu_f32_16n::ew());
    ops.gelu_f32 = Box::new(|| act::x86_64_avx512_gelu_f32_16n::ew());

    ops.sigmoid_f16 = Box::new(|| act_f16::x86_64_avx512_sigmoid_f16_16n::ew());
    ops.tanh_f16 = Box::new(|| act_f16::x86_64_avx512_tanh_f16_16n::ew());
    ops.hardswish_f16 = Box::new(|| act_f16::x86_64_avx512_hardswish_f16_64n::ew());
    ops.leaky_relu_f16 = Box::new(|| act_f16::x86_64_avx512_leaky_relu_f16_64n::ew());
    ops.silu_f16 = Box::new(|| act_f16::x86_64_avx512_silu_f16_16n::ew());
    ops.gelu_f16 = Box::new(|| act_f16::x86_64_avx512_gelu_f16_16n::ew());

    ops.max_f32 = Box::new(|| max::x86_64_avx512_max_f32_64n::red());
    ops.softmax2_fastcompact_f32 =
        Box::new(|| softmax::x86_64_avx512_softmax2_fastcompact_f32_64n::red());
    ops.softmax2_fastcompact_f16 = Box::new(|| x86_64_avx512_softmax2_fastcompact_f16_64n::red());

    ops.erf_f32 = Box::new(|| erf::x86_64_avx512_erf_f32_64n::ew());

    ops.rms_norm_f32 = Box::new(rms_norm::rms_norm_f32);

    log::info!(
        "sigmoid_f32, tanh_f32, hardswish_f32, leaky_relu_f32, \
         silu_f32, gelu_f32, \
         sigmoid_f16, tanh_f16, hardswish_f16, leaky_relu_f16, \
         silu_f16, gelu_f16, \
         max_f32, softmax2_fastcompact_f32, softmax2_fastcompact_f16, erf_f32, \
         rms_norm_f32: x86_64/avx512f activated"
    );
}

pub fn plug(ops: &mut Ops) {
    mmm::plug(ops);
    if is_x86_feature_detected!("avx2") {
        plug_avx2(ops);
        if is_x86_feature_detected!("fma") {
            plug_fma(ops);
            if is_x86_feature_detected!("avx512f") {
                plug_avx512f(ops);
                if is_x86_feature_detected!("avx512fp16") {
                    plug_avx512fp16(ops);
                }
            }
        }
    }
}
