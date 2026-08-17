use crate::Ops;
use crate::frame::element_wise::ElementWiseKer;
use crate::frame::reduce::{MapReduceKer, ReduceKer};
use crate::x86_64_fma::softmax::x86_64_avx512_softmax2_fastcompact_f16_64n;
use crate::x86_64_fma::softmax::x86_64_fma_softmax2_fastcompact_f32_32n;

pub mod mmm;

mod amd_avx512_linear;
mod amd_fma_linear;
mod intel_avx512_linear;
mod intel_avx512_mmv_linear;
mod intel_fma_linear;

/// CPU vendor, the axis (with the AVX-512-vs-FMA tier, decided by which plug runs)
/// that selects a per-target `LinearCostModel`. `TRACT_X86_KIND=intel|amd|other`
/// overrides the CPUID probe (for forcing a cohort under emulation or in CI).
#[derive(PartialEq, Clone, Copy)]
pub(crate) enum Vendor {
    Intel,
    Amd,
    Other,
}

pub(crate) fn vendor() -> Vendor {
    if let Ok(k) = std::env::var("TRACT_X86_KIND") {
        return match k.as_str() {
            "intel" => Vendor::Intel,
            "amd" => Vendor::Amd,
            _ => Vendor::Other,
        };
    }
    // `unsafe` is required on the MSRV (1.91); newer rustc deems it redundant.
    #[allow(unused_unsafe)]
    let id = unsafe { std::arch::x86_64::__cpuid(0) };
    let mut s = [0u8; 12];
    s[0..4].copy_from_slice(&id.ebx.to_le_bytes());
    s[4..8].copy_from_slice(&id.edx.to_le_bytes());
    s[8..12].copy_from_slice(&id.ecx.to_le_bytes());
    match &s {
        b"GenuineIntel" => Vendor::Intel,
        b"AuthenticAMD" => Vendor::Amd,
        _ => Vendor::Other,
    }
}

pub mod act;
pub mod act_f16;
pub mod act_f16_fp16;

pub mod amx;
pub mod amx_bf16;
pub mod avxvnni;
pub mod by_scalar;
pub mod erf;
#[cfg(tract_avx512vnni)]
pub mod fma_width;
pub mod max;
pub mod min;
pub mod panel_extract;
pub mod rms_norm;
pub mod softmax;

const AVX: fn() -> bool = || is_x86_feature_detected!("avx");
const AVX2: fn() -> bool = || is_x86_feature_detected!("avx2");
const FMA: fn() -> bool = || is_x86_feature_detected!("fma");
const AVX512F: fn() -> bool = || is_x86_feature_detected!("avx512f");
#[cfg(tract_avx512vnni)]
const AVX512VNNI: fn() -> bool = || is_x86_feature_detected!("avx512vnni");

tanh_impl!(f32, fma_tanh_f32, 8, 8, is_x86_feature_detected!("fma"));
sigmoid_impl!(f32, fma_sigmoid_f32, 8, 8, is_x86_feature_detected!("fma"));
silu_impl!(f32, fma_silu_f32, 8, 8, is_x86_feature_detected!("fma"));

// AVX-without-FMA ports of the fma kernels above (each vfmadd132ps expanded
// to an in-place vmulps+vaddps pair) for CPUs outside the fma tier.
tanh_impl!(f32, avx_tanh_f32, 8, 8, is_x86_feature_detected!("avx"));
sigmoid_impl!(f32, avx_sigmoid_f32, 8, 8, is_x86_feature_detected!("avx"));

// AVX-512 (zmm, 16-wide) variants. The assembly lives in x86_64/avx512/; the
// main loop handles 64 lanes (4 zmm) per iteration with a 16-lane tail, so
// nr()=16 (any multiple of 16 is safe).
tanh_impl!(f32, avx512_tanh_f32, 16, 16, is_x86_feature_detected!("avx512f"));
sigmoid_impl!(f32, avx512_sigmoid_f32, 16, 16, is_x86_feature_detected!("avx512f"));

fn plug_avx2(_ops: &mut Ops) {}

/// Element-wise kernels for AVX-capable CPUs outside the fma tier: the
/// mul_by_scalar / max / min asm is plain AVX. Activations are dispatched through
/// the activation registry. softmax keeps its generic fallback on this tier.
fn plug_avx(ops: &mut Ops) {
    ops.mul_by_scalar_f32 = Box::new(|| by_scalar::x86_64_avx_f32_mul_by_scalar_32n::ew());
    ops.max_f32 = Box::new(|| max::x86_64_fma_max_f32_32n::red());
    ops.min_f32 = Box::new(|| min::x86_64_fma_min_f32_32n::red());

    log::info!("mul_by_scalar_f32, max_f32, min_f32: x86_64/avx activated");
}

fn plug_fma(ops: &mut Ops) {
    panel_extract::plug(ops);

    ops.mul_by_scalar_f32 = Box::new(|| by_scalar::x86_64_avx_f32_mul_by_scalar_32n::ew());
    ops.max_f32 = Box::new(|| max::x86_64_fma_max_f32_32n::red());
    ops.min_f32 = Box::new(|| min::x86_64_fma_min_f32_32n::red());
    ops.softmax2_fastcompact_f32 = Box::new(|| x86_64_fma_softmax2_fastcompact_f32_32n::red());

    log::info!("mul_by_scalar_f32, max_f32, min_f32, softmax2: x86_64/fma activated");
}

/// AVX-512_FP16 tier. The activations it used to upgrade (hardswish_f16) are now
/// dispatched through the activation registry, so nothing is plugged here.
fn plug_avx512fp16(_ops: &mut Ops) {}

fn plug_avx512f(ops: &mut Ops) {
    ops.max_f32 = Box::new(|| max::x86_64_avx512_max_f32_64n::red());
    ops.softmax2_fastcompact_f32 =
        Box::new(|| softmax::x86_64_avx512_softmax2_fastcompact_f32_64n::red());
    ops.softmax2_fastcompact_f16 = Box::new(|| x86_64_avx512_softmax2_fastcompact_f16_64n::red());

    ops.rms_norm_f32 = Box::new(rms_norm::rms_norm_f32);

    log::info!(
        "max_f32, softmax2_fastcompact_f32, \
         softmax2_fastcompact_f16, rms_norm_f32: x86_64/avx512f activated"
    );
}

pub fn plug(ops: &mut Ops) {
    mmm::plug(ops);
    if is_x86_feature_detected!("avx")
        && !(is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma"))
    {
        plug_avx(ops);
    }
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
