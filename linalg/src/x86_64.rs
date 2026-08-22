use crate::Ops;
use crate::frame::element_wise::ElementWiseKer;
use crate::frame::reduce::{MapReduceKer, ReduceKer};
use crate::x86_64::softmax::x86_64_avx512_softmax2_f32_64n;
use crate::x86_64::softmax::x86_64_fma_softmax2_f32_32n;

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
    cpuid_vendor()
}

#[cfg(target_arch = "x86_64")]
fn cpuid_vendor() -> Vendor {
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

#[cfg(not(target_arch = "x86_64"))]
fn cpuid_vendor() -> Vendor {
    Vendor::Other
}

pub mod act;
pub mod act_f16;
pub mod act_f16_fp16;

// CPUID probes, tile permission syscalls and uarch burst measurements: host machinery
// with no kernel to enumerate, and only ever consulted by `plug`.
#[cfg(target_arch = "x86_64")]
pub mod amx;
#[cfg(target_arch = "x86_64")]
pub mod amx_bf16;
#[cfg(target_arch = "x86_64")]
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

/// A CPUID feature probe, answering false in a build that does not target x86_64: the
/// kernels it gates are bail stubs there, so nothing may select them.
macro_rules! cpu_feature {
    ($id:ident = $feature:tt) => {
        #[cfg(target_arch = "x86_64")]
        const $id: fn() -> bool = || is_x86_feature_detected!($feature);
        #[cfg(not(target_arch = "x86_64"))]
        const $id: fn() -> bool = || false;
    };
}

cpu_feature!(AVX = "avx");
cpu_feature!(AVX2 = "avx2");
cpu_feature!(FMA = "fma");
cpu_feature!(AVX512F = "avx512f");
cpu_feature!(AVX512FP16 = "avx512fp16");
cpu_feature!(F16C = "f16c");

/// The q40f16 and f16 packings of `fma_mmm_f32_32x1` need the f16 conversions on top of FMA.
const FMA_F16C: fn() -> bool = || FMA() && F16C();

#[cfg(tract_avx512vnni)]
cpu_feature!(AVX512VNNI = "avx512vnni");

tanh_impl!(x86_64; f32, fma_tanh_f32, 8, 8, FMA());
sigmoid_impl!(x86_64; f32, fma_sigmoid_f32, 8, 8, FMA());
silu_impl!(x86_64; f32, fma_silu_f32, 8, 8, FMA());

// AVX-without-FMA ports of the fma kernels above (each vfmadd132ps expanded
// to an in-place vmulps+vaddps pair) for CPUs outside the fma tier.
tanh_impl!(x86_64; f32, avx_tanh_f32, 8, 8, AVX());
sigmoid_impl!(x86_64; f32, avx_sigmoid_f32, 8, 8, AVX());

// AVX-512 (zmm, 16-wide) variants. The assembly lives in x86_64/avx512/; the
// main loop handles 64 lanes (4 zmm) per iteration with a 16-lane tail, so
// nr()=16 (any multiple of 16 is safe).
tanh_impl!(x86_64; f32, avx512_tanh_f32, 16, 16, AVX512F());
sigmoid_impl!(x86_64; f32, avx512_sigmoid_f32, 16, 16, AVX512F());
silu_impl!(x86_64; f32, avx512_silu_f32, 16, 16, AVX512F());

fn plug_avx2(_ops: &mut Ops) {}

/// Element-wise kernels for AVX-capable CPUs outside the fma tier: the
/// mul_by_scalar / max / min asm is plain AVX, and sigmoid / tanh have
/// dedicated mul+add ports. softmax uses fma asm and keeps its generic
/// fallback on this tier.
fn plug_avx(ops: &mut Ops) {
    ops.sigmoid_f32 = Box::new(|| avx_sigmoid_f32::ew());
    ops.tanh_f32 = Box::new(|| avx_tanh_f32::ew());

    ops.mul_by_scalar_f32 = Box::new(|| by_scalar::x86_64_avx_f32_mul_by_scalar_32n::ew());
    ops.max_f32 = Box::new(|| max::x86_64_fma_max_f32_32n::red());
    ops.min_f32 = Box::new(|| min::x86_64_fma_min_f32_32n::red());

    log::info!("sigmoid_f32, tanh_f32, mul_by_scalar_f32, max_f32, min_f32: x86_64/avx activated");
}

fn plug_fma(ops: &mut Ops) {
    panel_extract::plug(ops);

    ops.sigmoid_f32 = Box::new(|| fma_sigmoid_f32::ew());
    ops.tanh_f32 = Box::new(|| fma_tanh_f32::ew());
    ops.silu_f32 = Box::new(|| fma_silu_f32::ew());

    ops.mul_by_scalar_f32 = Box::new(|| by_scalar::x86_64_avx_f32_mul_by_scalar_32n::ew());
    ops.max_f32 = Box::new(|| max::x86_64_fma_max_f32_32n::red());
    ops.min_f32 = Box::new(|| min::x86_64_fma_min_f32_32n::red());
    ops.softmax2_f32 = Box::new(|| x86_64_fma_softmax2_f32_32n::red());

    log::info!("sigmoid_f32, tanh_f32, silu_f32: x86_64/fma activated");
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
    ops.silu_f32 = Box::new(|| avx512_silu_f32::ew());
    ops.gelu_f32 = Box::new(|| act::x86_64_avx512_gelu_f32_16n::ew());

    ops.sigmoid_f16 = Box::new(|| act_f16::x86_64_avx512_sigmoid_f16_16n::ew());
    ops.tanh_f16 = Box::new(|| act_f16::x86_64_avx512_tanh_f16_16n::ew());
    ops.hardswish_f16 = Box::new(|| act_f16::x86_64_avx512_hardswish_f16_64n::ew());
    ops.leaky_relu_f16 = Box::new(|| act_f16::x86_64_avx512_leaky_relu_f16_64n::ew());
    ops.silu_f16 = Box::new(|| act_f16::x86_64_avx512_silu_f16_16n::ew());
    ops.gelu_f16 = Box::new(|| act_f16::x86_64_avx512_gelu_f16_16n::ew());

    ops.max_f32 = Box::new(|| max::x86_64_avx512_max_f32_64n::red());
    ops.softmax2_f32 = Box::new(|| x86_64_avx512_softmax2_f32_64n::red());

    ops.erf_f32 = Box::new(|| erf::x86_64_avx512_erf_f32_64n::ew());

    ops.rms_norm_f32 = Box::new(rms_norm::rms_norm_f32);

    log::info!(
        "sigmoid_f32, tanh_f32, hardswish_f32, leaky_relu_f32, \
         silu_f32, gelu_f32, \
         sigmoid_f16, tanh_f16, hardswish_f16, leaky_relu_f16, \
         silu_f16, gelu_f16, \
         max_f32, softmax2_f32, erf_f32, \
         rms_norm_f32: x86_64/avx512f activated"
    );
}

pub fn plug(ops: &mut Ops) {
    mmm::plug(ops);
    if AVX() && !(AVX2() && FMA()) {
        plug_avx(ops);
    }
    if AVX2() {
        plug_avx2(ops);
        if FMA() {
            plug_fma(ops);
            if AVX512F() {
                plug_avx512f(ops);
                if AVX512FP16() {
                    plug_avx512fp16(ops);
                }
            }
        }
    }
}

inventory::submit! {
    crate::platform::PlatformSelector {
        target: crate::platform::Target::X86_64,
        plug,
    }
}
