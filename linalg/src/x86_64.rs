pub mod mmm;

mod amd_avx512_linear;
mod amd_fma_linear;
mod intel_avx512_linear;
mod intel_avx512_mmv_linear;
mod intel_fma_linear;

/// CPU vendor, the axis (with the AVX-512-vs-FMA tier, read from the instruction set)
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
// with no kernel to enumerate, and only ever consulted by `isa_set`.
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

#[cfg(tract_avx512vnni)]
cpu_feature!(AVX512VNNI = "avx512vnni");

ew_routine!(x86_64; Tanh, f32, fma_tanh_f32, 8, 8, isa(X86_64Avx2, X86_64Fma));
ew_routine!(x86_64; Sigmoid, f32, fma_sigmoid_f32, 8, 8, isa(X86_64Avx2, X86_64Fma));
ew_routine!(x86_64; Silu, f32, fma_silu_f32, 8, 8, isa(X86_64Avx2, X86_64Fma));

// AVX-without-FMA ports of the fma kernels above (each vfmadd132ps expanded
// to an in-place vmulps+vaddps pair) for CPUs outside the fma tier.
ew_routine!(x86_64; Tanh, f32, avx_tanh_f32, 8, 8, isa(X86_64Avx));
ew_routine!(x86_64; Sigmoid, f32, avx_sigmoid_f32, 8, 8, isa(X86_64Avx));

// AVX-512 (zmm, 16-wide) variants. The assembly lives in x86_64/avx512/; the
// main loop handles 64 lanes (4 zmm) per iteration with a 16-lane tail, so
// nr()=16 (any multiple of 16 is safe).
ew_routine!(x86_64; Tanh, f32, avx512_tanh_f32, 16, 16, isa(X86_64Avx512f));
ew_routine!(x86_64; Sigmoid, f32, avx512_sigmoid_f32, 16, 16, isa(X86_64Avx512f));
ew_routine!(x86_64; Silu, f32, avx512_silu_f32, 16, 16, isa(X86_64Avx512f));

routine!(x86_64; F32, Gelu, act::x86_64_avx512_gelu_f32_16n, isa(X86_64Avx512f));
routine!(x86_64; F32, Erf, erf::x86_64_avx512_erf_f32_64n, isa(X86_64Avx512f));
routine!(x86_64; F32, Hardswish, act::x86_64_avx512_hardswish_f32_64n, isa(X86_64Avx512f));

routine!(x86_64; F16, Sigmoid, act_f16::x86_64_avx512_sigmoid_f16_16n, isa(X86_64Avx512f));
routine!(x86_64; F16, Tanh, act_f16::x86_64_avx512_tanh_f16_16n, isa(X86_64Avx512f));
routine!(x86_64; F16, Silu, act_f16::x86_64_avx512_silu_f16_16n, isa(X86_64Avx512f));
routine!(x86_64; F16, Gelu, act_f16::x86_64_avx512_gelu_f16_16n, isa(X86_64Avx512f));
routine!(x86_64; F16, Hardswish, act_f16::x86_64_avx512_hardswish_f16_64n, isa(X86_64Avx512f));

routine!(x86_64; F16, Hardswish, act_f16_fp16::x86_64_avx512fp16_hardswish_f16_128n,
    isa(X86_64Avx512Fp16));

routine!(x86_64; F32Param, LeakyRelu, act::x86_64_avx512_leaky_relu_f32_64n, isa(X86_64Avx512f));
routine!(x86_64; F16Param, LeakyRelu, act_f16::x86_64_avx512_leaky_relu_f16_64n, isa(X86_64Avx512f));

// Correct, and slower than the f32 round-trip above on every AVX-512_FP16 part measured, so it
// is declared to keep its tests running and never preferred. A part where fp16 mul and max
// saturate their ports would want the boost dropped.
routine!(x86_64; F16Param, LeakyRelu, act_f16_fp16::x86_64_avx512fp16_leaky_relu_f16_128n,
    isa(X86_64Avx512Fp16), boost(crate::isa::NEVER_PREFERRED));

/// What CPUID says this core has, in the shared vocabulary.
pub fn isa_set() -> crate::isa::IsaSet {
    use crate::isa::{Isa, IsaSet};
    let mut set = IsaSet::of_arch(crate::isa::Arch::X86_64);
    for (isa, probe) in [
        (Isa::X86_64Avx, AVX),
        (Isa::X86_64Avx2, AVX2),
        (Isa::X86_64Fma, FMA),
        (Isa::X86_64F16c, F16C),
        (Isa::X86_64Avx512f, AVX512F),
        (Isa::X86_64Avx512Fp16, AVX512FP16),
    ] {
        if probe() {
            set = set.with(isa);
        }
    }
    #[cfg(tract_avx512vnni)]
    if AVX512VNNI() {
        set = set.with(Isa::X86_64Avx512Vnni);
    }
    #[cfg(tract_avxvnni)]
    if avxvnni::has_avxvnni() {
        set = set.with(Isa::X86_64AvxVnni);
    }
    #[cfg(tract_amx_int8)]
    if amx::has_amx_int8() {
        set = set.with(Isa::X86_64AmxInt8);
    }
    #[cfg(tract_amx_bf16)]
    if amx_bf16::has_amx_bf16() {
        set = set.with(Isa::X86_64AmxBf16);
    }
    set
}

routine!(x86_64; F32MapReduce, Softmax2, softmax::x86_64_fma_softmax2_f32_32n,
    isa(X86_64Avx2, X86_64Fma));
routine!(x86_64; F32MapReduce, Softmax2, softmax::x86_64_avx512_softmax2_f32_64n,
    isa(X86_64Avx512f));
routine!(x86_64; RmsNormF32, RmsNorm, "x86_64_avx512_rms_norm_f32", rms_norm::rms_norm_f32,
    isa(X86_64Avx512f));
