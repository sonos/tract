#![allow(clippy::excessive_precision)]
#[cfg(any(target_os = "macos", all(target_os = "ios", feature = "apple-amx-ios")))]
mod apple_amx;
#[cfg(target_os = "macos")]
mod apple_m1_linear;
#[cfg(target_os = "macos")]
mod apple_m4_linear;
mod arm64simd;
mod cortex_a53_linear;
mod cortex_a53_mmv_linear;
mod cortex_a55_linear;
mod cortex_a55_mmv_linear;
// `tract_sme` is set by build.rs only when the assembler can assemble SME
// (gates out e.g. the old Debian stretch aarch64 toolchain).
#[cfg(all(any(target_os = "macos", target_os = "linux"), tract_sme))]
mod sme;
mod sve;
pub use arm64simd::*;

#[cfg(not(feature = "no_fp16"))]
pub mod arm64fp16;
#[cfg(not(feature = "no_fp16"))]
pub use arm64fp16::*;

use crate::DatumType;
#[cfg(target_arch = "aarch64")]
use crate::f16;

use crate::isa::Isa;
use crate::isa::IsaSet;
use crate::mmm::{Query, Suitable};

// https://en.wikipedia.org/wiki/Comparison_of_ARMv8-A_cores
const PART_A53: &str = "0xd03";
const PART_A55: &str = "0xd05";
const PART_A72: &str = "0xd08";
const PART_A73: &str = "0xd09";
const PART_A75: &str = "0xd0a";
const PART_NEOVERSE_N1: &str = "0xd0c";
const PART_NEOVERSE_N2: &str = "0xd49";
const PART_NEOVERSE_N3: &str = "0xd8e";
const PART_NEOVERSE_V1: &str = "0xd40";
const PART_NEOVERSE_V2: &str = "0xd4f";
const PART_NEOVERSE_V3: &str = "0xd83";

fn max_cpuid() -> std::io::Result<String> {
    let cpu_info = std::fs::read_to_string("/proc/cpuinfo")?;
    let max = cpu_info
        .lines()
        .filter(|line| line.starts_with("CPU part"))
        .map(|line| line.split_whitespace().last().unwrap_or(""))
        .max();
    Ok(max.unwrap_or("").to_string())
}

lazy_static::lazy_static! {
    static ref KIND: Kind = Kind::choose();

    static ref CPU_FEATURES: Vec<String> = {
        #[cfg(test)] crate::setup_test_logger();
        let Ok(cpu_info) = std::fs::read_to_string("/proc/cpuinfo") else {
            log::warn!("Could not read /proc/cpuinfo. CPU Features detection may be impaired.");
            return vec!();
        };
        if let Some(line) = cpu_info
            .lines()
                .find(|line| line.starts_with("Features")) {
                    line.split_once(':').unwrap().1.split_whitespace().map(|s| s.to_string()).collect()
                } else {
                    log::warn!("Could not find \"Features  :\" lines in /proc/cpuinfo. CPU Features detection may be impaired.");
                    vec!()
        }
    };

    static ref HAS_FP16: bool = {
        CPU_FEATURES.iter().any(|s| &**s == "asimdhp")
    };
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn apple_string_from_c_bytes(buf: &[u8]) -> String {
    use std::ffi::CStr;

    CStr::from_bytes_until_nul(buf)
        .ok()
        .map(|s| s.to_string_lossy().into_owned())
        .unwrap_or_default()
}

#[cfg(any(target_os = "macos", target_os = "ios"))]
fn apple_get_syscall(key: &str) -> String {
    use std::ffi::{CString, c_char, c_int, c_void};
    use std::ptr::null_mut;

    unsafe extern "C" {
        fn sysctlbyname(
            name: *const c_char,
            oldp: *mut c_void,
            oldlenp: *mut usize,
            newp: *mut c_void,
            newlen: usize,
        ) -> c_int;
    }

    let Ok(name) = CString::new(key) else {
        return String::new();
    };

    unsafe {
        let mut len_needed: usize = 0;
        if sysctlbyname(name.as_ptr(), null_mut(), &mut len_needed, null_mut(), 0) != 0 {
            return String::new();
        }

        let mut buf = vec![0u8; len_needed.saturating_add(1)];
        let mut len: usize = buf.len();
        if sysctlbyname(name.as_ptr(), buf.as_mut_ptr() as _, &mut len, null_mut(), 0) != 0 {
            return String::new();
        }

        buf.truncate(len.min(buf.len()));
        if buf.last().copied() != Some(0) {
            buf.push(0);
        }

        apple_string_from_c_bytes(&buf)
    }
}

/// The Apple silicon generation, from the CPU brand string, for per-chip cost-model
/// selection. Returns `None` for chips without a fitted model (they keep the default
/// dispatch). Distinct chips need distinct models: e.g. M1 has AMX, M4 has SME.
#[cfg(target_os = "macos")]
fn apple_chip() -> Option<&'static str> {
    let brand = apple_get_syscall("machdep.cpu.brand_string");
    [("M1", "m1"), ("M2", "m2"), ("M3", "m3"), ("M4", "m4")]
        .into_iter()
        .find_map(|(needle, id)| brand.contains(needle).then_some(id))
}

#[cfg(all(test, any(target_os = "macos", target_os = "ios")))]
mod tests {
    use super::*;

    #[test]
    fn apple_string_from_c_bytes_returns_empty_without_nul() {
        assert_eq!(apple_string_from_c_bytes(b"hello"), "");
    }

    #[test]
    fn apple_string_from_c_bytes_stops_at_first_nul() {
        assert_eq!(apple_string_from_c_bytes(b"hello\0world\0"), "hello");
    }

    #[test]
    fn apple_get_syscall_does_not_panic() {
        let _ = apple_get_syscall("machdep.cpu.brand_string");
    }
}

#[cfg(target_os = "macos")]
pub fn has_amx() -> bool {
    !apple_get_syscall("machdep.cpu.brand_string").contains("(Virtual)")
}

#[cfg(target_os = "ios")]
lazy_static::lazy_static! {
    static ref IPHONE_MODEL_MAJOR:Option<usize> = {
        let version = apple_get_syscall("hw.machine");
        let Some((major, _)) = version.trim_start_matches("iPhone").split_once(",") else { return None };
        major.parse::<usize>().ok()
    };
}

#[cfg(all(target_os = "ios", feature = "apple-amx-ios"))]
fn has_amx() -> bool {
    // iPhone12,1 is the one branded "iPhone 11", with Apple A13 bionic, first CPU featuring amx
    IPHONE_MODEL_MAJOR.map(|it| it >= 12).unwrap_or(false)
}

#[inline]
#[cfg(target_os = "ios")]
pub fn has_fp16() -> bool {
    // iPhone10,1 is the one branded "iPhone 8", with Apple A11 bionic, first CPU featuring fp16
    IPHONE_MODEL_MAJOR.map(|it| it >= 10).unwrap_or(false)
}

/// True when the running CPU implements FEAT_FP16, hence when the native f16 kernels are
/// legal. Always false in a build that does not target aarch64: the module compiles
/// everywhere, but its kernels are only assembled natively.
#[inline]
#[cfg(not(target_os = "ios"))]
pub fn has_fp16() -> bool {
    cfg!(target_arch = "aarch64")
        && (cfg!(target_os = "macos")
            || cfg!(feature_cpu = "fp16")
            || *KIND == Kind::CortexA55
            || *KIND == Kind::CortexA75
            || *HAS_FP16)
}

// FEAT_DotProd (SDOT/UDOT), ARMv8.2. TRACT_DOTPROD_DISABLE=1 forces it off so
// callers can A/B the SDOT kernel against the SMLAL 8x8 fallback on one binary.
#[cfg(all(target_os = "macos", target_arch = "aarch64"))]
pub fn has_dotprod() -> bool {
    // Every Apple arm64 CPU (M1+/A11+) implements FEAT_DotProd.
    !crate::knobs::TRACT_DOTPROD_DISABLE.get()
}

#[cfg(all(target_os = "linux", target_arch = "aarch64"))]
pub fn has_dotprod() -> bool {
    if crate::knobs::TRACT_DOTPROD_DISABLE.get() {
        return false;
    }
    // HWCAP_ASIMDDP = 1 << 20 on aarch64.
    const HWCAP_ASIMDDP: u64 = 1 << 20;
    const AT_HWCAP: u64 = 16;
    unsafe extern "C" {
        fn getauxval(t: u64) -> u64;
    }
    unsafe { (getauxval(AT_HWCAP) & HWCAP_ASIMDDP) != 0 }
}

#[cfg(not(all(
    any(target_os = "macos", target_os = "linux", target_os = "ios"),
    target_arch = "aarch64"
)))]
pub fn has_dotprod() -> bool {
    false
}

#[cfg(all(target_os = "ios", target_arch = "aarch64"))]
pub fn has_dotprod() -> bool {
    // A11+ (iPhone10,1+) implement FEAT_DotProd.
    !crate::knobs::TRACT_DOTPROD_DISABLE.get()
        && IPHONE_MODEL_MAJOR.map(|it| it >= 10).unwrap_or(false)
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "fp16")]
#[inline]
pub unsafe fn add_f16(a: f16, b: f16) -> f16 {
    unsafe {
        let result: u16;
        std::arch::asm!(
        "fadd {0:h}, {1:h}, {2:h}",
        lateout(vreg) result,
        in(vreg) a.to_bits(),
        in(vreg) b.to_bits(),
        options(pure, nomem, nostack, preserves_flags));
        f16::from_bits(result)
    }
}

#[cfg(target_arch = "aarch64")]
#[target_feature(enable = "fp16")]
#[inline]
pub unsafe fn mul_f16(a: f16, b: f16) -> f16 {
    unsafe {
        let result: u16;
        std::arch::asm!(
        "fmul {0:h}, {1:h}, {2:h}",
        lateout(vreg) result,
        in(vreg) a.to_bits(),
        in(vreg) b.to_bits(),
        options(pure, nomem, nostack, preserves_flags));
        f16::from_bits(result)
    }
}

#[derive(Debug, PartialEq, Eq, Copy, Clone)]
pub enum Kind {
    Generic,
    AppleM,
    Neoverse,
    CortexA53,
    CortexA55,
    CortexA72,
    CortexA73,
    CortexA75,
}

impl Kind {
    pub fn choose() -> Kind {
        #[cfg(test)]
        crate::setup_test_logger();
        let kind = if let Some(kind) = crate::knobs::TRACT_CPU_AARCH64_KIND.get() {
            log::info!("CPU kind forced with TRACT_CPU_AARCH64_KIND: {}", kind);
            let kind = kind.to_lowercase();
            if kind.contains("a53") {
                Kind::CortexA53
            } else if kind.contains("a55") {
                Kind::CortexA55
            } else if kind.contains("a72") {
                Kind::CortexA72
            } else if kind.contains("a73") {
                Kind::CortexA73
            } else if kind.contains("a75") {
                Kind::CortexA75
            } else if kind.contains("neoverse") {
                Kind::Neoverse
            } else if kind.contains("applem") {
                Kind::AppleM
            } else {
                Kind::Generic
            }
        } else if cfg!(target_os = "macos") {
            Kind::AppleM
        } else {
            let part = if let Some(part) = crate::knobs::TRACT_CPU_AARCH64_OVERRIDE_CPU_PART.get() {
                log::info!("CPU part forced with TRACT_CPU_AARCH64_OVERRIDE_CPU_PART: {}", part);
                part
            } else if cfg!(target_os = "linux") {
                let part = max_cpuid().unwrap_or_else(|_| "0x00".to_string());
                log::info!("CPU part auto detected: {}", part);
                part
            } else {
                log::info!("Unknown CPU part");
                "0x00".to_string()
            };
            match &*part {
                PART_A53 => Kind::CortexA53,
                PART_A55 => Kind::CortexA55,
                PART_A72 => Kind::CortexA72,
                PART_A73 => Kind::CortexA73,
                PART_A75 => Kind::CortexA75,
                PART_NEOVERSE_N1 | PART_NEOVERSE_N2 | PART_NEOVERSE_N3 | PART_NEOVERSE_V1
                | PART_NEOVERSE_V2 | PART_NEOVERSE_V3 => Kind::Neoverse,
                _ => Kind::Generic,
            }
        };
        log::info!("CPU optimisation: {:?}", kind);
        kind
    }
}

/// SDOT (~4x the SMLAL 8x8) when FEAT_DotProd is present, else the SMLAL 8x8 fallback. The SDOT
/// kernel only exists when the assembler could encode `sdot` (`tract_arm64_dotprod`, set by
/// build.rs); otherwise always use the SMLAL 8x8.
fn neon_qmmm_i32(isa: &IsaSet, _suitable: &[Suitable]) -> Option<&'static str> {
    let _ = isa;
    #[cfg(tract_arm64_dotprod)]
    if isa.has(Isa::Aarch64DotProd) {
        return Some(arm64simd_mmm_i32_8x8_dot.name.as_str());
    }
    Some(arm64simd_mmm_i32_8x8.name.as_str())
}

/// n==1: below the fixed kernel's mr, a narrower/better-fitting kernel wins (the 64x1 pays full
/// mr-padding), so consult the cost model; at or above mr the fixed 64x1 is already optimal and
/// the model only second-guesses it into knife-edge mispicks, so keep it.
fn neon_mmv_f32(suitable: &[Suitable], query: &Query) -> Option<&'static str> {
    match *KIND {
        Kind::CortexA53 => match query.m {
            Some(m) if m < 64 => {
                cortex_a53_mmv_linear::linear_model().preferred(suitable, Some(m), query.k, Some(1))
            }
            _ => Some(arm64simd_mmm_f32_64x1_a53.name.as_str()),
        },
        Kind::CortexA55 => match query.m {
            Some(m) if m < 64 => {
                cortex_a55_mmv_linear::linear_model().preferred(suitable, Some(m), query.k, Some(1))
            }
            _ => Some(arm64simd_mmm_f32_64x1_a55.name.as_str()),
        },
        _ => Some(arm64simd_mmm_f32_64x1_gen.name.as_str()),
    }
}

fn neon_mmm_f32(suitable: &[Suitable], query: &Query) -> Option<&'static str> {
    match *KIND {
        Kind::CortexA53 => {
            cortex_a53_linear::linear_model().preferred(suitable, query.m, query.k, query.n)
        }
        Kind::CortexA55 => {
            cortex_a55_linear::linear_model().preferred(suitable, query.m, query.k, query.n)
        }
        _ => Some(if query.n.unwrap_or(8) < 8 {
            arm64simd_mmm_f32_16x4_gen.name.as_str()
        } else {
            arm64simd_mmm_f32_8x8_gen.name.as_str()
        }),
    }
}

/// Baseline aarch64: NEON is always there, so this tier always applies and every extension above
/// it answers first.
fn neon_preferred(
    isa: &IsaSet,
    dt: DatumType,
    query: &Query,
    suitable: &[Suitable],
) -> Option<&'static str> {
    match (dt, query.n) {
        (DatumType::F32, Some(1)) => neon_mmv_f32(suitable, query),
        (DatumType::F32, _) => neon_mmm_f32(suitable, query),
        (DatumType::I32, Some(1)) => Some(arm64simd_mmm_i32_64x1.name.as_str()),
        (DatumType::I32, _) => neon_qmmm_i32(isa, suitable),
        _ => None,
    }
}

inventory::submit! {
    crate::mmm_tiers::MmmTier {
        arch: Some(crate::isa::Arch::Aarch64),
        precedence: 1,
        name: "arm64simd",
        applies: |_| true,
        preferred: neon_preferred,
    }
}

#[cfg(not(feature = "no_fp16"))]
fn fp16_preferred(
    _isa: &IsaSet,
    dt: DatumType,
    query: &Query,
    _suitable: &[Suitable],
) -> Option<&'static str> {
    let a55 = *KIND == Kind::CortexA55;
    match (dt, query.n) {
        (DatumType::F16, Some(1)) if a55 => Some(arm64fp16_mmm_f16_128x1_a55.name.as_str()),
        (DatumType::F16, Some(1)) => Some(arm64fp16_mmm_f16_128x1_gen.name.as_str()),
        (DatumType::F16, _) => {
            use tract_data::internal::DimLike;
            let n = query.n.unwrap_or(1024);
            let narrow = n.divceil(4) * 4 < n.divceil(8) * 8;
            Some(match (a55, narrow) {
                (true, true) => &arm64fp16_mmm_f16_32x4_a55.name,
                (true, false) => &arm64fp16_mmm_f16_16x8_a55.name,
                (false, true) => &arm64fp16_mmm_f16_32x4_gen.name,
                (false, false) => &arm64fp16_mmm_f16_16x8_gen.name,
            })
        }
        _ => None,
    }
}

#[cfg(not(feature = "no_fp16"))]
inventory::submit! {
    crate::mmm_tiers::MmmTier {
        arch: Some(crate::isa::Arch::Aarch64),
        precedence: 2,
        name: "arm64fp16",
        applies: |isa| isa.has(Isa::Aarch64Fp16),
        preferred: fp16_preferred,
    }
}

routine!(aarch64; F32, Silu, arm64simd_silu_f32_4n_fused);
routine!(aarch64; F32, Gelu, arm64simd_gelu_f32_4n_fused);
routine!(aarch64; F32, Hardswish, arm64simd_hardswish_f32_8n);

routine!(aarch64; F16, Sigmoid, arm64simd_sigmoid_f16_4n);
routine!(aarch64; F16, Tanh, arm64simd_tanh_f16_4n);
routine!(aarch64; F16, Silu, arm64simd_silu_f16_lut_8n);

routine!(aarch64; F32Param, LeakyRelu, arm64simd_leaky_relu_f32_8n);
#[cfg(not(feature = "no_fp16"))]
routine!(aarch64; F16Param, LeakyRelu, arm64fp16_leaky_relu_f16_16n, isa(Aarch64Fp16));

routine!(aarch64; F32Reduce, ReduceMax, arm64simd_max_f32_16n);
routine!(aarch64; F32Reduce, ReduceMin, arm64simd_min_f32_16n);
routine!(aarch64; F32Reduce, ReduceSum, arm64simd_sum_f32_16n);
routine!(aarch64; RmsNormF32, RmsNorm, "arm64simd_rms_norm_f32", arm64simd_rms_norm_f32);
#[cfg(not(feature = "no_fp16"))]
routine!(aarch64; F16Reduce, ReduceMax, arm64fp16_max_f16_32n, isa(Aarch64Fp16));
#[cfg(not(feature = "no_fp16"))]
routine!(aarch64; F16Reduce, ReduceSum, arm64fp16_sum_f16_32n, isa(Aarch64Fp16));

routine!(aarch64; BinF32, BinUnicast(Mul), arm64simd_unicast_mul_f32_16n);
#[cfg(not(feature = "no_fp16"))]
routine!(aarch64; BinF16, BinUnicast(Mul), arm64fp16_unicast_mul_f16_32n, isa(Aarch64Fp16));
routine!(aarch64; BinF32, BinUnicast(Add), arm64simd_unicast_add_f32_16n);
#[cfg(not(feature = "no_fp16"))]
routine!(aarch64; BinF16, BinUnicast(Add), arm64fp16_unicast_add_f16_32n, isa(Aarch64Fp16));
routine!(aarch64; BinF32, BinUnicast(Sub), arm64simd_unicast_sub_f32_16n);
#[cfg(not(feature = "no_fp16"))]
routine!(aarch64; BinF16, BinUnicast(Sub), arm64fp16_unicast_sub_f16_32n, isa(Aarch64Fp16));
routine!(aarch64; BinF32, BinUnicast(SubF), arm64simd_unicast_subf_f32_16n);
#[cfg(not(feature = "no_fp16"))]
routine!(aarch64; BinF16, BinUnicast(SubF), arm64fp16_unicast_subf_f16_32n, isa(Aarch64Fp16));
routine!(aarch64; BinF32, BinUnicast(Min), arm64simd_unicast_min_f32_16n);
#[cfg(not(feature = "no_fp16"))]
routine!(aarch64; BinF16, BinUnicast(Min), arm64fp16_unicast_min_f16_32n, isa(Aarch64Fp16));
routine!(aarch64; BinF32, BinUnicast(Max), arm64simd_unicast_max_f32_16n);
#[cfg(not(feature = "no_fp16"))]
routine!(aarch64; BinF16, BinUnicast(Max), arm64fp16_unicast_max_f16_32n, isa(Aarch64Fp16));

/// The per-chip Apple f32 cost model, the top rung: it refines the AMX heuristic and the
/// always-SME default wherever the shape is pinned. Which chip this is and what the chip can run
/// are separate questions — the model is fitted per microarchitecture, and whether its AMX or SME
/// cohort is runnable is the instruction set's business, so on a virtualised host the same model
/// still speaks for the NEON kernels it was fitted over.
///
/// Every term it weighs is a shape term, so a dim the caller could not pin leaves it nothing to
/// say: it declines, and the tier below states what a wide AMX or SME tile is worth at an unknown
/// shape.
#[cfg(target_os = "macos")]
fn apple_chip_preferred(
    _isa: &IsaSet,
    dt: DatumType,
    query: &Query,
    suitable: &[Suitable],
) -> Option<&'static str> {
    let pinned = query.m.is_some() && query.k.is_some() && query.n.is_some();
    if dt != DatumType::F32 || query.n == Some(1) || !pinned {
        return None;
    }
    let model = match apple_chip()? {
        "m1" => apple_m1_linear::linear_model(),
        "m4" => apple_m4_linear::linear_model(),
        _ => return None,
    };
    model.preferred(suitable, query.m, query.k, query.n)
}

#[cfg(target_os = "macos")]
inventory::submit! {
    crate::mmm_tiers::MmmTier {
        arch: Some(crate::isa::Arch::Aarch64),
        precedence: 8,
        name: "apple-chip-model",
        applies: |_| matches!(apple_chip(), Some("m1") | Some("m4")),
        preferred: apple_chip_preferred,
    }
}

/// What this core has, in the shared vocabulary.
pub fn isa_set() -> crate::isa::IsaSet {
    use crate::isa::IsaSet;
    let mut set = IsaSet::of_arch(crate::isa::Arch::Aarch64);
    if has_fp16() {
        set = set.with(Isa::Aarch64Fp16);
        #[cfg(feature = "no_fp16")]
        log::warn!(
            "This is a build with fp16 disabled, while your platform CPU seems to support it."
        );
    }
    if has_dotprod() {
        set = set.with(Isa::Aarch64DotProd);
    }
    if sve::has_sve2() {
        set = set.with(Isa::Aarch64Sve2);
        #[cfg(all(target_os = "linux", target_arch = "aarch64"))]
        log::info!("SVE2 available, VL = {} bytes", sve::rdvl_bytes());
    } else if sve::has_sve() {
        log::info!("SVE (v1) present; SVE2 kernels not enabled");
    }
    #[cfg(all(any(target_os = "macos", target_os = "linux"), tract_sme))]
    {
        if sme::has_sme() {
            set = set.with(Isa::Aarch64Sme);
        }
        if sme::has_sme2() {
            set = set.with(Isa::Aarch64Sme2);
        }
    }
    #[cfg(any(target_os = "macos", all(target_os = "ios", feature = "apple-amx-ios")))]
    if has_amx() {
        set = set.with(Isa::Aarch64AppleAmx);
    }
    set
}
