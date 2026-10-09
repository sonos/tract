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
#[cfg(all(any(target_os = "macos", target_os = "linux"), tract_sme))]
pub use sme::has_sme;
#[cfg(not(all(any(target_os = "macos", target_os = "linux"), tract_sme)))]
pub fn has_sme() -> bool {
    false
}
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

/// The per-chip Apple f32 cost model, the top rung: it refines the AMX heuristic and the
/// always-SME default wherever the shape is pinned. Which chip this is and what the chip can run
/// are separate questions — the model is fitted per microarchitecture, and whether its AMX or SME
/// cohort is runnable is the instruction set's business, so on a virtualised host the same model
/// still speaks for the NEON kernels it was fitted over.
///
/// The coefficients are single-thread. A wider pool does not get a flat divisor. Each candidate
/// is scored by [`crate::mmm::schedule_workers`]: streaming and per-tile time shrink with the
/// worker count, the fixed term does not, and leaving the inline path pays
/// [`crate::mmm::PARALLEL_ENTRY_SECONDS`] once. An AMX kernel that already fills its pipe takes
/// one worker per matrix pipe: the first thread fills the first pipe, the next thread the next
/// pipe, and a pool larger than the pipe count does not stack more workers onto a pipe that is
/// already full. A short-K AMX kernel, or one whose M or N does not cover a tile, does not fill
/// the pipe, so it may take one worker per performance core of that same kernel. SME and NEON
/// kernels scale across the performance cores, one worker per core. Threads past the
/// performance cluster are efficiency cores and are in neither roof. The roofs come from
/// sysctl, so an M1 and an M4, a Max, and an Ultra differ only in that count and in which
/// coefficient table `apple_chip` selects. x86 reads the same pipe type at launch: one worker
/// per physical core, and one TMUL per physical core when the kernel is Intel AMX. Those cost
/// models stay on the one-thread pick.
///
/// Every term it weighs is a shape term, so a dim the caller could not pin leaves it nothing to
/// say: it declines, and the tier below states what a wide AMX or SME tile is worth at an unknown
/// shape. `n == 1` declines too: the nr==1 rows in this table were not fitted as a GEMV model.
#[cfg(target_os = "macos")]
fn apple_chip_preferred(
    _isa: &IsaSet,
    dt: DatumType,
    query: &Query,
    suitable: &[Suitable],
) -> Option<&'static str> {
    let (Some(m), Some(k), Some(n)) = (query.m, query.k, query.n) else {
        return None;
    };
    if dt != DatumType::F32 || n == 1 {
        return None;
    }
    let model = apple_cost_model()?;
    let pool = crate::multithread::current_executor_threads().max(1);
    let cluster = crate::topology::perf_cluster();
    model.preferred_parallel(
        suitable,
        m,
        k,
        n,
        |mmm, stream_dominates| match cluster {
            Some(cluster) => kernel_roof(
                pool,
                cluster,
                crate::topology::uses_amx_cluster_pipe(mmm.isa()),
                stream_dominates,
                m,
                k,
                n,
                mmm.mr(),
                mmm.nr(),
            ),
            None => pool,
        },
        0,
        crate::mmm::PARALLEL_ENTRY_SECONDS,
    )
}

/// Fitted table for this chip. Built once: the launch path asks on every
/// non-vector product, and building it reads the brand string.
#[cfg(target_os = "macos")]
fn apple_cost_model() -> Option<&'static crate::mmm::LinearCostModel<'static>> {
    use std::sync::OnceLock;
    static MODEL: OnceLock<Option<crate::mmm::LinearCostModel<'static>>> = OnceLock::new();
    MODEL
        .get_or_init(|| match apple_chip() {
            Some("m1") => Some(apple_m1_linear::linear_model()),
            Some("m4") => Some(apple_m4_linear::linear_model()),
            _ => None,
        })
        .as_ref()
}

/// Whether one thread of this AMX kernel already fills its cluster's matrix
/// pipe.
///
/// A side shorter than the tile is one partial panel. The fit's streaming
/// coefficient was measured on full tiles, so it calls a long-K partial panel
/// stream-dominated and full; a second thread still reduces that time, so a
/// partial panel does not count as a full pipe.
///
/// On a full tile, `stream_dominates` is the fitted split, with a floor: a K
/// of at least an eighth of the tile area counts as full even when the fit
/// still calls the shape tile-dominated, so a long-K full tile keeps the
/// per-cluster roof rather than taking a worker per core.
#[cfg(target_os = "macos")]
fn matrix_pipe_is_full(
    stream_dominates: bool,
    m: usize,
    k: usize,
    n: usize,
    mr: usize,
    nr: usize,
) -> bool {
    if m < mr || n < nr {
        return false;
    }
    stream_dominates || k >= mr.saturating_mul(nr) / 8
}

/// Most workers this kernel may use on `cluster` for a pool of `pool`.
///
/// `matrix` marks a kernel on the per-cluster matrix unit, Apple AMX. When it
/// fills its pipe it gets one worker per pipe: the pool fills the next free
/// pipe and stops when every pipe has a worker. Every other kernel — NEON,
/// SME, or an AMX kernel that does not fill its pipe — gets one worker per
/// performance core. The short-K case, and a partial panel, are more workers of
/// the same kernel. It does not pack the matmul a second time for NEON, and it
/// does not run on efficiency cores.
#[cfg(target_os = "macos")]
fn worker_roof(
    pool: usize,
    cluster: crate::topology::PerfCluster,
    matrix: bool,
    fills_pipe: bool,
) -> usize {
    let budget = crate::topology::dispatch_threads(pool, cluster);
    if matrix && fills_pipe { budget.matrix } else { budget.neon }
}

/// [`worker_roof`] for a kernel whose shape terms are known; `matrix` is the
/// per-cluster (AMX) flag.
#[cfg(target_os = "macos")]
#[allow(clippy::too_many_arguments)]
fn kernel_roof(
    pool: usize,
    cluster: crate::topology::PerfCluster,
    matrix: bool,
    stream_dominates: bool,
    m: usize,
    k: usize,
    n: usize,
    mr: usize,
    nr: usize,
) -> usize {
    let fills = matrix && matrix_pipe_is_full(stream_dominates, m, k, n, mr, nr);
    worker_roof(pool, cluster, matrix, fills)
}

#[cfg(all(target_os = "macos", feature = "multithread-mm"))]
fn threading_panel_threshold() -> usize {
    crate::multithread::current_threading_panel_threshold()
}

/// Workers for a kernel the dispatch has already chosen.
///
/// Same roof and same entry cost as [`apple_chip_preferred`]: `matrix` marks an
/// AMX kernel, capped at one worker per cluster pipe, while SME and NEON take
/// one worker per performance core. A GEMV, a kernel the Apple table does not
/// name, or a chip without that table keeps that hardware roof, and the flat
/// panel gate inside `hardware` is the only economics those paths have — they
/// have no cost model to price the split. A GEMV has one column and cannot be
/// split on N, so it does not take the short-K core roof. No performance
/// cluster leaves the pool as it is.
///
/// A kernel the model can score takes the entry-cost comparison in
/// [`crate::mmm::schedule_workers`] instead of that gate: a panel count cannot
/// tell a 60-panel K=864 grid from a 60-panel K=4 one.
#[cfg(all(target_os = "macos", feature = "multithread-mm"))]
pub fn matmul_workers(
    name: &str,
    m: usize,
    k: usize,
    n: usize,
    mr: usize,
    nr: usize,
    matrix: bool,
) -> usize {
    let pool = crate::multithread::current_executor_threads().max(1);
    let Some(cluster) = crate::topology::perf_cluster() else {
        return pool;
    };
    let panels = m.div_ceil(mr) * n.div_ceil(nr.max(1));
    let threshold = threading_panel_threshold();
    let hardware = |matrix: bool| {
        // No shape term to say the pipe is idle, so an AMX kernel stays at
        // one worker per pipe.
        if panels < threshold { 1 } else { worker_roof(pool, cluster, matrix, true) }
    };
    // nr==1 rows were fit at n >= 2, so they are not a model of this path.
    // A gemv has one column and cannot be split on N.
    if nr == 1 || n == 1 {
        return hardware(matrix);
    }
    let Some(model) = apple_cost_model() else {
        return hardware(matrix);
    };
    let Some(ix) = model.kernels.iter().position(|kernel| *kernel == name) else {
        return hardware(matrix);
    };
    let (stream, tile, fixed) = model.terms(ix, m, k, n, mr, nr);
    let roof = kernel_roof(pool, cluster, matrix, stream > tile + fixed, m, k, n, mr, nr);
    crate::mmm::schedule_workers(
        stream,
        tile,
        fixed,
        roof,
        panels,
        0,
        crate::mmm::PARALLEL_ENTRY_SECONDS,
    )
    .0
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

#[cfg(all(test, target_os = "macos"))]
mod apple_thread_policy {
    use super::*;
    use crate::mmm::{PARALLEL_ENTRY_SECONDS, Query, schedule_workers};
    use crate::topology::{PerfCluster, uses_amx_cluster_pipe, uses_apple_matrix_pipe};

    fn suitable_f32(m: usize, k: usize, n: usize) -> Vec<Suitable> {
        let query = Query::plain(DatumType::F32, Some(m), Some(k), Some(n));
        crate::MmmDispatch::for_isa(crate::isa::native()).suitable(&query)
    }

    fn is_matrix(suitable: &[Suitable], name: &str) -> bool {
        suitable
            .iter()
            .find(|(mmm, _, _)| mmm.name() == name)
            .is_some_and(|(mmm, _, _)| uses_apple_matrix_pipe(mmm.isa()))
    }

    /// Kernel the parallel model picks, and how many workers that kernel gets,
    /// on an explicit cluster. Same fill test as [`super::kernel_roof`].
    fn scored(
        model: &crate::mmm::LinearCostModel<'_>,
        suitable: &[Suitable],
        m: usize,
        k: usize,
        n: usize,
        pool: usize,
        cluster: PerfCluster,
    ) -> (String, usize) {
        let name = model
            .preferred_parallel(
                suitable,
                m,
                k,
                n,
                |mmm, stream_dominates| {
                    kernel_roof(
                        pool,
                        cluster,
                        uses_amx_cluster_pipe(mmm.isa()),
                        stream_dominates,
                        m,
                        k,
                        n,
                        mmm.mr(),
                        mmm.nr(),
                    )
                },
                0,
                PARALLEL_ENTRY_SECONDS,
            )
            .unwrap()
            .to_string();
        let (mmm, _, _) = suitable.iter().find(|(mmm, _, _)| mmm.name() == name).unwrap();
        let ix = model.kernels.iter().position(|kernel| *kernel == name).unwrap();
        let (stream, tile, fixed) = model.terms(ix, m, k, n, mmm.mr(), mmm.nr());
        let roof = kernel_roof(
            pool,
            cluster,
            uses_amx_cluster_pipe(mmm.isa()),
            stream > tile + fixed,
            m,
            k,
            n,
            mmm.mr(),
            mmm.nr(),
        );
        let panels = m.div_ceil(mmm.mr()) * n.div_ceil(mmm.nr());
        let (workers, _) =
            schedule_workers(stream, tile, fixed, roof, panels, 0, PARALLEL_ENTRY_SECONDS);
        (name, workers)
    }

    #[test]
    fn one_thread_matches_the_unscaled_table() {
        let Some(chip) = apple_chip() else { return };
        let model = match chip {
            "m1" => apple_m1_linear::linear_model(),
            "m4" => apple_m4_linear::linear_model(),
            _ => return,
        };
        for (m, k, n) in [(32, 32, 32), (64, 8, 64), (128, 16, 128), (512, 512, 120), (7, 13, 9)] {
            let suitable = suitable_f32(m, k, n);
            let unscaled = model.preferred(&suitable, Some(m), Some(k), Some(n)).unwrap();
            let parallel = model
                .preferred_parallel(&suitable, m, k, n, |_, _| 1, 64, PARALLEL_ENTRY_SECONDS)
                .unwrap();
            assert_eq!(parallel, unscaled, "{chip} {m}x{k}x{n}");
        }
    }

    /// The M4 table, scored as a base M4 (one L2 cluster, four performance
    /// cores) and as a wider part with the same table (three clusters, twelve
    /// performance cores). SME scales across the performance cores: a fat GEMM
    /// takes one worker per core, past the cluster-pipe count that caps AMX.
    /// A short-K matmul keeps the same performance-core roof: two threads do
    /// not earn the entry cost, four do. A handful of panels stays at one
    /// worker and at the one-thread kernel.
    #[test]
    fn m4_sme_workers_scale_one_per_performance_core() {
        if apple_chip() != Some("m4") {
            return;
        }
        let model = apple_m4_linear::linear_model();
        let base = PerfCluster { physical_cpus: 4, cpus_per_l2: Some(4) };
        let wide = PerfCluster { physical_cpus: 12, cpus_per_l2: Some(4) };
        let big = suitable_f32(512, 512, 512);

        for pool in [4usize, 8] {
            let (name, workers) = scored(&model, &big, 512, 512, 512, pool, base);
            assert!(is_matrix(&big, &name), "512³ pool {pool} picked {name}");
            assert!(!name.contains("amx"), "picked {name}");
            assert_eq!(workers, 4, "512³ takes the four performance cores, pool {pool}");
        }
        let (name, workers) = scored(&model, &big, 512, 512, 512, 2, wide);
        assert!(is_matrix(&big, &name), "512³ at 2 threads picked {name}");
        assert!(!name.contains("amx"), "picked {name}");
        assert_eq!(workers, 2);
        for pool in [8usize, 12] {
            let (name, workers) = scored(&model, &big, 512, 512, 512, pool, wide);
            assert!(is_matrix(&big, &name), "512³ pool {pool} picked {name}");
            assert!(!name.contains("amx"), "picked {name}");
            assert_eq!(workers, pool, "512³ scales past the pipe count, pool {pool}");
        }

        // K = 48 does not fill the pipe. Two threads do not earn the entry cost.
        // Four do, and the roof is the performance-core count of this same SME
        // kernel: four on the base, and one per core on the wide part.
        let short = suitable_f32(960, 48, 96);
        let (name, workers) = scored(&model, &short, 960, 48, 96, 2, base);
        assert!(is_matrix(&short, &name), "short-K at 2 threads picked {name}");
        assert!(!name.contains("amx"), "picked {name}");
        assert_eq!(workers, 1, "two threads do not earn the entry");
        for pool in [4usize, 8] {
            let (name, workers) = scored(&model, &short, 960, 48, 96, pool, base);
            assert!(is_matrix(&short, &name), "short-K pool {pool} picked {name}");
            assert!(!name.contains("amx"), "picked {name}");
            assert_eq!(workers, 4, "short-K stacks on the idle pipe, pool {pool}");
        }
        let (name, workers) = scored(&model, &short, 960, 48, 96, 2, wide);
        assert!(is_matrix(&short, &name), "wide short-K at 2 threads picked {name}");
        assert_eq!(workers, 1, "two threads do not earn the entry on a wide part");
        for (pool, expect) in [(4usize, 4), (8, 8), (12, 12)] {
            let (name, workers) = scored(&model, &short, 960, 48, 96, pool, wide);
            assert!(is_matrix(&short, &name), "wide short-K pool {pool} picked {name}");
            assert!(!name.contains("amx"), "picked {name}");
            assert_eq!(workers, expect, "wide short-K pool {pool}");
        }

        // 16×576×25600 is a tiny-det convolution: K is long, the stream term
        // dominates, and the channel side is one partial 32-wide panel. That
        // does not fill the pipe. Two threads earn the split, four take the
        // performance cores, and the kernel stays SME. Swapping M and N is the
        // same roof. One thread stays on the unscaled pick.
        let partial = suitable_f32(16, 576, 25600);
        let unscaled = model.preferred(&partial, Some(16), Some(576), Some(25600)).unwrap();
        let (name, workers) = scored(&model, &partial, 16, 576, 25600, 1, base);
        assert_eq!(name, unscaled, "partial panel at one thread");
        assert_eq!(workers, 1);
        let (name, workers) = scored(&model, &partial, 16, 576, 25600, 2, base);
        assert!(is_matrix(&partial, &name), "partial panel at 2 threads picked {name}");
        assert!(!name.contains("amx"), "picked {name}");
        assert_eq!(workers, 2, "the second thread earns its entry");
        for pool in [4usize, 8] {
            let (name, workers) = scored(&model, &partial, 16, 576, 25600, pool, base);
            assert!(is_matrix(&partial, &name), "partial panel pool {pool} picked {name}");
            assert!(!name.contains("amx"), "picked {name}");
            assert_eq!(workers, 4, "partial panel pool {pool}");
        }
        let (name, workers) = scored(&model, &partial, 16, 576, 25600, 12, wide);
        assert!(is_matrix(&partial, &name), "wide partial panel picked {name}");
        assert!(!name.contains("amx"), "picked {name}");
        assert_eq!(workers, 12, "a partial panel takes the performance cores");
        let swapped = suitable_f32(25600, 576, 16);
        for (pool, expect) in [(2usize, 2), (4, 4), (8, 4)] {
            let (name, workers) = scored(&model, &swapped, 25600, 576, 16, pool, base);
            assert!(is_matrix(&swapped, &name), "swapped partial pool {pool} picked {name}");
            assert!(!name.contains("amx"), "picked {name}");
            assert_eq!(workers, expect, "swapped partial pool {pool}");
        }

        // 32x1568x400 is thirteen panels of a K-heavy conv. A panel count
        // alone calls that too small to split; the stream term says each of
        // the four performance cores earns its entry cost.
        let kheavy = suitable_f32(32, 1568, 400);
        let (name, workers) = scored(&model, &kheavy, 32, 1568, 400, 4, base);
        assert_eq!(workers, 4, "32x1568x400 earns four workers, picked {name}");

        let few = suitable_f32(64, 8, 64);
        let unscaled = model.preferred(&few, Some(64), Some(8), Some(64)).unwrap();
        for (pool, cluster) in [(8usize, base), (12, wide)] {
            let (name, workers) = scored(&model, &few, 64, 8, 64, pool, cluster);
            assert_eq!(workers, 1, "64x8x64 does not earn the entry cost");
            assert_eq!(name, unscaled);
        }
    }

    /// The M1 coefficient table on the same pipe roof. This host may be an M4:
    /// SME kernels are absent from the M1 table, so they are not candidates, and
    /// the M1 AMX and NEON kernels are ones this machine can run. A fat GEMM
    /// stays on AMX and takes one worker per pipe: one on a base M1, two on a
    /// Pro or Max, four on an Ultra, and no more when the pool is larger.
    /// 256³ is not asserted to stay there. Once the pool can pay, NEON on the
    /// performance cores beats AMX held to the pipe count.
    #[test]
    fn m1_workers_fill_each_pipe_and_then_stop() {
        let big = suitable_f32(512, 512, 512);
        if !big.iter().any(|(mmm, _, _)| mmm.name().contains("apple_amx")) {
            return;
        }
        let model = apple_m1_linear::linear_model();
        let base = PerfCluster { physical_cpus: 4, cpus_per_l2: Some(4) };
        let pro = PerfCluster { physical_cpus: 8, cpus_per_l2: Some(4) };
        let ultra = PerfCluster { physical_cpus: 16, cpus_per_l2: Some(4) };

        for (m, k, n) in [(512, 512, 512), (128, 32, 128), (64, 8, 64)] {
            let suitable = suitable_f32(m, k, n);
            let unscaled = model.preferred(&suitable, Some(m), Some(k), Some(n)).unwrap();
            let (name, workers) = scored(&model, &suitable, m, k, n, 1, base);
            assert_eq!(name, unscaled, "m1 one thread {m}x{k}x{n}");
            assert_eq!(workers, 1);
        }

        for pool in [1usize, 4, 8] {
            let (name, workers) = scored(&model, &big, 512, 512, 512, pool, base);
            assert_eq!(name, "apple_amx_mmm_f32_32x32", "512³ on one M1 pipe, pool {pool}");
            assert_eq!(workers, 1, "pool {pool}");
        }
        for pool in [2usize, 8, 16] {
            let (name, workers) = scored(&model, &big, 512, 512, 512, pool, pro);
            assert_eq!(name, "apple_amx_mmm_f32_32x32", "512³ on two M1 pipes, pool {pool}");
            assert_eq!(workers, 2, "pool {pool}");
        }
        for pool in [4usize, 8, 16] {
            let (name, workers) = scored(&model, &big, 512, 512, 512, pool, ultra);
            assert_eq!(name, "apple_amx_mmm_f32_32x32", "512³ on four M1 pipes, pool {pool}");
            assert_eq!(workers, 4, "pool {pool}");
        }

        // The split does not earn its entry cost, so the one-thread NEON kernel stays.
        let narrow = suitable_f32(128, 32, 128);
        let unscaled = model.preferred(&narrow, Some(128), Some(32), Some(128)).unwrap();
        let (name, workers) = scored(&model, &narrow, 128, 32, 128, 8, base);
        assert_eq!(name, unscaled);
        assert_eq!(name, "arm64simd_mmm_f32_12x8_gen");
        assert_eq!(workers, 1);

        // Honest leave: NEON at the performance-core roof beats AMX stuck on one pipe.
        let mid = suitable_f32(256, 256, 256);
        let (name, workers) = scored(&model, &mid, 256, 256, 256, 4, base);
        assert_eq!(name, "arm64simd_mmm_f32_8x8_gen");
        assert_eq!(workers, 4);
        let (name, workers) = scored(&model, &mid, 256, 256, 256, 8, pro);
        assert_eq!(name, "arm64simd_mmm_f32_8x8_gen");
        assert_eq!(workers, 8);
    }

    /// `apple_chip_preferred` and [`super::matmul_workers`] have to see the
    /// executor installed for the run. The TLS scope is what the CLI's
    /// `--threads` installs before optimisation.
    #[cfg(feature = "multithread-mm")]
    #[test]
    fn the_pick_reads_the_executor_pool() {
        use crate::multithread::{Executor, multithread_tract_scope};
        use crate::topology::perf_cluster;

        let Some(chip) = apple_chip() else { return };
        if chip != "m4" {
            return;
        }
        let Some(cluster) = perf_cluster() else { return };
        let query = Query::plain(DatumType::F32, Some(960), Some(48), Some(96));
        let dispatch = crate::MmmDispatch::for_isa(crate::isa::native());
        let model = apple_m4_linear::linear_model();
        let suitable = dispatch.suitable(&query);

        let expect = |pool: usize| {
            let (name, workers) = scored(&model, &suitable, 960, 48, 96, pool, cluster);
            (name, workers)
        };
        let check = |pool: Executor, nth: usize| {
            multithread_tract_scope(pool, || {
                let from_tier =
                    apple_chip_preferred(&crate::isa::native(), DatumType::F32, &query, &suitable)
                        .unwrap()
                        .to_string();
                let (name, workers) = expect(nth);
                assert_eq!(from_tier, name);
                assert_eq!(
                    matmul_workers("sme_mmm_f32_32x32", 960, 48, 96, 32, 32, false),
                    workers
                );
            })
        };
        check(Executor::SingleThread, 1);
        check(Executor::multithread(2), 2);
        check(Executor::multithread(4), 4);
        check(Executor::multithread(8), 8);
        if cluster.matrix_units() == 1 {
            assert_eq!(expect(2).1, 1);
            assert_eq!(expect(4).1, 4.min(cluster.physical_cpus));
            assert_eq!(expect(8).1, 8.min(cluster.physical_cpus));
        }
    }
}
