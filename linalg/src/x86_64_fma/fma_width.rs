// How many 512-bit FMA / VNNI issue ports does the running core have?
//
// The zmm-wide `avx512vnni_mmm_i32_16x16` kernel is only worth selecting over
// the ymm-wide `avx512vnni_mmm_i32_8x8` on cores that can retire a 512-bit
// VPDPBUSD as fast as a 256-bit one -- i.e. cores with *two* 512-bit FMA units.
// That is a server-class property:
//
//   * 2x 512-bit FMA: Skylake-SP/X, Cascade Lake, Cooper Lake, Ice Lake-SP/-X,
//     Sapphire Rapids, Emerald Rapids, Granite Rapids (Xeon Gold/Platinum/W).
//   * 1x 512-bit FMA: every AVX-512 *client* core -- Cannon Lake, Ice Lake-U/Y,
//     Tiger Lake, Rocket Lake -- where the port-0 + port-5 256-bit FMAs fuse
//     into one 512-bit unit. AMD Zen4/Zen5 also behave as 1 (their 512-bit ops
//     are double-pumped over the 256-bit datapath).
//
// Why it matters: on a 1-FMA core, one 512-bit VPDPBUSD/cycle delivers the same
// MAC/s as two 256-bit VPDPBUSD/cycle. So the 16x16 tile gives *zero* throughput
// upside there while adding A-packing, the 16-column +128 bias correction and a
// bigger epilogue -- a net regression on real matmuls (measured on an
// i9-11900KB: int8 LLM/TDNN prefill -4..-11%). On a 2-FMA core the 16x16 issues
// two 512-bit VPDPBUSD/cycle = ~2x the 8x8's MAC/s, which is the win the kernel
// exists for. tract therefore only enables the 16x16 VNNI kernel when this
// returns >= 2 (see `plug_avx512vnni`).
//
// Detection is a deterministic CPUID DisplayFamily/DisplayModel allowlist -- the
// same model-table approach oneDNN/BLIS use for SKU-specific dispatch -- chosen
// over a runtime microbenchmark on purpose: it is noise-free (important on the
// shared/virtualised hosts where a timing probe is unreliable) and auditable.
// It defaults to the *safe* 1-FMA answer for any model not on the allowlist
// (all client parts, AMD, and unknown future silicon stay on the
// never-regressing 8x8 kernel).
//
// `TRACT_AVX512_FMA_UNITS` (`1` or `2`) overrides the result -- use `2` to opt a
// dual-FMA SKU we have not catalogued yet into the wide kernel, or `1` to force
// 8x8 on a low-end (Bronze/Silver) server SKU that ships only one FMA unit
// (those share a DisplayModel with their dual-FMA Gold/Platinum siblings, so
// CPUID alone cannot tell them apart).

use std::sync::OnceLock;

/// Intel DisplayFamily-6 models known to ship two 512-bit FMA units.
const DUAL_FMA_INTEL_MODELS: &[u32] = &[
    0x55, // Skylake-SP/X, Cascade Lake-SP/X, Cooper Lake
    0x6A, // Ice Lake-SP
    0x6C, // Ice Lake-DE / -X
    0x8F, // Sapphire Rapids
    0xCF, // Emerald Rapids
    0xAD, // Granite Rapids
    0xAE, // Granite Rapids-D
];

fn detect_fma_units() -> u8 {
    if let Ok(v) = std::env::var("TRACT_AVX512_FMA_UNITS") {
        match v.trim() {
            "1" => return 1,
            "2" => return 2,
            other => log::warn!(
                "TRACT_AVX512_FMA_UNITS={other:?} not understood (expected 1 or 2); using CPUID"
            ),
        }
    }
    #[allow(unused_unsafe)]
    let eax = unsafe { std::arch::x86_64::__cpuid(1) }.eax;
    let base_family = (eax >> 8) & 0xF;
    let base_model = (eax >> 4) & 0xF;
    let ext_model = (eax >> 16) & 0xF;
    // Intel CPUID.1: DisplayModel folds in the extended-model nibble for
    // families 6 and 15 (Intel SDM Vol 2, "INPUT EAX = 1"). AMD reports
    // base_family == 0xF for Zen, so it never matches the family-6 allowlist
    // and correctly falls through to the 1-FMA default.
    let display_model = if base_family == 0x6 || base_family == 0xF {
        (ext_model << 4) | base_model
    } else {
        base_model
    };
    if base_family == 0x6 && DUAL_FMA_INTEL_MODELS.contains(&display_model) { 2 } else { 1 }
}

/// Number of 512-bit FMA/VNNI issue ports on this core (1 or 2). Memoised.
pub fn avx512_fma_units() -> u8 {
    static UNITS: OnceLock<u8> = OnceLock::new();
    *UNITS.get_or_init(detect_fma_units)
}

/// True iff the core has two 512-bit FMA ports, i.e. the zmm 16x16 VNNI kernel
/// can out-run the ymm 8x8 kernel. See module docs for the full rationale.
pub fn has_dual_avx512_fma() -> bool {
    avx512_fma_units() >= 2
}
