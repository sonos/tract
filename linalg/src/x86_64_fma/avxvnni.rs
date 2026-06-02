// AVX-VNNI int8 GEMM runtime gate.
//
// AVX-VNNI (CPUID leaf 7 sub-leaf 1 EAX bit 4) is the VEX-encoded sibling of
// AVX-512-VNNI's VPDPBUSD: same i32 += u8 * s8 dot4 semantics, but addressable
// from VEX (= AVX2-class) decoders. It exists primarily for Atom-class
// server / E-core SKUs that have AVX2 + AVX-VNNI but no AVX-512:
//
//   * Alder Lake / Raptor Lake / Meteor Lake E-cores (Gracemont, Crestmont)
//   * Sierra Forest (Sierra Glen)
//   * Clearwater Forest (Darkmont)
//
// On a CPU with AVX-512-VNNI (Cascade Lake, Ice Lake, Sapphire Rapids+), this
// detector still returns true if CPUID leaf 7.1 EAX.4 is set -- some big-core
// SKUs report AVX-VNNI alongside AVX-512-VNNI -- but the dispatch in mmm.rs
// prefers the EVEX-encoded avx512vnni kernel in that case (same throughput,
// 32 zmm registers available for unrolling). The AVX-VNNI kernel is only
// selected when AVX-512-VNNI is absent.

use std::sync::OnceLock;

/// CPUID leaf 7 sub-leaf 1, EAX bit 4 = AVX-VNNI (Intel SDM Vol 2 Table 1-7).
/// Sub-leaf 1 is only valid when CPUID.7.0.EAX (the max sub-leaf field) >= 1;
/// older CPUs return zeroed structures. We check the max-sub-leaf first to
/// avoid a misleading bit on pre-AVX-VNNI silicon.
fn cpu_has_avxvnni() -> bool {
    if !std::is_x86_feature_detected!("avx2") {
        return false;
    }
    let max_sub = std::arch::x86_64::__cpuid_count(7, 0).eax;
    if max_sub < 1 {
        return false;
    }
    let r = std::arch::x86_64::__cpuid_count(7, 1);
    (r.eax & (1 << 4)) != 0
}

/// Returns true iff CPUID reports AVX-VNNI on this CPU. Memoised; no OS
/// permission gate is required (unlike AMX, AVX-VNNI uses no extended state).
pub fn has_avxvnni() -> bool {
    static GATE: OnceLock<bool> = OnceLock::new();
    *GATE.get_or_init(cpu_has_avxvnni)
}
