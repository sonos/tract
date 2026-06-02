// Build-time capability probe for the assembler, used by build.rs
// (assembler_supports_avxvnni). Checks that the assembler accepts the
// `{vex}` prefix on VPDPBUSD, which forces the AVX-VNNI (VEX-encoded)
// form instead of the AVX-512-VNNI (EVEX-encoded) form gas defaults to.
// Requires binutils >= 2.36 (which added `{vex}`/`{evex}` prefixes for
// explicit encoding selection). When the probe fails the AVX-VNNI kernel
// is skipped and dispatch falls back to AVX2 emulation on AVX-VNNI-only
// hardware (Clearwater Forest / Sierra Forest / Alder Lake E-cores).
// Not linked into anything.
.intel_syntax noprefix
.text
.globl tract_avxvnni_probe
tract_avxvnni_probe:
    // AVX-VNNI: u8 x s8 -> i32 dot4 (VEX-encoded)
    {vex} vpdpbusd ymm0, ymm1, ymm2
    ret
