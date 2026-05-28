// Build-time capability probe for the assembler, used by build.rs
// (assembler_supports_avx512vnni). Older binutils — notably the Debian stretch
// x86_64 toolchain in CI — predate AVX-512 VNNI (added in binutils ~2.30) and
// cannot assemble `vpdpbusd ymm` even when targeting a VNNI-capable CPU. If
// this file fails to assemble, build.rs skips the VNNI kernel and the
// `tract_avx512vnni` cfg, and the runtime falls back to the AVX2 i32 path.
// Not linked into anything.
.intel_syntax noprefix
.text
.globl tract_avx512vnni_probe
tract_avx512vnni_probe:
    vpdpbusd ymm0, ymm1, ymm2
    ret
