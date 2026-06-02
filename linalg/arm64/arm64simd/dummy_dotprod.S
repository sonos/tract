// Build-time capability probe for the assembler, used by build.rs
// (assembler_supports_dotprod). Older binutils — notably the Debian stretch
// aarch64 cross-toolchain in CI — predate FEAT_DotProd and cannot assemble
// `sdot` even with `.cpu generic+fp+simd+dotprod`. If this file fails to
// assemble, build.rs skips the SDOT kernel and the `tract_arm64_dotprod` cfg,
// and the runtime falls back to the SMLAL 8x8 i32 kernel. Not linked into
// anything.
.cpu generic+fp+simd+dotprod
.text
.globl tract_dotprod_probe
tract_dotprod_probe:
    sdot v0.4s, v1.16b, v2.4b[0]
    ret
