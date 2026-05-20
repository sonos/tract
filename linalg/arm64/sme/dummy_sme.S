// Build-time capability probe for the assembler, used by build.rs
// (assembler_supports_sme). Older binutils — notably the Debian stretch
// aarch64 cross-toolchain in CI — predate SME and cannot assemble these
// mnemonics even with `.arch armv9-a+sme2`. If this file fails to assemble,
// build.rs skips the SME kernels and the `tract_sme` cfg, and the runtime
// falls back to the portable path. Not linked into anything.
.arch armv9-a+sme2
.text
.globl tract_sme_probe
tract_sme_probe:
    smstart
    zero    {za}
    smstop
    ret
