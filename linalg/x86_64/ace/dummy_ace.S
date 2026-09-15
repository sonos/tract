// Build-time capability probe for ACE (AI Compute Extensions), used by build.rs
// (assembler_supports_ace). ACE has no silicon and no assembler support yet
// (hardware expected ~2028), so this probe is EXPECTED TO FAIL on every current
// toolchain: that leaves cfg(tract_ace) unset and keeps the portable emulated ACE
// kernels (pure Rust, in src/ace/) in place. When binutils/LLVM gain ACE support
// the probe passes, cfg(tract_ace) is set, and the real ACE .S kernels light up.
//
// The mnemonics below are based on the ACE v1.15 specification from the x86
// Ecosystem Advisory Group. These are the documented ACE instruction names:
// - TILEZERO, LDTILECFG, STTILECFG, TILERELEASE (tile management)
// - TILEMOVROW, TILEMOVCOL (data movement)
// - TOP4BSSD, TOP2BF16PS, TOP4MX* variants (outer products)
// - BSRINIT, BSRMOVF/H/L (block scale register operations)
//
// Not linked into anything.
.intel_syntax noprefix
.text
.globl tract_ace_probe
tract_ace_probe:
    push rbp
    mov rbp, rsp
    sub rsp, 64
    mov byte ptr [rsp], 1            // palette = 1 (ACE is an AMX-framework palette)
    ldtilecfg [rsp]
    // ACE outer products consume two AVX10 ZMM vectors into a tile accumulator.
    // Using documented ACE v1 mnemonics from the specification:
    tilezero tmm0                   // Zero tile register
    top4bssd  tmm0, zmm1, zmm2     // INT8 outer product (16x16x4 MACs)
    top2bf16ps tmm0, zmm1, zmm2     // BF16 outer product (16x16x2 MACs)
    // MX block-scaled variants (E4M3/E5M2 FP8, INT8 with block scales)
    // Note: exact MX mnemonics may vary slightly in final assembler
    tilerelease
    mov rsp, rbp
    pop rbp
    ret
