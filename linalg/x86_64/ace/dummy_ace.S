// Build-time capability probe for ACE (AI Compute Extensions), used by build.rs
// (assembler_supports_ace). ACE has no silicon and no assembler support yet
// (hardware expected ~2028), so this probe is EXPECTED TO FAIL on every current
// toolchain: that leaves cfg(tract_ace) unset and keeps the portable emulated ACE
// kernels (pure Rust, in src/ace/) in place. When binutils/LLVM gain ACE support
// the probe passes, cfg(tract_ace) is set, and the real ACE .S kernels light up.
//
// The mnemonics below are SPECULATIVE: the public ACE whitepaper gives intrinsic
// names (`__tile_top4bssd`, `__tile_top2bf16ps`) but not assembler spellings.
// Replace these with the ratified mnemonics once the ACE ISA reference is public.
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
    ttop4bssd   tmm0, zmm1, zmm2     // INT8 outer product   (speculative spelling)
    ttop2bf16ps tmm0, zmm1, zmm2     // BF16 outer product   (speculative spelling)
    ttopmxfp8   tmm0, zmm1, zmm2     // MXFP8 block-scaled    (speculative spelling)
    tilerelease
    mov rsp, rbp
    pop rbp
    ret
