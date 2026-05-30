// Build-time capability probe for the assembler, used by build.rs
// (assembler_supports_amx_int8). Older binutils — notably the Debian stretch
// x86_64 cross-toolchain in CI — predate AMX and cannot assemble these
// mnemonics. If this file fails to assemble, build.rs skips the AMX kernels
// and the `tract_amx_int8` cfg, and the runtime falls back to VNNI (or AVX2)
// for `qmmm_i32`. Not linked into anything.
.intel_syntax noprefix
.text
.globl tract_amx_int8_probe
tract_amx_int8_probe:
    push rbp
    mov rbp, rsp
    sub rsp, 64                  // room for the tilecfg block
    mov qword ptr [rsp], 0
    mov qword ptr [rsp+8], 0
    mov qword ptr [rsp+16], 0
    mov qword ptr [rsp+24], 0
    mov qword ptr [rsp+32], 0
    mov qword ptr [rsp+40], 0
    mov qword ptr [rsp+48], 0
    mov qword ptr [rsp+56], 0
    mov byte ptr [rsp], 1        // palette = 1
    ldtilecfg [rsp]
    tilezero tmm0
    tdpbusd  tmm0, tmm1, tmm2
    tilerelease
    mov rsp, rbp
    pop rbp
    ret
