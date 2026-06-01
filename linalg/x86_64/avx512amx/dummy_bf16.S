// Build-time capability probe for the assembler, used by build.rs
// (assembler_supports_amx_bf16). Checks that the assembler accepts the
// TDPBF16PS mnemonic (AMX bf16 dot-product). Same binutils version requirement
// as AMX int8 (>= 2.34); provided as a separate probe so the two cfgs can be
// set independently if needed. Not linked into anything.
.intel_syntax noprefix
.text
.globl tract_amx_bf16_probe
tract_amx_bf16_probe:
    push rbp
    mov rbp, rsp
    sub rsp, 64
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
    tdpbf16ps tmm0, tmm1, tmm2   // AMX bf16: the instruction this probe checks
    tilerelease
    mov rsp, rbp
    pop rbp
    ret
