.intel_syntax noprefix

    .text
    .align 32;
    .globl fma_stile16x6

/* tile 16 x 6:

    ymm0 ymm2 ymm4 ymm6 ymm8 ymm10
    ymm1 ymm3 ymm5 ymm7 ymm8 ymm11

System V ABI:
    args: rdi, rsi, rdx, rcx, r8, r9
    preserve: rbx, rsp, rbp, r12, r13, r14, r15 
    scratch: rax, rdi, rsi, rdx, rcx, r8, r9, r10, r11
    return: rax (+rdx)
*/

fma_stile16x6:
    push        rbx
    push        r12
    push        r13
    push        r14
    push        r15
    mov         rax, 1

    vzeroall

    mov     rsi,    [rdi]       // A
    mov     rbx,    [rdi + 8]   // B

    mov     rcx,    [rdi + 24]  // Linear spec
    mov     rcx,    [rcx + 8]   // k
    test    rcx,    rcx
    je      .non_linear

    mov     r8, [rbx]   // B discriminant
    cmpq    r8,  1
    je      .packed_packed
    cmpq    r8,  2
    je      .packed_tops_and_offsets

    jp      .return

.packed_tops_and_offsets:
    mov     rsi,    [rsi + 8] // A
    mov     r14,    [rbx + 8] // B rows offsets
    mov     r15,    [rbx + 16] // B cols head

    mov     r8,     [r15]
    mov     r9,     [r15 + 8]
    mov     r10,    [r15 + 16]
    mov     r11,    [r15 + 24]
    mov     r12,    [r15 + 32]
    mov     r13,    [r15 + 40]

.main_loop_packed_tops_and_offsets:
    vmovaps         ymm12,  [rsi]
    vmovaps         ymm13,  [rsi + 32]
    mov     r15,    [r14]

    vbroadcastss    ymm14,  [r8 + r15]
    vbroadcastss    ymm15,  [r9 + r15]

    vfmadd231ps     ymm0,   ymm12, ymm14
    vfmadd231ps     ymm1,   ymm13, ymm14

    vbroadcastss    ymm14,  [r10 + r15]

    vfmadd231ps     ymm2,   ymm12, ymm15
    vfmadd231ps     ymm3,   ymm13, ymm15

    vbroadcastss    ymm15,  [r11 + r15]

    vfmadd231ps     ymm4,   ymm12, ymm14
    vfmadd231ps     ymm5,   ymm13, ymm14

    vbroadcastss    ymm14,  [r12 + r15]

    vfmadd231ps     ymm6,   ymm12, ymm15
    vfmadd231ps     ymm7,   ymm13, ymm15

    vbroadcastss    ymm15,  [r13 + r15]

    vfmadd231ps     ymm8,   ymm12, ymm14
    vfmadd231ps     ymm9,   ymm13, ymm14

    vfmadd231ps     ymm10,   ymm12, ymm15
    vfmadd231ps     ymm11,   ymm13, ymm15

    add     r14,    8
    add             rsi,    64
    dec             rcx
    jnz             .main_loop_packed_tops_and_offsets

    jp              .non_linear

.packed_packed:

    mov     rsi,   [rsi + 8] // A
    mov     r8,    [rbx + 8] // B 

.main_loop_packed_packed:
    vbroadcastss    ymm14,  [r8]
    vbroadcastss    ymm15,  [r8 + 4]

    vmovaps         ymm12,  [rsi]
    vmovaps         ymm13,  [rsi + 32]

    vfmadd231ps     ymm0,   ymm12, ymm14
    vfmadd231ps     ymm1,   ymm13, ymm14

    vbroadcastss    ymm14,  [r8 + 8]

    vfmadd231ps     ymm2,   ymm12, ymm15
    vfmadd231ps     ymm3,   ymm13, ymm15

    vbroadcastss    ymm15,  [r8 + 12]

    vfmadd231ps     ymm4,   ymm12, ymm14
    vfmadd231ps     ymm5,   ymm13, ymm14

    vbroadcastss    ymm14,  [r8 + 16]

    vfmadd231ps     ymm6,   ymm12, ymm15
    vfmadd231ps     ymm7,   ymm13, ymm15

    vbroadcastss    ymm15,  [r8 + 20]

    vfmadd231ps     ymm8,   ymm12, ymm14
    vfmadd231ps     ymm9,   ymm13, ymm14

    vfmadd231ps     ymm10,   ymm12, ymm15
    vfmadd231ps     ymm11,   ymm13, ymm15

    add             r8,     24
    add             rsi,    64
    dec             rcx
    jnz             .main_loop_packed_packed

.non_linear:

.store:
    mov     rcx,    [rdi + 16]

    // FIXME: assume Strides storage
    mov     r8,     [rcx + 8]           // c ptr
    mov     rsi,    [rcx + 16]          // row stride
    mov     rbx,    [rcx + 24]          // col stride

    // tops of cols
    lea     r9,     [ r8 + rbx ]
    lea     r10,    [ r8 + 2 * rbx ]
    lea     r12,    [ r8 + 4 * rbx ]
    lea     r11,    [ r10 + rbx ]
    lea     r13,    [ r12 + rbx ]

.macro push_6_values_to_memory item
    vextractps  [r8],     xmm0, \item
    add         r8,     rsi
    vextractps  [r9],     xmm2, \item
    add         r9,     rsi
    vextractps  [r10],    xmm4, \item
    add         r10,    rsi
    vextractps  [r11],    xmm6, \item
    add         r11,    rsi
    vextractps  [r12],    xmm8, \item
    add         r12,    rsi
    vextractps  [r13],    xmm10, \item
    add         r13,    rsi
.endm

.macro push_24_values_to_memory
    push_6_values_to_memory(0)
    push_6_values_to_memory(1)
    push_6_values_to_memory(2)
    push_6_values_to_memory(3)
.endm

.macro move_24_values_to_top_of_tile control
    vperm2f128  ymm0,   ymm0,   ymm1,   \control
    vperm2f128  ymm2,   ymm2,   ymm3,   \control
    vperm2f128  ymm4,   ymm4,   ymm5,   \control
    vperm2f128  ymm6,   ymm6,   ymm7,   \control
    vperm2f128  ymm8,   ymm8,   ymm9,   \control
    vperm2f128  ymm10,  ymm10,  ymm11,  \control
.endm

    push_24_values_to_memory
    move_24_values_to_top_of_tile(1)
    push_24_values_to_memory
    move_24_values_to_top_of_tile(2)
    push_24_values_to_memory
    move_24_values_to_top_of_tile(3)
    push_24_values_to_memory

    mov     rax,    0

.return:

    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    ret
