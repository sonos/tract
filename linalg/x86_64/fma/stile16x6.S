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

    jp      .unimplemented

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
    mov     rcx,    [rdi + 32]          // non linear spec
    test    rcx,    rcx
    jz      .store

    sub     rcx,    16
.non_linear_loop:
    add     rcx,    16
    mov     rax,    [rcx]

    cmpq    rax,    0
    je      .store

    cmpq    rax,    3
    je      .non_linear_addc

.non_linear_addc:
    mov     rax,    [rdi + 16]

    // FIXME: assume Strides storage
    mov     r10,    [rax + 8]           // c ptr
    mov     rsi,    [rax + 16]          // row stride
    mov     rbx,    [rax + 24]          // col stride

.macro gather_8_from_r8_to_ymm14
    lea     r9,     [ r8 + 2 * rsi]
    vinsertps       xmm14,  xmm14,  [r8],         0b00000000
    vinsertps       xmm14,  xmm14,  [r8 + rsi],   0b01010000
    vinsertps       xmm14,  xmm14,  [r9],         0b10100000
    vinsertps       xmm14,  xmm14,  [r9 + rsi],   0b11110000

    lea     r8,     [ r8 + 4 * rsi ]
    lea     r9,     [ r8 + 2 * rsi ]
    vinsertps       xmm15,  xmm15,  [r8],         0b00000000
    vinsertps       xmm15,  xmm15,  [r8 + rsi],   0b01010000
    vinsertps       xmm15,  xmm15,  [r9],         0b10100000
    vinsertps       xmm15,  xmm15,  [r9 + rsi],   0b11110000

    vperm2f128      ymm14,  ymm14, ymm15,         0b00100000
.endm

    mov     r8,     r10
    gather_8_from_r8_to_ymm14
    vaddps          ymm0,   ymm0,   ymm14
    lea     r8,     [r10 + 8 * rsi ]
    gather_8_from_r8_to_ymm14
    vaddps          ymm1,   ymm1,   ymm14

    add     r10,    rbx
    mov     r8,     r10
    gather_8_from_r8_to_ymm14
    vaddps          ymm2,   ymm2,   ymm14
    lea     r8,     [r10 + 8 * rsi ]
    gather_8_from_r8_to_ymm14
    vaddps          ymm3,   ymm3,   ymm14

    add     r10,    rbx
    mov     r8,     r10
    gather_8_from_r8_to_ymm14
    vaddps          ymm4,   ymm4,   ymm14
    lea     r8,     [r10 + 8 * rsi ]
    gather_8_from_r8_to_ymm14
    vaddps          ymm5,   ymm5,   ymm14

    add     r10,    rbx
    mov     r8,     r10
    gather_8_from_r8_to_ymm14
    vaddps          ymm6,   ymm6,   ymm14
    lea     r8,     [r10 + 8 * rsi ]
    gather_8_from_r8_to_ymm14
    vaddps          ymm7,   ymm7,   ymm14

    add     r10,    rbx
    mov     r8,     r10
    gather_8_from_r8_to_ymm14
    vaddps          ymm8,   ymm8,   ymm14
    lea     r8,     [r10 + 8 * rsi ]
    gather_8_from_r8_to_ymm14
    vaddps          ymm9,   ymm9,   ymm14

    add     r10,    rbx
    mov     r8,     r10
    gather_8_from_r8_to_ymm14
    vaddps          ymm10,   ymm10,   ymm14
    lea     r8,     [r10 + 8 * rsi ]
    gather_8_from_r8_to_ymm14
    vaddps          ymm11,   ymm11,   ymm14

    jp      .non_linear_loop

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

.ok:
    mov     rax,    0
    jp      .return

.unimplemented:
    mov     rax,    1

.return:
    pop r15
    pop r14
    pop r13
    pop r12
    pop rbx
    ret
