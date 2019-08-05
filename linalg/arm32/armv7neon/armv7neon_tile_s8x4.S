// vim: ft=arm

// C tile regs
// 
//      q8[0]    q10[0]   q12[0]    q14[0]
//      q8[1]    q10[1]   q12[1]    q14[1]
//      q8[2]    q10[2]   q12[2]    q14[2]
//      q8[3]    q10[3]   q12[3]    q14[3]
//
//      q9[0]    q11[0]   q13[0]    q15[0]
//      q9[1]    q11[1]   q13[1]    q15[1]
//      q9[2]    q11[2]   q13[2]    q15[2]
//      q9[3]    q11[3]   q13[3]    q15[3]

// packed A buffering (2x8 values): alternating q0, q1 with q2, q3
// packed B buffering (2x4 values): alternating q4 with q5

// q6 and q7 are left alone -> no need to preserve s24-s31

    .arm
    .text
    .global neon_stile8x4
    .type neon_stile8x4, %function

neon_stile8x4:

    push        { r4-r10 }               // no lr (we're a leaf), no fp. #24 bytes
    vpush       { q4-q5 }

    veor      q8, q8 ,q8
    veor      q9, q9 ,q9
    veor      q10, q10 ,q10
    veor      q11, q11 ,q11
    veor      q12, q12 ,q12
    veor      q13, q13 ,q13
    veor      q14, q14 ,q14
    veor      q15, q15 ,q15

    ldr     r4, [r0]
    // check a->discriminant == 1 (packed)
    ldr     r5, [r4]
    cmp     r5, #1
    bne     .unsupported
    ldr     r1, [r4, #4]  // packed A ptr
    pld     [r1]

    // check linear
    ldr     r4, [r0, #12]
    ldr     r5, [r4]
    cmp     r5, #0
    bne     .unsupported
    ldr     r3, [r4, #4] // k

    cmp r3, #0
    beq .non_linear

    // B
    ldr     r4, [r0, #8]
    ldr     r5, [r4]
    cmp     r5, #0
    bne     .unsupported
    cmp     r5, #1
    bne     .packed_packed
    cmp     r5, #1
    bne     .packed_tops_and_offsets
    b       .unsupported

    .packed_tops_and_offsets:
    ldr             r2, [r4, #8] // B cols
    ldm             r2, {r5, r6, r7, r8}
    ldr             r3, [r4, #4] // B row offsets

    .packed_tops_and_offsets_loop_1:

    ldr             r2, [r3], #4

    add             r9, r5, r4
    add             r10, r6, r4
    add             r11, r7, r4
    add             r12, r8, r4

    vldr            s16, [r9]
    vldr            s17, [r10]
    vldr            s18, [r11]
    vldr            s19, [r12]

    vmla.f32        q8, q0, d8[0]
    vmla.f32        q9, q1, d8[0]

    vmla.f32        q10, q0, d8[1]
    vmla.f32        q11, q1, d8[1]

    vmla.f32        q12, q0, d9[0]
    vmla.f32        q13, q1, d9[0]

    vmla.f32        q14, q0, d9[1]
    vmla.f32        q15, q1, d9[1]

    subs r3, r3, #1
    bne .packed_packed_loop_1

    .packed_packed:
    ldr     r2, [r4, #4] // packed B ptr
    pld     [r2]

    cmp r3, #4
    blt .packed_packed_loop_1

    .packed_packed_loop_4:

    // 1
    vldmia          r1!, { q0, q1 }
    vldmia          r2!, { q4 }

    vmla.f32        q8, q0, d8[0]
    vmla.f32        q9, q1, d8[0]

    vldmia          r1!, { q2, q3 }

    vmla.f32        q10, q0, d8[1]
    vmla.f32        q11, q1, d8[1]

    vldmia          r2!, { q5 }

    vmla.f32        q12, q0, d9[0]
    vmla.f32        q13, q1, d9[0]

    vmla.f32        q14, q0, d9[1]
    vmla.f32        q15, q1, d9[1]

    // 2
    vldmia          r1!, { q0, q1 }

    vmla.f32        q8, q2, d10[0]
    vmla.f32        q9, q3, d10[0]

    vldmia          r2!, { q4 }

    vmla.f32        q10, q2, d10[1]
    vmla.f32        q11, q3, d10[1]

    vmla.f32        q12, q2, d11[0]
    vmla.f32        q13, q3, d11[0]

    vmla.f32        q14, q2, d11[1]
    vmla.f32        q15, q3, d11[1]

    // 3
    vldmia          r1!, { q2, q3 }

    vmla.f32        q8, q0, d8[0]
    vmla.f32        q9, q1, d8[0]

    vldmia          r2!, { q5 }

    vmla.f32        q10, q0, d8[1]
    vmla.f32        q11, q1, d8[1]

    pld [r1]

    vmla.f32        q12, q0, d9[0]
    vmla.f32        q13, q1, d9[0]

    pld [r1, #8]

    vmla.f32        q14, q0, d9[1]
    vmla.f32        q15, q1, d9[1]
    pld [r1, #16]

    // 4

    vmla.f32        q8, q2, d10[0]
    vmla.f32        q9, q3, d10[0]

    pld [r1, #24]

    vmla.f32        q10, q2, d10[1]
    vmla.f32        q11, q3, d10[1]

    pld [r2]

    vmla.f32        q12, q2, d11[0]
    vmla.f32        q13, q3, d11[0]

    pld [r2, #8]

    vmla.f32        q14, q2, d11[1]
    vmla.f32        q15, q3, d11[1]

    sub r3, r3, #4
    cmp r3, #4
    bge .packed_packed_loop_4

    cmp r3, #0
    beq .non_linear

    .packed_packed_loop_1:

    vldmia          r1!, { q0, q1 }
    vldmia          r2!, { q4 }

    vmla.f32        q8, q0, d8[0]
    vmla.f32        q9, q1, d8[0]

    vmla.f32        q10, q0, d8[1]
    vmla.f32        q11, q1, d8[1]

    vmla.f32        q12, q0, d9[0]
    vmla.f32        q13, q1, d9[0]

    vmla.f32        q14, q0, d9[1]
    vmla.f32        q15, q1, d9[1]

    subs r3, r3, #1
    bne .packed_packed_loop_1
    b  .non_linear

    .non_linear:

    ldr     r4, [r0, #16]
    cmp     r4, #0
    beq     .STORE

    .non_linear_loop:
    ldr     r5, [r4, #4]!


    .STORE:

    ldr     r4, [r0, #8]
    ldr     r5, [r4]
    cmp     r5, #0
    bne     .unsupported

    ldr     r3, [r4, #4]   // C ptr
    ldr     r9, [r4, #8]   // rsc
    ldr     r8, [r4, #12]  // csc

    add r4, r3, r8
    add r5, r4, r8
    add r6, r5, r8 // r3,r4,r5,r6 are now addr for cols of C

    vst1.f32    d16[0], [ r3 ]
    add r3 , r3, r9
    vst1.f32    d16[1], [ r3 ]
    add r3 , r3, r9
    vst1.f32    d17[0], [ r3 ]
    add r3 , r3, r9
    vst1.f32    d17[1], [ r3 ]
    add r3 , r3, r9

    vst1.f32   d18[0], [ r3 ]
    add r3 , r3, r9
    vst1.f32   d18[1], [ r3 ]
    add r3 , r3, r9
    vst1.f32   d19[0], [ r3 ]
    add r3 , r3, r9
    vst1.f32   d19[1], [ r3 ]

    vst1.f32   d20[0], [ r4 ]
    add r4 , r4, r9
    vst1.f32   d20[1], [ r4 ]
    add r4 , r4, r9
    vst1.f32   d21[0], [ r4 ]
    add r4 , r4, r9
    vst1.f32   d21[1], [ r4 ]
    add r4 , r4, r9

    vst1.f32   d22[0], [ r4 ]
    add r4 , r4, r9
    vst1.f32   d22[1], [ r4 ]
    add r4 , r4, r9
    vst1.f32   d23[0], [ r4 ]
    add r4 , r4, r9
    vst1.f32   d23[1], [ r4 ]

    vst1.f32   d24[0], [ r5 ]
    add r5 , r5, r9
    vst1.f32   d24[1], [ r5 ]
    add r5 , r5, r9
    vst1.f32   d25[0], [ r5 ]
    add r5 , r5, r9
    vst1.f32   d25[1], [ r5 ]
    add r5 , r5, r9

    vst1.f32   d26[0], [ r5 ]
    add r5 , r5, r9
    vst1.f32   d26[1], [ r5 ]
    add r5 , r5, r9
    vst1.f32   d27[0], [ r5 ]
    add r5 , r5, r9
    vst1.f32   d27[1], [ r5 ]

    vst1.f32   d28[0], [ r6 ]
    add r6 , r6, r9
    vst1.f32   d28[1], [ r6 ]
    add r6 , r6, r9
    vst1.f32   d29[0], [ r6 ]
    add r6 , r6, r9
    vst1.f32   d29[1], [ r6 ]
    add r6 , r6, r9

    vst1.f32   d30[0], [ r6 ]
    add r6 , r6, r9
    vst1.f32   d30[1], [ r6 ]
    add r6 , r6, r9
    vst1.f32   d31[0], [ r6 ]
    add r6 , r6, r9
    vst1.f32   d31[1], [ r6 ]

.ok:
    mov         r0,     #0
    b           .return

.unsupported:
    mov         r0,     #1

.return:

    vpop        { q4-q5 }
    pop         { r4-r10 }

    bx          lr
