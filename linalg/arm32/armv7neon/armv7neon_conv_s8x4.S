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

// q6 and q7 are left alone -> no need to preserve s24-s31

    .arm
    .text
    .global arm7neon_conv_s8x4
    .type arm7neon_conv_s8x4, %function
/*
    fn arm7neon_mm_s8x4(
        k: size_t,
        a: *const f32,
        b_tops: *const *const f32,
        b_offsets: *const isize,
        c: *mut f32,
        rsc: size_t,
        csc: size_t,
    );
*/

arm7neon_conv_s8x4:

//  r0=k, r1=a, r2=b_tops, r3=b_offsets
//  c, rsc, csc on stack

    pld [r1]
    pld [r1, #8]
    pld [r1, #16]
    pld [r1, #24]
    pld [r2]
    pld [r2, #8]

    push        { r4-r12 }               // no lr (we're a leaf), no fp. #36 bytes

    vpush       { q4-q5 }               // 2*4*4 = 32 bytes -> total 68bytes

    veor      q8, q8 ,q8
    veor      q9, q9 ,q9
    veor      q10, q10 ,q10
    veor      q11, q11 ,q11
    veor      q12, q12 ,q12
    veor      q13, q13 ,q13
    veor      q14, q14 ,q14
    veor      q15, q15 ,q15

    cmp r0, #0
    beq .STORE

    ldm             r2, { r5, r6, r7, r8 }
    ldr             r4, [r3], #4

    add             r9, r5, r4
    add             r10, r6, r4
    add             r11, r7, r4
    add             r12, r8, r4

    .LOOP:

    vldmia          r1!, { q0, q1 }
    ldr             r4, [r3], #4

    vldr            s16, [r9]
    vldr            s17, [r10]
    vldr            s18, [r11]
    vldr            s19, [r12]

    vmla.f32        q8, q0, d8[0]
    add             r9, r5, r4
    vmla.f32        q9, q1, d8[0]

    vmla.f32        q10, q0, d8[1]
    add             r10, r6, r4
    vmla.f32        q11, q1, d8[1]

    vmla.f32        q12, q0, d9[0]
    add             r11, r7, r4
    vmla.f32        q13, q1, d9[0]

    vmla.f32        q14, q0, d9[1]
    add             r12, r8, r4
    vmla.f32        q15, q1, d9[1]

    subs r0, r0, #1
    bne .LOOP

    .STORE:

    ldr         r3, [sp, #68]
    ldr         r9, [sp, #72]
    ldr         r8, [sp, #76]

    lsl r8, r8, #2 // r8 *= sizeof(float) // csc
    lsl r9, r9, #2 // r9 *= sizeof(float) // rsc

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

    vpop        { q4-q5 }
    pop         { r4-r12 }

    bx          lr
