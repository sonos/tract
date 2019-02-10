// vim: ft=arm

    .arm
    .text
    .global arm_vfpv2_mm_s4x4
    .type arm_vfpv2_mm_s4x4, %function

// C tile:

//  s16 s20 s24 s28
//  s17 s21 s25 s29
//  s18 s22 s26 s30
//  s19 s23 s27 s31

// packed A: (2x4) alternating between (s0-s3) and (s4-s7)
// packed B: (2x4) alternating between (s8-s11) and (s12-15)

// all vfp registers in use.

/*
    fn arm_vfpv2_mm_s4x4(
        k: size_t,
        a: *const f32,
        b: *const f32,
        c: *mut f32,
        rsc: size_t,
        csc: size_t,
    );
*/

arm_vfpv2_mm_s4x4:

//  r0=k, r1=a, r2=b, r3=c

    pld [r1]
    pld [r1, #8]
    pld [r2]
    pld [r2, #8]

    push        { r4-r9 }               // no lr (we're a leaf), no fp. #24 bytes

    ldr         r8, [sp, #28]
    ldr         r9, [sp, #24]

//  r8=rsc, r9=csc

    vmrs        r6, FPSCR
    bic         r6, r6, #0x00370000
    vmsr        FPSCR, r6

    vpush       { s16-s31 }
    eor         r6, r6

    vmov        s16, r6
    vmov.f32    s17, s16
    vmov.f32    s18, s16
    vmov.f32    s19, s16
    vmov.f32    s20, s16
    vmov.f32    s21, s16
    vmov.f32    s22, s16
    vmov.f32    s23, s16
    vmov.f32    s24, s16
    vmov.f32    s25, s16
    vmov.f32    s26, s16
    vmov.f32    s27, s16
    vmov.f32    s28, s16
    vmov.f32    s29, s16
    vmov.f32    s30, s16
    vmov.f32    s31, s16

    cmp r0, #0
    beq .STORE

    cmp r0, #4
    blt .LOOP

    .LOOP4:

    // 1
    vldmia          r1!, { s0, s1 }
    vldmia          r2!, { s8, s9 }

    vmla.f32        s16, s0, s8
    vldmia          r1!, { s2, s3 }
    vmla.f32        s17, s1, s8
    vldmia          r2!, { s10, s11 }
    vmla.f32        s18, s2, s8
    vmla.f32        s19, s3, s8

    vmla.f32        s20, s0, s9
    vmla.f32        s21, s1, s9
    vmla.f32        s22, s2, s9
    vmla.f32        s23, s3, s9

    vldmia          r1!, { s4-s7 }
    vmla.f32        s24, s0, s10
    vmla.f32        s25, s1, s10
    vmla.f32        s26, s2, s10
    vmla.f32        s27, s3, s10

    vldmia          r2!, { s12-s15 }
    vmla.f32        s28, s0, s11
    vmla.f32        s29, s1, s11
    vmla.f32        s30, s2, s11
    vmla.f32        s31, s3, s11

    // 2
    vmla.f32        s16, s4, s12
    vmla.f32        s17, s5, s12
    vmla.f32        s18, s6, s12
    vmla.f32        s19, s7, s12

    vldmia          r1!, { s0-s3 }

    vmla.f32        s20, s4, s13
    vmla.f32        s21, s5, s13
    vmla.f32        s22, s6, s13
    vmla.f32        s23, s7, s13

    vldmia          r2!, { s8-s11 }

    vmla.f32        s24, s4, s14
    vmla.f32        s25, s5, s14
    vmla.f32        s26, s6, s14
    vmla.f32        s27, s7, s14

    vmla.f32        s28, s4, s15
    vmla.f32        s29, s5, s15
    vmla.f32        s30, s6, s15
    vmla.f32        s31, s7, s15

    // 3
    vmla.f32        s16, s0, s8
    vmla.f32        s17, s1, s8
    vmla.f32        s18, s2, s8
    vmla.f32        s19, s3, s8

    vldmia          r1!, { s4-s7 }

    vmla.f32        s20, s0, s9
    vmla.f32        s21, s1, s9
    vmla.f32        s22, s2, s9
    vmla.f32        s23, s3, s9

    vldmia          r2!, { s12-s15 }

    vmla.f32        s24, s0, s10
    vmla.f32        s25, s1, s10
    vmla.f32        s26, s2, s10
    vmla.f32        s27, s3, s10

    pld [r1]

    vmla.f32        s28, s0, s11
    vmla.f32        s29, s1, s11
    vmla.f32        s30, s2, s11
    vmla.f32        s31, s3, s11

    pld [r2]

    // 4
    vmla.f32        s16, s4, s12
    vmla.f32        s17, s5, s12
    vmla.f32        s18, s6, s12
    vmla.f32        s19, s7, s12

    vmla.f32        s20, s4, s13
    vmla.f32        s21, s5, s13
    vmla.f32        s22, s6, s13
    vmla.f32        s23, s7, s13

    vmla.f32        s24, s4, s14
    vmla.f32        s25, s5, s14
    vmla.f32        s26, s6, s14
    vmla.f32        s27, s7, s14

    vmla.f32        s28, s4, s15
    vmla.f32        s29, s5, s15
    vmla.f32        s30, s6, s15
    vmla.f32        s31, s7, s15

    sub r0, r0, #4
    cmp r0, #4
    bge .LOOP4

    cmp r0, #0
    beq .STORE

    .LOOP:

    vldmia          r1!, { s0, s1 }
    vldmia          r2!, { s8, s9 }

    vmla.f32        s16, s0, s8
    vldmia          r1!, { s2, s3 }
    vmla.f32        s17, s1, s8
    vldmia          r2!, { s10, s11 }
    vmla.f32        s18, s2, s8
    vmla.f32        s19, s3, s8

    vmla.f32        s20, s0, s9
    vmla.f32        s21, s1, s9
    vmla.f32        s22, s2, s9
    vmla.f32        s23, s3, s9

    vmla.f32        s24, s0, s10
    vmla.f32        s25, s1, s10
    vmla.f32        s26, s2, s10
    vmla.f32        s27, s3, s10

    vmla.f32        s28, s0, s11
    vmla.f32        s29, s1, s11
    vmla.f32        s30, s2, s11
    vmla.f32        s31, s3, s11

    subs r0, r0, #1
    bne .LOOP


    .STORE:

    lsl         r8, r8, #2              // r8 *= sizeof(float) // csc
    lsl         r9, r9, #2              // r9 *= sizeof(float) // rsc

    add         r4, r3, r8
    add         r5, r4, r8
    add         r6, r5, r8              // r3,r4,r5,r6 are now addr for cols of C

    fsts        s16, [ r3 ]
    add         r3, r3, r9
    fsts        s17, [ r3 ]
    add         r3, r3, r9
    fsts        s18, [ r3 ]
    add         r3, r3, r9
    fsts        s19, [ r3 ]

    fsts        s20, [ r4 ]
    add         r4, r4, r9
    fsts        s21, [ r4 ]
    add         r4, r4, r9
    fsts        s22, [ r4 ]
    add         r4, r4, r9
    fsts        s23, [ r4 ]

    fsts        s24, [ r5 ]
    add         r5, r5, r9
    fsts        s25, [ r5 ]
    add         r5, r5, r9
    fsts        s26, [ r5 ]
    add         r5, r5, r9
    fsts        s27, [ r5 ]

    fsts        s28, [ r6 ]
    add         r6, r6, r9
    fsts        s29, [ r6 ]
    add         r6, r6, r9
    fsts        s30, [ r6 ]
    add         r6, r6, r9
    fsts        s31, [ r6 ]

    vpop        { s16-s31 }
    pop         { r4-r9 }

    bx          lr
