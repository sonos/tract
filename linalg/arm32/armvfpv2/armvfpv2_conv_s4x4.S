// vim: ft=arm

    .arm
    .text
    .global arm_vfpv2_conv_s4x4
    .type arm_vfpv2_conv_s4x4, %function

// C tile:

//  s16 s20 s24 s28
//  s17 s21 s25 s29
//  s18 s22 s26 s30
//  s19 s23 s27 s31

// packed A: (2x4) alternating between (s0-s3) and (s4-s7)

// all vfp registers in use.

/*
    fn arm_vfpv2_conv_s4x4(
        k: size_t,
        a: *const f32,
        b_tops: *const *const f32,
        b_offsets: *const isize,
        c: *mut f32,
        rsc: size_t,
        csc: size_t,
    );
*/

arm_vfpv2_conv_s4x4:

//  r0=k, r1=a, r2=b_tops, r3=b_offsets
//  c, rsc, csc on stack

    pld         [r1]
    pld         [r1, #8]
    pld         [r2]
    pld         [r2, #8]

    push        { r4-r10 }               // no lr (we're a leaf), no fp. #28 bytes


    vmrs        r4, FPSCR
    bic         r4, r4, #0x00370000
    vmsr        FPSCR, r4

    vpush       { s16-s31 }             // push 16*4 = 64bytes. stacked 92 bytes

    eor         r4, r4
    vmov        s16, r4
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

    ldm             r2, { r5, r6, r7, r8 }

    .LOOP:

    ldr             r4, [r3], #4
    ldr             r2, [r3]
    vldmia          r1!, { s0, s1, s2, s3 }
    pld             [r3, #4]
    pld             [r1]
    pld             [r1, #8]

    add             r9, r4, r5
    vldr            s8, [r9]
    add             r10, r4, r6
    vldr            s9, [r10]
    add             r9, r4, r7
    vldr            s10, [r9]
    add             r10, r4, r8
    vldr            s11, [r10]

    add             r9, r2, r5
    pld             [r9]
    add             r10, r2, r6
    pld             [r10]
    add             r9, r2, r7
    pld             [r9]
    add             r10, r2, r8
    pld             [r10]

    vmla.f32        s16, s0, s8
    vmla.f32        s17, s1, s8
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

    ldr         r3, [sp, #92]
    ldr         r9, [sp, #96]
    ldr         r8, [sp, #100]

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
    pop         { r4-r10 }

    bx          lr
