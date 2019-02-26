// vim: ft=arm

/*
    fn arm64simd_conv_s8x8(
        k: size_t,
        a: *const f32,
        b_tops: *const *const f32,
        b_offsets: *const isize,
        c: *mut f32,
        rsc: size_t,
        csc: size_t,
    );
*/

// C tile regs: v16 to v31, no need to preserve
// 
//      v16[0] v18[0] v20[0] v22[0] v24[0] v26[0] v28[0] v30[0]
//      v16[1] v18[1] 
//      v16[2] v18[2] 
//      v16[3] v18[3]
//                     
//      v17[0] v19[0] v21[0] v23[0] v25[0] v27[0] v29[0] v31[0]
//      v17[1] v19[1] 
//      v17[2] v19[2] 
//      v17[3] v19[3] 

// no preservation either for v0-v7...
// packed A buffering (2x8 values): alternating v0, v1 with v2, v3

.text
.align 4
#ifdef __MACH__
    .global _arm64simd_conv_s8x8
    _arm64simd_conv_s8x8:
#else
    .cpu generic+fp+simd
    .global arm64simd_conv_s8x8
    arm64simd_conv_s8x8:
#endif

    
//  x0=k, x1=a, x2=b_tops, x3=b_offsets
//  x4=c, x5=rsc, x6=csc

    prfm        pldl1keep, [x1]
    prfm        pldl1keep, [x2]

    stp         x19, x20, [sp, #-16]!
    stp         x21, x22, [sp, #-16]!
    stp         x23, x24, [sp, #-16]!
    stp         x25, x26, [sp, #-16]!

    eor         v16.8b, v16.8b, v16.8b
    eor         v17.8b, v17.8b, v17.8b
    eor         v18.8b, v18.8b, v18.8b
    eor         v19.8b, v19.8b, v19.8b

    eor         v20.8b, v20.8b, v20.8b
    eor         v21.8b, v21.8b, v21.8b
    eor         v22.8b, v22.8b, v22.8b
    eor         v23.8b, v23.8b, v23.8b

    eor         v24.8b, v24.8b, v24.8b
    eor         v25.8b, v25.8b, v25.8b
    eor         v26.8b, v26.8b, v26.8b
    eor         v27.8b, v27.8b, v27.8b

    eor         v28.8b, v28.8b, v28.8b
    eor         v29.8b, v29.8b, v29.8b
    eor         v30.8b, v30.8b, v30.8b
    eor         v31.8b, v31.8b, v31.8b

    cmp         x0, #0
    beq .STORE

    ldp         x19, x20, [x2], #16
    ldp         x21, x22, [x2], #16
    ldp         x23, x24, [x2], #16
    ldp         x25, x26, [x2], #16

    .LOOP:

    ldr         q0, [ x1 ], #16
    ldr         q1, [ x1 ], #16

    ldr         x2, [ x3 ], #8

    add         x9, x2, x19
    ld1         {v4.s}[0], [ x9 ]
    add         x10, x2, x20
    ld1         {v4.s}[1], [ x10 ]
    add         x11, x2, x21
    ld1         {v4.s}[2], [ x11 ]
    add         x12, x2, x22
    ld1         {v4.s}[3], [ x12 ]
    add         x13, x2, x23
    ld1         {v5.s}[0], [ x13 ]
    add         x14, x2, x24
    ld1         {v5.s}[1], [ x14 ]
    add         x15, x2, x25
    ld1         {v5.s}[2], [ x15 ]
    add         x9, x2, x26
    ld1         {v5.s}[3], [ x9 ]


    fmla        v16.4s, v0.4s, v4.s[0]
    fmla        v17.4s, v1.4s, v4.s[0]
    fmla        v18.4s, v0.4s, v4.s[1]
    fmla        v19.4s, v1.4s, v4.s[1]
    fmla        v20.4s, v0.4s, v4.s[2]
    fmla        v21.4s, v1.4s, v4.s[2]
    fmla        v22.4s, v0.4s, v4.s[3]
    fmla        v23.4s, v1.4s, v4.s[3]

    fmla        v24.4s, v0.4s, v5.s[0]
    fmla        v25.4s, v1.4s, v5.s[0]
    fmla        v26.4s, v0.4s, v5.s[1]
    fmla        v27.4s, v1.4s, v5.s[1]
    fmla        v28.4s, v0.4s, v5.s[2]
    fmla        v29.4s, v1.4s, v5.s[2]
    fmla        v30.4s, v0.4s, v5.s[3]
    fmla        v31.4s, v1.4s, v5.s[3]

    subs        x0, x0, #1
    bne .LOOP

    .STORE:

    lsl         x5, x5, #2 /// rows
    lsl         x6, x6, #2 /// cols

    // x4 (arg), plus x9..x15 holds columns
    add         x9, x4, x6
    add         x10, x9, x6
    add         x11, x10, x6
    add         x12, x11, x6
    add         x13, x12, x6
    add         x14, x13, x6
    add         x15, x14, x6


    // col 1
    st1         { v16.s }[0], [ x4 ], x5
    st1         { v16.s }[1], [ x4 ], x5
    st1         { v16.s }[2], [ x4 ], x5
    st1         { v16.s }[3], [ x4 ], x5

    st1         { v17.s }[0], [ x4 ], x5
    st1         { v17.s }[1], [ x4 ], x5
    st1         { v17.s }[2], [ x4 ], x5
    st1         { v17.s }[3], [ x4 ], x5

    // col 2
    st1         { v18.s }[0], [ x9 ], x5
    st1         { v18.s }[1], [ x9 ], x5
    st1         { v18.s }[2], [ x9 ], x5
    st1         { v18.s }[3], [ x9 ], x5

    st1         { v19.s }[0], [ x9 ], x5
    st1         { v19.s }[1], [ x9 ], x5
    st1         { v19.s }[2], [ x9 ], x5
    st1         { v19.s }[3], [ x9 ], x5

    // col 3
    st1         { v20.s }[0], [ x10 ], x5
    st1         { v20.s }[1], [ x10 ], x5
    st1         { v20.s }[2], [ x10 ], x5
    st1         { v20.s }[3], [ x10 ], x5

    st1         { v21.s }[0], [ x10 ], x5
    st1         { v21.s }[1], [ x10 ], x5
    st1         { v21.s }[2], [ x10 ], x5
    st1         { v21.s }[3], [ x10 ], x5

    // col 4
    st1         { v22.s }[0], [ x11 ], x5
    st1         { v22.s }[1], [ x11 ], x5
    st1         { v22.s }[2], [ x11 ], x5
    st1         { v22.s }[3], [ x11 ], x5

    st1         { v23.s }[0], [ x11 ], x5
    st1         { v23.s }[1], [ x11 ], x5
    st1         { v23.s }[2], [ x11 ], x5
    st1         { v23.s }[3], [ x11 ], x5

    // col 5
    st1         { v24.s }[0], [ x12 ], x5
    st1         { v24.s }[1], [ x12 ], x5
    st1         { v24.s }[2], [ x12 ], x5
    st1         { v24.s }[3], [ x12 ], x5

    st1         { v25.s }[0], [ x12 ], x5
    st1         { v25.s }[1], [ x12 ], x5
    st1         { v25.s }[2], [ x12 ], x5
    st1         { v25.s }[3], [ x12 ], x5

    // col 6
    st1         { v26.s }[0], [ x13 ], x5
    st1         { v26.s }[1], [ x13 ], x5
    st1         { v26.s }[2], [ x13 ], x5
    st1         { v26.s }[3], [ x13 ], x5

    st1         { v27.s }[0], [ x13 ], x5
    st1         { v27.s }[1], [ x13 ], x5
    st1         { v27.s }[2], [ x13 ], x5
    st1         { v27.s }[3], [ x13 ], x5

    // col 7
    st1         { v28.s }[0], [ x14 ], x5
    st1         { v28.s }[1], [ x14 ], x5
    st1         { v28.s }[2], [ x14 ], x5
    st1         { v28.s }[3], [ x14 ], x5

    st1         { v29.s }[0], [ x14 ], x5
    st1         { v29.s }[1], [ x14 ], x5
    st1         { v29.s }[2], [ x14 ], x5
    st1         { v29.s }[3], [ x14 ], x5

    // col 8
    st1         { v30.s }[0], [ x15 ], x5
    st1         { v30.s }[1], [ x15 ], x5
    st1         { v30.s }[2], [ x15 ], x5
    st1         { v30.s }[3], [ x15 ], x5

    st1         { v31.s }[0], [ x15 ], x5
    st1         { v31.s }[1], [ x15 ], x5
    st1         { v31.s }[2], [ x15 ], x5
    st1         { v31.s }[3], [ x15 ], x5

    ldp         x25, x26, [sp], #16
    ldp         x23, x24, [sp], #16
    ldp         x21, x22, [sp], #16
    ldp         x19, x20, [sp], #16

    ret
