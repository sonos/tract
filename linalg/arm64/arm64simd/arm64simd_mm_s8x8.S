// vim: ft=arm

/*
    fn arm64simd_mm_s8x8(
        k: size_t,
        a: *const f32,
        b: *const f32,
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
// packed B buffering (2x8 values): alternating v4, v5 with v6, v7

.text
.align 4
#ifdef __MACH__
    .global _arm64simd_mm_s8x8
    _arm64simd_mm_s8x8:
#else
    .cpu generic+fp+simd
    .global arm64simd_mm_s8x8
    arm64simd_mm_s8x8:
#endif

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

    cmp         x0, #4
    blt         .LOOP

    .LOOP4:

    ldr         q0, [ x1 ], #16
    ldr         q1, [ x1 ], #16

    ldr         q4, [ x2 ], #16
    ldr         q5, [ x2 ], #16

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

    ldr         q2, [ x1 ], #16
    ldr         q3, [ x1 ], #16

    ldr         q6, [ x2 ], #16
    ldr         q7, [ x2 ], #16

    fmla        v16.4s, v2.4s, v6.s[0]
    fmla        v17.4s, v3.4s, v6.s[0]
    fmla        v18.4s, v2.4s, v6.s[1]
    fmla        v19.4s, v3.4s, v6.s[1]
    fmla        v20.4s, v2.4s, v6.s[2]
    fmla        v21.4s, v3.4s, v6.s[2]
    fmla        v22.4s, v2.4s, v6.s[3]
    fmla        v23.4s, v3.4s, v6.s[3]

    fmla        v24.4s, v2.4s, v7.s[0]
    fmla        v25.4s, v3.4s, v7.s[0]
    fmla        v26.4s, v2.4s, v7.s[1]
    fmla        v27.4s, v3.4s, v7.s[1]
    fmla        v28.4s, v2.4s, v7.s[2]
    fmla        v29.4s, v3.4s, v7.s[2]
    fmla        v30.4s, v2.4s, v7.s[3]
    fmla        v31.4s, v3.4s, v7.s[3]

    ldr         q0, [ x1 ], #16
    ldr         q1, [ x1 ], #16

    ldr         q4, [ x2 ], #16
    ldr         q5, [ x2 ], #16

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

    ldr         q2, [ x1 ], #16
    ldr         q3, [ x1 ], #16

    ldr         q6, [ x2 ], #16
    ldr         q7, [ x2 ], #16

    fmla        v16.4s, v2.4s, v6.s[0]
    fmla        v17.4s, v3.4s, v6.s[0]
    fmla        v18.4s, v2.4s, v6.s[1]
    fmla        v19.4s, v3.4s, v6.s[1]
    fmla        v20.4s, v2.4s, v6.s[2]
    fmla        v21.4s, v3.4s, v6.s[2]
    fmla        v22.4s, v2.4s, v6.s[3]
    fmla        v23.4s, v3.4s, v6.s[3]

    fmla        v24.4s, v2.4s, v7.s[0]
    fmla        v25.4s, v3.4s, v7.s[0]
    fmla        v26.4s, v2.4s, v7.s[1]
    fmla        v27.4s, v3.4s, v7.s[1]
    fmla        v28.4s, v2.4s, v7.s[2]
    fmla        v29.4s, v3.4s, v7.s[2]
    fmla        v30.4s, v2.4s, v7.s[3]
    fmla        v31.4s, v3.4s, v7.s[3]

    sub         x0, x0, #4
    cmp         x0, #4
    bge         .LOOP4

    cmp x0, #0
    beq .STORE

    .LOOP:

    ldr         q0, [ x1 ], #16
    ldr         q1, [ x1 ], #16

    ldr         q4, [ x2 ], #16
    ldr         q5, [ x2 ], #16

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

    lsl         x4, x4, #2 /// rows
    lsl         x5, x5, #2 /// cols

    // x3, plus x9..x15 holds columns
    add         x9, x3, x5
    add         x10, x9, x5
    add         x11, x10, x5
    add         x12, x11, x5
    add         x13, x12, x5
    add         x14, x13, x5
    add         x15, x14, x5


    // col 1
    st1         { v16.s }[0], [ x3 ], x4
    st1         { v16.s }[1], [ x3 ], x4
    st1         { v16.s }[2], [ x3 ], x4
    st1         { v16.s }[3], [ x3 ], x4

    st1         { v17.s }[0], [ x3 ], x4
    st1         { v17.s }[1], [ x3 ], x4
    st1         { v17.s }[2], [ x3 ], x4
    st1         { v17.s }[3], [ x3 ], x4

    // col 2
    st1         { v18.s }[0], [ x9 ], x4
    st1         { v18.s }[1], [ x9 ], x4
    st1         { v18.s }[2], [ x9 ], x4
    st1         { v18.s }[3], [ x9 ], x4

    st1         { v19.s }[0], [ x9 ], x4
    st1         { v19.s }[1], [ x9 ], x4
    st1         { v19.s }[2], [ x9 ], x4
    st1         { v19.s }[3], [ x9 ], x4

    // col 3
    st1         { v20.s }[0], [ x10 ], x4
    st1         { v20.s }[1], [ x10 ], x4
    st1         { v20.s }[2], [ x10 ], x4
    st1         { v20.s }[3], [ x10 ], x4

    st1         { v21.s }[0], [ x10 ], x4
    st1         { v21.s }[1], [ x10 ], x4
    st1         { v21.s }[2], [ x10 ], x4
    st1         { v21.s }[3], [ x10 ], x4

    // col 4
    st1         { v22.s }[0], [ x11 ], x4
    st1         { v22.s }[1], [ x11 ], x4
    st1         { v22.s }[2], [ x11 ], x4
    st1         { v22.s }[3], [ x11 ], x4

    st1         { v23.s }[0], [ x11 ], x4
    st1         { v23.s }[1], [ x11 ], x4
    st1         { v23.s }[2], [ x11 ], x4
    st1         { v23.s }[3], [ x11 ], x4

    // col 5
    st1         { v24.s }[0], [ x12 ], x4
    st1         { v24.s }[1], [ x12 ], x4
    st1         { v24.s }[2], [ x12 ], x4
    st1         { v24.s }[3], [ x12 ], x4

    st1         { v25.s }[0], [ x12 ], x4
    st1         { v25.s }[1], [ x12 ], x4
    st1         { v25.s }[2], [ x12 ], x4
    st1         { v25.s }[3], [ x12 ], x4

    // col 6
    st1         { v26.s }[0], [ x13 ], x4
    st1         { v26.s }[1], [ x13 ], x4
    st1         { v26.s }[2], [ x13 ], x4
    st1         { v26.s }[3], [ x13 ], x4

    st1         { v27.s }[0], [ x13 ], x4
    st1         { v27.s }[1], [ x13 ], x4
    st1         { v27.s }[2], [ x13 ], x4
    st1         { v27.s }[3], [ x13 ], x4

    // col 7
    st1         { v28.s }[0], [ x14 ], x4
    st1         { v28.s }[1], [ x14 ], x4
    st1         { v28.s }[2], [ x14 ], x4
    st1         { v28.s }[3], [ x14 ], x4

    st1         { v29.s }[0], [ x14 ], x4
    st1         { v29.s }[1], [ x14 ], x4
    st1         { v29.s }[2], [ x14 ], x4
    st1         { v29.s }[3], [ x14 ], x4

    // col 8
    st1         { v30.s }[0], [ x15 ], x4
    st1         { v30.s }[1], [ x15 ], x4
    st1         { v30.s }[2], [ x15 ], x4
    st1         { v30.s }[3], [ x15 ], x4

    st1         { v31.s }[0], [ x15 ], x4
    st1         { v31.s }[1], [ x15 ], x4
    st1         { v31.s }[2], [ x15 ], x4
    st1         { v31.s }[3], [ x15 ], x4

    ret
