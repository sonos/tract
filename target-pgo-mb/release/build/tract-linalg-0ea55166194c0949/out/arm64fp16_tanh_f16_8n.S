// vim: ft=arm

// no preservation either for v0-v7 and v16-v31

.text
.align 4


.global _arm64fp16_tanh_f16_8n_0_23_4_pre
_arm64fp16_tanh_f16_8n_0_23_4_pre:

    cmp         x1, #0
    beq         .return

    adr         x2, .coeffs_num
    ld1         { v0.8h }, [x2]
    dup         v5.8h, v0.h[0]              // v5 <- low, broadcasted
    dup         v6.8h, v0.h[1]              // v6 <- high, broadcasted

    cmp         x1, #32
    blt         .loop

.loop4:
    ld1         { v16.8h, v17.8h, v18.8h, v19.8h }, [x0]

    fmax        v16.8h, v16.8h, v5.8h
    fmax        v17.8h, v17.8h, v5.8h
    fmax        v18.8h, v18.8h, v5.8h
    fmax        v19.8h, v19.8h, v5.8h

    fmin        v16.8h, v16.8h, v6.8h
    fmin        v17.8h, v17.8h, v6.8h
    fmin        v18.8h, v18.8h, v6.8h
    fmin        v19.8h, v19.8h, v6.8h       // v16 <- x

    fmul        v20.8h, v16.8h, v16.8h
    fmul        v21.8h, v17.8h, v17.8h
    fmul        v22.8h, v18.8h, v18.8h
    fmul        v23.8h, v19.8h, v19.8h      // v20 <- x2

    dup         v24.8h, v0.h[3]
    fmla        v24.8h, v20.8h, v0.h[2]
    dup         v25.8h, v0.h[3]
    fmla        v25.8h, v21.8h, v0.h[2]
    dup         v26.8h, v0.h[3]
    fmla        v26.8h, v22.8h, v0.h[2]
    dup         v27.8h, v0.h[3]
    fmla        v27.8h, v23.8h, v0.h[2]

    fmul        v16.8h, v16.8h, v24.8h
    fmul        v17.8h, v17.8h, v25.8h
    fmul        v18.8h, v18.8h, v26.8h
    fmul        v19.8h, v19.8h, v27.8h      // v16 <- numerator

    dup         v28.8h, v0.h[5]
    fmla        v28.8h, v20.8h, v0.h[4]
    dup         v29.8h, v0.h[5]
    fmla        v29.8h, v21.8h, v0.h[4]
    dup         v30.8h, v0.h[5]
    fmla        v30.8h, v22.8h, v0.h[4]
    dup         v31.8h, v0.h[5]
    fmla        v31.8h, v23.8h, v0.h[4]

    dup         v24.8h, v0.h[6]
    fmla        v24.8h, v20.8h, v28.8h
    dup         v25.8h, v0.h[6]
    fmla        v25.8h, v21.8h, v29.8h
    dup         v26.8h, v0.h[6]
    fmla        v26.8h, v22.8h, v30.8h
    dup         v27.8h, v0.h[6]
    fmla        v27.8h, v23.8h, v31.8h      // v24 <- denum

    fdiv        v16.8h, v16.8h, v24.8h
    fdiv        v17.8h, v17.8h, v25.8h
    fdiv        v18.8h, v18.8h, v26.8h
    fdiv        v19.8h, v19.8h, v27.8h

    st1         { v16.8h, v17.8h, v18.8h, v19.8h }, [x0], #64

    subs        x1, x1, #32
    cmp         x1, #32
    bge         .loop4

    cmp         x1, #0
    beq         .return

.loop:
    ld1         { v16.8h }, [x0]

    fmax        v16.8h, v16.8h, v5.8h
    fmin        v16.8h, v16.8h, v6.8h       // v16 <- x
    fmul        v20.8h, v16.8h, v16.8h      // v20 <- x2

    dup         v24.8h, v0.h[3]
    fmla        v24.8h, v20.8h, v0.h[2]
    fmul        v16.8h, v16.8h, v24.8h      // v16 <- numerator

    dup         v28.8h, v0.h[5]
    fmla        v28.8h, v20.8h, v0.h[4]
    dup         v24.8h, v0.h[6]
    fmla        v24.8h, v20.8h, v28.8h      // v24 <- denum

    fdiv        v16.8h, v16.8h, v24.8h
    
    st1         { v16.8h }, [x0], #16

    subs        x1, x1, #8
    bne         .loop

.return:
    ret

.coeffs_num:
    .short 50094
    .short 17326
    .short 11594              // alpha
    .short 15359

    .short 7858             // beta
    .short 13984   
    .short 15360
    .short 0                        // padding