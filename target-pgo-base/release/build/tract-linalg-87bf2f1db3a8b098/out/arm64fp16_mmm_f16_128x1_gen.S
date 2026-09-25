// vim: ft=arm

// C tile regs: v16 to v31, no need to preserve

// no preservation either for v0-v7...
// v8..v15 are callee-preserved
// packed A buffering (2x8 values): alternating v0, v1 with v2, v3
// packed B buffering (2x8 values): alternating v4, v5 with v6, v7

.text
.align 4


.global _arm64fp16_mmm_f16_128x1_gen_0_23_4_pre
_arm64fp16_mmm_f16_128x1_gen_0_23_4_pre:

    stp         x20, x21, [sp, #-16]!
    stp         x22, x23, [sp, #-16]!
    stp         x24, x25, [sp, #-16]!

    stp         d8, d9, [sp, #-16]!
    stp         d10, d11, [sp, #-16]!
    stp         d12, d13, [sp, #-16]!
    stp         d14, d15, [sp, #-16]!

// vim: ft=arm

.non_linear:
    sub         x0, x0, 40

.non_linear_loop:
    add         x0, x0, 40
    ldr         x2, [x0]

    mov         x4, #29

    cmp         x2, #29
    csel        x2, x2, x4, lt
    cmp         x2, #0
    csel        x2, x4, x2, lt

    adr         x3, .jmp_table
    add         x3, x3, x2, LSL#2
    br          x3

.jmp_table:

    b   .done

    b   .clear

    b   .load_tile

    b   .scalar_min

    b   .scalar_max

    b   .scalar_add

    b   .scalar_mul

    b   .scalar_sub

    b   .scalar_sub_flipped

    b   .leaky_relu

    b   .per_row_min

    b   .per_row_max

    b   .per_row_add

    b   .per_row_mul

    b   .per_row_sub

    b   .per_row_sub_flipped

    b   .per_col_min

    b   .per_col_max

    b   .per_col_add

    b   .per_col_mul

    b   .per_col_sub

    b   .per_col_sub_flipped

    b   .q_scale

    b   .q_shr

    b   .q_shl

    b   .add_unicast

    b   .add_row_col_products

    b   .store

    b   .add_mat_mul

    b   .unsupported

    add x0, x2, #4000
    b .return

.unsupported:
    mov         x0, #1
    b           .return

.done:
    mov         x0, 0
    b           .return


.add_mat_mul:
    ldr         x2, [x0, #24]       // b
    ldp         x3, x1, [x0, #8]    // k, a

    cmp         x3, #0
    beq         .non_linear_loop
    sub         x3, x3, #1


    ld1         { v8.h }[0], [ x2 ], #2
    ld1         { v0.4s, v1.4s, v2.4s, v3.4s }, [ x1 ], #64
    ld1         { v4.4s, v5.4s, v6.4s, v7.4s }, [ x1 ], #64

    cmp         x3, #0
    beq         .packed_packed_loop_1_last

    cmp         x3, #4
    blt        .packed_packed_loop_1





.p2align 4
.packed_packed_loop_4:
    
        ld1         { v9.4s, v10.4s, v11.4s }, [x1], #48

    fmla        v16.8h, v0.8h, v8.h[0]
    ldr         w8, [x2], #4
    fmla        v17.8h, v1.8h, v8.h[0]
    ldr         d12, [x1], #8
    fmla        v18.8h, v2.8h, v8.h[0]
    ldr         x12, [x1], #8
    fmla        v19.8h, v3.8h, v8.h[0]
    ldr         d13, [x1], #8
    fmla        v20.8h, v4.8h, v8.h[0]
    ldr         x13, [x1], #8
    fmla        v21.8h, v5.8h, v8.h[0]
    ldr         d14, [x1], #8
    fmla        v22.8h, v6.8h, v8.h[0]
    ldr         x14, [x1], #8
    fmla        v23.8h, v7.8h, v8.h[0]
    ldr         d15, [x1], #8
    fmla        v24.8h, v9.8h, v8.h[0]
    ldr         x15, [x1], #8

    ld1         { v0.8h, v1.8h, v2.8h, v3.8h }, [x1], #64
    ins         v8.s[1], w8
    ld1         { v4.8h, v5.8h, v6.8h, v7.8h }, [x1], #64

    fmla        v25.8h, v10.8h, v8.h[0]
    ins         v12.d[1], x12
    fmla        v26.8h, v11.8h, v8.h[0]
    ins         v13.d[1], x13
    fmla        v27.8h, v12.8h, v8.h[0]
    ins         v14.d[1], x14
    fmla        v28.8h, v13.8h, v8.h[0]
    ins         v15.d[1], x15

    ld1         { v9.8h, v10.8h, v11.8h, v12.8h }, [x1], #64

    fmla        v29.8h, v14.8h, v8.h[0]
    ldr         d13, [x1], #8
    fmla        v30.8h, v15.8h, v8.h[0]
    ldr         x13, [x1], #8
    fmla        v31.8h, v0.8h, v8.h[0]
    ldr         d14, [x1], #8

    fmla        v16.8h, v1.8h, v8.h[2]
    ldr         x14, [x1], #8
    fmla        v17.8h, v2.8h, v8.h[2]
    ldr         d15, [x1], #8
    fmla        v18.8h, v3.8h, v8.h[2]
    ldr         x15, [x1], #8
    fmla        v19.8h, v4.8h, v8.h[2]

    ld1         { v0.8h }, [x1], #16

    fmla        v20.8h, v5.8h, v8.h[2]
    ldr         d1, [x1], #8
    fmla        v21.8h, v6.8h, v8.h[2]
    ldr         x10, [x1], #8

    fmla        v22.8h, v7.8h, v8.h[2]

    fmla        v23.8h, v9.8h, v8.h[2]
    ins         v13.d[1], x13
    fmla        v24.8h, v10.8h, v8.h[2]
    ins         v14.d[1], x14
    fmla        v25.8h, v11.8h, v8.h[2]
    ins         v15.d[1], x15

    fmla        v26.8h, v12.8h, v8.h[2]
    prfm        pldl1keep, [x1, #1024]
    fmla        v27.8h, v13.8h, v8.h[2]
    ins         v1.d[1], x10
    fmla        v28.8h, v14.8h, v8.h[2]
    prfm        pldl1keep, [x1, #1088]
    fmla        v29.8h, v15.8h, v8.h[2]
    prfm        pldl1keep, [x1, #1152]
    fmla        v30.8h, v0.8h, v8.h[2]
    prfm        pldl1keep, [x1, #1216]
    fmla        v31.8h, v1.8h, v8.h[2]
    prfm        pldl1keep, [x2, #256]

    ld1         { v0.4s, v1.4s, v2.4s, v3.4s }, [x1], #64
    ins         v8.h[0], v8.h[3]
    ld1         { v4.4s, v5.4s, v6.4s, v7.4s }, [x1], #64



    
        ld1         { v9.4s, v10.4s, v11.4s }, [x1], #48

    fmla        v16.8h, v0.8h, v8.h[0]
    ldr         w8, [x2], #4
    fmla        v17.8h, v1.8h, v8.h[0]
    ldr         d12, [x1], #8
    fmla        v18.8h, v2.8h, v8.h[0]
    ldr         x12, [x1], #8
    fmla        v19.8h, v3.8h, v8.h[0]
    ldr         d13, [x1], #8
    fmla        v20.8h, v4.8h, v8.h[0]
    ldr         x13, [x1], #8
    fmla        v21.8h, v5.8h, v8.h[0]
    ldr         d14, [x1], #8
    fmla        v22.8h, v6.8h, v8.h[0]
    ldr         x14, [x1], #8
    fmla        v23.8h, v7.8h, v8.h[0]
    ldr         d15, [x1], #8
    fmla        v24.8h, v9.8h, v8.h[0]
    ldr         x15, [x1], #8

    ld1         { v0.8h, v1.8h, v2.8h, v3.8h }, [x1], #64
    ins         v8.s[1], w8
    ld1         { v4.8h, v5.8h, v6.8h, v7.8h }, [x1], #64

    fmla        v25.8h, v10.8h, v8.h[0]
    ins         v12.d[1], x12
    fmla        v26.8h, v11.8h, v8.h[0]
    ins         v13.d[1], x13
    fmla        v27.8h, v12.8h, v8.h[0]
    ins         v14.d[1], x14
    fmla        v28.8h, v13.8h, v8.h[0]
    ins         v15.d[1], x15

    ld1         { v9.8h, v10.8h, v11.8h, v12.8h }, [x1], #64

    fmla        v29.8h, v14.8h, v8.h[0]
    ldr         d13, [x1], #8
    fmla        v30.8h, v15.8h, v8.h[0]
    ldr         x13, [x1], #8
    fmla        v31.8h, v0.8h, v8.h[0]
    ldr         d14, [x1], #8

    fmla        v16.8h, v1.8h, v8.h[2]
    ldr         x14, [x1], #8
    fmla        v17.8h, v2.8h, v8.h[2]
    ldr         d15, [x1], #8
    fmla        v18.8h, v3.8h, v8.h[2]
    ldr         x15, [x1], #8
    fmla        v19.8h, v4.8h, v8.h[2]

    ld1         { v0.8h }, [x1], #16

    fmla        v20.8h, v5.8h, v8.h[2]
    ldr         d1, [x1], #8
    fmla        v21.8h, v6.8h, v8.h[2]
    ldr         x10, [x1], #8

    fmla        v22.8h, v7.8h, v8.h[2]

    fmla        v23.8h, v9.8h, v8.h[2]
    ins         v13.d[1], x13
    fmla        v24.8h, v10.8h, v8.h[2]
    ins         v14.d[1], x14
    fmla        v25.8h, v11.8h, v8.h[2]
    ins         v15.d[1], x15

    fmla        v26.8h, v12.8h, v8.h[2]
    prfm        pldl1keep, [x1, #1024]
    fmla        v27.8h, v13.8h, v8.h[2]
    ins         v1.d[1], x10
    fmla        v28.8h, v14.8h, v8.h[2]
    prfm        pldl1keep, [x1, #1088]
    fmla        v29.8h, v15.8h, v8.h[2]
    prfm        pldl1keep, [x1, #1152]
    fmla        v30.8h, v0.8h, v8.h[2]
    prfm        pldl1keep, [x1, #1216]
    fmla        v31.8h, v1.8h, v8.h[2]
    prfm        pldl1keep, [x2, #256]

    ld1         { v0.4s, v1.4s, v2.4s, v3.4s }, [x1], #64
    ins         v8.h[0], v8.h[3]
    ld1         { v4.4s, v5.4s, v6.4s, v7.4s }, [x1], #64




    sub         x3, x3, #4
    cmp         x3, #4
    bge         .packed_packed_loop_4

    cmp         x3, #0
    beq         .packed_packed_loop_1_last

.p2align 4
.packed_packed_loop_1:
    
        ld1         { v9.8h, v10.8h, v11.8h, v12.8h }, [x1], #64
    ld1         { v13.8h, v14.8h, v15.8h }, [x1], #48

    fmla        v16.8h, v0.8h, v8.h[0]
    fmla        v17.8h, v1.8h, v8.h[0]
    fmla        v18.8h, v2.8h, v8.h[0]
    fmla        v19.8h, v3.8h, v8.h[0]
    fmla        v20.8h, v4.8h, v8.h[0]
    fmla        v21.8h, v5.8h, v8.h[0]
    fmla        v22.8h, v6.8h, v8.h[0]
    fmla        v23.8h, v7.8h, v8.h[0]
    fmla        v24.8h, v9.8h, v8.h[0]
    ld1         { v9.8h }, [ x1 ], #16
    ld1         { v0.8h, v1.8h, v2.8h, v3.8h }, [x1], #64
    ld1         { v4.8h, v5.8h, v6.8h, v7.8h }, [x1], #64
    fmla        v25.8h, v10.8h, v8.h[0]
    fmla        v26.8h, v11.8h, v8.h[0]
    fmla        v27.8h, v12.8h, v8.h[0]
    fmla        v28.8h, v13.8h, v8.h[0]
    fmla        v29.8h, v14.8h, v8.h[0]
    fmla        v30.8h, v15.8h, v8.h[0]

    fmla        v31.8h, v9.8h, v8.h[0]

    ld1         { v8.h }[0], [ x2 ], #2

    prfm        pldl1keep, [x1, #1024]
    prfm        pldl1keep, [x1, #1088]
    prfm        pldl1keep, [x1, #1152]
    prfm        pldl1keep, [x1, #1216]
    prfm        pldl1keep, [x2, #256]



    subs        x3, x3, #1
    bne         .packed_packed_loop_1

// last loop can't read beyond actual input as it's likely not packed and padded
.packed_packed_loop_1_last:
    ld1         { v9.8h, v10.8h, v11.8h, v12.8h }, [x1], #64
    ld1         { v13.8h, v14.8h, v15.8h }, [x1], #48

    fmla        v16.8h, v0.8h, v8.h[0]
    fmla        v17.8h, v1.8h, v8.h[0]
    ld1         { v0.8h }, [ x1 ]
    fmla        v18.8h, v2.8h, v8.h[0]
    fmla        v19.8h, v3.8h, v8.h[0]
    fmla        v20.8h, v4.8h, v8.h[0]
    fmla        v21.8h, v5.8h, v8.h[0]
    fmla        v22.8h, v6.8h, v8.h[0]
    fmla        v23.8h, v7.8h, v8.h[0]

    fmla        v24.8h, v9.8h, v8.h[0]
    fmla        v25.8h, v10.8h, v8.h[0]
    fmla        v26.8h, v11.8h, v8.h[0]
    fmla        v27.8h, v12.8h, v8.h[0]
    fmla        v28.8h, v13.8h, v8.h[0]
    fmla        v29.8h, v14.8h, v8.h[0]
    fmla        v30.8h, v15.8h, v8.h[0]
    fmla        v31.8h, v0.8h, v8.h[0]

    b           .non_linear_loop

// vim: ft=arm




.scalar_min:
    add         x2, x0, #8
    ld1         {v0.h}[0], [ x2 ]
    dup         v0.8h, v0.h[0]
    
        
            fmin       v16.8h, v0.8h, v16.8h
        
            fmin       v17.8h, v0.8h, v17.8h
        
            fmin       v18.8h, v0.8h, v18.8h
        
            fmin       v19.8h, v0.8h, v19.8h
        
            fmin       v20.8h, v0.8h, v20.8h
        
            fmin       v21.8h, v0.8h, v21.8h
        
            fmin       v22.8h, v0.8h, v22.8h
        
            fmin       v23.8h, v0.8h, v23.8h
        
            fmin       v24.8h, v0.8h, v24.8h
        
            fmin       v25.8h, v0.8h, v25.8h
        
            fmin       v26.8h, v0.8h, v26.8h
        
            fmin       v27.8h, v0.8h, v27.8h
        
            fmin       v28.8h, v0.8h, v28.8h
        
            fmin       v29.8h, v0.8h, v29.8h
        
            fmin       v30.8h, v0.8h, v30.8h
        
            fmin       v31.8h, v0.8h, v31.8h
        
    

    b           .non_linear_loop


.scalar_max:
    add         x2, x0, #8
    ld1         {v0.h}[0], [ x2 ]
    dup         v0.8h, v0.h[0]
    
        
            fmax       v16.8h, v0.8h, v16.8h
        
            fmax       v17.8h, v0.8h, v17.8h
        
            fmax       v18.8h, v0.8h, v18.8h
        
            fmax       v19.8h, v0.8h, v19.8h
        
            fmax       v20.8h, v0.8h, v20.8h
        
            fmax       v21.8h, v0.8h, v21.8h
        
            fmax       v22.8h, v0.8h, v22.8h
        
            fmax       v23.8h, v0.8h, v23.8h
        
            fmax       v24.8h, v0.8h, v24.8h
        
            fmax       v25.8h, v0.8h, v25.8h
        
            fmax       v26.8h, v0.8h, v26.8h
        
            fmax       v27.8h, v0.8h, v27.8h
        
            fmax       v28.8h, v0.8h, v28.8h
        
            fmax       v29.8h, v0.8h, v29.8h
        
            fmax       v30.8h, v0.8h, v30.8h
        
            fmax       v31.8h, v0.8h, v31.8h
        
    

    b           .non_linear_loop


.scalar_mul:
    add         x2, x0, #8
    ld1         {v0.h}[0], [ x2 ]
    dup         v0.8h, v0.h[0]
    
        
            fmul       v16.8h, v0.8h, v16.8h
        
            fmul       v17.8h, v0.8h, v17.8h
        
            fmul       v18.8h, v0.8h, v18.8h
        
            fmul       v19.8h, v0.8h, v19.8h
        
            fmul       v20.8h, v0.8h, v20.8h
        
            fmul       v21.8h, v0.8h, v21.8h
        
            fmul       v22.8h, v0.8h, v22.8h
        
            fmul       v23.8h, v0.8h, v23.8h
        
            fmul       v24.8h, v0.8h, v24.8h
        
            fmul       v25.8h, v0.8h, v25.8h
        
            fmul       v26.8h, v0.8h, v26.8h
        
            fmul       v27.8h, v0.8h, v27.8h
        
            fmul       v28.8h, v0.8h, v28.8h
        
            fmul       v29.8h, v0.8h, v29.8h
        
            fmul       v30.8h, v0.8h, v30.8h
        
            fmul       v31.8h, v0.8h, v31.8h
        
    

    b           .non_linear_loop


.scalar_add:
    add         x2, x0, #8
    ld1         {v0.h}[0], [ x2 ]
    dup         v0.8h, v0.h[0]
    
        
            fadd       v16.8h, v0.8h, v16.8h
        
            fadd       v17.8h, v0.8h, v17.8h
        
            fadd       v18.8h, v0.8h, v18.8h
        
            fadd       v19.8h, v0.8h, v19.8h
        
            fadd       v20.8h, v0.8h, v20.8h
        
            fadd       v21.8h, v0.8h, v21.8h
        
            fadd       v22.8h, v0.8h, v22.8h
        
            fadd       v23.8h, v0.8h, v23.8h
        
            fadd       v24.8h, v0.8h, v24.8h
        
            fadd       v25.8h, v0.8h, v25.8h
        
            fadd       v26.8h, v0.8h, v26.8h
        
            fadd       v27.8h, v0.8h, v27.8h
        
            fadd       v28.8h, v0.8h, v28.8h
        
            fadd       v29.8h, v0.8h, v29.8h
        
            fadd       v30.8h, v0.8h, v30.8h
        
            fadd       v31.8h, v0.8h, v31.8h
        
    

    b           .non_linear_loop


.scalar_sub:
    add         x2, x0, #8
    ld1         {v0.h}[0], [ x2 ]
    dup         v0.8h, v0.h[0]
    
        
            fsub       v16.8h, v0.8h, v16.8h
        
            fsub       v17.8h, v0.8h, v17.8h
        
            fsub       v18.8h, v0.8h, v18.8h
        
            fsub       v19.8h, v0.8h, v19.8h
        
            fsub       v20.8h, v0.8h, v20.8h
        
            fsub       v21.8h, v0.8h, v21.8h
        
            fsub       v22.8h, v0.8h, v22.8h
        
            fsub       v23.8h, v0.8h, v23.8h
        
            fsub       v24.8h, v0.8h, v24.8h
        
            fsub       v25.8h, v0.8h, v25.8h
        
            fsub       v26.8h, v0.8h, v26.8h
        
            fsub       v27.8h, v0.8h, v27.8h
        
            fsub       v28.8h, v0.8h, v28.8h
        
            fsub       v29.8h, v0.8h, v29.8h
        
            fsub       v30.8h, v0.8h, v30.8h
        
            fsub       v31.8h, v0.8h, v31.8h
        
    

    b           .non_linear_loop


.scalar_sub_flipped:
    add         x2, x0, #8
    ld1         {v0.h}[0], [ x2 ]
    dup         v0.8h, v0.h[0]
    
        
            fsub       v16.8h, v16.8h, v0.8h
        
            fsub       v17.8h, v17.8h, v0.8h
        
            fsub       v18.8h, v18.8h, v0.8h
        
            fsub       v19.8h, v19.8h, v0.8h
        
            fsub       v20.8h, v20.8h, v0.8h
        
            fsub       v21.8h, v21.8h, v0.8h
        
            fsub       v22.8h, v22.8h, v0.8h
        
            fsub       v23.8h, v23.8h, v0.8h
        
            fsub       v24.8h, v24.8h, v0.8h
        
            fsub       v25.8h, v25.8h, v0.8h
        
            fsub       v26.8h, v26.8h, v0.8h
        
            fsub       v27.8h, v27.8h, v0.8h
        
            fsub       v28.8h, v28.8h, v0.8h
        
            fsub       v29.8h, v29.8h, v0.8h
        
            fsub       v30.8h, v30.8h, v0.8h
        
            fsub       v31.8h, v31.8h, v0.8h
        
    

    b           .non_linear_loop


.clear:

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

    b .non_linear_loop

.leaky_relu:
    add         x2, x0, #8
    ld1         {v4.s}[0], [ x2 ]
    dup         v4.8h, v4.h[0]

    // bsl cond/dst, then, else
    // fcmge dst, src, #0.0
    
        fmul  v0.8h, v16.8h, v4.8h
        fcmge v1.8h, v16.8h, #0.0
        bsl   v1.16b, v16.16b, v0.16b
        and   v16.16b, v1.16b, v1.16b
    
        fmul  v0.8h, v17.8h, v4.8h
        fcmge v1.8h, v17.8h, #0.0
        bsl   v1.16b, v17.16b, v0.16b
        and   v17.16b, v1.16b, v1.16b
    
        fmul  v0.8h, v18.8h, v4.8h
        fcmge v1.8h, v18.8h, #0.0
        bsl   v1.16b, v18.16b, v0.16b
        and   v18.16b, v1.16b, v1.16b
    
        fmul  v0.8h, v19.8h, v4.8h
        fcmge v1.8h, v19.8h, #0.0
        bsl   v1.16b, v19.16b, v0.16b
        and   v19.16b, v1.16b, v1.16b
    
        fmul  v0.8h, v20.8h, v4.8h
        fcmge v1.8h, v20.8h, #0.0
        bsl   v1.16b, v20.16b, v0.16b
        and   v20.16b, v1.16b, v1.16b
    
        fmul  v0.8h, v21.8h, v4.8h
        fcmge v1.8h, v21.8h, #0.0
        bsl   v1.16b, v21.16b, v0.16b
        and   v21.16b, v1.16b, v1.16b
    
        fmul  v0.8h, v22.8h, v4.8h
        fcmge v1.8h, v22.8h, #0.0
        bsl   v1.16b, v22.16b, v0.16b
        and   v22.16b, v1.16b, v1.16b
    
        fmul  v0.8h, v23.8h, v4.8h
        fcmge v1.8h, v23.8h, #0.0
        bsl   v1.16b, v23.16b, v0.16b
        and   v23.16b, v1.16b, v1.16b
    
        fmul  v0.8h, v24.8h, v4.8h
        fcmge v1.8h, v24.8h, #0.0
        bsl   v1.16b, v24.16b, v0.16b
        and   v24.16b, v1.16b, v1.16b
    
        fmul  v0.8h, v25.8h, v4.8h
        fcmge v1.8h, v25.8h, #0.0
        bsl   v1.16b, v25.16b, v0.16b
        and   v25.16b, v1.16b, v1.16b
    
        fmul  v0.8h, v26.8h, v4.8h
        fcmge v1.8h, v26.8h, #0.0
        bsl   v1.16b, v26.16b, v0.16b
        and   v26.16b, v1.16b, v1.16b
    
        fmul  v0.8h, v27.8h, v4.8h
        fcmge v1.8h, v27.8h, #0.0
        bsl   v1.16b, v27.16b, v0.16b
        and   v27.16b, v1.16b, v1.16b
    
        fmul  v0.8h, v28.8h, v4.8h
        fcmge v1.8h, v28.8h, #0.0
        bsl   v1.16b, v28.16b, v0.16b
        and   v28.16b, v1.16b, v1.16b
    
        fmul  v0.8h, v29.8h, v4.8h
        fcmge v1.8h, v29.8h, #0.0
        bsl   v1.16b, v29.16b, v0.16b
        and   v29.16b, v1.16b, v1.16b
    
        fmul  v0.8h, v30.8h, v4.8h
        fcmge v1.8h, v30.8h, #0.0
        bsl   v1.16b, v30.16b, v0.16b
        and   v30.16b, v1.16b, v1.16b
    
        fmul  v0.8h, v31.8h, v4.8h
        fcmge v1.8h, v31.8h, #0.0
        bsl   v1.16b, v31.16b, v0.16b
        and   v31.16b, v1.16b, v1.16b
    

    b .non_linear_loop


.q_scale:
.q_shl:
.q_shr:
    b .unsupported
// vim: ft=arm




.per_row_min:
    ldr         x2, [x0, #8]





    ldr         q0, [ x2 ], #16

    ldr         q1, [ x2 ], #16

    ldr         q2, [ x2 ], #16

    ldr         q3, [ x2 ], #16

    ldr         q4, [ x2 ], #16

    ldr         q5, [ x2 ], #16

    ldr         q6, [ x2 ], #16

    ldr         q7, [ x2 ], #16

    ldr         q8, [ x2 ], #16

    ldr         q9, [ x2 ], #16

    ldr         q10, [ x2 ], #16

    ldr         q11, [ x2 ], #16

    ldr         q12, [ x2 ], #16

    ldr         q13, [ x2 ], #16

    ldr         q14, [ x2 ], #16

    ldr         q15, [ x2 ], #16



    
        
        fmin v16.8h, v0.8h, v16.8h
    
        
        fmin v17.8h, v1.8h, v17.8h
    
        
        fmin v18.8h, v2.8h, v18.8h
    
        
        fmin v19.8h, v3.8h, v19.8h
    
        
        fmin v20.8h, v4.8h, v20.8h
    
        
        fmin v21.8h, v5.8h, v21.8h
    
        
        fmin v22.8h, v6.8h, v22.8h
    
        
        fmin v23.8h, v7.8h, v23.8h
    
        
        fmin v24.8h, v8.8h, v24.8h
    
        
        fmin v25.8h, v9.8h, v25.8h
    
        
        fmin v26.8h, v10.8h, v26.8h
    
        
        fmin v27.8h, v11.8h, v27.8h
    
        
        fmin v28.8h, v12.8h, v28.8h
    
        
        fmin v29.8h, v13.8h, v29.8h
    
        
        fmin v30.8h, v14.8h, v30.8h
    
        
        fmin v31.8h, v15.8h, v31.8h
    


b           .non_linear_loop


.per_row_max:
    ldr         x2, [x0, #8]





    ldr         q0, [ x2 ], #16

    ldr         q1, [ x2 ], #16

    ldr         q2, [ x2 ], #16

    ldr         q3, [ x2 ], #16

    ldr         q4, [ x2 ], #16

    ldr         q5, [ x2 ], #16

    ldr         q6, [ x2 ], #16

    ldr         q7, [ x2 ], #16

    ldr         q8, [ x2 ], #16

    ldr         q9, [ x2 ], #16

    ldr         q10, [ x2 ], #16

    ldr         q11, [ x2 ], #16

    ldr         q12, [ x2 ], #16

    ldr         q13, [ x2 ], #16

    ldr         q14, [ x2 ], #16

    ldr         q15, [ x2 ], #16



    
        
        fmax v16.8h, v0.8h, v16.8h
    
        
        fmax v17.8h, v1.8h, v17.8h
    
        
        fmax v18.8h, v2.8h, v18.8h
    
        
        fmax v19.8h, v3.8h, v19.8h
    
        
        fmax v20.8h, v4.8h, v20.8h
    
        
        fmax v21.8h, v5.8h, v21.8h
    
        
        fmax v22.8h, v6.8h, v22.8h
    
        
        fmax v23.8h, v7.8h, v23.8h
    
        
        fmax v24.8h, v8.8h, v24.8h
    
        
        fmax v25.8h, v9.8h, v25.8h
    
        
        fmax v26.8h, v10.8h, v26.8h
    
        
        fmax v27.8h, v11.8h, v27.8h
    
        
        fmax v28.8h, v12.8h, v28.8h
    
        
        fmax v29.8h, v13.8h, v29.8h
    
        
        fmax v30.8h, v14.8h, v30.8h
    
        
        fmax v31.8h, v15.8h, v31.8h
    


b           .non_linear_loop


.per_row_mul:
    ldr         x2, [x0, #8]





    ldr         q0, [ x2 ], #16

    ldr         q1, [ x2 ], #16

    ldr         q2, [ x2 ], #16

    ldr         q3, [ x2 ], #16

    ldr         q4, [ x2 ], #16

    ldr         q5, [ x2 ], #16

    ldr         q6, [ x2 ], #16

    ldr         q7, [ x2 ], #16

    ldr         q8, [ x2 ], #16

    ldr         q9, [ x2 ], #16

    ldr         q10, [ x2 ], #16

    ldr         q11, [ x2 ], #16

    ldr         q12, [ x2 ], #16

    ldr         q13, [ x2 ], #16

    ldr         q14, [ x2 ], #16

    ldr         q15, [ x2 ], #16



    
        
        fmul v16.8h, v0.8h, v16.8h
    
        
        fmul v17.8h, v1.8h, v17.8h
    
        
        fmul v18.8h, v2.8h, v18.8h
    
        
        fmul v19.8h, v3.8h, v19.8h
    
        
        fmul v20.8h, v4.8h, v20.8h
    
        
        fmul v21.8h, v5.8h, v21.8h
    
        
        fmul v22.8h, v6.8h, v22.8h
    
        
        fmul v23.8h, v7.8h, v23.8h
    
        
        fmul v24.8h, v8.8h, v24.8h
    
        
        fmul v25.8h, v9.8h, v25.8h
    
        
        fmul v26.8h, v10.8h, v26.8h
    
        
        fmul v27.8h, v11.8h, v27.8h
    
        
        fmul v28.8h, v12.8h, v28.8h
    
        
        fmul v29.8h, v13.8h, v29.8h
    
        
        fmul v30.8h, v14.8h, v30.8h
    
        
        fmul v31.8h, v15.8h, v31.8h
    


b           .non_linear_loop


.per_row_add:
    ldr         x2, [x0, #8]





    ldr         q0, [ x2 ], #16

    ldr         q1, [ x2 ], #16

    ldr         q2, [ x2 ], #16

    ldr         q3, [ x2 ], #16

    ldr         q4, [ x2 ], #16

    ldr         q5, [ x2 ], #16

    ldr         q6, [ x2 ], #16

    ldr         q7, [ x2 ], #16

    ldr         q8, [ x2 ], #16

    ldr         q9, [ x2 ], #16

    ldr         q10, [ x2 ], #16

    ldr         q11, [ x2 ], #16

    ldr         q12, [ x2 ], #16

    ldr         q13, [ x2 ], #16

    ldr         q14, [ x2 ], #16

    ldr         q15, [ x2 ], #16



    
        
        fadd v16.8h, v0.8h, v16.8h
    
        
        fadd v17.8h, v1.8h, v17.8h
    
        
        fadd v18.8h, v2.8h, v18.8h
    
        
        fadd v19.8h, v3.8h, v19.8h
    
        
        fadd v20.8h, v4.8h, v20.8h
    
        
        fadd v21.8h, v5.8h, v21.8h
    
        
        fadd v22.8h, v6.8h, v22.8h
    
        
        fadd v23.8h, v7.8h, v23.8h
    
        
        fadd v24.8h, v8.8h, v24.8h
    
        
        fadd v25.8h, v9.8h, v25.8h
    
        
        fadd v26.8h, v10.8h, v26.8h
    
        
        fadd v27.8h, v11.8h, v27.8h
    
        
        fadd v28.8h, v12.8h, v28.8h
    
        
        fadd v29.8h, v13.8h, v29.8h
    
        
        fadd v30.8h, v14.8h, v30.8h
    
        
        fadd v31.8h, v15.8h, v31.8h
    


b           .non_linear_loop


.per_row_sub:
    ldr         x2, [x0, #8]





    ldr         q0, [ x2 ], #16

    ldr         q1, [ x2 ], #16

    ldr         q2, [ x2 ], #16

    ldr         q3, [ x2 ], #16

    ldr         q4, [ x2 ], #16

    ldr         q5, [ x2 ], #16

    ldr         q6, [ x2 ], #16

    ldr         q7, [ x2 ], #16

    ldr         q8, [ x2 ], #16

    ldr         q9, [ x2 ], #16

    ldr         q10, [ x2 ], #16

    ldr         q11, [ x2 ], #16

    ldr         q12, [ x2 ], #16

    ldr         q13, [ x2 ], #16

    ldr         q14, [ x2 ], #16

    ldr         q15, [ x2 ], #16



    
        
        fsub v16.8h, v0.8h, v16.8h
    
        
        fsub v17.8h, v1.8h, v17.8h
    
        
        fsub v18.8h, v2.8h, v18.8h
    
        
        fsub v19.8h, v3.8h, v19.8h
    
        
        fsub v20.8h, v4.8h, v20.8h
    
        
        fsub v21.8h, v5.8h, v21.8h
    
        
        fsub v22.8h, v6.8h, v22.8h
    
        
        fsub v23.8h, v7.8h, v23.8h
    
        
        fsub v24.8h, v8.8h, v24.8h
    
        
        fsub v25.8h, v9.8h, v25.8h
    
        
        fsub v26.8h, v10.8h, v26.8h
    
        
        fsub v27.8h, v11.8h, v27.8h
    
        
        fsub v28.8h, v12.8h, v28.8h
    
        
        fsub v29.8h, v13.8h, v29.8h
    
        
        fsub v30.8h, v14.8h, v30.8h
    
        
        fsub v31.8h, v15.8h, v31.8h
    


b           .non_linear_loop


.per_row_sub_flipped:
    ldr         x2, [x0, #8]





    ldr         q0, [ x2 ], #16

    ldr         q1, [ x2 ], #16

    ldr         q2, [ x2 ], #16

    ldr         q3, [ x2 ], #16

    ldr         q4, [ x2 ], #16

    ldr         q5, [ x2 ], #16

    ldr         q6, [ x2 ], #16

    ldr         q7, [ x2 ], #16

    ldr         q8, [ x2 ], #16

    ldr         q9, [ x2 ], #16

    ldr         q10, [ x2 ], #16

    ldr         q11, [ x2 ], #16

    ldr         q12, [ x2 ], #16

    ldr         q13, [ x2 ], #16

    ldr         q14, [ x2 ], #16

    ldr         q15, [ x2 ], #16



    
        
        fsub v16.8h, v16.8h, v0.8h
    
        
        fsub v17.8h, v17.8h, v1.8h
    
        
        fsub v18.8h, v18.8h, v2.8h
    
        
        fsub v19.8h, v19.8h, v3.8h
    
        
        fsub v20.8h, v20.8h, v4.8h
    
        
        fsub v21.8h, v21.8h, v5.8h
    
        
        fsub v22.8h, v22.8h, v6.8h
    
        
        fsub v23.8h, v23.8h, v7.8h
    
        
        fsub v24.8h, v24.8h, v8.8h
    
        
        fsub v25.8h, v25.8h, v9.8h
    
        
        fsub v26.8h, v26.8h, v10.8h
    
        
        fsub v27.8h, v27.8h, v11.8h
    
        
        fsub v28.8h, v28.8h, v12.8h
    
        
        fsub v29.8h, v29.8h, v13.8h
    
        
        fsub v30.8h, v30.8h, v14.8h
    
        
        fsub v31.8h, v31.8h, v15.8h
    


b           .non_linear_loop

// vim: ft=arm




.per_col_min:
    ldr         x2, [x0, #8]







        ld1         {v0.h}[0], [ x2 ]


// mr:128 0 1


    dup v3.8h, v0.h[0]
    
        
        
            fmin v16.8h, v3.8h, v16.8h
        
    
        
        
            fmin v17.8h, v3.8h, v17.8h
        
    
        
        
            fmin v18.8h, v3.8h, v18.8h
        
    
        
        
            fmin v19.8h, v3.8h, v19.8h
        
    
        
        
            fmin v20.8h, v3.8h, v20.8h
        
    
        
        
            fmin v21.8h, v3.8h, v21.8h
        
    
        
        
            fmin v22.8h, v3.8h, v22.8h
        
    
        
        
            fmin v23.8h, v3.8h, v23.8h
        
    
        
        
            fmin v24.8h, v3.8h, v24.8h
        
    
        
        
            fmin v25.8h, v3.8h, v25.8h
        
    
        
        
            fmin v26.8h, v3.8h, v26.8h
        
    
        
        
            fmin v27.8h, v3.8h, v27.8h
        
    
        
        
            fmin v28.8h, v3.8h, v28.8h
        
    
        
        
            fmin v29.8h, v3.8h, v29.8h
        
    
        
        
            fmin v30.8h, v3.8h, v30.8h
        
    
        
        
            fmin v31.8h, v3.8h, v31.8h
        
    


b           .non_linear_loop


.per_col_max:
    ldr         x2, [x0, #8]







        ld1         {v0.h}[0], [ x2 ]


// mr:128 0 1


    dup v3.8h, v0.h[0]
    
        
        
            fmax v16.8h, v3.8h, v16.8h
        
    
        
        
            fmax v17.8h, v3.8h, v17.8h
        
    
        
        
            fmax v18.8h, v3.8h, v18.8h
        
    
        
        
            fmax v19.8h, v3.8h, v19.8h
        
    
        
        
            fmax v20.8h, v3.8h, v20.8h
        
    
        
        
            fmax v21.8h, v3.8h, v21.8h
        
    
        
        
            fmax v22.8h, v3.8h, v22.8h
        
    
        
        
            fmax v23.8h, v3.8h, v23.8h
        
    
        
        
            fmax v24.8h, v3.8h, v24.8h
        
    
        
        
            fmax v25.8h, v3.8h, v25.8h
        
    
        
        
            fmax v26.8h, v3.8h, v26.8h
        
    
        
        
            fmax v27.8h, v3.8h, v27.8h
        
    
        
        
            fmax v28.8h, v3.8h, v28.8h
        
    
        
        
            fmax v29.8h, v3.8h, v29.8h
        
    
        
        
            fmax v30.8h, v3.8h, v30.8h
        
    
        
        
            fmax v31.8h, v3.8h, v31.8h
        
    


b           .non_linear_loop


.per_col_mul:
    ldr         x2, [x0, #8]







        ld1         {v0.h}[0], [ x2 ]


// mr:128 0 1


    dup v3.8h, v0.h[0]
    
        
        
            fmul v16.8h, v3.8h, v16.8h
        
    
        
        
            fmul v17.8h, v3.8h, v17.8h
        
    
        
        
            fmul v18.8h, v3.8h, v18.8h
        
    
        
        
            fmul v19.8h, v3.8h, v19.8h
        
    
        
        
            fmul v20.8h, v3.8h, v20.8h
        
    
        
        
            fmul v21.8h, v3.8h, v21.8h
        
    
        
        
            fmul v22.8h, v3.8h, v22.8h
        
    
        
        
            fmul v23.8h, v3.8h, v23.8h
        
    
        
        
            fmul v24.8h, v3.8h, v24.8h
        
    
        
        
            fmul v25.8h, v3.8h, v25.8h
        
    
        
        
            fmul v26.8h, v3.8h, v26.8h
        
    
        
        
            fmul v27.8h, v3.8h, v27.8h
        
    
        
        
            fmul v28.8h, v3.8h, v28.8h
        
    
        
        
            fmul v29.8h, v3.8h, v29.8h
        
    
        
        
            fmul v30.8h, v3.8h, v30.8h
        
    
        
        
            fmul v31.8h, v3.8h, v31.8h
        
    


b           .non_linear_loop


.per_col_add:
    ldr         x2, [x0, #8]







        ld1         {v0.h}[0], [ x2 ]


// mr:128 0 1


    dup v3.8h, v0.h[0]
    
        
        
            fadd v16.8h, v3.8h, v16.8h
        
    
        
        
            fadd v17.8h, v3.8h, v17.8h
        
    
        
        
            fadd v18.8h, v3.8h, v18.8h
        
    
        
        
            fadd v19.8h, v3.8h, v19.8h
        
    
        
        
            fadd v20.8h, v3.8h, v20.8h
        
    
        
        
            fadd v21.8h, v3.8h, v21.8h
        
    
        
        
            fadd v22.8h, v3.8h, v22.8h
        
    
        
        
            fadd v23.8h, v3.8h, v23.8h
        
    
        
        
            fadd v24.8h, v3.8h, v24.8h
        
    
        
        
            fadd v25.8h, v3.8h, v25.8h
        
    
        
        
            fadd v26.8h, v3.8h, v26.8h
        
    
        
        
            fadd v27.8h, v3.8h, v27.8h
        
    
        
        
            fadd v28.8h, v3.8h, v28.8h
        
    
        
        
            fadd v29.8h, v3.8h, v29.8h
        
    
        
        
            fadd v30.8h, v3.8h, v30.8h
        
    
        
        
            fadd v31.8h, v3.8h, v31.8h
        
    


b           .non_linear_loop


.per_col_sub:
    ldr         x2, [x0, #8]







        ld1         {v0.h}[0], [ x2 ]


// mr:128 0 1


    dup v3.8h, v0.h[0]
    
        
        
            fsub v16.8h, v3.8h, v16.8h
        
    
        
        
            fsub v17.8h, v3.8h, v17.8h
        
    
        
        
            fsub v18.8h, v3.8h, v18.8h
        
    
        
        
            fsub v19.8h, v3.8h, v19.8h
        
    
        
        
            fsub v20.8h, v3.8h, v20.8h
        
    
        
        
            fsub v21.8h, v3.8h, v21.8h
        
    
        
        
            fsub v22.8h, v3.8h, v22.8h
        
    
        
        
            fsub v23.8h, v3.8h, v23.8h
        
    
        
        
            fsub v24.8h, v3.8h, v24.8h
        
    
        
        
            fsub v25.8h, v3.8h, v25.8h
        
    
        
        
            fsub v26.8h, v3.8h, v26.8h
        
    
        
        
            fsub v27.8h, v3.8h, v27.8h
        
    
        
        
            fsub v28.8h, v3.8h, v28.8h
        
    
        
        
            fsub v29.8h, v3.8h, v29.8h
        
    
        
        
            fsub v30.8h, v3.8h, v30.8h
        
    
        
        
            fsub v31.8h, v3.8h, v31.8h
        
    


b           .non_linear_loop


.per_col_sub_flipped:
    ldr         x2, [x0, #8]







        ld1         {v0.h}[0], [ x2 ]


// mr:128 0 1


    dup v3.8h, v0.h[0]
    
        
        
            fsub v16.8h, v16.8h, v3.8h
        
    
        
        
            fsub v17.8h, v17.8h, v3.8h
        
    
        
        
            fsub v18.8h, v18.8h, v3.8h
        
    
        
        
            fsub v19.8h, v19.8h, v3.8h
        
    
        
        
            fsub v20.8h, v20.8h, v3.8h
        
    
        
        
            fsub v21.8h, v21.8h, v3.8h
        
    
        
        
            fsub v22.8h, v22.8h, v3.8h
        
    
        
        
            fsub v23.8h, v23.8h, v3.8h
        
    
        
        
            fsub v24.8h, v24.8h, v3.8h
        
    
        
        
            fsub v25.8h, v25.8h, v3.8h
        
    
        
        
            fsub v26.8h, v26.8h, v3.8h
        
    
        
        
            fsub v27.8h, v27.8h, v3.8h
        
    
        
        
            fsub v28.8h, v28.8h, v3.8h
        
    
        
        
            fsub v29.8h, v29.8h, v3.8h
        
    
        
        
            fsub v30.8h, v30.8h, v3.8h
        
    
        
        
            fsub v31.8h, v31.8h, v3.8h
        
    


b           .non_linear_loop

// vim: ft=arm

.load_tile:
    ldr         x2, [ x0, #8 ]
    
        ld1         { v16.4s }, [ x2 ], #16
    
        ld1         { v17.4s }, [ x2 ], #16
    
        ld1         { v18.4s }, [ x2 ], #16
    
        ld1         { v19.4s }, [ x2 ], #16
    
        ld1         { v20.4s }, [ x2 ], #16
    
        ld1         { v21.4s }, [ x2 ], #16
    
        ld1         { v22.4s }, [ x2 ], #16
    
        ld1         { v23.4s }, [ x2 ], #16
    
        ld1         { v24.4s }, [ x2 ], #16
    
        ld1         { v25.4s }, [ x2 ], #16
    
        ld1         { v26.4s }, [ x2 ], #16
    
        ld1         { v27.4s }, [ x2 ], #16
    
        ld1         { v28.4s }, [ x2 ], #16
    
        ld1         { v29.4s }, [ x2 ], #16
    
        ld1         { v30.4s }, [ x2 ], #16
    
        ld1         { v31.4s }, [ x2 ], #16
    

    b           .non_linear_loop


.add_unicast:
    ldp         x5, x6, [x0, #8]           // c base ptr, rsc
    cmp         x6, #2
    beq         .do_per_row_add

    
        
            ld1 {v0.h}[0], [ x5 ], x6
        
            ld1 {v0.h}[1], [ x5 ], x6
        
            ld1 {v0.h}[2], [ x5 ], x6
        
            ld1 {v0.h}[3], [ x5 ], x6
        
            ld1 {v0.h}[4], [ x5 ], x6
        
            ld1 {v0.h}[5], [ x5 ], x6
        
            ld1 {v0.h}[6], [ x5 ], x6
        
            ld1 {v0.h}[7], [ x5 ], x6
        
        fadd v16.8h, v16.8h, v0.8h
    
        
            ld1 {v0.h}[0], [ x5 ], x6
        
            ld1 {v0.h}[1], [ x5 ], x6
        
            ld1 {v0.h}[2], [ x5 ], x6
        
            ld1 {v0.h}[3], [ x5 ], x6
        
            ld1 {v0.h}[4], [ x5 ], x6
        
            ld1 {v0.h}[5], [ x5 ], x6
        
            ld1 {v0.h}[6], [ x5 ], x6
        
            ld1 {v0.h}[7], [ x5 ], x6
        
        fadd v17.8h, v17.8h, v0.8h
    
        
            ld1 {v0.h}[0], [ x5 ], x6
        
            ld1 {v0.h}[1], [ x5 ], x6
        
            ld1 {v0.h}[2], [ x5 ], x6
        
            ld1 {v0.h}[3], [ x5 ], x6
        
            ld1 {v0.h}[4], [ x5 ], x6
        
            ld1 {v0.h}[5], [ x5 ], x6
        
            ld1 {v0.h}[6], [ x5 ], x6
        
            ld1 {v0.h}[7], [ x5 ], x6
        
        fadd v18.8h, v18.8h, v0.8h
    
        
            ld1 {v0.h}[0], [ x5 ], x6
        
            ld1 {v0.h}[1], [ x5 ], x6
        
            ld1 {v0.h}[2], [ x5 ], x6
        
            ld1 {v0.h}[3], [ x5 ], x6
        
            ld1 {v0.h}[4], [ x5 ], x6
        
            ld1 {v0.h}[5], [ x5 ], x6
        
            ld1 {v0.h}[6], [ x5 ], x6
        
            ld1 {v0.h}[7], [ x5 ], x6
        
        fadd v19.8h, v19.8h, v0.8h
    
        
            ld1 {v0.h}[0], [ x5 ], x6
        
            ld1 {v0.h}[1], [ x5 ], x6
        
            ld1 {v0.h}[2], [ x5 ], x6
        
            ld1 {v0.h}[3], [ x5 ], x6
        
            ld1 {v0.h}[4], [ x5 ], x6
        
            ld1 {v0.h}[5], [ x5 ], x6
        
            ld1 {v0.h}[6], [ x5 ], x6
        
            ld1 {v0.h}[7], [ x5 ], x6
        
        fadd v20.8h, v20.8h, v0.8h
    
        
            ld1 {v0.h}[0], [ x5 ], x6
        
            ld1 {v0.h}[1], [ x5 ], x6
        
            ld1 {v0.h}[2], [ x5 ], x6
        
            ld1 {v0.h}[3], [ x5 ], x6
        
            ld1 {v0.h}[4], [ x5 ], x6
        
            ld1 {v0.h}[5], [ x5 ], x6
        
            ld1 {v0.h}[6], [ x5 ], x6
        
            ld1 {v0.h}[7], [ x5 ], x6
        
        fadd v21.8h, v21.8h, v0.8h
    
        
            ld1 {v0.h}[0], [ x5 ], x6
        
            ld1 {v0.h}[1], [ x5 ], x6
        
            ld1 {v0.h}[2], [ x5 ], x6
        
            ld1 {v0.h}[3], [ x5 ], x6
        
            ld1 {v0.h}[4], [ x5 ], x6
        
            ld1 {v0.h}[5], [ x5 ], x6
        
            ld1 {v0.h}[6], [ x5 ], x6
        
            ld1 {v0.h}[7], [ x5 ], x6
        
        fadd v22.8h, v22.8h, v0.8h
    
        
            ld1 {v0.h}[0], [ x5 ], x6
        
            ld1 {v0.h}[1], [ x5 ], x6
        
            ld1 {v0.h}[2], [ x5 ], x6
        
            ld1 {v0.h}[3], [ x5 ], x6
        
            ld1 {v0.h}[4], [ x5 ], x6
        
            ld1 {v0.h}[5], [ x5 ], x6
        
            ld1 {v0.h}[6], [ x5 ], x6
        
            ld1 {v0.h}[7], [ x5 ], x6
        
        fadd v23.8h, v23.8h, v0.8h
    
        
            ld1 {v0.h}[0], [ x5 ], x6
        
            ld1 {v0.h}[1], [ x5 ], x6
        
            ld1 {v0.h}[2], [ x5 ], x6
        
            ld1 {v0.h}[3], [ x5 ], x6
        
            ld1 {v0.h}[4], [ x5 ], x6
        
            ld1 {v0.h}[5], [ x5 ], x6
        
            ld1 {v0.h}[6], [ x5 ], x6
        
            ld1 {v0.h}[7], [ x5 ], x6
        
        fadd v24.8h, v24.8h, v0.8h
    
        
            ld1 {v0.h}[0], [ x5 ], x6
        
            ld1 {v0.h}[1], [ x5 ], x6
        
            ld1 {v0.h}[2], [ x5 ], x6
        
            ld1 {v0.h}[3], [ x5 ], x6
        
            ld1 {v0.h}[4], [ x5 ], x6
        
            ld1 {v0.h}[5], [ x5 ], x6
        
            ld1 {v0.h}[6], [ x5 ], x6
        
            ld1 {v0.h}[7], [ x5 ], x6
        
        fadd v25.8h, v25.8h, v0.8h
    
        
            ld1 {v0.h}[0], [ x5 ], x6
        
            ld1 {v0.h}[1], [ x5 ], x6
        
            ld1 {v0.h}[2], [ x5 ], x6
        
            ld1 {v0.h}[3], [ x5 ], x6
        
            ld1 {v0.h}[4], [ x5 ], x6
        
            ld1 {v0.h}[5], [ x5 ], x6
        
            ld1 {v0.h}[6], [ x5 ], x6
        
            ld1 {v0.h}[7], [ x5 ], x6
        
        fadd v26.8h, v26.8h, v0.8h
    
        
            ld1 {v0.h}[0], [ x5 ], x6
        
            ld1 {v0.h}[1], [ x5 ], x6
        
            ld1 {v0.h}[2], [ x5 ], x6
        
            ld1 {v0.h}[3], [ x5 ], x6
        
            ld1 {v0.h}[4], [ x5 ], x6
        
            ld1 {v0.h}[5], [ x5 ], x6
        
            ld1 {v0.h}[6], [ x5 ], x6
        
            ld1 {v0.h}[7], [ x5 ], x6
        
        fadd v27.8h, v27.8h, v0.8h
    
        
            ld1 {v0.h}[0], [ x5 ], x6
        
            ld1 {v0.h}[1], [ x5 ], x6
        
            ld1 {v0.h}[2], [ x5 ], x6
        
            ld1 {v0.h}[3], [ x5 ], x6
        
            ld1 {v0.h}[4], [ x5 ], x6
        
            ld1 {v0.h}[5], [ x5 ], x6
        
            ld1 {v0.h}[6], [ x5 ], x6
        
            ld1 {v0.h}[7], [ x5 ], x6
        
        fadd v28.8h, v28.8h, v0.8h
    
        
            ld1 {v0.h}[0], [ x5 ], x6
        
            ld1 {v0.h}[1], [ x5 ], x6
        
            ld1 {v0.h}[2], [ x5 ], x6
        
            ld1 {v0.h}[3], [ x5 ], x6
        
            ld1 {v0.h}[4], [ x5 ], x6
        
            ld1 {v0.h}[5], [ x5 ], x6
        
            ld1 {v0.h}[6], [ x5 ], x6
        
            ld1 {v0.h}[7], [ x5 ], x6
        
        fadd v29.8h, v29.8h, v0.8h
    
        
            ld1 {v0.h}[0], [ x5 ], x6
        
            ld1 {v0.h}[1], [ x5 ], x6
        
            ld1 {v0.h}[2], [ x5 ], x6
        
            ld1 {v0.h}[3], [ x5 ], x6
        
            ld1 {v0.h}[4], [ x5 ], x6
        
            ld1 {v0.h}[5], [ x5 ], x6
        
            ld1 {v0.h}[6], [ x5 ], x6
        
            ld1 {v0.h}[7], [ x5 ], x6
        
        fadd v30.8h, v30.8h, v0.8h
    
        
            ld1 {v0.h}[0], [ x5 ], x6
        
            ld1 {v0.h}[1], [ x5 ], x6
        
            ld1 {v0.h}[2], [ x5 ], x6
        
            ld1 {v0.h}[3], [ x5 ], x6
        
            ld1 {v0.h}[4], [ x5 ], x6
        
            ld1 {v0.h}[5], [ x5 ], x6
        
            ld1 {v0.h}[6], [ x5 ], x6
        
            ld1 {v0.h}[7], [ x5 ], x6
        
        fadd v31.8h, v31.8h, v0.8h
    

    b           .non_linear_loop

.do_per_row_add:
    ld1     {v0.8h-v3.8h}, [x5], #64
    ld1     {v4.8h-v7.8h}, [x5], #64
    ld1     {v8.8h-v11.8h}, [x5], #64
    ld1     {v12.8h-v15.8h}, [x5], #64

    
        fadd v16.8h, v16.8h, v0.8h
    
        fadd v17.8h, v17.8h, v1.8h
    
        fadd v18.8h, v18.8h, v2.8h
    
        fadd v19.8h, v19.8h, v3.8h
    
        fadd v20.8h, v20.8h, v4.8h
    
        fadd v21.8h, v21.8h, v5.8h
    
        fadd v22.8h, v22.8h, v6.8h
    
        fadd v23.8h, v23.8h, v7.8h
    
        fadd v24.8h, v24.8h, v8.8h
    
        fadd v25.8h, v25.8h, v9.8h
    
        fadd v26.8h, v26.8h, v10.8h
    
        fadd v27.8h, v27.8h, v11.8h
    
        fadd v28.8h, v28.8h, v12.8h
    
        fadd v29.8h, v29.8h, v13.8h
    
        fadd v30.8h, v30.8h, v14.8h
    
        fadd v31.8h, v31.8h, v15.8h
    

    b           .non_linear_loop

.add_row_col_products:
    ldr     x3, [x0, #16]
    ldr     x2, [x0, #8]

    ld1         {v8.h}[0], [ x3 ]

    
        ldr     q0, [x2], #16
    
        ldr     q1, [x2], #16
    
        ldr     q2, [x2], #16
    
        ldr     q3, [x2], #16
    
        ldr     q4, [x2], #16
    
        ldr     q5, [x2], #16
    
        ldr     q6, [x2], #16
    
        ldr     q7, [x2], #16
    

    fmla        v16.8h, v0.8h, v8.h[0]
    ldr         q0, [x2], #16
    fmla        v17.8h, v1.8h, v8.h[0] 
    ldr         q1, [x2], #16
    fmla        v18.8h, v2.8h, v8.h[0] 
    ldr         q2, [x2], #16
    fmla        v19.8h, v3.8h, v8.h[0] 
    ldr         q3, [x2], #16
    fmla        v20.8h, v4.8h, v8.h[0] 
    ldr         q4, [x2], #16
    fmla        v21.8h, v5.8h, v8.h[0] 
    ldr         q5, [x2], #16
    fmla        v22.8h, v6.8h, v8.h[0] 
    ldr         q6, [x2], #16
    fmla        v23.8h, v7.8h, v8.h[0] 
    ldr         q7, [x2], #16

    fmla        v24.8h, v0.8h, v8.h[0]
    fmla        v25.8h, v1.8h, v8.h[0] 
    fmla        v26.8h, v2.8h, v8.h[0] 
    fmla        v27.8h, v3.8h, v8.h[0] 
    fmla        v28.8h, v4.8h, v8.h[0] 
    fmla        v29.8h, v5.8h, v8.h[0] 
    fmla        v30.8h, v6.8h, v8.h[0] 
    fmla        v31.8h, v7.8h, v8.h[0] 

    b           .non_linear_loop

.store:
    ldp         x5, x6, [x0, #8]                // c base ptr, rsc$

    cmp         x6, #2
    beq         .store_strides_contig

    
        
            st1 { v16.h }[0], [ x5 ], x6
        
            st1 { v16.h }[1], [ x5 ], x6
        
            st1 { v16.h }[2], [ x5 ], x6
        
            st1 { v16.h }[3], [ x5 ], x6
        
            st1 { v16.h }[4], [ x5 ], x6
        
            st1 { v16.h }[5], [ x5 ], x6
        
            st1 { v16.h }[6], [ x5 ], x6
        
            st1 { v16.h }[7], [ x5 ], x6
        
    
        
            st1 { v17.h }[0], [ x5 ], x6
        
            st1 { v17.h }[1], [ x5 ], x6
        
            st1 { v17.h }[2], [ x5 ], x6
        
            st1 { v17.h }[3], [ x5 ], x6
        
            st1 { v17.h }[4], [ x5 ], x6
        
            st1 { v17.h }[5], [ x5 ], x6
        
            st1 { v17.h }[6], [ x5 ], x6
        
            st1 { v17.h }[7], [ x5 ], x6
        
    
        
            st1 { v18.h }[0], [ x5 ], x6
        
            st1 { v18.h }[1], [ x5 ], x6
        
            st1 { v18.h }[2], [ x5 ], x6
        
            st1 { v18.h }[3], [ x5 ], x6
        
            st1 { v18.h }[4], [ x5 ], x6
        
            st1 { v18.h }[5], [ x5 ], x6
        
            st1 { v18.h }[6], [ x5 ], x6
        
            st1 { v18.h }[7], [ x5 ], x6
        
    
        
            st1 { v19.h }[0], [ x5 ], x6
        
            st1 { v19.h }[1], [ x5 ], x6
        
            st1 { v19.h }[2], [ x5 ], x6
        
            st1 { v19.h }[3], [ x5 ], x6
        
            st1 { v19.h }[4], [ x5 ], x6
        
            st1 { v19.h }[5], [ x5 ], x6
        
            st1 { v19.h }[6], [ x5 ], x6
        
            st1 { v19.h }[7], [ x5 ], x6
        
    
        
            st1 { v20.h }[0], [ x5 ], x6
        
            st1 { v20.h }[1], [ x5 ], x6
        
            st1 { v20.h }[2], [ x5 ], x6
        
            st1 { v20.h }[3], [ x5 ], x6
        
            st1 { v20.h }[4], [ x5 ], x6
        
            st1 { v20.h }[5], [ x5 ], x6
        
            st1 { v20.h }[6], [ x5 ], x6
        
            st1 { v20.h }[7], [ x5 ], x6
        
    
        
            st1 { v21.h }[0], [ x5 ], x6
        
            st1 { v21.h }[1], [ x5 ], x6
        
            st1 { v21.h }[2], [ x5 ], x6
        
            st1 { v21.h }[3], [ x5 ], x6
        
            st1 { v21.h }[4], [ x5 ], x6
        
            st1 { v21.h }[5], [ x5 ], x6
        
            st1 { v21.h }[6], [ x5 ], x6
        
            st1 { v21.h }[7], [ x5 ], x6
        
    
        
            st1 { v22.h }[0], [ x5 ], x6
        
            st1 { v22.h }[1], [ x5 ], x6
        
            st1 { v22.h }[2], [ x5 ], x6
        
            st1 { v22.h }[3], [ x5 ], x6
        
            st1 { v22.h }[4], [ x5 ], x6
        
            st1 { v22.h }[5], [ x5 ], x6
        
            st1 { v22.h }[6], [ x5 ], x6
        
            st1 { v22.h }[7], [ x5 ], x6
        
    
        
            st1 { v23.h }[0], [ x5 ], x6
        
            st1 { v23.h }[1], [ x5 ], x6
        
            st1 { v23.h }[2], [ x5 ], x6
        
            st1 { v23.h }[3], [ x5 ], x6
        
            st1 { v23.h }[4], [ x5 ], x6
        
            st1 { v23.h }[5], [ x5 ], x6
        
            st1 { v23.h }[6], [ x5 ], x6
        
            st1 { v23.h }[7], [ x5 ], x6
        
    
        
            st1 { v24.h }[0], [ x5 ], x6
        
            st1 { v24.h }[1], [ x5 ], x6
        
            st1 { v24.h }[2], [ x5 ], x6
        
            st1 { v24.h }[3], [ x5 ], x6
        
            st1 { v24.h }[4], [ x5 ], x6
        
            st1 { v24.h }[5], [ x5 ], x6
        
            st1 { v24.h }[6], [ x5 ], x6
        
            st1 { v24.h }[7], [ x5 ], x6
        
    
        
            st1 { v25.h }[0], [ x5 ], x6
        
            st1 { v25.h }[1], [ x5 ], x6
        
            st1 { v25.h }[2], [ x5 ], x6
        
            st1 { v25.h }[3], [ x5 ], x6
        
            st1 { v25.h }[4], [ x5 ], x6
        
            st1 { v25.h }[5], [ x5 ], x6
        
            st1 { v25.h }[6], [ x5 ], x6
        
            st1 { v25.h }[7], [ x5 ], x6
        
    
        
            st1 { v26.h }[0], [ x5 ], x6
        
            st1 { v26.h }[1], [ x5 ], x6
        
            st1 { v26.h }[2], [ x5 ], x6
        
            st1 { v26.h }[3], [ x5 ], x6
        
            st1 { v26.h }[4], [ x5 ], x6
        
            st1 { v26.h }[5], [ x5 ], x6
        
            st1 { v26.h }[6], [ x5 ], x6
        
            st1 { v26.h }[7], [ x5 ], x6
        
    
        
            st1 { v27.h }[0], [ x5 ], x6
        
            st1 { v27.h }[1], [ x5 ], x6
        
            st1 { v27.h }[2], [ x5 ], x6
        
            st1 { v27.h }[3], [ x5 ], x6
        
            st1 { v27.h }[4], [ x5 ], x6
        
            st1 { v27.h }[5], [ x5 ], x6
        
            st1 { v27.h }[6], [ x5 ], x6
        
            st1 { v27.h }[7], [ x5 ], x6
        
    
        
            st1 { v28.h }[0], [ x5 ], x6
        
            st1 { v28.h }[1], [ x5 ], x6
        
            st1 { v28.h }[2], [ x5 ], x6
        
            st1 { v28.h }[3], [ x5 ], x6
        
            st1 { v28.h }[4], [ x5 ], x6
        
            st1 { v28.h }[5], [ x5 ], x6
        
            st1 { v28.h }[6], [ x5 ], x6
        
            st1 { v28.h }[7], [ x5 ], x6
        
    
        
            st1 { v29.h }[0], [ x5 ], x6
        
            st1 { v29.h }[1], [ x5 ], x6
        
            st1 { v29.h }[2], [ x5 ], x6
        
            st1 { v29.h }[3], [ x5 ], x6
        
            st1 { v29.h }[4], [ x5 ], x6
        
            st1 { v29.h }[5], [ x5 ], x6
        
            st1 { v29.h }[6], [ x5 ], x6
        
            st1 { v29.h }[7], [ x5 ], x6
        
    
        
            st1 { v30.h }[0], [ x5 ], x6
        
            st1 { v30.h }[1], [ x5 ], x6
        
            st1 { v30.h }[2], [ x5 ], x6
        
            st1 { v30.h }[3], [ x5 ], x6
        
            st1 { v30.h }[4], [ x5 ], x6
        
            st1 { v30.h }[5], [ x5 ], x6
        
            st1 { v30.h }[6], [ x5 ], x6
        
            st1 { v30.h }[7], [ x5 ], x6
        
    
        
            st1 { v31.h }[0], [ x5 ], x6
        
            st1 { v31.h }[1], [ x5 ], x6
        
            st1 { v31.h }[2], [ x5 ], x6
        
            st1 { v31.h }[3], [ x5 ], x6
        
            st1 { v31.h }[4], [ x5 ], x6
        
            st1 { v31.h }[5], [ x5 ], x6
        
            st1 { v31.h }[6], [ x5 ], x6
        
            st1 { v31.h }[7], [ x5 ], x6
        
    
    b           .non_linear_loop

.store_strides_contig:

    
        st1 { v16.8h }, [ x5 ], #16
    
        st1 { v17.8h }, [ x5 ], #16
    
        st1 { v18.8h }, [ x5 ], #16
    
        st1 { v19.8h }, [ x5 ], #16
    
        st1 { v20.8h }, [ x5 ], #16
    
        st1 { v21.8h }, [ x5 ], #16
    
        st1 { v22.8h }, [ x5 ], #16
    
        st1 { v23.8h }, [ x5 ], #16
    
        st1 { v24.8h }, [ x5 ], #16
    
        st1 { v25.8h }, [ x5 ], #16
    
        st1 { v26.8h }, [ x5 ], #16
    
        st1 { v27.8h }, [ x5 ], #16
    
        st1 { v28.8h }, [ x5 ], #16
    
        st1 { v29.8h }, [ x5 ], #16
    
        st1 { v30.8h }, [ x5 ], #16
    
        st1 { v31.8h }, [ x5 ], #16
    
    b           .non_linear_loop

.return:

    ldp         d14, d15, [sp], #16
    ldp         d12, d13, [sp], #16
    ldp         d10, d11, [sp], #16
    ldp         d8, d9, [sp], #16

    ldp         x24, x25, [sp], #16
    ldp         x22, x23, [sp], #16
    ldp         x20, x21, [sp], #16

    ret
