// vim: ft=arm

// C tile regs:
// - x19-x29 to preserve (but x19, x28, x29 not used) 
// - d8..d15 to preserve
// - v16 to v31, no need to preserve
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

// packed A buffering (2x8 values): alternating v0, v1 with v2, v3
// packed B buffering (2x8 values): alternating v4, v5 with v6, v7

.text
.align 4

.cpu generic+fp+simd
.global _arm64simd_mmm_f32_64x1_gen_0_23_4_pre
_arm64simd_mmm_f32_64x1_gen_0_23_4_pre:

    stp         x20, x21, [sp, #-16]!
    stp         x22, x23, [sp, #-16]!
    stp         x24, x25, [sp, #-16]!
    stp         x26, x27, [sp, #-16]!

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


    ld1         { v8.s }[0], [ x2 ], #4
    ld1         { v0.4s, v1.4s, v2.4s, v3.4s }, [ x1 ], #64
    ld1         { v4.4s, v5.4s, v6.4s, v7.4s }, [ x1 ], #64

    cmp         x3, #0
    beq         .packed_packed_loop_1_last

    cmp         x3, #4
    blt        .packed_packed_loop_1





.p2align 4
.packed_packed_loop_4:
    
    
        // load a: v9, v10, v11, v12, v13, v14, v15
// load a: v0, v1, v2, v3, v4, v4, v6, v7

    ld1         { v9.4s, v10.4s, v11.4s, v12.4s }, [x1], #64
    ld1         { v13.4s, v14.4s, v15.4s }, [x1], #48

    fmla        v16.4s, v0.4s, v8.s[0]
    fmla        v17.4s, v1.4s, v8.s[0]
    fmla        v18.4s, v2.4s, v8.s[0]
    fmla        v19.4s, v3.4s, v8.s[0]

    ld1         { v0.4s, v1.4s }, [x1], #32

    fmla        v20.4s, v4.4s, v8.s[0]
    fmla        v21.4s, v5.4s, v8.s[0]

    ld1         {  v2.4s, v3.4s, v4.4s, v5.4s }, [x1], #64
    fmla        v22.4s, v6.4s, v8.s[0]
    fmla        v23.4s, v7.4s, v8.s[0]

    ld1         {  v6.4s, v7.4s }, [x1], #32

    fmla        v24.4s, v9.4s, v8.s[0]
    fmla        v25.4s, v10.4s, v8.s[0]
    fmla        v26.4s, v11.4s, v8.s[0]
    fmla        v27.4s, v12.4s, v8.s[0]
    fmla        v28.4s, v13.4s, v8.s[0]
    fmla        v29.4s, v14.4s, v8.s[0]
    fmla        v30.4s, v15.4s, v8.s[0]

    ld1         { v9.4s, v10.4s, v11.4s, v12.4s }, [x1], #64
    ld1         { v13.4s, v14.4s, v15.4s }, [x1], #48

    fmla        v31.4s, v0.4s, v8.s[0]
    ld1         { v8.s }[0], [ x2 ], #4

    fmla        v16.4s, v1.4s, v8.s[0]
    ld1         { v0.4s, v1.4s }, [x1], #32
    fmla        v17.4s, v2.4s, v8.s[0]
    fmla        v18.4s, v3.4s, v8.s[0]
    fmla        v19.4s, v4.4s, v8.s[0]

    fmla        v20.4s, v5.4s, v8.s[0]
    fmla        v21.4s, v6.4s, v8.s[0]
    fmla        v22.4s, v7.4s, v8.s[0]
    fmla        v23.4s, v9.4s, v8.s[0]

    fmla        v24.4s, v10.4s, v8.s[0]
    fmla        v25.4s, v11.4s, v8.s[0]
    fmla        v26.4s, v12.4s, v8.s[0]
    fmla        v27.4s, v13.4s, v8.s[0]
    fmla        v28.4s, v14.4s, v8.s[0]
    fmla        v29.4s, v15.4s, v8.s[0]
    fmla        v30.4s, v0.4s, v8.s[0]
    fmla        v31.4s, v1.4s, v8.s[0]
    ld1         { v8.s }[0], [ x2 ], #4

    ld1         { v0.4s, v1.4s, v2.4s, v3.4s }, [x1], #64
    ld1         { v4.4s, v5.4s, v6.4s, v7.4s }, [x1], #64

    prfm        pldl1keep, [x1, #1024]
    prfm        pldl1keep, [x1, #1088]
    prfm        pldl1keep, [x1, #1152]
    prfm        pldl1keep, [x1, #1216]
    prfm        pldl1keep, [x2, #256]

    

    
    
        // load a: v9, v10, v11, v12, v13, v14, v15
// load a: v0, v1, v2, v3, v4, v4, v6, v7

    ld1         { v9.4s, v10.4s, v11.4s, v12.4s }, [x1], #64
    ld1         { v13.4s, v14.4s, v15.4s }, [x1], #48

    fmla        v16.4s, v0.4s, v8.s[0]
    fmla        v17.4s, v1.4s, v8.s[0]
    fmla        v18.4s, v2.4s, v8.s[0]
    fmla        v19.4s, v3.4s, v8.s[0]

    ld1         { v0.4s, v1.4s }, [x1], #32

    fmla        v20.4s, v4.4s, v8.s[0]
    fmla        v21.4s, v5.4s, v8.s[0]

    ld1         {  v2.4s, v3.4s, v4.4s, v5.4s }, [x1], #64
    fmla        v22.4s, v6.4s, v8.s[0]
    fmla        v23.4s, v7.4s, v8.s[0]

    ld1         {  v6.4s, v7.4s }, [x1], #32

    fmla        v24.4s, v9.4s, v8.s[0]
    fmla        v25.4s, v10.4s, v8.s[0]
    fmla        v26.4s, v11.4s, v8.s[0]
    fmla        v27.4s, v12.4s, v8.s[0]
    fmla        v28.4s, v13.4s, v8.s[0]
    fmla        v29.4s, v14.4s, v8.s[0]
    fmla        v30.4s, v15.4s, v8.s[0]

    ld1         { v9.4s, v10.4s, v11.4s, v12.4s }, [x1], #64
    ld1         { v13.4s, v14.4s, v15.4s }, [x1], #48

    fmla        v31.4s, v0.4s, v8.s[0]
    ld1         { v8.s }[0], [ x2 ], #4

    fmla        v16.4s, v1.4s, v8.s[0]
    ld1         { v0.4s, v1.4s }, [x1], #32
    fmla        v17.4s, v2.4s, v8.s[0]
    fmla        v18.4s, v3.4s, v8.s[0]
    fmla        v19.4s, v4.4s, v8.s[0]

    fmla        v20.4s, v5.4s, v8.s[0]
    fmla        v21.4s, v6.4s, v8.s[0]
    fmla        v22.4s, v7.4s, v8.s[0]
    fmla        v23.4s, v9.4s, v8.s[0]

    fmla        v24.4s, v10.4s, v8.s[0]
    fmla        v25.4s, v11.4s, v8.s[0]
    fmla        v26.4s, v12.4s, v8.s[0]
    fmla        v27.4s, v13.4s, v8.s[0]
    fmla        v28.4s, v14.4s, v8.s[0]
    fmla        v29.4s, v15.4s, v8.s[0]
    fmla        v30.4s, v0.4s, v8.s[0]
    fmla        v31.4s, v1.4s, v8.s[0]
    ld1         { v8.s }[0], [ x2 ], #4

    ld1         { v0.4s, v1.4s, v2.4s, v3.4s }, [x1], #64
    ld1         { v4.4s, v5.4s, v6.4s, v7.4s }, [x1], #64

    prfm        pldl1keep, [x1, #1024]
    prfm        pldl1keep, [x1, #1088]
    prfm        pldl1keep, [x1, #1152]
    prfm        pldl1keep, [x1, #1216]
    prfm        pldl1keep, [x2, #256]

    


    sub         x3, x3, #4
    cmp         x3, #4
    bge         .packed_packed_loop_4

    cmp         x3, #0
    beq         .packed_packed_loop_1_last

.p2align 4
.packed_packed_loop_1:
    
    
            ld1         { v9.4s, v10.4s, v11.4s, v12.4s }, [x1], #64
    ld1         { v13.4s, v14.4s, v15.4s }, [x1], #48

    fmla        v16.4s, v0.4s, v8.s[0]
    fmla        v17.4s, v1.4s, v8.s[0]
    fmla        v18.4s, v2.4s, v8.s[0]
    fmla        v19.4s, v3.4s, v8.s[0]
    fmla        v20.4s, v4.4s, v8.s[0]
    fmla        v21.4s, v5.4s, v8.s[0]
    fmla        v22.4s, v6.4s, v8.s[0]
    fmla        v23.4s, v7.4s, v8.s[0]
    fmla        v24.4s, v9.4s, v8.s[0]
    ld1         { v9.4s }, [ x1 ], #16
    ld1         { v0.4s, v1.4s, v2.4s, v3.4s }, [x1], #64
    ld1         { v4.4s, v5.4s, v6.4s, v7.4s }, [x1], #64
    fmla        v25.4s, v10.4s, v8.s[0]
    fmla        v26.4s, v11.4s, v8.s[0]
    fmla        v27.4s, v12.4s, v8.s[0]
    fmla        v28.4s, v13.4s, v8.s[0]
    fmla        v29.4s, v14.4s, v8.s[0]
    fmla        v30.4s, v15.4s, v8.s[0]

    fmla        v31.4s, v9.4s, v8.s[0]

    ld1         { v8.s }[0], [ x2 ], #4

    prfm        pldl1keep, [x1, #1024]
    prfm        pldl1keep, [x1, #1088]
    prfm        pldl1keep, [x1, #1152]
    prfm        pldl1keep, [x1, #1216]
    prfm        pldl1keep, [x2, #256]

    


    subs        x3, x3, #1
    bne         .packed_packed_loop_1

// last loop can't read beyond actual input as it's likely not packed and padded
.packed_packed_loop_1_last:
    ld1         { v9.4s, v10.4s, v11.4s, v12.4s }, [x1], #64
    ld1         { v13.4s, v14.4s, v15.4s }, [x1], #48

    fmla        v16.4s, v0.4s, v8.s[0]
    fmla        v17.4s, v1.4s, v8.s[0]
    ld1         { v0.4s }, [ x1 ]
    fmla        v18.4s, v2.4s, v8.s[0]
    fmla        v19.4s, v3.4s, v8.s[0]
    fmla        v20.4s, v4.4s, v8.s[0]
    fmla        v21.4s, v5.4s, v8.s[0]
    fmla        v22.4s, v6.4s, v8.s[0]
    fmla        v23.4s, v7.4s, v8.s[0]

    fmla        v24.4s, v9.4s, v8.s[0]
    fmla        v25.4s, v10.4s, v8.s[0]
    fmla        v26.4s, v11.4s, v8.s[0]
    fmla        v27.4s, v12.4s, v8.s[0]
    fmla        v28.4s, v13.4s, v8.s[0]
    fmla        v29.4s, v14.4s, v8.s[0]
    fmla        v30.4s, v15.4s, v8.s[0]
    fmla        v31.4s, v0.4s, v8.s[0]

    b           .non_linear_loop

// vim: ft=arm




.scalar_min:
    add         x2, x0, #8
    ld1         {v0.s}[0], [ x2 ]
    dup         v0.4s, v0.s[0]
    
        
            fmin       v16.4s, v0.4s, v16.4s
        
            fmin       v17.4s, v0.4s, v17.4s
        
            fmin       v18.4s, v0.4s, v18.4s
        
            fmin       v19.4s, v0.4s, v19.4s
        
            fmin       v20.4s, v0.4s, v20.4s
        
            fmin       v21.4s, v0.4s, v21.4s
        
            fmin       v22.4s, v0.4s, v22.4s
        
            fmin       v23.4s, v0.4s, v23.4s
        
            fmin       v24.4s, v0.4s, v24.4s
        
            fmin       v25.4s, v0.4s, v25.4s
        
            fmin       v26.4s, v0.4s, v26.4s
        
            fmin       v27.4s, v0.4s, v27.4s
        
            fmin       v28.4s, v0.4s, v28.4s
        
            fmin       v29.4s, v0.4s, v29.4s
        
            fmin       v30.4s, v0.4s, v30.4s
        
            fmin       v31.4s, v0.4s, v31.4s
        
    

    b           .non_linear_loop


.scalar_max:
    add         x2, x0, #8
    ld1         {v0.s}[0], [ x2 ]
    dup         v0.4s, v0.s[0]
    
        
            fmax       v16.4s, v0.4s, v16.4s
        
            fmax       v17.4s, v0.4s, v17.4s
        
            fmax       v18.4s, v0.4s, v18.4s
        
            fmax       v19.4s, v0.4s, v19.4s
        
            fmax       v20.4s, v0.4s, v20.4s
        
            fmax       v21.4s, v0.4s, v21.4s
        
            fmax       v22.4s, v0.4s, v22.4s
        
            fmax       v23.4s, v0.4s, v23.4s
        
            fmax       v24.4s, v0.4s, v24.4s
        
            fmax       v25.4s, v0.4s, v25.4s
        
            fmax       v26.4s, v0.4s, v26.4s
        
            fmax       v27.4s, v0.4s, v27.4s
        
            fmax       v28.4s, v0.4s, v28.4s
        
            fmax       v29.4s, v0.4s, v29.4s
        
            fmax       v30.4s, v0.4s, v30.4s
        
            fmax       v31.4s, v0.4s, v31.4s
        
    

    b           .non_linear_loop


.scalar_mul:
    add         x2, x0, #8
    ld1         {v0.s}[0], [ x2 ]
    dup         v0.4s, v0.s[0]
    
        
            fmul       v16.4s, v0.4s, v16.4s
        
            fmul       v17.4s, v0.4s, v17.4s
        
            fmul       v18.4s, v0.4s, v18.4s
        
            fmul       v19.4s, v0.4s, v19.4s
        
            fmul       v20.4s, v0.4s, v20.4s
        
            fmul       v21.4s, v0.4s, v21.4s
        
            fmul       v22.4s, v0.4s, v22.4s
        
            fmul       v23.4s, v0.4s, v23.4s
        
            fmul       v24.4s, v0.4s, v24.4s
        
            fmul       v25.4s, v0.4s, v25.4s
        
            fmul       v26.4s, v0.4s, v26.4s
        
            fmul       v27.4s, v0.4s, v27.4s
        
            fmul       v28.4s, v0.4s, v28.4s
        
            fmul       v29.4s, v0.4s, v29.4s
        
            fmul       v30.4s, v0.4s, v30.4s
        
            fmul       v31.4s, v0.4s, v31.4s
        
    

    b           .non_linear_loop


.scalar_add:
    add         x2, x0, #8
    ld1         {v0.s}[0], [ x2 ]
    dup         v0.4s, v0.s[0]
    
        
            fadd       v16.4s, v0.4s, v16.4s
        
            fadd       v17.4s, v0.4s, v17.4s
        
            fadd       v18.4s, v0.4s, v18.4s
        
            fadd       v19.4s, v0.4s, v19.4s
        
            fadd       v20.4s, v0.4s, v20.4s
        
            fadd       v21.4s, v0.4s, v21.4s
        
            fadd       v22.4s, v0.4s, v22.4s
        
            fadd       v23.4s, v0.4s, v23.4s
        
            fadd       v24.4s, v0.4s, v24.4s
        
            fadd       v25.4s, v0.4s, v25.4s
        
            fadd       v26.4s, v0.4s, v26.4s
        
            fadd       v27.4s, v0.4s, v27.4s
        
            fadd       v28.4s, v0.4s, v28.4s
        
            fadd       v29.4s, v0.4s, v29.4s
        
            fadd       v30.4s, v0.4s, v30.4s
        
            fadd       v31.4s, v0.4s, v31.4s
        
    

    b           .non_linear_loop


.scalar_sub:
    add         x2, x0, #8
    ld1         {v0.s}[0], [ x2 ]
    dup         v0.4s, v0.s[0]
    
        
            fsub       v16.4s, v0.4s, v16.4s
        
            fsub       v17.4s, v0.4s, v17.4s
        
            fsub       v18.4s, v0.4s, v18.4s
        
            fsub       v19.4s, v0.4s, v19.4s
        
            fsub       v20.4s, v0.4s, v20.4s
        
            fsub       v21.4s, v0.4s, v21.4s
        
            fsub       v22.4s, v0.4s, v22.4s
        
            fsub       v23.4s, v0.4s, v23.4s
        
            fsub       v24.4s, v0.4s, v24.4s
        
            fsub       v25.4s, v0.4s, v25.4s
        
            fsub       v26.4s, v0.4s, v26.4s
        
            fsub       v27.4s, v0.4s, v27.4s
        
            fsub       v28.4s, v0.4s, v28.4s
        
            fsub       v29.4s, v0.4s, v29.4s
        
            fsub       v30.4s, v0.4s, v30.4s
        
            fsub       v31.4s, v0.4s, v31.4s
        
    

    b           .non_linear_loop


.scalar_sub_flipped:
    add         x2, x0, #8
    ld1         {v0.s}[0], [ x2 ]
    dup         v0.4s, v0.s[0]
    
        
            fsub       v16.4s, v16.4s, v0.4s
        
            fsub       v17.4s, v17.4s, v0.4s
        
            fsub       v18.4s, v18.4s, v0.4s
        
            fsub       v19.4s, v19.4s, v0.4s
        
            fsub       v20.4s, v20.4s, v0.4s
        
            fsub       v21.4s, v21.4s, v0.4s
        
            fsub       v22.4s, v22.4s, v0.4s
        
            fsub       v23.4s, v23.4s, v0.4s
        
            fsub       v24.4s, v24.4s, v0.4s
        
            fsub       v25.4s, v25.4s, v0.4s
        
            fsub       v26.4s, v26.4s, v0.4s
        
            fsub       v27.4s, v27.4s, v0.4s
        
            fsub       v28.4s, v28.4s, v0.4s
        
            fsub       v29.4s, v29.4s, v0.4s
        
            fsub       v30.4s, v30.4s, v0.4s
        
            fsub       v31.4s, v31.4s, v0.4s
        
    

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
    dup         v4.4s, v4.s[0]

    // bsl cond/dst, then, else
    // fcmge dst, src, #0.0
    
        fmul  v0.4s, v16.4s, v4.4s
        fcmge v1.4s, v16.4s, #0.0
        bsl   v1.16b, v16.16b, v0.16b
        and   v16.16b, v1.16b, v1.16b
    
        fmul  v0.4s, v17.4s, v4.4s
        fcmge v1.4s, v17.4s, #0.0
        bsl   v1.16b, v17.16b, v0.16b
        and   v17.16b, v1.16b, v1.16b
    
        fmul  v0.4s, v18.4s, v4.4s
        fcmge v1.4s, v18.4s, #0.0
        bsl   v1.16b, v18.16b, v0.16b
        and   v18.16b, v1.16b, v1.16b
    
        fmul  v0.4s, v19.4s, v4.4s
        fcmge v1.4s, v19.4s, #0.0
        bsl   v1.16b, v19.16b, v0.16b
        and   v19.16b, v1.16b, v1.16b
    
        fmul  v0.4s, v20.4s, v4.4s
        fcmge v1.4s, v20.4s, #0.0
        bsl   v1.16b, v20.16b, v0.16b
        and   v20.16b, v1.16b, v1.16b
    
        fmul  v0.4s, v21.4s, v4.4s
        fcmge v1.4s, v21.4s, #0.0
        bsl   v1.16b, v21.16b, v0.16b
        and   v21.16b, v1.16b, v1.16b
    
        fmul  v0.4s, v22.4s, v4.4s
        fcmge v1.4s, v22.4s, #0.0
        bsl   v1.16b, v22.16b, v0.16b
        and   v22.16b, v1.16b, v1.16b
    
        fmul  v0.4s, v23.4s, v4.4s
        fcmge v1.4s, v23.4s, #0.0
        bsl   v1.16b, v23.16b, v0.16b
        and   v23.16b, v1.16b, v1.16b
    
        fmul  v0.4s, v24.4s, v4.4s
        fcmge v1.4s, v24.4s, #0.0
        bsl   v1.16b, v24.16b, v0.16b
        and   v24.16b, v1.16b, v1.16b
    
        fmul  v0.4s, v25.4s, v4.4s
        fcmge v1.4s, v25.4s, #0.0
        bsl   v1.16b, v25.16b, v0.16b
        and   v25.16b, v1.16b, v1.16b
    
        fmul  v0.4s, v26.4s, v4.4s
        fcmge v1.4s, v26.4s, #0.0
        bsl   v1.16b, v26.16b, v0.16b
        and   v26.16b, v1.16b, v1.16b
    
        fmul  v0.4s, v27.4s, v4.4s
        fcmge v1.4s, v27.4s, #0.0
        bsl   v1.16b, v27.16b, v0.16b
        and   v27.16b, v1.16b, v1.16b
    
        fmul  v0.4s, v28.4s, v4.4s
        fcmge v1.4s, v28.4s, #0.0
        bsl   v1.16b, v28.16b, v0.16b
        and   v28.16b, v1.16b, v1.16b
    
        fmul  v0.4s, v29.4s, v4.4s
        fcmge v1.4s, v29.4s, #0.0
        bsl   v1.16b, v29.16b, v0.16b
        and   v29.16b, v1.16b, v1.16b
    
        fmul  v0.4s, v30.4s, v4.4s
        fcmge v1.4s, v30.4s, #0.0
        bsl   v1.16b, v30.16b, v0.16b
        and   v30.16b, v1.16b, v1.16b
    
        fmul  v0.4s, v31.4s, v4.4s
        fcmge v1.4s, v31.4s, #0.0
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



    
        
        fmin v16.4s, v0.4s, v16.4s
    
        
        fmin v17.4s, v1.4s, v17.4s
    
        
        fmin v18.4s, v2.4s, v18.4s
    
        
        fmin v19.4s, v3.4s, v19.4s
    
        
        fmin v20.4s, v4.4s, v20.4s
    
        
        fmin v21.4s, v5.4s, v21.4s
    
        
        fmin v22.4s, v6.4s, v22.4s
    
        
        fmin v23.4s, v7.4s, v23.4s
    
        
        fmin v24.4s, v8.4s, v24.4s
    
        
        fmin v25.4s, v9.4s, v25.4s
    
        
        fmin v26.4s, v10.4s, v26.4s
    
        
        fmin v27.4s, v11.4s, v27.4s
    
        
        fmin v28.4s, v12.4s, v28.4s
    
        
        fmin v29.4s, v13.4s, v29.4s
    
        
        fmin v30.4s, v14.4s, v30.4s
    
        
        fmin v31.4s, v15.4s, v31.4s
    


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



    
        
        fmax v16.4s, v0.4s, v16.4s
    
        
        fmax v17.4s, v1.4s, v17.4s
    
        
        fmax v18.4s, v2.4s, v18.4s
    
        
        fmax v19.4s, v3.4s, v19.4s
    
        
        fmax v20.4s, v4.4s, v20.4s
    
        
        fmax v21.4s, v5.4s, v21.4s
    
        
        fmax v22.4s, v6.4s, v22.4s
    
        
        fmax v23.4s, v7.4s, v23.4s
    
        
        fmax v24.4s, v8.4s, v24.4s
    
        
        fmax v25.4s, v9.4s, v25.4s
    
        
        fmax v26.4s, v10.4s, v26.4s
    
        
        fmax v27.4s, v11.4s, v27.4s
    
        
        fmax v28.4s, v12.4s, v28.4s
    
        
        fmax v29.4s, v13.4s, v29.4s
    
        
        fmax v30.4s, v14.4s, v30.4s
    
        
        fmax v31.4s, v15.4s, v31.4s
    


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



    
        
        fmul v16.4s, v0.4s, v16.4s
    
        
        fmul v17.4s, v1.4s, v17.4s
    
        
        fmul v18.4s, v2.4s, v18.4s
    
        
        fmul v19.4s, v3.4s, v19.4s
    
        
        fmul v20.4s, v4.4s, v20.4s
    
        
        fmul v21.4s, v5.4s, v21.4s
    
        
        fmul v22.4s, v6.4s, v22.4s
    
        
        fmul v23.4s, v7.4s, v23.4s
    
        
        fmul v24.4s, v8.4s, v24.4s
    
        
        fmul v25.4s, v9.4s, v25.4s
    
        
        fmul v26.4s, v10.4s, v26.4s
    
        
        fmul v27.4s, v11.4s, v27.4s
    
        
        fmul v28.4s, v12.4s, v28.4s
    
        
        fmul v29.4s, v13.4s, v29.4s
    
        
        fmul v30.4s, v14.4s, v30.4s
    
        
        fmul v31.4s, v15.4s, v31.4s
    


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



    
        
        fadd v16.4s, v0.4s, v16.4s
    
        
        fadd v17.4s, v1.4s, v17.4s
    
        
        fadd v18.4s, v2.4s, v18.4s
    
        
        fadd v19.4s, v3.4s, v19.4s
    
        
        fadd v20.4s, v4.4s, v20.4s
    
        
        fadd v21.4s, v5.4s, v21.4s
    
        
        fadd v22.4s, v6.4s, v22.4s
    
        
        fadd v23.4s, v7.4s, v23.4s
    
        
        fadd v24.4s, v8.4s, v24.4s
    
        
        fadd v25.4s, v9.4s, v25.4s
    
        
        fadd v26.4s, v10.4s, v26.4s
    
        
        fadd v27.4s, v11.4s, v27.4s
    
        
        fadd v28.4s, v12.4s, v28.4s
    
        
        fadd v29.4s, v13.4s, v29.4s
    
        
        fadd v30.4s, v14.4s, v30.4s
    
        
        fadd v31.4s, v15.4s, v31.4s
    


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



    
        
        fsub v16.4s, v0.4s, v16.4s
    
        
        fsub v17.4s, v1.4s, v17.4s
    
        
        fsub v18.4s, v2.4s, v18.4s
    
        
        fsub v19.4s, v3.4s, v19.4s
    
        
        fsub v20.4s, v4.4s, v20.4s
    
        
        fsub v21.4s, v5.4s, v21.4s
    
        
        fsub v22.4s, v6.4s, v22.4s
    
        
        fsub v23.4s, v7.4s, v23.4s
    
        
        fsub v24.4s, v8.4s, v24.4s
    
        
        fsub v25.4s, v9.4s, v25.4s
    
        
        fsub v26.4s, v10.4s, v26.4s
    
        
        fsub v27.4s, v11.4s, v27.4s
    
        
        fsub v28.4s, v12.4s, v28.4s
    
        
        fsub v29.4s, v13.4s, v29.4s
    
        
        fsub v30.4s, v14.4s, v30.4s
    
        
        fsub v31.4s, v15.4s, v31.4s
    


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



    
        
        fsub v16.4s, v16.4s, v0.4s
    
        
        fsub v17.4s, v17.4s, v1.4s
    
        
        fsub v18.4s, v18.4s, v2.4s
    
        
        fsub v19.4s, v19.4s, v3.4s
    
        
        fsub v20.4s, v20.4s, v4.4s
    
        
        fsub v21.4s, v21.4s, v5.4s
    
        
        fsub v22.4s, v22.4s, v6.4s
    
        
        fsub v23.4s, v23.4s, v7.4s
    
        
        fsub v24.4s, v24.4s, v8.4s
    
        
        fsub v25.4s, v25.4s, v9.4s
    
        
        fsub v26.4s, v26.4s, v10.4s
    
        
        fsub v27.4s, v27.4s, v11.4s
    
        
        fsub v28.4s, v28.4s, v12.4s
    
        
        fsub v29.4s, v29.4s, v13.4s
    
        
        fsub v30.4s, v30.4s, v14.4s
    
        
        fsub v31.4s, v31.4s, v15.4s
    


b           .non_linear_loop

// vim: ft=arm




.per_col_min:
    ldr         x2, [x0, #8]







        ld1         {v0.s}[0], [ x2 ]


// 64 1


    dup v3.4s, v0.s[0]
    
        
        
            fmin v16.4s, v3.4s, v16.4s
        
    
        
        
            fmin v17.4s, v3.4s, v17.4s
        
    
        
        
            fmin v18.4s, v3.4s, v18.4s
        
    
        
        
            fmin v19.4s, v3.4s, v19.4s
        
    
        
        
            fmin v20.4s, v3.4s, v20.4s
        
    
        
        
            fmin v21.4s, v3.4s, v21.4s
        
    
        
        
            fmin v22.4s, v3.4s, v22.4s
        
    
        
        
            fmin v23.4s, v3.4s, v23.4s
        
    
        
        
            fmin v24.4s, v3.4s, v24.4s
        
    
        
        
            fmin v25.4s, v3.4s, v25.4s
        
    
        
        
            fmin v26.4s, v3.4s, v26.4s
        
    
        
        
            fmin v27.4s, v3.4s, v27.4s
        
    
        
        
            fmin v28.4s, v3.4s, v28.4s
        
    
        
        
            fmin v29.4s, v3.4s, v29.4s
        
    
        
        
            fmin v30.4s, v3.4s, v30.4s
        
    
        
        
            fmin v31.4s, v3.4s, v31.4s
        
    


b           .non_linear_loop


.per_col_max:
    ldr         x2, [x0, #8]







        ld1         {v0.s}[0], [ x2 ]


// 64 1


    dup v3.4s, v0.s[0]
    
        
        
            fmax v16.4s, v3.4s, v16.4s
        
    
        
        
            fmax v17.4s, v3.4s, v17.4s
        
    
        
        
            fmax v18.4s, v3.4s, v18.4s
        
    
        
        
            fmax v19.4s, v3.4s, v19.4s
        
    
        
        
            fmax v20.4s, v3.4s, v20.4s
        
    
        
        
            fmax v21.4s, v3.4s, v21.4s
        
    
        
        
            fmax v22.4s, v3.4s, v22.4s
        
    
        
        
            fmax v23.4s, v3.4s, v23.4s
        
    
        
        
            fmax v24.4s, v3.4s, v24.4s
        
    
        
        
            fmax v25.4s, v3.4s, v25.4s
        
    
        
        
            fmax v26.4s, v3.4s, v26.4s
        
    
        
        
            fmax v27.4s, v3.4s, v27.4s
        
    
        
        
            fmax v28.4s, v3.4s, v28.4s
        
    
        
        
            fmax v29.4s, v3.4s, v29.4s
        
    
        
        
            fmax v30.4s, v3.4s, v30.4s
        
    
        
        
            fmax v31.4s, v3.4s, v31.4s
        
    


b           .non_linear_loop


.per_col_mul:
    ldr         x2, [x0, #8]







        ld1         {v0.s}[0], [ x2 ]


// 64 1


    dup v3.4s, v0.s[0]
    
        
        
            fmul v16.4s, v3.4s, v16.4s
        
    
        
        
            fmul v17.4s, v3.4s, v17.4s
        
    
        
        
            fmul v18.4s, v3.4s, v18.4s
        
    
        
        
            fmul v19.4s, v3.4s, v19.4s
        
    
        
        
            fmul v20.4s, v3.4s, v20.4s
        
    
        
        
            fmul v21.4s, v3.4s, v21.4s
        
    
        
        
            fmul v22.4s, v3.4s, v22.4s
        
    
        
        
            fmul v23.4s, v3.4s, v23.4s
        
    
        
        
            fmul v24.4s, v3.4s, v24.4s
        
    
        
        
            fmul v25.4s, v3.4s, v25.4s
        
    
        
        
            fmul v26.4s, v3.4s, v26.4s
        
    
        
        
            fmul v27.4s, v3.4s, v27.4s
        
    
        
        
            fmul v28.4s, v3.4s, v28.4s
        
    
        
        
            fmul v29.4s, v3.4s, v29.4s
        
    
        
        
            fmul v30.4s, v3.4s, v30.4s
        
    
        
        
            fmul v31.4s, v3.4s, v31.4s
        
    


b           .non_linear_loop


.per_col_add:
    ldr         x2, [x0, #8]







        ld1         {v0.s}[0], [ x2 ]


// 64 1


    dup v3.4s, v0.s[0]
    
        
        
            fadd v16.4s, v3.4s, v16.4s
        
    
        
        
            fadd v17.4s, v3.4s, v17.4s
        
    
        
        
            fadd v18.4s, v3.4s, v18.4s
        
    
        
        
            fadd v19.4s, v3.4s, v19.4s
        
    
        
        
            fadd v20.4s, v3.4s, v20.4s
        
    
        
        
            fadd v21.4s, v3.4s, v21.4s
        
    
        
        
            fadd v22.4s, v3.4s, v22.4s
        
    
        
        
            fadd v23.4s, v3.4s, v23.4s
        
    
        
        
            fadd v24.4s, v3.4s, v24.4s
        
    
        
        
            fadd v25.4s, v3.4s, v25.4s
        
    
        
        
            fadd v26.4s, v3.4s, v26.4s
        
    
        
        
            fadd v27.4s, v3.4s, v27.4s
        
    
        
        
            fadd v28.4s, v3.4s, v28.4s
        
    
        
        
            fadd v29.4s, v3.4s, v29.4s
        
    
        
        
            fadd v30.4s, v3.4s, v30.4s
        
    
        
        
            fadd v31.4s, v3.4s, v31.4s
        
    


b           .non_linear_loop


.per_col_sub:
    ldr         x2, [x0, #8]







        ld1         {v0.s}[0], [ x2 ]


// 64 1


    dup v3.4s, v0.s[0]
    
        
        
            fsub v16.4s, v3.4s, v16.4s
        
    
        
        
            fsub v17.4s, v3.4s, v17.4s
        
    
        
        
            fsub v18.4s, v3.4s, v18.4s
        
    
        
        
            fsub v19.4s, v3.4s, v19.4s
        
    
        
        
            fsub v20.4s, v3.4s, v20.4s
        
    
        
        
            fsub v21.4s, v3.4s, v21.4s
        
    
        
        
            fsub v22.4s, v3.4s, v22.4s
        
    
        
        
            fsub v23.4s, v3.4s, v23.4s
        
    
        
        
            fsub v24.4s, v3.4s, v24.4s
        
    
        
        
            fsub v25.4s, v3.4s, v25.4s
        
    
        
        
            fsub v26.4s, v3.4s, v26.4s
        
    
        
        
            fsub v27.4s, v3.4s, v27.4s
        
    
        
        
            fsub v28.4s, v3.4s, v28.4s
        
    
        
        
            fsub v29.4s, v3.4s, v29.4s
        
    
        
        
            fsub v30.4s, v3.4s, v30.4s
        
    
        
        
            fsub v31.4s, v3.4s, v31.4s
        
    


b           .non_linear_loop


.per_col_sub_flipped:
    ldr         x2, [x0, #8]







        ld1         {v0.s}[0], [ x2 ]


// 64 1


    dup v3.4s, v0.s[0]
    
        
        
            fsub v16.4s, v16.4s, v3.4s
        
    
        
        
            fsub v17.4s, v17.4s, v3.4s
        
    
        
        
            fsub v18.4s, v18.4s, v3.4s
        
    
        
        
            fsub v19.4s, v19.4s, v3.4s
        
    
        
        
            fsub v20.4s, v20.4s, v3.4s
        
    
        
        
            fsub v21.4s, v21.4s, v3.4s
        
    
        
        
            fsub v22.4s, v22.4s, v3.4s
        
    
        
        
            fsub v23.4s, v23.4s, v3.4s
        
    
        
        
            fsub v24.4s, v24.4s, v3.4s
        
    
        
        
            fsub v25.4s, v25.4s, v3.4s
        
    
        
        
            fsub v26.4s, v26.4s, v3.4s
        
    
        
        
            fsub v27.4s, v27.4s, v3.4s
        
    
        
        
            fsub v28.4s, v28.4s, v3.4s
        
    
        
        
            fsub v29.4s, v29.4s, v3.4s
        
    
        
        
            fsub v30.4s, v30.4s, v3.4s
        
    
        
        
            fsub v31.4s, v31.4s, v3.4s
        
    


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
    cmp         x6, #4
    beq         .do_per_row_add

    
        
            ld1 {v0.s}[0], [ x5 ], x6
        
            ld1 {v0.s}[1], [ x5 ], x6
        
            ld1 {v0.s}[2], [ x5 ], x6
        
            ld1 {v0.s}[3], [ x5 ], x6
        
        fadd v16.4s, v16.4s, v0.4s
    
        
            ld1 {v0.s}[0], [ x5 ], x6
        
            ld1 {v0.s}[1], [ x5 ], x6
        
            ld1 {v0.s}[2], [ x5 ], x6
        
            ld1 {v0.s}[3], [ x5 ], x6
        
        fadd v17.4s, v17.4s, v0.4s
    
        
            ld1 {v0.s}[0], [ x5 ], x6
        
            ld1 {v0.s}[1], [ x5 ], x6
        
            ld1 {v0.s}[2], [ x5 ], x6
        
            ld1 {v0.s}[3], [ x5 ], x6
        
        fadd v18.4s, v18.4s, v0.4s
    
        
            ld1 {v0.s}[0], [ x5 ], x6
        
            ld1 {v0.s}[1], [ x5 ], x6
        
            ld1 {v0.s}[2], [ x5 ], x6
        
            ld1 {v0.s}[3], [ x5 ], x6
        
        fadd v19.4s, v19.4s, v0.4s
    
        
            ld1 {v0.s}[0], [ x5 ], x6
        
            ld1 {v0.s}[1], [ x5 ], x6
        
            ld1 {v0.s}[2], [ x5 ], x6
        
            ld1 {v0.s}[3], [ x5 ], x6
        
        fadd v20.4s, v20.4s, v0.4s
    
        
            ld1 {v0.s}[0], [ x5 ], x6
        
            ld1 {v0.s}[1], [ x5 ], x6
        
            ld1 {v0.s}[2], [ x5 ], x6
        
            ld1 {v0.s}[3], [ x5 ], x6
        
        fadd v21.4s, v21.4s, v0.4s
    
        
            ld1 {v0.s}[0], [ x5 ], x6
        
            ld1 {v0.s}[1], [ x5 ], x6
        
            ld1 {v0.s}[2], [ x5 ], x6
        
            ld1 {v0.s}[3], [ x5 ], x6
        
        fadd v22.4s, v22.4s, v0.4s
    
        
            ld1 {v0.s}[0], [ x5 ], x6
        
            ld1 {v0.s}[1], [ x5 ], x6
        
            ld1 {v0.s}[2], [ x5 ], x6
        
            ld1 {v0.s}[3], [ x5 ], x6
        
        fadd v23.4s, v23.4s, v0.4s
    
        
            ld1 {v0.s}[0], [ x5 ], x6
        
            ld1 {v0.s}[1], [ x5 ], x6
        
            ld1 {v0.s}[2], [ x5 ], x6
        
            ld1 {v0.s}[3], [ x5 ], x6
        
        fadd v24.4s, v24.4s, v0.4s
    
        
            ld1 {v0.s}[0], [ x5 ], x6
        
            ld1 {v0.s}[1], [ x5 ], x6
        
            ld1 {v0.s}[2], [ x5 ], x6
        
            ld1 {v0.s}[3], [ x5 ], x6
        
        fadd v25.4s, v25.4s, v0.4s
    
        
            ld1 {v0.s}[0], [ x5 ], x6
        
            ld1 {v0.s}[1], [ x5 ], x6
        
            ld1 {v0.s}[2], [ x5 ], x6
        
            ld1 {v0.s}[3], [ x5 ], x6
        
        fadd v26.4s, v26.4s, v0.4s
    
        
            ld1 {v0.s}[0], [ x5 ], x6
        
            ld1 {v0.s}[1], [ x5 ], x6
        
            ld1 {v0.s}[2], [ x5 ], x6
        
            ld1 {v0.s}[3], [ x5 ], x6
        
        fadd v27.4s, v27.4s, v0.4s
    
        
            ld1 {v0.s}[0], [ x5 ], x6
        
            ld1 {v0.s}[1], [ x5 ], x6
        
            ld1 {v0.s}[2], [ x5 ], x6
        
            ld1 {v0.s}[3], [ x5 ], x6
        
        fadd v28.4s, v28.4s, v0.4s
    
        
            ld1 {v0.s}[0], [ x5 ], x6
        
            ld1 {v0.s}[1], [ x5 ], x6
        
            ld1 {v0.s}[2], [ x5 ], x6
        
            ld1 {v0.s}[3], [ x5 ], x6
        
        fadd v29.4s, v29.4s, v0.4s
    
        
            ld1 {v0.s}[0], [ x5 ], x6
        
            ld1 {v0.s}[1], [ x5 ], x6
        
            ld1 {v0.s}[2], [ x5 ], x6
        
            ld1 {v0.s}[3], [ x5 ], x6
        
        fadd v30.4s, v30.4s, v0.4s
    
        
            ld1 {v0.s}[0], [ x5 ], x6
        
            ld1 {v0.s}[1], [ x5 ], x6
        
            ld1 {v0.s}[2], [ x5 ], x6
        
            ld1 {v0.s}[3], [ x5 ], x6
        
        fadd v31.4s, v31.4s, v0.4s
    

    b           .non_linear_loop

.do_per_row_add:
    ld1     {v0.4s-v3.4s}, [x5], #64
    ld1     {v4.4s-v7.4s}, [x5], #64
    ld1     {v8.4s-v11.4s}, [x5], #64
    ld1     {v12.4s-v15.4s}, [x5], #64

    
        fadd v16.4s, v16.4s, v0.4s
    
        fadd v17.4s, v17.4s, v1.4s
    
        fadd v18.4s, v18.4s, v2.4s
    
        fadd v19.4s, v19.4s, v3.4s
    
        fadd v20.4s, v20.4s, v4.4s
    
        fadd v21.4s, v21.4s, v5.4s
    
        fadd v22.4s, v22.4s, v6.4s
    
        fadd v23.4s, v23.4s, v7.4s
    
        fadd v24.4s, v24.4s, v8.4s
    
        fadd v25.4s, v25.4s, v9.4s
    
        fadd v26.4s, v26.4s, v10.4s
    
        fadd v27.4s, v27.4s, v11.4s
    
        fadd v28.4s, v28.4s, v12.4s
    
        fadd v29.4s, v29.4s, v13.4s
    
        fadd v30.4s, v30.4s, v14.4s
    
        fadd v31.4s, v31.4s, v15.4s
    

    b           .non_linear_loop

.add_row_col_products:
    ldr     x3, [x0, #16]
    ldr     x2, [x0, #8]

    ld1         {v8.s}[0], [ x3 ]

    
        ldr     q0, [x2], #16
    
        ldr     q1, [x2], #16
    
        ldr     q2, [x2], #16
    
        ldr     q3, [x2], #16
    
        ldr     q4, [x2], #16
    
        ldr     q5, [x2], #16
    
        ldr     q6, [x2], #16
    
        ldr     q7, [x2], #16
    

    fmla        v16.4s, v0.4s, v8.s[0]
    ldr         q0, [x2], #16
    fmla        v17.4s, v1.4s, v8.s[0] 
    ldr         q1, [x2], #16
    fmla        v18.4s, v2.4s, v8.s[0] 
    ldr         q2, [x2], #16
    fmla        v19.4s, v3.4s, v8.s[0] 
    ldr         q3, [x2], #16
    fmla        v20.4s, v4.4s, v8.s[0] 
    ldr         q4, [x2], #16
    fmla        v21.4s, v5.4s, v8.s[0] 
    ldr         q5, [x2], #16
    fmla        v22.4s, v6.4s, v8.s[0] 
    ldr         q6, [x2], #16
    fmla        v23.4s, v7.4s, v8.s[0] 
    ldr         q7, [x2], #16

    fmla        v24.4s, v0.4s, v8.s[0]
    fmla        v25.4s, v1.4s, v8.s[0] 
    fmla        v26.4s, v2.4s, v8.s[0] 
    fmla        v27.4s, v3.4s, v8.s[0] 
    fmla        v28.4s, v4.4s, v8.s[0] 
    fmla        v29.4s, v5.4s, v8.s[0] 
    fmla        v30.4s, v6.4s, v8.s[0] 
    fmla        v31.4s, v7.4s, v8.s[0] 

    b           .non_linear_loop

.store:
    ldp         x5, x6, [x0, #8]                // c base ptr, rsc$

    cmp         x6, #4
    beq         .store_strides_contig

    
        
            st1 { v16.s }[0], [ x5 ], x6
        
            st1 { v16.s }[1], [ x5 ], x6
        
            st1 { v16.s }[2], [ x5 ], x6
        
            st1 { v16.s }[3], [ x5 ], x6
        
    
        
            st1 { v17.s }[0], [ x5 ], x6
        
            st1 { v17.s }[1], [ x5 ], x6
        
            st1 { v17.s }[2], [ x5 ], x6
        
            st1 { v17.s }[3], [ x5 ], x6
        
    
        
            st1 { v18.s }[0], [ x5 ], x6
        
            st1 { v18.s }[1], [ x5 ], x6
        
            st1 { v18.s }[2], [ x5 ], x6
        
            st1 { v18.s }[3], [ x5 ], x6
        
    
        
            st1 { v19.s }[0], [ x5 ], x6
        
            st1 { v19.s }[1], [ x5 ], x6
        
            st1 { v19.s }[2], [ x5 ], x6
        
            st1 { v19.s }[3], [ x5 ], x6
        
    
        
            st1 { v20.s }[0], [ x5 ], x6
        
            st1 { v20.s }[1], [ x5 ], x6
        
            st1 { v20.s }[2], [ x5 ], x6
        
            st1 { v20.s }[3], [ x5 ], x6
        
    
        
            st1 { v21.s }[0], [ x5 ], x6
        
            st1 { v21.s }[1], [ x5 ], x6
        
            st1 { v21.s }[2], [ x5 ], x6
        
            st1 { v21.s }[3], [ x5 ], x6
        
    
        
            st1 { v22.s }[0], [ x5 ], x6
        
            st1 { v22.s }[1], [ x5 ], x6
        
            st1 { v22.s }[2], [ x5 ], x6
        
            st1 { v22.s }[3], [ x5 ], x6
        
    
        
            st1 { v23.s }[0], [ x5 ], x6
        
            st1 { v23.s }[1], [ x5 ], x6
        
            st1 { v23.s }[2], [ x5 ], x6
        
            st1 { v23.s }[3], [ x5 ], x6
        
    
        
            st1 { v24.s }[0], [ x5 ], x6
        
            st1 { v24.s }[1], [ x5 ], x6
        
            st1 { v24.s }[2], [ x5 ], x6
        
            st1 { v24.s }[3], [ x5 ], x6
        
    
        
            st1 { v25.s }[0], [ x5 ], x6
        
            st1 { v25.s }[1], [ x5 ], x6
        
            st1 { v25.s }[2], [ x5 ], x6
        
            st1 { v25.s }[3], [ x5 ], x6
        
    
        
            st1 { v26.s }[0], [ x5 ], x6
        
            st1 { v26.s }[1], [ x5 ], x6
        
            st1 { v26.s }[2], [ x5 ], x6
        
            st1 { v26.s }[3], [ x5 ], x6
        
    
        
            st1 { v27.s }[0], [ x5 ], x6
        
            st1 { v27.s }[1], [ x5 ], x6
        
            st1 { v27.s }[2], [ x5 ], x6
        
            st1 { v27.s }[3], [ x5 ], x6
        
    
        
            st1 { v28.s }[0], [ x5 ], x6
        
            st1 { v28.s }[1], [ x5 ], x6
        
            st1 { v28.s }[2], [ x5 ], x6
        
            st1 { v28.s }[3], [ x5 ], x6
        
    
        
            st1 { v29.s }[0], [ x5 ], x6
        
            st1 { v29.s }[1], [ x5 ], x6
        
            st1 { v29.s }[2], [ x5 ], x6
        
            st1 { v29.s }[3], [ x5 ], x6
        
    
        
            st1 { v30.s }[0], [ x5 ], x6
        
            st1 { v30.s }[1], [ x5 ], x6
        
            st1 { v30.s }[2], [ x5 ], x6
        
            st1 { v30.s }[3], [ x5 ], x6
        
    
        
            st1 { v31.s }[0], [ x5 ], x6
        
            st1 { v31.s }[1], [ x5 ], x6
        
            st1 { v31.s }[2], [ x5 ], x6
        
            st1 { v31.s }[3], [ x5 ], x6
        
    
    b           .non_linear_loop

.store_strides_contig:

    
        st1 { v16.4s }, [ x5 ], #16
    
        st1 { v17.4s }, [ x5 ], #16
    
        st1 { v18.4s }, [ x5 ], #16
    
        st1 { v19.4s }, [ x5 ], #16
    
        st1 { v20.4s }, [ x5 ], #16
    
        st1 { v21.4s }, [ x5 ], #16
    
        st1 { v22.4s }, [ x5 ], #16
    
        st1 { v23.4s }, [ x5 ], #16
    
        st1 { v24.4s }, [ x5 ], #16
    
        st1 { v25.4s }, [ x5 ], #16
    
        st1 { v26.4s }, [ x5 ], #16
    
        st1 { v27.4s }, [ x5 ], #16
    
        st1 { v28.4s }, [ x5 ], #16
    
        st1 { v29.4s }, [ x5 ], #16
    
        st1 { v30.4s }, [ x5 ], #16
    
        st1 { v31.4s }, [ x5 ], #16
    
    b           .non_linear_loop

.return:

    ldp         d14, d15, [sp], #16
    ldp         d12, d13, [sp], #16
    ldp         d10, d11, [sp], #16
    ldp         d8, d9, [sp], #16

    ldp         x26, x27, [sp], #16
    ldp         x24, x25, [sp], #16
    ldp         x22, x23, [sp], #16
    ldp         x20, x21, [sp], #16

    ret
