// vim: ft=arm

// C tile regs: v16 to v31, no need to preserve

// no preservation either for v0-v7...
// v8..v15 are callee-preserved
// packed A buffering (2x8 values): alternating v0, v1 with v2, v3
// packed B buffering (2x8 values): alternating v4, v5 with v6, v7

.text
.align 4

.global _arm64simd_mmm_f32_32x3_a53_0_23_4_pre
_arm64simd_mmm_f32_32x3_a53_0_23_4_pre:

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
    ldp         x2, x4, [x0, #24]   // b, packing
    ldp         x3, x1, [x0, #8]    // k, a

    cmp         x3, #0
    beq         .non_linear_loop

    cmp         x4, #1
    beq         .f32f16
    cmp         x4, #2
    beq         .f16f32
    cmp         x4, #3
    beq         .f16f16

.p2align 4
.packed_packed_loop_1:
    ld1         { v7.4s }, [ x2 ]
    ld1         { v0.4s, v1.4s, v2.4s, v3.4s }, [ x1 ], #64
    ld1         { v4.4s, v5.4s, v6.4s }, [ x1 ], #48
    add         x2, x2, #12


    fmla        v8.4s, v0.4s, v7.s[0]

    fmla        v16.4s, v0.4s, v7.s[1]

    fmla        v24.4s, v0.4s, v7.s[2]


    ld1         { v0.4s }, [ x1 ], #16


    
        fmla        v9.4s, v1.4s, v7.s[0]
    
        fmla        v17.4s, v1.4s, v7.s[1]
    
        fmla        v25.4s, v1.4s, v7.s[2]
    

    
        fmla        v10.4s, v2.4s, v7.s[0]
    
        fmla        v18.4s, v2.4s, v7.s[1]
    
        fmla        v26.4s, v2.4s, v7.s[2]
    

    
        fmla        v11.4s, v3.4s, v7.s[0]
    
        fmla        v19.4s, v3.4s, v7.s[1]
    
        fmla        v27.4s, v3.4s, v7.s[2]
    

    
        fmla        v12.4s, v4.4s, v7.s[0]
    
        fmla        v20.4s, v4.4s, v7.s[1]
    
        fmla        v28.4s, v4.4s, v7.s[2]
    

    
        fmla        v13.4s, v5.4s, v7.s[0]
    
        fmla        v21.4s, v5.4s, v7.s[1]
    
        fmla        v29.4s, v5.4s, v7.s[2]
    

    
        fmla        v14.4s, v6.4s, v7.s[0]
    
        fmla        v22.4s, v6.4s, v7.s[1]
    
        fmla        v30.4s, v6.4s, v7.s[2]
    



    fmla        v15.4s, v0.4s, v7.s[0]

    fmla        v23.4s, v0.4s, v7.s[1]

    fmla        v31.4s, v0.4s, v7.s[2]


    subs        x3, x3, #1
    bne         .packed_packed_loop_1

    b           .non_linear_loop

.p2align 4
.f32f16:
    ld1         { v7.4h }, [ x2 ]
    ld1         { v0.4s, v1.4s, v2.4s, v3.4s }, [ x1 ], #64
    ld1         { v4.4s, v5.4s, v6.4s }, [ x1 ], #48
    fcvtl       v7.4s, v7.4h
    add         x2, x2, #6


    fmla        v8.4s, v0.4s, v7.s[0]

    fmla        v16.4s, v0.4s, v7.s[1]

    fmla        v24.4s, v0.4s, v7.s[2]


    ld1         { v0.4s }, [ x1 ], #16


    
        fmla        v9.4s, v1.4s, v7.s[0]
    
        fmla        v17.4s, v1.4s, v7.s[1]
    
        fmla        v25.4s, v1.4s, v7.s[2]
    

    
        fmla        v10.4s, v2.4s, v7.s[0]
    
        fmla        v18.4s, v2.4s, v7.s[1]
    
        fmla        v26.4s, v2.4s, v7.s[2]
    

    
        fmla        v11.4s, v3.4s, v7.s[0]
    
        fmla        v19.4s, v3.4s, v7.s[1]
    
        fmla        v27.4s, v3.4s, v7.s[2]
    

    
        fmla        v12.4s, v4.4s, v7.s[0]
    
        fmla        v20.4s, v4.4s, v7.s[1]
    
        fmla        v28.4s, v4.4s, v7.s[2]
    

    
        fmla        v13.4s, v5.4s, v7.s[0]
    
        fmla        v21.4s, v5.4s, v7.s[1]
    
        fmla        v29.4s, v5.4s, v7.s[2]
    

    
        fmla        v14.4s, v6.4s, v7.s[0]
    
        fmla        v22.4s, v6.4s, v7.s[1]
    
        fmla        v30.4s, v6.4s, v7.s[2]
    



    fmla        v15.4s, v0.4s, v7.s[0]

    fmla        v23.4s, v0.4s, v7.s[1]

    fmla        v31.4s, v0.4s, v7.s[2]


    subs        x3, x3, #1
    bne         .f32f16

    b           .non_linear_loop

.p2align 4
.f16f32:
    ld1         { v7.4s }, [ x2 ]
    ld1         { v0.8h, v1.8h, v2.8h, v3.8h }, [ x1 ], #64
    add         x2, x2, #12

    fcvtl       v4.4s, v0.4h
    fcvtl2      v5.4s, v0.8h
    fcvtl       v6.4s, v1.4h
    fcvtl2      v0.4s, v1.8h

    
        fmla        v8.4s, v4.4s, v7.s[0]
        fmla        v9.4s, v5.4s, v7.s[0]
        fmla        v10.4s, v6.4s, v7.s[0]
        fmla        v11.4s, v0.4s, v7.s[0]
    
        fmla        v16.4s, v4.4s, v7.s[1]
        fmla        v17.4s, v5.4s, v7.s[1]
        fmla        v18.4s, v6.4s, v7.s[1]
        fmla        v19.4s, v0.4s, v7.s[1]
    
        fmla        v24.4s, v4.4s, v7.s[2]
        fmla        v25.4s, v5.4s, v7.s[2]
        fmla        v26.4s, v6.4s, v7.s[2]
        fmla        v27.4s, v0.4s, v7.s[2]
    

    fcvtl       v4.4s, v2.4h
    fcvtl2      v5.4s, v2.8h
    fcvtl       v6.4s, v3.4h
    fcvtl2      v1.4s, v3.8h
    
    
        fmla        v12.4s, v4.4s, v7.s[0]
        fmla        v13.4s, v5.4s, v7.s[0]
        fmla        v14.4s, v6.4s, v7.s[0]
        fmla        v15.4s, v1.4s, v7.s[0]
    
        fmla        v20.4s, v4.4s, v7.s[1]
        fmla        v21.4s, v5.4s, v7.s[1]
        fmla        v22.4s, v6.4s, v7.s[1]
        fmla        v23.4s, v1.4s, v7.s[1]
    
        fmla        v28.4s, v4.4s, v7.s[2]
        fmla        v29.4s, v5.4s, v7.s[2]
        fmla        v30.4s, v6.4s, v7.s[2]
        fmla        v31.4s, v1.4s, v7.s[2]
    

    subs        x3, x3, #1
    bne         .f16f32

    b           .non_linear_loop

.p2align 4
.f16f16:
    ld1         { v7.4h }, [ x2 ]
    ld1         { v0.8h, v1.8h, v2.8h, v3.8h }, [ x1 ], #64
    add         x2, x2, #6

    fcvtl       v7.4s, v7.4h

    fcvtl       v4.4s, v0.4h
    fcvtl2      v5.4s, v0.8h
    fcvtl       v6.4s, v1.4h
    fcvtl2      v0.4s, v1.8h

    
        fmla        v8.4s, v4.4s, v7.s[0]
        fmla        v9.4s, v5.4s, v7.s[0]
        fmla        v10.4s, v6.4s, v7.s[0]
        fmla        v11.4s, v0.4s, v7.s[0]
    
        fmla        v16.4s, v4.4s, v7.s[1]
        fmla        v17.4s, v5.4s, v7.s[1]
        fmla        v18.4s, v6.4s, v7.s[1]
        fmla        v19.4s, v0.4s, v7.s[1]
    
        fmla        v24.4s, v4.4s, v7.s[2]
        fmla        v25.4s, v5.4s, v7.s[2]
        fmla        v26.4s, v6.4s, v7.s[2]
        fmla        v27.4s, v0.4s, v7.s[2]
    

    fcvtl       v4.4s, v2.4h
    fcvtl2      v5.4s, v2.8h
    fcvtl       v6.4s, v3.4h
    fcvtl2      v1.4s, v3.8h
    
    
        fmla        v12.4s, v4.4s, v7.s[0]
        fmla        v13.4s, v5.4s, v7.s[0]
        fmla        v14.4s, v6.4s, v7.s[0]
        fmla        v15.4s, v1.4s, v7.s[0]
    
        fmla        v20.4s, v4.4s, v7.s[1]
        fmla        v21.4s, v5.4s, v7.s[1]
        fmla        v22.4s, v6.4s, v7.s[1]
        fmla        v23.4s, v1.4s, v7.s[1]
    
        fmla        v28.4s, v4.4s, v7.s[2]
        fmla        v29.4s, v5.4s, v7.s[2]
        fmla        v30.4s, v6.4s, v7.s[2]
        fmla        v31.4s, v1.4s, v7.s[2]
    

    subs        x3, x3, #1
    bne         .f16f16

    b           .non_linear_loop


// vim: ft=arm




.scalar_min:
    add         x2, x0, #8
    ld1         {v0.s}[0], [ x2 ]
    dup         v0.4s, v0.s[0]
    
        
            fmin       v8.4s, v0.4s, v8.4s
        
            fmin       v9.4s, v0.4s, v9.4s
        
            fmin       v10.4s, v0.4s, v10.4s
        
            fmin       v11.4s, v0.4s, v11.4s
        
            fmin       v12.4s, v0.4s, v12.4s
        
            fmin       v13.4s, v0.4s, v13.4s
        
            fmin       v14.4s, v0.4s, v14.4s
        
            fmin       v15.4s, v0.4s, v15.4s
        
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
    
        
            fmax       v8.4s, v0.4s, v8.4s
        
            fmax       v9.4s, v0.4s, v9.4s
        
            fmax       v10.4s, v0.4s, v10.4s
        
            fmax       v11.4s, v0.4s, v11.4s
        
            fmax       v12.4s, v0.4s, v12.4s
        
            fmax       v13.4s, v0.4s, v13.4s
        
            fmax       v14.4s, v0.4s, v14.4s
        
            fmax       v15.4s, v0.4s, v15.4s
        
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
    
        
            fmul       v8.4s, v0.4s, v8.4s
        
            fmul       v9.4s, v0.4s, v9.4s
        
            fmul       v10.4s, v0.4s, v10.4s
        
            fmul       v11.4s, v0.4s, v11.4s
        
            fmul       v12.4s, v0.4s, v12.4s
        
            fmul       v13.4s, v0.4s, v13.4s
        
            fmul       v14.4s, v0.4s, v14.4s
        
            fmul       v15.4s, v0.4s, v15.4s
        
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
    
        
            fadd       v8.4s, v0.4s, v8.4s
        
            fadd       v9.4s, v0.4s, v9.4s
        
            fadd       v10.4s, v0.4s, v10.4s
        
            fadd       v11.4s, v0.4s, v11.4s
        
            fadd       v12.4s, v0.4s, v12.4s
        
            fadd       v13.4s, v0.4s, v13.4s
        
            fadd       v14.4s, v0.4s, v14.4s
        
            fadd       v15.4s, v0.4s, v15.4s
        
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
    
        
            fsub       v8.4s, v0.4s, v8.4s
        
            fsub       v9.4s, v0.4s, v9.4s
        
            fsub       v10.4s, v0.4s, v10.4s
        
            fsub       v11.4s, v0.4s, v11.4s
        
            fsub       v12.4s, v0.4s, v12.4s
        
            fsub       v13.4s, v0.4s, v13.4s
        
            fsub       v14.4s, v0.4s, v14.4s
        
            fsub       v15.4s, v0.4s, v15.4s
        
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
    
        
            fsub       v8.4s, v8.4s, v0.4s
        
            fsub       v9.4s, v9.4s, v0.4s
        
            fsub       v10.4s, v10.4s, v0.4s
        
            fsub       v11.4s, v11.4s, v0.4s
        
            fsub       v12.4s, v12.4s, v0.4s
        
            fsub       v13.4s, v13.4s, v0.4s
        
            fsub       v14.4s, v14.4s, v0.4s
        
            fsub       v15.4s, v15.4s, v0.4s
        
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

    eor         v8.8b, v8.8b, v8.8b

    eor         v9.8b, v9.8b, v9.8b

    eor         v10.8b, v10.8b, v10.8b

    eor         v11.8b, v11.8b, v11.8b

    eor         v12.8b, v12.8b, v12.8b

    eor         v13.8b, v13.8b, v13.8b

    eor         v14.8b, v14.8b, v14.8b

    eor         v15.8b, v15.8b, v15.8b

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
    
        fmul  v0.4s, v8.4s, v4.4s
        fcmge v1.4s, v8.4s, #0.0
        bsl   v1.16b, v8.16b, v0.16b
        and   v8.16b, v1.16b, v1.16b
    
        fmul  v0.4s, v9.4s, v4.4s
        fcmge v1.4s, v9.4s, #0.0
        bsl   v1.16b, v9.16b, v0.16b
        and   v9.16b, v1.16b, v1.16b
    
        fmul  v0.4s, v10.4s, v4.4s
        fcmge v1.4s, v10.4s, #0.0
        bsl   v1.16b, v10.16b, v0.16b
        and   v10.16b, v1.16b, v1.16b
    
        fmul  v0.4s, v11.4s, v4.4s
        fcmge v1.4s, v11.4s, #0.0
        bsl   v1.16b, v11.16b, v0.16b
        and   v11.16b, v1.16b, v1.16b
    
        fmul  v0.4s, v12.4s, v4.4s
        fcmge v1.4s, v12.4s, #0.0
        bsl   v1.16b, v12.16b, v0.16b
        and   v12.16b, v1.16b, v1.16b
    
        fmul  v0.4s, v13.4s, v4.4s
        fcmge v1.4s, v13.4s, #0.0
        bsl   v1.16b, v13.16b, v0.16b
        and   v13.16b, v1.16b, v1.16b
    
        fmul  v0.4s, v14.4s, v4.4s
        fcmge v1.4s, v14.4s, #0.0
        bsl   v1.16b, v14.16b, v0.16b
        and   v14.16b, v1.16b, v1.16b
    
        fmul  v0.4s, v15.4s, v4.4s
        fcmge v1.4s, v15.4s, #0.0
        bsl   v1.16b, v15.16b, v0.16b
        and   v15.16b, v1.16b, v1.16b
    
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



    
        
        fmin v8.4s, v0.4s, v8.4s
    
        
        fmin v9.4s, v1.4s, v9.4s
    
        
        fmin v10.4s, v2.4s, v10.4s
    
        
        fmin v11.4s, v3.4s, v11.4s
    
        
        fmin v12.4s, v4.4s, v12.4s
    
        
        fmin v13.4s, v5.4s, v13.4s
    
        
        fmin v14.4s, v6.4s, v14.4s
    
        
        fmin v15.4s, v7.4s, v15.4s
    
        
        fmin v16.4s, v0.4s, v16.4s
    
        
        fmin v17.4s, v1.4s, v17.4s
    
        
        fmin v18.4s, v2.4s, v18.4s
    
        
        fmin v19.4s, v3.4s, v19.4s
    
        
        fmin v20.4s, v4.4s, v20.4s
    
        
        fmin v21.4s, v5.4s, v21.4s
    
        
        fmin v22.4s, v6.4s, v22.4s
    
        
        fmin v23.4s, v7.4s, v23.4s
    
        
        fmin v24.4s, v0.4s, v24.4s
    
        
        fmin v25.4s, v1.4s, v25.4s
    
        
        fmin v26.4s, v2.4s, v26.4s
    
        
        fmin v27.4s, v3.4s, v27.4s
    
        
        fmin v28.4s, v4.4s, v28.4s
    
        
        fmin v29.4s, v5.4s, v29.4s
    
        
        fmin v30.4s, v6.4s, v30.4s
    
        
        fmin v31.4s, v7.4s, v31.4s
    


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



    
        
        fmax v8.4s, v0.4s, v8.4s
    
        
        fmax v9.4s, v1.4s, v9.4s
    
        
        fmax v10.4s, v2.4s, v10.4s
    
        
        fmax v11.4s, v3.4s, v11.4s
    
        
        fmax v12.4s, v4.4s, v12.4s
    
        
        fmax v13.4s, v5.4s, v13.4s
    
        
        fmax v14.4s, v6.4s, v14.4s
    
        
        fmax v15.4s, v7.4s, v15.4s
    
        
        fmax v16.4s, v0.4s, v16.4s
    
        
        fmax v17.4s, v1.4s, v17.4s
    
        
        fmax v18.4s, v2.4s, v18.4s
    
        
        fmax v19.4s, v3.4s, v19.4s
    
        
        fmax v20.4s, v4.4s, v20.4s
    
        
        fmax v21.4s, v5.4s, v21.4s
    
        
        fmax v22.4s, v6.4s, v22.4s
    
        
        fmax v23.4s, v7.4s, v23.4s
    
        
        fmax v24.4s, v0.4s, v24.4s
    
        
        fmax v25.4s, v1.4s, v25.4s
    
        
        fmax v26.4s, v2.4s, v26.4s
    
        
        fmax v27.4s, v3.4s, v27.4s
    
        
        fmax v28.4s, v4.4s, v28.4s
    
        
        fmax v29.4s, v5.4s, v29.4s
    
        
        fmax v30.4s, v6.4s, v30.4s
    
        
        fmax v31.4s, v7.4s, v31.4s
    


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



    
        
        fmul v8.4s, v0.4s, v8.4s
    
        
        fmul v9.4s, v1.4s, v9.4s
    
        
        fmul v10.4s, v2.4s, v10.4s
    
        
        fmul v11.4s, v3.4s, v11.4s
    
        
        fmul v12.4s, v4.4s, v12.4s
    
        
        fmul v13.4s, v5.4s, v13.4s
    
        
        fmul v14.4s, v6.4s, v14.4s
    
        
        fmul v15.4s, v7.4s, v15.4s
    
        
        fmul v16.4s, v0.4s, v16.4s
    
        
        fmul v17.4s, v1.4s, v17.4s
    
        
        fmul v18.4s, v2.4s, v18.4s
    
        
        fmul v19.4s, v3.4s, v19.4s
    
        
        fmul v20.4s, v4.4s, v20.4s
    
        
        fmul v21.4s, v5.4s, v21.4s
    
        
        fmul v22.4s, v6.4s, v22.4s
    
        
        fmul v23.4s, v7.4s, v23.4s
    
        
        fmul v24.4s, v0.4s, v24.4s
    
        
        fmul v25.4s, v1.4s, v25.4s
    
        
        fmul v26.4s, v2.4s, v26.4s
    
        
        fmul v27.4s, v3.4s, v27.4s
    
        
        fmul v28.4s, v4.4s, v28.4s
    
        
        fmul v29.4s, v5.4s, v29.4s
    
        
        fmul v30.4s, v6.4s, v30.4s
    
        
        fmul v31.4s, v7.4s, v31.4s
    


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



    
        
        fadd v8.4s, v0.4s, v8.4s
    
        
        fadd v9.4s, v1.4s, v9.4s
    
        
        fadd v10.4s, v2.4s, v10.4s
    
        
        fadd v11.4s, v3.4s, v11.4s
    
        
        fadd v12.4s, v4.4s, v12.4s
    
        
        fadd v13.4s, v5.4s, v13.4s
    
        
        fadd v14.4s, v6.4s, v14.4s
    
        
        fadd v15.4s, v7.4s, v15.4s
    
        
        fadd v16.4s, v0.4s, v16.4s
    
        
        fadd v17.4s, v1.4s, v17.4s
    
        
        fadd v18.4s, v2.4s, v18.4s
    
        
        fadd v19.4s, v3.4s, v19.4s
    
        
        fadd v20.4s, v4.4s, v20.4s
    
        
        fadd v21.4s, v5.4s, v21.4s
    
        
        fadd v22.4s, v6.4s, v22.4s
    
        
        fadd v23.4s, v7.4s, v23.4s
    
        
        fadd v24.4s, v0.4s, v24.4s
    
        
        fadd v25.4s, v1.4s, v25.4s
    
        
        fadd v26.4s, v2.4s, v26.4s
    
        
        fadd v27.4s, v3.4s, v27.4s
    
        
        fadd v28.4s, v4.4s, v28.4s
    
        
        fadd v29.4s, v5.4s, v29.4s
    
        
        fadd v30.4s, v6.4s, v30.4s
    
        
        fadd v31.4s, v7.4s, v31.4s
    


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



    
        
        fsub v8.4s, v0.4s, v8.4s
    
        
        fsub v9.4s, v1.4s, v9.4s
    
        
        fsub v10.4s, v2.4s, v10.4s
    
        
        fsub v11.4s, v3.4s, v11.4s
    
        
        fsub v12.4s, v4.4s, v12.4s
    
        
        fsub v13.4s, v5.4s, v13.4s
    
        
        fsub v14.4s, v6.4s, v14.4s
    
        
        fsub v15.4s, v7.4s, v15.4s
    
        
        fsub v16.4s, v0.4s, v16.4s
    
        
        fsub v17.4s, v1.4s, v17.4s
    
        
        fsub v18.4s, v2.4s, v18.4s
    
        
        fsub v19.4s, v3.4s, v19.4s
    
        
        fsub v20.4s, v4.4s, v20.4s
    
        
        fsub v21.4s, v5.4s, v21.4s
    
        
        fsub v22.4s, v6.4s, v22.4s
    
        
        fsub v23.4s, v7.4s, v23.4s
    
        
        fsub v24.4s, v0.4s, v24.4s
    
        
        fsub v25.4s, v1.4s, v25.4s
    
        
        fsub v26.4s, v2.4s, v26.4s
    
        
        fsub v27.4s, v3.4s, v27.4s
    
        
        fsub v28.4s, v4.4s, v28.4s
    
        
        fsub v29.4s, v5.4s, v29.4s
    
        
        fsub v30.4s, v6.4s, v30.4s
    
        
        fsub v31.4s, v7.4s, v31.4s
    


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



    
        
        fsub v8.4s, v8.4s, v0.4s
    
        
        fsub v9.4s, v9.4s, v1.4s
    
        
        fsub v10.4s, v10.4s, v2.4s
    
        
        fsub v11.4s, v11.4s, v3.4s
    
        
        fsub v12.4s, v12.4s, v4.4s
    
        
        fsub v13.4s, v13.4s, v5.4s
    
        
        fsub v14.4s, v14.4s, v6.4s
    
        
        fsub v15.4s, v15.4s, v7.4s
    
        
        fsub v16.4s, v16.4s, v0.4s
    
        
        fsub v17.4s, v17.4s, v1.4s
    
        
        fsub v18.4s, v18.4s, v2.4s
    
        
        fsub v19.4s, v19.4s, v3.4s
    
        
        fsub v20.4s, v20.4s, v4.4s
    
        
        fsub v21.4s, v21.4s, v5.4s
    
        
        fsub v22.4s, v22.4s, v6.4s
    
        
        fsub v23.4s, v23.4s, v7.4s
    
        
        fsub v24.4s, v24.4s, v0.4s
    
        
        fsub v25.4s, v25.4s, v1.4s
    
        
        fsub v26.4s, v26.4s, v2.4s
    
        
        fsub v27.4s, v27.4s, v3.4s
    
        
        fsub v28.4s, v28.4s, v4.4s
    
        
        fsub v29.4s, v29.4s, v5.4s
    
        
        fsub v30.4s, v30.4s, v6.4s
    
        
        fsub v31.4s, v31.4s, v7.4s
    


b           .non_linear_loop

// vim: ft=arm




.per_col_min:
    ldr         x2, [x0, #8]







        ldr         d0, [ x2 ], #8
        ld1         {v0.s}[2], [ x2 ]


// 32 3


    dup v3.4s, v0.s[0]
    
        
        
            fmin v8.4s, v3.4s, v8.4s
        
    
        
        
            fmin v9.4s, v3.4s, v9.4s
        
    
        
        
            fmin v10.4s, v3.4s, v10.4s
        
    
        
        
            fmin v11.4s, v3.4s, v11.4s
        
    
        
        
            fmin v12.4s, v3.4s, v12.4s
        
    
        
        
            fmin v13.4s, v3.4s, v13.4s
        
    
        
        
            fmin v14.4s, v3.4s, v14.4s
        
    
        
        
            fmin v15.4s, v3.4s, v15.4s
        
    

    dup v3.4s, v0.s[1]
    
        
        
            fmin v16.4s, v3.4s, v16.4s
        
    
        
        
            fmin v17.4s, v3.4s, v17.4s
        
    
        
        
            fmin v18.4s, v3.4s, v18.4s
        
    
        
        
            fmin v19.4s, v3.4s, v19.4s
        
    
        
        
            fmin v20.4s, v3.4s, v20.4s
        
    
        
        
            fmin v21.4s, v3.4s, v21.4s
        
    
        
        
            fmin v22.4s, v3.4s, v22.4s
        
    
        
        
            fmin v23.4s, v3.4s, v23.4s
        
    

    dup v3.4s, v0.s[2]
    
        
        
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







        ldr         d0, [ x2 ], #8
        ld1         {v0.s}[2], [ x2 ]


// 32 3


    dup v3.4s, v0.s[0]
    
        
        
            fmax v8.4s, v3.4s, v8.4s
        
    
        
        
            fmax v9.4s, v3.4s, v9.4s
        
    
        
        
            fmax v10.4s, v3.4s, v10.4s
        
    
        
        
            fmax v11.4s, v3.4s, v11.4s
        
    
        
        
            fmax v12.4s, v3.4s, v12.4s
        
    
        
        
            fmax v13.4s, v3.4s, v13.4s
        
    
        
        
            fmax v14.4s, v3.4s, v14.4s
        
    
        
        
            fmax v15.4s, v3.4s, v15.4s
        
    

    dup v3.4s, v0.s[1]
    
        
        
            fmax v16.4s, v3.4s, v16.4s
        
    
        
        
            fmax v17.4s, v3.4s, v17.4s
        
    
        
        
            fmax v18.4s, v3.4s, v18.4s
        
    
        
        
            fmax v19.4s, v3.4s, v19.4s
        
    
        
        
            fmax v20.4s, v3.4s, v20.4s
        
    
        
        
            fmax v21.4s, v3.4s, v21.4s
        
    
        
        
            fmax v22.4s, v3.4s, v22.4s
        
    
        
        
            fmax v23.4s, v3.4s, v23.4s
        
    

    dup v3.4s, v0.s[2]
    
        
        
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







        ldr         d0, [ x2 ], #8
        ld1         {v0.s}[2], [ x2 ]


// 32 3


    dup v3.4s, v0.s[0]
    
        
        
            fmul v8.4s, v3.4s, v8.4s
        
    
        
        
            fmul v9.4s, v3.4s, v9.4s
        
    
        
        
            fmul v10.4s, v3.4s, v10.4s
        
    
        
        
            fmul v11.4s, v3.4s, v11.4s
        
    
        
        
            fmul v12.4s, v3.4s, v12.4s
        
    
        
        
            fmul v13.4s, v3.4s, v13.4s
        
    
        
        
            fmul v14.4s, v3.4s, v14.4s
        
    
        
        
            fmul v15.4s, v3.4s, v15.4s
        
    

    dup v3.4s, v0.s[1]
    
        
        
            fmul v16.4s, v3.4s, v16.4s
        
    
        
        
            fmul v17.4s, v3.4s, v17.4s
        
    
        
        
            fmul v18.4s, v3.4s, v18.4s
        
    
        
        
            fmul v19.4s, v3.4s, v19.4s
        
    
        
        
            fmul v20.4s, v3.4s, v20.4s
        
    
        
        
            fmul v21.4s, v3.4s, v21.4s
        
    
        
        
            fmul v22.4s, v3.4s, v22.4s
        
    
        
        
            fmul v23.4s, v3.4s, v23.4s
        
    

    dup v3.4s, v0.s[2]
    
        
        
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







        ldr         d0, [ x2 ], #8
        ld1         {v0.s}[2], [ x2 ]


// 32 3


    dup v3.4s, v0.s[0]
    
        
        
            fadd v8.4s, v3.4s, v8.4s
        
    
        
        
            fadd v9.4s, v3.4s, v9.4s
        
    
        
        
            fadd v10.4s, v3.4s, v10.4s
        
    
        
        
            fadd v11.4s, v3.4s, v11.4s
        
    
        
        
            fadd v12.4s, v3.4s, v12.4s
        
    
        
        
            fadd v13.4s, v3.4s, v13.4s
        
    
        
        
            fadd v14.4s, v3.4s, v14.4s
        
    
        
        
            fadd v15.4s, v3.4s, v15.4s
        
    

    dup v3.4s, v0.s[1]
    
        
        
            fadd v16.4s, v3.4s, v16.4s
        
    
        
        
            fadd v17.4s, v3.4s, v17.4s
        
    
        
        
            fadd v18.4s, v3.4s, v18.4s
        
    
        
        
            fadd v19.4s, v3.4s, v19.4s
        
    
        
        
            fadd v20.4s, v3.4s, v20.4s
        
    
        
        
            fadd v21.4s, v3.4s, v21.4s
        
    
        
        
            fadd v22.4s, v3.4s, v22.4s
        
    
        
        
            fadd v23.4s, v3.4s, v23.4s
        
    

    dup v3.4s, v0.s[2]
    
        
        
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







        ldr         d0, [ x2 ], #8
        ld1         {v0.s}[2], [ x2 ]


// 32 3


    dup v3.4s, v0.s[0]
    
        
        
            fsub v8.4s, v3.4s, v8.4s
        
    
        
        
            fsub v9.4s, v3.4s, v9.4s
        
    
        
        
            fsub v10.4s, v3.4s, v10.4s
        
    
        
        
            fsub v11.4s, v3.4s, v11.4s
        
    
        
        
            fsub v12.4s, v3.4s, v12.4s
        
    
        
        
            fsub v13.4s, v3.4s, v13.4s
        
    
        
        
            fsub v14.4s, v3.4s, v14.4s
        
    
        
        
            fsub v15.4s, v3.4s, v15.4s
        
    

    dup v3.4s, v0.s[1]
    
        
        
            fsub v16.4s, v3.4s, v16.4s
        
    
        
        
            fsub v17.4s, v3.4s, v17.4s
        
    
        
        
            fsub v18.4s, v3.4s, v18.4s
        
    
        
        
            fsub v19.4s, v3.4s, v19.4s
        
    
        
        
            fsub v20.4s, v3.4s, v20.4s
        
    
        
        
            fsub v21.4s, v3.4s, v21.4s
        
    
        
        
            fsub v22.4s, v3.4s, v22.4s
        
    
        
        
            fsub v23.4s, v3.4s, v23.4s
        
    

    dup v3.4s, v0.s[2]
    
        
        
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







        ldr         d0, [ x2 ], #8
        ld1         {v0.s}[2], [ x2 ]


// 32 3


    dup v3.4s, v0.s[0]
    
        
        
            fsub v8.4s, v8.4s, v3.4s
        
    
        
        
            fsub v9.4s, v9.4s, v3.4s
        
    
        
        
            fsub v10.4s, v10.4s, v3.4s
        
    
        
        
            fsub v11.4s, v11.4s, v3.4s
        
    
        
        
            fsub v12.4s, v12.4s, v3.4s
        
    
        
        
            fsub v13.4s, v13.4s, v3.4s
        
    
        
        
            fsub v14.4s, v14.4s, v3.4s
        
    
        
        
            fsub v15.4s, v15.4s, v3.4s
        
    

    dup v3.4s, v0.s[1]
    
        
        
            fsub v16.4s, v16.4s, v3.4s
        
    
        
        
            fsub v17.4s, v17.4s, v3.4s
        
    
        
        
            fsub v18.4s, v18.4s, v3.4s
        
    
        
        
            fsub v19.4s, v19.4s, v3.4s
        
    
        
        
            fsub v20.4s, v20.4s, v3.4s
        
    
        
        
            fsub v21.4s, v21.4s, v3.4s
        
    
        
        
            fsub v22.4s, v22.4s, v3.4s
        
    
        
        
            fsub v23.4s, v23.4s, v3.4s
        
    

    dup v3.4s, v0.s[2]
    
        
        
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
    
        ld1         { v8.4s }, [ x2 ], #16
    
        ld1         { v9.4s }, [ x2 ], #16
    
        ld1         { v10.4s }, [ x2 ], #16
    
        ld1         { v11.4s }, [ x2 ], #16
    
        ld1         { v12.4s }, [ x2 ], #16
    
        ld1         { v13.4s }, [ x2 ], #16
    
        ld1         { v14.4s }, [ x2 ], #16
    
        ld1         { v15.4s }, [ x2 ], #16
    
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
    ldp         x5, x6, [x0, #8]
    ldp         x7, x8, [x0, #24]

    
        mov x4, x5
        
            
                ld1 {v0.s}[0], [ x4 ], x6
            
                ld1 {v0.s}[1], [ x4 ], x6
            
                ld1 {v0.s}[2], [ x4 ], x6
            
                ld1 {v0.s}[3], [ x4 ], x6
            
            fadd v8.4s, v8.4s, v0.4s
        
            
                ld1 {v0.s}[0], [ x4 ], x6
            
                ld1 {v0.s}[1], [ x4 ], x6
            
                ld1 {v0.s}[2], [ x4 ], x6
            
                ld1 {v0.s}[3], [ x4 ], x6
            
            fadd v9.4s, v9.4s, v0.4s
        
            
                ld1 {v0.s}[0], [ x4 ], x6
            
                ld1 {v0.s}[1], [ x4 ], x6
            
                ld1 {v0.s}[2], [ x4 ], x6
            
                ld1 {v0.s}[3], [ x4 ], x6
            
            fadd v10.4s, v10.4s, v0.4s
        
            
                ld1 {v0.s}[0], [ x4 ], x6
            
                ld1 {v0.s}[1], [ x4 ], x6
            
                ld1 {v0.s}[2], [ x4 ], x6
            
                ld1 {v0.s}[3], [ x4 ], x6
            
            fadd v11.4s, v11.4s, v0.4s
        
            
                ld1 {v0.s}[0], [ x4 ], x6
            
                ld1 {v0.s}[1], [ x4 ], x6
            
                ld1 {v0.s}[2], [ x4 ], x6
            
                ld1 {v0.s}[3], [ x4 ], x6
            
            fadd v12.4s, v12.4s, v0.4s
        
            
                ld1 {v0.s}[0], [ x4 ], x6
            
                ld1 {v0.s}[1], [ x4 ], x6
            
                ld1 {v0.s}[2], [ x4 ], x6
            
                ld1 {v0.s}[3], [ x4 ], x6
            
            fadd v13.4s, v13.4s, v0.4s
        
            
                ld1 {v0.s}[0], [ x4 ], x6
            
                ld1 {v0.s}[1], [ x4 ], x6
            
                ld1 {v0.s}[2], [ x4 ], x6
            
                ld1 {v0.s}[3], [ x4 ], x6
            
            fadd v14.4s, v14.4s, v0.4s
        
            
                ld1 {v0.s}[0], [ x4 ], x6
            
                ld1 {v0.s}[1], [ x4 ], x6
            
                ld1 {v0.s}[2], [ x4 ], x6
            
                ld1 {v0.s}[3], [ x4 ], x6
            
            fadd v15.4s, v15.4s, v0.4s
        
        add x5, x5, x7
    
        mov x4, x5
        
            
                ld1 {v0.s}[0], [ x4 ], x6
            
                ld1 {v0.s}[1], [ x4 ], x6
            
                ld1 {v0.s}[2], [ x4 ], x6
            
                ld1 {v0.s}[3], [ x4 ], x6
            
            fadd v16.4s, v16.4s, v0.4s
        
            
                ld1 {v0.s}[0], [ x4 ], x6
            
                ld1 {v0.s}[1], [ x4 ], x6
            
                ld1 {v0.s}[2], [ x4 ], x6
            
                ld1 {v0.s}[3], [ x4 ], x6
            
            fadd v17.4s, v17.4s, v0.4s
        
            
                ld1 {v0.s}[0], [ x4 ], x6
            
                ld1 {v0.s}[1], [ x4 ], x6
            
                ld1 {v0.s}[2], [ x4 ], x6
            
                ld1 {v0.s}[3], [ x4 ], x6
            
            fadd v18.4s, v18.4s, v0.4s
        
            
                ld1 {v0.s}[0], [ x4 ], x6
            
                ld1 {v0.s}[1], [ x4 ], x6
            
                ld1 {v0.s}[2], [ x4 ], x6
            
                ld1 {v0.s}[3], [ x4 ], x6
            
            fadd v19.4s, v19.4s, v0.4s
        
            
                ld1 {v0.s}[0], [ x4 ], x6
            
                ld1 {v0.s}[1], [ x4 ], x6
            
                ld1 {v0.s}[2], [ x4 ], x6
            
                ld1 {v0.s}[3], [ x4 ], x6
            
            fadd v20.4s, v20.4s, v0.4s
        
            
                ld1 {v0.s}[0], [ x4 ], x6
            
                ld1 {v0.s}[1], [ x4 ], x6
            
                ld1 {v0.s}[2], [ x4 ], x6
            
                ld1 {v0.s}[3], [ x4 ], x6
            
            fadd v21.4s, v21.4s, v0.4s
        
            
                ld1 {v0.s}[0], [ x4 ], x6
            
                ld1 {v0.s}[1], [ x4 ], x6
            
                ld1 {v0.s}[2], [ x4 ], x6
            
                ld1 {v0.s}[3], [ x4 ], x6
            
            fadd v22.4s, v22.4s, v0.4s
        
            
                ld1 {v0.s}[0], [ x4 ], x6
            
                ld1 {v0.s}[1], [ x4 ], x6
            
                ld1 {v0.s}[2], [ x4 ], x6
            
                ld1 {v0.s}[3], [ x4 ], x6
            
            fadd v23.4s, v23.4s, v0.4s
        
        add x5, x5, x7
    
        mov x4, x5
        
            
                ld1 {v0.s}[0], [ x4 ], x6
            
                ld1 {v0.s}[1], [ x4 ], x6
            
                ld1 {v0.s}[2], [ x4 ], x6
            
                ld1 {v0.s}[3], [ x4 ], x6
            
            fadd v24.4s, v24.4s, v0.4s
        
            
                ld1 {v0.s}[0], [ x4 ], x6
            
                ld1 {v0.s}[1], [ x4 ], x6
            
                ld1 {v0.s}[2], [ x4 ], x6
            
                ld1 {v0.s}[3], [ x4 ], x6
            
            fadd v25.4s, v25.4s, v0.4s
        
            
                ld1 {v0.s}[0], [ x4 ], x6
            
                ld1 {v0.s}[1], [ x4 ], x6
            
                ld1 {v0.s}[2], [ x4 ], x6
            
                ld1 {v0.s}[3], [ x4 ], x6
            
            fadd v26.4s, v26.4s, v0.4s
        
            
                ld1 {v0.s}[0], [ x4 ], x6
            
                ld1 {v0.s}[1], [ x4 ], x6
            
                ld1 {v0.s}[2], [ x4 ], x6
            
                ld1 {v0.s}[3], [ x4 ], x6
            
            fadd v27.4s, v27.4s, v0.4s
        
            
                ld1 {v0.s}[0], [ x4 ], x6
            
                ld1 {v0.s}[1], [ x4 ], x6
            
                ld1 {v0.s}[2], [ x4 ], x6
            
                ld1 {v0.s}[3], [ x4 ], x6
            
            fadd v28.4s, v28.4s, v0.4s
        
            
                ld1 {v0.s}[0], [ x4 ], x6
            
                ld1 {v0.s}[1], [ x4 ], x6
            
                ld1 {v0.s}[2], [ x4 ], x6
            
                ld1 {v0.s}[3], [ x4 ], x6
            
            fadd v29.4s, v29.4s, v0.4s
        
            
                ld1 {v0.s}[0], [ x4 ], x6
            
                ld1 {v0.s}[1], [ x4 ], x6
            
                ld1 {v0.s}[2], [ x4 ], x6
            
                ld1 {v0.s}[3], [ x4 ], x6
            
            fadd v30.4s, v30.4s, v0.4s
        
            
                ld1 {v0.s}[0], [ x4 ], x6
            
                ld1 {v0.s}[1], [ x4 ], x6
            
                ld1 {v0.s}[2], [ x4 ], x6
            
                ld1 {v0.s}[3], [ x4 ], x6
            
            fadd v31.4s, v31.4s, v0.4s
        
        add x5, x5, x7
    

    b           .non_linear_loop

.add_row_col_products:
    ldp         x2, x3, [x0, #8]

    ld1         { v7.d }[0], [ x3 ], #8
    ld1         { v7.s }[2], [ x3 ], #4
    ld1         { v0.4s, v1.4s, v2.4s, v3.4s }, [ x2 ], #64
    ld1         { v4.4s, v5.4s, v6.4s }, [ x2 ], #48


    fmla        v8.4s, v0.4s, v7.s[0]

    fmla        v16.4s, v0.4s, v7.s[1]

    fmla        v24.4s, v0.4s, v7.s[2]


    ld1         { v0.4s }, [ x2 ], #16


    
        fmla        v9.4s, v1.4s, v7.s[0]
    
        fmla        v17.4s, v1.4s, v7.s[1]
    
        fmla        v25.4s, v1.4s, v7.s[2]
    

    
        fmla        v10.4s, v2.4s, v7.s[0]
    
        fmla        v18.4s, v2.4s, v7.s[1]
    
        fmla        v26.4s, v2.4s, v7.s[2]
    

    
        fmla        v11.4s, v3.4s, v7.s[0]
    
        fmla        v19.4s, v3.4s, v7.s[1]
    
        fmla        v27.4s, v3.4s, v7.s[2]
    

    
        fmla        v12.4s, v4.4s, v7.s[0]
    
        fmla        v20.4s, v4.4s, v7.s[1]
    
        fmla        v28.4s, v4.4s, v7.s[2]
    

    
        fmla        v13.4s, v5.4s, v7.s[0]
    
        fmla        v21.4s, v5.4s, v7.s[1]
    
        fmla        v29.4s, v5.4s, v7.s[2]
    

    
        fmla        v14.4s, v6.4s, v7.s[0]
    
        fmla        v22.4s, v6.4s, v7.s[1]
    
        fmla        v30.4s, v6.4s, v7.s[2]
    



    fmla        v15.4s, v0.4s, v7.s[0]

    fmla        v23.4s, v0.4s, v7.s[1]

    fmla        v31.4s, v0.4s, v7.s[2]


    b           .non_linear_loop

.store:
    ldp         x5, x6, [x0, #8]                // c base ptr, rsc
    ldp         x7, x8, [x0, #24]               // csc, item_size

    cmp         x8, #2
    beq         .store_f16

    cmp         x6, #4
    beq         .store_strides_contig


    
        mov x4, x5
        
            
                st1 { v8.s }[0], [ x4 ], x6
            
                st1 { v8.s }[1], [ x4 ], x6
            
                st1 { v8.s }[2], [ x4 ], x6
            
                st1 { v8.s }[3], [ x4 ], x6
            
        
            
                st1 { v9.s }[0], [ x4 ], x6
            
                st1 { v9.s }[1], [ x4 ], x6
            
                st1 { v9.s }[2], [ x4 ], x6
            
                st1 { v9.s }[3], [ x4 ], x6
            
        
            
                st1 { v10.s }[0], [ x4 ], x6
            
                st1 { v10.s }[1], [ x4 ], x6
            
                st1 { v10.s }[2], [ x4 ], x6
            
                st1 { v10.s }[3], [ x4 ], x6
            
        
            
                st1 { v11.s }[0], [ x4 ], x6
            
                st1 { v11.s }[1], [ x4 ], x6
            
                st1 { v11.s }[2], [ x4 ], x6
            
                st1 { v11.s }[3], [ x4 ], x6
            
        
            
                st1 { v12.s }[0], [ x4 ], x6
            
                st1 { v12.s }[1], [ x4 ], x6
            
                st1 { v12.s }[2], [ x4 ], x6
            
                st1 { v12.s }[3], [ x4 ], x6
            
        
            
                st1 { v13.s }[0], [ x4 ], x6
            
                st1 { v13.s }[1], [ x4 ], x6
            
                st1 { v13.s }[2], [ x4 ], x6
            
                st1 { v13.s }[3], [ x4 ], x6
            
        
            
                st1 { v14.s }[0], [ x4 ], x6
            
                st1 { v14.s }[1], [ x4 ], x6
            
                st1 { v14.s }[2], [ x4 ], x6
            
                st1 { v14.s }[3], [ x4 ], x6
            
        
            
                st1 { v15.s }[0], [ x4 ], x6
            
                st1 { v15.s }[1], [ x4 ], x6
            
                st1 { v15.s }[2], [ x4 ], x6
            
                st1 { v15.s }[3], [ x4 ], x6
            
        
        add x5, x5, x7
    
        mov x4, x5
        
            
                st1 { v16.s }[0], [ x4 ], x6
            
                st1 { v16.s }[1], [ x4 ], x6
            
                st1 { v16.s }[2], [ x4 ], x6
            
                st1 { v16.s }[3], [ x4 ], x6
            
        
            
                st1 { v17.s }[0], [ x4 ], x6
            
                st1 { v17.s }[1], [ x4 ], x6
            
                st1 { v17.s }[2], [ x4 ], x6
            
                st1 { v17.s }[3], [ x4 ], x6
            
        
            
                st1 { v18.s }[0], [ x4 ], x6
            
                st1 { v18.s }[1], [ x4 ], x6
            
                st1 { v18.s }[2], [ x4 ], x6
            
                st1 { v18.s }[3], [ x4 ], x6
            
        
            
                st1 { v19.s }[0], [ x4 ], x6
            
                st1 { v19.s }[1], [ x4 ], x6
            
                st1 { v19.s }[2], [ x4 ], x6
            
                st1 { v19.s }[3], [ x4 ], x6
            
        
            
                st1 { v20.s }[0], [ x4 ], x6
            
                st1 { v20.s }[1], [ x4 ], x6
            
                st1 { v20.s }[2], [ x4 ], x6
            
                st1 { v20.s }[3], [ x4 ], x6
            
        
            
                st1 { v21.s }[0], [ x4 ], x6
            
                st1 { v21.s }[1], [ x4 ], x6
            
                st1 { v21.s }[2], [ x4 ], x6
            
                st1 { v21.s }[3], [ x4 ], x6
            
        
            
                st1 { v22.s }[0], [ x4 ], x6
            
                st1 { v22.s }[1], [ x4 ], x6
            
                st1 { v22.s }[2], [ x4 ], x6
            
                st1 { v22.s }[3], [ x4 ], x6
            
        
            
                st1 { v23.s }[0], [ x4 ], x6
            
                st1 { v23.s }[1], [ x4 ], x6
            
                st1 { v23.s }[2], [ x4 ], x6
            
                st1 { v23.s }[3], [ x4 ], x6
            
        
        add x5, x5, x7
    
        mov x4, x5
        
            
                st1 { v24.s }[0], [ x4 ], x6
            
                st1 { v24.s }[1], [ x4 ], x6
            
                st1 { v24.s }[2], [ x4 ], x6
            
                st1 { v24.s }[3], [ x4 ], x6
            
        
            
                st1 { v25.s }[0], [ x4 ], x6
            
                st1 { v25.s }[1], [ x4 ], x6
            
                st1 { v25.s }[2], [ x4 ], x6
            
                st1 { v25.s }[3], [ x4 ], x6
            
        
            
                st1 { v26.s }[0], [ x4 ], x6
            
                st1 { v26.s }[1], [ x4 ], x6
            
                st1 { v26.s }[2], [ x4 ], x6
            
                st1 { v26.s }[3], [ x4 ], x6
            
        
            
                st1 { v27.s }[0], [ x4 ], x6
            
                st1 { v27.s }[1], [ x4 ], x6
            
                st1 { v27.s }[2], [ x4 ], x6
            
                st1 { v27.s }[3], [ x4 ], x6
            
        
            
                st1 { v28.s }[0], [ x4 ], x6
            
                st1 { v28.s }[1], [ x4 ], x6
            
                st1 { v28.s }[2], [ x4 ], x6
            
                st1 { v28.s }[3], [ x4 ], x6
            
        
            
                st1 { v29.s }[0], [ x4 ], x6
            
                st1 { v29.s }[1], [ x4 ], x6
            
                st1 { v29.s }[2], [ x4 ], x6
            
                st1 { v29.s }[3], [ x4 ], x6
            
        
            
                st1 { v30.s }[0], [ x4 ], x6
            
                st1 { v30.s }[1], [ x4 ], x6
            
                st1 { v30.s }[2], [ x4 ], x6
            
                st1 { v30.s }[3], [ x4 ], x6
            
        
            
                st1 { v31.s }[0], [ x4 ], x6
            
                st1 { v31.s }[1], [ x4 ], x6
            
                st1 { v31.s }[2], [ x4 ], x6
            
                st1 { v31.s }[3], [ x4 ], x6
            
        
        add x5, x5, x7
    
    b           .non_linear_loop

.store_strides_contig:

    
        mov x4, x5
        
            st1 { v8.4s }, [ x4 ], 16
        
            st1 { v9.4s }, [ x4 ], 16
        
            st1 { v10.4s }, [ x4 ], 16
        
            st1 { v11.4s }, [ x4 ], 16
        
            st1 { v12.4s }, [ x4 ], 16
        
            st1 { v13.4s }, [ x4 ], 16
        
            st1 { v14.4s }, [ x4 ], 16
        
            st1 { v15.4s }, [ x4 ], 16
        
        add x5, x5, x7
    
        mov x4, x5
        
            st1 { v16.4s }, [ x4 ], 16
        
            st1 { v17.4s }, [ x4 ], 16
        
            st1 { v18.4s }, [ x4 ], 16
        
            st1 { v19.4s }, [ x4 ], 16
        
            st1 { v20.4s }, [ x4 ], 16
        
            st1 { v21.4s }, [ x4 ], 16
        
            st1 { v22.4s }, [ x4 ], 16
        
            st1 { v23.4s }, [ x4 ], 16
        
        add x5, x5, x7
    
        mov x4, x5
        
            st1 { v24.4s }, [ x4 ], 16
        
            st1 { v25.4s }, [ x4 ], 16
        
            st1 { v26.4s }, [ x4 ], 16
        
            st1 { v27.4s }, [ x4 ], 16
        
            st1 { v28.4s }, [ x4 ], 16
        
            st1 { v29.4s }, [ x4 ], 16
        
            st1 { v30.4s }, [ x4 ], 16
        
            st1 { v31.4s }, [ x4 ], 16
        
        add x5, x5, x7
    

    b           .non_linear_loop

.store_f16:

    cmp         x6, #2
    beq         .store_strides_contig_f16

    
        
            fcvtn  v0.4h, v8.4s
            fcvtn2 v0.8h, v9.4s
        
            fcvtn  v1.4h, v10.4s
            fcvtn2 v1.8h, v11.4s
        
            fcvtn  v2.4h, v12.4s
            fcvtn2 v2.8h, v13.4s
        
            fcvtn  v3.4h, v14.4s
            fcvtn2 v3.8h, v15.4s
        

        mov x4, x5
        
            
                st1 { v0.h }[0], [ x4 ], x6
            
                st1 { v0.h }[1], [ x4 ], x6
            
                st1 { v0.h }[2], [ x4 ], x6
            
                st1 { v0.h }[3], [ x4 ], x6
            
                st1 { v0.h }[4], [ x4 ], x6
            
                st1 { v0.h }[5], [ x4 ], x6
            
                st1 { v0.h }[6], [ x4 ], x6
            
                st1 { v0.h }[7], [ x4 ], x6
            
        
            
                st1 { v1.h }[0], [ x4 ], x6
            
                st1 { v1.h }[1], [ x4 ], x6
            
                st1 { v1.h }[2], [ x4 ], x6
            
                st1 { v1.h }[3], [ x4 ], x6
            
                st1 { v1.h }[4], [ x4 ], x6
            
                st1 { v1.h }[5], [ x4 ], x6
            
                st1 { v1.h }[6], [ x4 ], x6
            
                st1 { v1.h }[7], [ x4 ], x6
            
        
            
                st1 { v2.h }[0], [ x4 ], x6
            
                st1 { v2.h }[1], [ x4 ], x6
            
                st1 { v2.h }[2], [ x4 ], x6
            
                st1 { v2.h }[3], [ x4 ], x6
            
                st1 { v2.h }[4], [ x4 ], x6
            
                st1 { v2.h }[5], [ x4 ], x6
            
                st1 { v2.h }[6], [ x4 ], x6
            
                st1 { v2.h }[7], [ x4 ], x6
            
        
            
                st1 { v3.h }[0], [ x4 ], x6
            
                st1 { v3.h }[1], [ x4 ], x6
            
                st1 { v3.h }[2], [ x4 ], x6
            
                st1 { v3.h }[3], [ x4 ], x6
            
                st1 { v3.h }[4], [ x4 ], x6
            
                st1 { v3.h }[5], [ x4 ], x6
            
                st1 { v3.h }[6], [ x4 ], x6
            
                st1 { v3.h }[7], [ x4 ], x6
            
        
        add x5, x5, x7

    
        
            fcvtn  v0.4h, v16.4s
            fcvtn2 v0.8h, v17.4s
        
            fcvtn  v1.4h, v18.4s
            fcvtn2 v1.8h, v19.4s
        
            fcvtn  v2.4h, v20.4s
            fcvtn2 v2.8h, v21.4s
        
            fcvtn  v3.4h, v22.4s
            fcvtn2 v3.8h, v23.4s
        

        mov x4, x5
        
            
                st1 { v0.h }[0], [ x4 ], x6
            
                st1 { v0.h }[1], [ x4 ], x6
            
                st1 { v0.h }[2], [ x4 ], x6
            
                st1 { v0.h }[3], [ x4 ], x6
            
                st1 { v0.h }[4], [ x4 ], x6
            
                st1 { v0.h }[5], [ x4 ], x6
            
                st1 { v0.h }[6], [ x4 ], x6
            
                st1 { v0.h }[7], [ x4 ], x6
            
        
            
                st1 { v1.h }[0], [ x4 ], x6
            
                st1 { v1.h }[1], [ x4 ], x6
            
                st1 { v1.h }[2], [ x4 ], x6
            
                st1 { v1.h }[3], [ x4 ], x6
            
                st1 { v1.h }[4], [ x4 ], x6
            
                st1 { v1.h }[5], [ x4 ], x6
            
                st1 { v1.h }[6], [ x4 ], x6
            
                st1 { v1.h }[7], [ x4 ], x6
            
        
            
                st1 { v2.h }[0], [ x4 ], x6
            
                st1 { v2.h }[1], [ x4 ], x6
            
                st1 { v2.h }[2], [ x4 ], x6
            
                st1 { v2.h }[3], [ x4 ], x6
            
                st1 { v2.h }[4], [ x4 ], x6
            
                st1 { v2.h }[5], [ x4 ], x6
            
                st1 { v2.h }[6], [ x4 ], x6
            
                st1 { v2.h }[7], [ x4 ], x6
            
        
            
                st1 { v3.h }[0], [ x4 ], x6
            
                st1 { v3.h }[1], [ x4 ], x6
            
                st1 { v3.h }[2], [ x4 ], x6
            
                st1 { v3.h }[3], [ x4 ], x6
            
                st1 { v3.h }[4], [ x4 ], x6
            
                st1 { v3.h }[5], [ x4 ], x6
            
                st1 { v3.h }[6], [ x4 ], x6
            
                st1 { v3.h }[7], [ x4 ], x6
            
        
        add x5, x5, x7

    
        
            fcvtn  v0.4h, v24.4s
            fcvtn2 v0.8h, v25.4s
        
            fcvtn  v1.4h, v26.4s
            fcvtn2 v1.8h, v27.4s
        
            fcvtn  v2.4h, v28.4s
            fcvtn2 v2.8h, v29.4s
        
            fcvtn  v3.4h, v30.4s
            fcvtn2 v3.8h, v31.4s
        

        mov x4, x5
        
            
                st1 { v0.h }[0], [ x4 ], x6
            
                st1 { v0.h }[1], [ x4 ], x6
            
                st1 { v0.h }[2], [ x4 ], x6
            
                st1 { v0.h }[3], [ x4 ], x6
            
                st1 { v0.h }[4], [ x4 ], x6
            
                st1 { v0.h }[5], [ x4 ], x6
            
                st1 { v0.h }[6], [ x4 ], x6
            
                st1 { v0.h }[7], [ x4 ], x6
            
        
            
                st1 { v1.h }[0], [ x4 ], x6
            
                st1 { v1.h }[1], [ x4 ], x6
            
                st1 { v1.h }[2], [ x4 ], x6
            
                st1 { v1.h }[3], [ x4 ], x6
            
                st1 { v1.h }[4], [ x4 ], x6
            
                st1 { v1.h }[5], [ x4 ], x6
            
                st1 { v1.h }[6], [ x4 ], x6
            
                st1 { v1.h }[7], [ x4 ], x6
            
        
            
                st1 { v2.h }[0], [ x4 ], x6
            
                st1 { v2.h }[1], [ x4 ], x6
            
                st1 { v2.h }[2], [ x4 ], x6
            
                st1 { v2.h }[3], [ x4 ], x6
            
                st1 { v2.h }[4], [ x4 ], x6
            
                st1 { v2.h }[5], [ x4 ], x6
            
                st1 { v2.h }[6], [ x4 ], x6
            
                st1 { v2.h }[7], [ x4 ], x6
            
        
            
                st1 { v3.h }[0], [ x4 ], x6
            
                st1 { v3.h }[1], [ x4 ], x6
            
                st1 { v3.h }[2], [ x4 ], x6
            
                st1 { v3.h }[3], [ x4 ], x6
            
                st1 { v3.h }[4], [ x4 ], x6
            
                st1 { v3.h }[5], [ x4 ], x6
            
                st1 { v3.h }[6], [ x4 ], x6
            
                st1 { v3.h }[7], [ x4 ], x6
            
        
        add x5, x5, x7

    


    b           .non_linear_loop

.store_strides_contig_f16:

    
        
            fcvtn  v0.4h, v8.4s
            fcvtn2 v0.8h, v9.4s
        
            fcvtn  v1.4h, v10.4s
            fcvtn2 v1.8h, v11.4s
        
            fcvtn  v2.4h, v12.4s
            fcvtn2 v2.8h, v13.4s
        
            fcvtn  v3.4h, v14.4s
            fcvtn2 v3.8h, v15.4s
        

        mov x4, x5
        
            st1 { v0.4s }, [ x4 ], #16
        
            st1 { v1.4s }, [ x4 ], #16
        
            st1 { v2.4s }, [ x4 ], #16
        
            st1 { v3.4s }, [ x4 ], #16
        
        add x5, x5, x7

    
        
            fcvtn  v0.4h, v16.4s
            fcvtn2 v0.8h, v17.4s
        
            fcvtn  v1.4h, v18.4s
            fcvtn2 v1.8h, v19.4s
        
            fcvtn  v2.4h, v20.4s
            fcvtn2 v2.8h, v21.4s
        
            fcvtn  v3.4h, v22.4s
            fcvtn2 v3.8h, v23.4s
        

        mov x4, x5
        
            st1 { v0.4s }, [ x4 ], #16
        
            st1 { v1.4s }, [ x4 ], #16
        
            st1 { v2.4s }, [ x4 ], #16
        
            st1 { v3.4s }, [ x4 ], #16
        
        add x5, x5, x7

    
        
            fcvtn  v0.4h, v24.4s
            fcvtn2 v0.8h, v25.4s
        
            fcvtn  v1.4h, v26.4s
            fcvtn2 v1.8h, v27.4s
        
            fcvtn  v2.4h, v28.4s
            fcvtn2 v2.8h, v29.4s
        
            fcvtn  v3.4h, v30.4s
            fcvtn2 v3.8h, v31.4s
        

        mov x4, x5
        
            st1 { v0.4s }, [ x4 ], #16
        
            st1 { v1.4s }, [ x4 ], #16
        
            st1 { v2.4s }, [ x4 ], #16
        
            st1 { v3.4s }, [ x4 ], #16
        
        add x5, x5, x7

    
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
