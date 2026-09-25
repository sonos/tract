// vim: ft=arm

// C tile regs: 
// - x19-x29 to preserve (but x19, x28, x29 not used) 
// - d8..d15 to preserve
// - v16 to v31, no need to preserve

// no preservation either for v0-v7...
// packed A buffering (2x8 values): alternating v0, v1 with v2, v3
// packed B buffering (2x8 values): alternating v4, v5 with v6, v7

.text
.align 4

.cpu generic+fp+simd
.global _arm64simd_mmm_i32_64x1_0_23_4_pre
_arm64simd_mmm_i32_64x1_0_23_4_pre:

/*
    prfm        pldl1keep, [x1]
    prfm        pldl1keep, [x2]
*/
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
    ldp         x2, x4, [x0, #24]   // b, packing
    ldp         x3, x1, [x0, #8]    // k, a

    cmp         x3, #0
    beq         .non_linear_loop

    cmp         x4, #1
    beq         .packed_packed_loop_1_i8i8

.packed_packed_loop_1:
    ld1         {v9.s}[0], [ x2 ], 4

    ld1	        { v0.4s-v3.4s }, [ x1 ], #64
    ld1	        { v4.4s-v7.4s }, [ x1 ], #64
    
        mla      v16.4s, v0.4s, v9.s[0]
        mla      v17.4s, v1.4s, v9.s[0]
    
        mla      v18.4s, v2.4s, v9.s[0]
        mla      v19.4s, v3.4s, v9.s[0]
    
        mla      v20.4s, v4.4s, v9.s[0]
        mla      v21.4s, v5.4s, v9.s[0]
    
        mla      v22.4s, v6.4s, v9.s[0]
        mla      v23.4s, v7.4s, v9.s[0]
    

    ld1	        { v0.4s-v3.4s }, [ x1 ], #64
    ld1	        { v4.4s-v7.4s }, [ x1 ], #64
    
        mla      v24.4s, v0.4s, v9.s[0]
        mla      v25.4s, v1.4s, v9.s[0]
    
        mla      v26.4s, v2.4s, v9.s[0]
        mla      v27.4s, v3.4s, v9.s[0]
    
        mla      v28.4s, v4.4s, v9.s[0]
        mla      v29.4s, v5.4s, v9.s[0]
    
        mla      v30.4s, v6.4s, v9.s[0]
        mla      v31.4s, v7.4s, v9.s[0]
    

    subs        x3, x3, #1
    bne .packed_packed_loop_1
    
    b .non_linear_loop

.packed_packed_loop_1_i8i8:
    ld1         {v9.b}[0], [ x2 ], 1
    sshll       v9.8h, v9.8b, 0

    ld1	        { v0.8b-v3.8b }, [ x1 ], #32
    ld1	        { v4.8b-v7.8b }, [ x1 ], #32

    
        sshll       v10.8h, v0.8b, 0
        smlal       v16.4s, v10.4h, v9.h[0]
        smlal2      v17.4s, v10.8h, v9.h[0]
    
        sshll       v10.8h, v1.8b, 0
        smlal       v18.4s, v10.4h, v9.h[0]
        smlal2      v19.4s, v10.8h, v9.h[0]
    
        sshll       v10.8h, v2.8b, 0
        smlal       v20.4s, v10.4h, v9.h[0]
        smlal2      v21.4s, v10.8h, v9.h[0]
    
        sshll       v10.8h, v3.8b, 0
        smlal       v22.4s, v10.4h, v9.h[0]
        smlal2      v23.4s, v10.8h, v9.h[0]
    
        sshll       v10.8h, v4.8b, 0
        smlal       v24.4s, v10.4h, v9.h[0]
        smlal2      v25.4s, v10.8h, v9.h[0]
    
        sshll       v10.8h, v5.8b, 0
        smlal       v26.4s, v10.4h, v9.h[0]
        smlal2      v27.4s, v10.8h, v9.h[0]
    
        sshll       v10.8h, v6.8b, 0
        smlal       v28.4s, v10.4h, v9.h[0]
        smlal2      v29.4s, v10.8h, v9.h[0]
    
        sshll       v10.8h, v7.8b, 0
        smlal       v30.4s, v10.4h, v9.h[0]
        smlal2      v31.4s, v10.8h, v9.h[0]
    

    subs        x3, x3, #1
    bne .packed_packed_loop_1_i8i8

    b .non_linear_loop

.add_unicast:
    ldp         x5, x6, [x0, #8]
    ldp         x7, x8, [x0, #24]

    cmp         x8, #4
    beq         non_linear_addc_i32

    
        
            ld1 {v0.b}[0], [ x5 ], x6
        
            ld1 {v0.b}[1], [ x5 ], x6
        
            ld1 {v0.b}[2], [ x5 ], x6
        
            ld1 {v0.b}[3], [ x5 ], x6
        
        sshll v0.8h, v0.8b, 0
        sshll v0.4s, v0.4h, 0
        add v16.4s, v16.4s, v0.4s
    
        
            ld1 {v0.b}[0], [ x5 ], x6
        
            ld1 {v0.b}[1], [ x5 ], x6
        
            ld1 {v0.b}[2], [ x5 ], x6
        
            ld1 {v0.b}[3], [ x5 ], x6
        
        sshll v0.8h, v0.8b, 0
        sshll v0.4s, v0.4h, 0
        add v17.4s, v17.4s, v0.4s
    
        
            ld1 {v0.b}[0], [ x5 ], x6
        
            ld1 {v0.b}[1], [ x5 ], x6
        
            ld1 {v0.b}[2], [ x5 ], x6
        
            ld1 {v0.b}[3], [ x5 ], x6
        
        sshll v0.8h, v0.8b, 0
        sshll v0.4s, v0.4h, 0
        add v18.4s, v18.4s, v0.4s
    
        
            ld1 {v0.b}[0], [ x5 ], x6
        
            ld1 {v0.b}[1], [ x5 ], x6
        
            ld1 {v0.b}[2], [ x5 ], x6
        
            ld1 {v0.b}[3], [ x5 ], x6
        
        sshll v0.8h, v0.8b, 0
        sshll v0.4s, v0.4h, 0
        add v19.4s, v19.4s, v0.4s
    
        
            ld1 {v0.b}[0], [ x5 ], x6
        
            ld1 {v0.b}[1], [ x5 ], x6
        
            ld1 {v0.b}[2], [ x5 ], x6
        
            ld1 {v0.b}[3], [ x5 ], x6
        
        sshll v0.8h, v0.8b, 0
        sshll v0.4s, v0.4h, 0
        add v20.4s, v20.4s, v0.4s
    
        
            ld1 {v0.b}[0], [ x5 ], x6
        
            ld1 {v0.b}[1], [ x5 ], x6
        
            ld1 {v0.b}[2], [ x5 ], x6
        
            ld1 {v0.b}[3], [ x5 ], x6
        
        sshll v0.8h, v0.8b, 0
        sshll v0.4s, v0.4h, 0
        add v21.4s, v21.4s, v0.4s
    
        
            ld1 {v0.b}[0], [ x5 ], x6
        
            ld1 {v0.b}[1], [ x5 ], x6
        
            ld1 {v0.b}[2], [ x5 ], x6
        
            ld1 {v0.b}[3], [ x5 ], x6
        
        sshll v0.8h, v0.8b, 0
        sshll v0.4s, v0.4h, 0
        add v22.4s, v22.4s, v0.4s
    
        
            ld1 {v0.b}[0], [ x5 ], x6
        
            ld1 {v0.b}[1], [ x5 ], x6
        
            ld1 {v0.b}[2], [ x5 ], x6
        
            ld1 {v0.b}[3], [ x5 ], x6
        
        sshll v0.8h, v0.8b, 0
        sshll v0.4s, v0.4h, 0
        add v23.4s, v23.4s, v0.4s
    
        
            ld1 {v0.b}[0], [ x5 ], x6
        
            ld1 {v0.b}[1], [ x5 ], x6
        
            ld1 {v0.b}[2], [ x5 ], x6
        
            ld1 {v0.b}[3], [ x5 ], x6
        
        sshll v0.8h, v0.8b, 0
        sshll v0.4s, v0.4h, 0
        add v24.4s, v24.4s, v0.4s
    
        
            ld1 {v0.b}[0], [ x5 ], x6
        
            ld1 {v0.b}[1], [ x5 ], x6
        
            ld1 {v0.b}[2], [ x5 ], x6
        
            ld1 {v0.b}[3], [ x5 ], x6
        
        sshll v0.8h, v0.8b, 0
        sshll v0.4s, v0.4h, 0
        add v25.4s, v25.4s, v0.4s
    
        
            ld1 {v0.b}[0], [ x5 ], x6
        
            ld1 {v0.b}[1], [ x5 ], x6
        
            ld1 {v0.b}[2], [ x5 ], x6
        
            ld1 {v0.b}[3], [ x5 ], x6
        
        sshll v0.8h, v0.8b, 0
        sshll v0.4s, v0.4h, 0
        add v26.4s, v26.4s, v0.4s
    
        
            ld1 {v0.b}[0], [ x5 ], x6
        
            ld1 {v0.b}[1], [ x5 ], x6
        
            ld1 {v0.b}[2], [ x5 ], x6
        
            ld1 {v0.b}[3], [ x5 ], x6
        
        sshll v0.8h, v0.8b, 0
        sshll v0.4s, v0.4h, 0
        add v27.4s, v27.4s, v0.4s
    
        
            ld1 {v0.b}[0], [ x5 ], x6
        
            ld1 {v0.b}[1], [ x5 ], x6
        
            ld1 {v0.b}[2], [ x5 ], x6
        
            ld1 {v0.b}[3], [ x5 ], x6
        
        sshll v0.8h, v0.8b, 0
        sshll v0.4s, v0.4h, 0
        add v28.4s, v28.4s, v0.4s
    
        
            ld1 {v0.b}[0], [ x5 ], x6
        
            ld1 {v0.b}[1], [ x5 ], x6
        
            ld1 {v0.b}[2], [ x5 ], x6
        
            ld1 {v0.b}[3], [ x5 ], x6
        
        sshll v0.8h, v0.8b, 0
        sshll v0.4s, v0.4h, 0
        add v29.4s, v29.4s, v0.4s
    
        
            ld1 {v0.b}[0], [ x5 ], x6
        
            ld1 {v0.b}[1], [ x5 ], x6
        
            ld1 {v0.b}[2], [ x5 ], x6
        
            ld1 {v0.b}[3], [ x5 ], x6
        
        sshll v0.8h, v0.8b, 0
        sshll v0.4s, v0.4h, 0
        add v30.4s, v30.4s, v0.4s
    
        
            ld1 {v0.b}[0], [ x5 ], x6
        
            ld1 {v0.b}[1], [ x5 ], x6
        
            ld1 {v0.b}[2], [ x5 ], x6
        
            ld1 {v0.b}[3], [ x5 ], x6
        
        sshll v0.8h, v0.8b, 0
        sshll v0.4s, v0.4h, 0
        add v31.4s, v31.4s, v0.4s
    

    b           .non_linear_loop

non_linear_addc_i32:
    
        
            ld1 {v0.s}[0], [ x5 ], x6
        
            ld1 {v0.s}[1], [ x5 ], x6
        
            ld1 {v0.s}[2], [ x5 ], x6
        
            ld1 {v0.s}[3], [ x5 ], x6
        
        add v16.4s, v16.4s, v0.4s
    
        
            ld1 {v0.s}[0], [ x5 ], x6
        
            ld1 {v0.s}[1], [ x5 ], x6
        
            ld1 {v0.s}[2], [ x5 ], x6
        
            ld1 {v0.s}[3], [ x5 ], x6
        
        add v17.4s, v17.4s, v0.4s
    
        
            ld1 {v0.s}[0], [ x5 ], x6
        
            ld1 {v0.s}[1], [ x5 ], x6
        
            ld1 {v0.s}[2], [ x5 ], x6
        
            ld1 {v0.s}[3], [ x5 ], x6
        
        add v18.4s, v18.4s, v0.4s
    
        
            ld1 {v0.s}[0], [ x5 ], x6
        
            ld1 {v0.s}[1], [ x5 ], x6
        
            ld1 {v0.s}[2], [ x5 ], x6
        
            ld1 {v0.s}[3], [ x5 ], x6
        
        add v19.4s, v19.4s, v0.4s
    
        
            ld1 {v0.s}[0], [ x5 ], x6
        
            ld1 {v0.s}[1], [ x5 ], x6
        
            ld1 {v0.s}[2], [ x5 ], x6
        
            ld1 {v0.s}[3], [ x5 ], x6
        
        add v20.4s, v20.4s, v0.4s
    
        
            ld1 {v0.s}[0], [ x5 ], x6
        
            ld1 {v0.s}[1], [ x5 ], x6
        
            ld1 {v0.s}[2], [ x5 ], x6
        
            ld1 {v0.s}[3], [ x5 ], x6
        
        add v21.4s, v21.4s, v0.4s
    
        
            ld1 {v0.s}[0], [ x5 ], x6
        
            ld1 {v0.s}[1], [ x5 ], x6
        
            ld1 {v0.s}[2], [ x5 ], x6
        
            ld1 {v0.s}[3], [ x5 ], x6
        
        add v22.4s, v22.4s, v0.4s
    
        
            ld1 {v0.s}[0], [ x5 ], x6
        
            ld1 {v0.s}[1], [ x5 ], x6
        
            ld1 {v0.s}[2], [ x5 ], x6
        
            ld1 {v0.s}[3], [ x5 ], x6
        
        add v23.4s, v23.4s, v0.4s
    
        
            ld1 {v0.s}[0], [ x5 ], x6
        
            ld1 {v0.s}[1], [ x5 ], x6
        
            ld1 {v0.s}[2], [ x5 ], x6
        
            ld1 {v0.s}[3], [ x5 ], x6
        
        add v24.4s, v24.4s, v0.4s
    
        
            ld1 {v0.s}[0], [ x5 ], x6
        
            ld1 {v0.s}[1], [ x5 ], x6
        
            ld1 {v0.s}[2], [ x5 ], x6
        
            ld1 {v0.s}[3], [ x5 ], x6
        
        add v25.4s, v25.4s, v0.4s
    
        
            ld1 {v0.s}[0], [ x5 ], x6
        
            ld1 {v0.s}[1], [ x5 ], x6
        
            ld1 {v0.s}[2], [ x5 ], x6
        
            ld1 {v0.s}[3], [ x5 ], x6
        
        add v26.4s, v26.4s, v0.4s
    
        
            ld1 {v0.s}[0], [ x5 ], x6
        
            ld1 {v0.s}[1], [ x5 ], x6
        
            ld1 {v0.s}[2], [ x5 ], x6
        
            ld1 {v0.s}[3], [ x5 ], x6
        
        add v27.4s, v27.4s, v0.4s
    
        
            ld1 {v0.s}[0], [ x5 ], x6
        
            ld1 {v0.s}[1], [ x5 ], x6
        
            ld1 {v0.s}[2], [ x5 ], x6
        
            ld1 {v0.s}[3], [ x5 ], x6
        
        add v28.4s, v28.4s, v0.4s
    
        
            ld1 {v0.s}[0], [ x5 ], x6
        
            ld1 {v0.s}[1], [ x5 ], x6
        
            ld1 {v0.s}[2], [ x5 ], x6
        
            ld1 {v0.s}[3], [ x5 ], x6
        
        add v29.4s, v29.4s, v0.4s
    
        
            ld1 {v0.s}[0], [ x5 ], x6
        
            ld1 {v0.s}[1], [ x5 ], x6
        
            ld1 {v0.s}[2], [ x5 ], x6
        
            ld1 {v0.s}[3], [ x5 ], x6
        
        add v30.4s, v30.4s, v0.4s
    
        
            ld1 {v0.s}[0], [ x5 ], x6
        
            ld1 {v0.s}[1], [ x5 ], x6
        
            ld1 {v0.s}[2], [ x5 ], x6
        
            ld1 {v0.s}[3], [ x5 ], x6
        
        add v31.4s, v31.4s, v0.4s
    

    b           .non_linear_loop

.add_row_col_products:
    ldr     x2, [x0, #8]
    ldr     x3, [x0, #16]

    ld1         { v15.s }[0], [ x3 ]
    xtn         v15.4h, v15.4s

    ld1         { v0.4s-v3.4s }, [ x2 ], #64
    ld1         { v4.4s-v7.4s }, [ x2 ], #64

    
        xtn         v0.4h, v0.4s
        smlal        v16.4s, v0.4h, v15.h[0]
    
        xtn         v1.4h, v1.4s
        smlal        v17.4s, v1.4h, v15.h[0]
    
        xtn         v2.4h, v2.4s
        smlal        v18.4s, v2.4h, v15.h[0]
    
        xtn         v3.4h, v3.4s
        smlal        v19.4s, v3.4h, v15.h[0]
    
        xtn         v4.4h, v4.4s
        smlal        v20.4s, v4.4h, v15.h[0]
    
        xtn         v5.4h, v5.4s
        smlal        v21.4s, v5.4h, v15.h[0]
    
        xtn         v6.4h, v6.4s
        smlal        v22.4s, v6.4h, v15.h[0]
    
        xtn         v7.4h, v7.4s
        smlal        v23.4s, v7.4h, v15.h[0]
    

    ld1         { v0.4s-v3.4s }, [ x2 ], #64
    ld1         { v4.4s-v7.4s }, [ x2 ], #64

    
        xtn         v0.4h, v0.4s
        smlal        v24.4s, v0.4h, v15.h[0]
    
        xtn         v1.4h, v1.4s
        smlal        v25.4s, v1.4h, v15.h[0]
    
        xtn         v2.4h, v2.4s
        smlal        v26.4s, v2.4h, v15.h[0]
    
        xtn         v3.4h, v3.4s
        smlal        v27.4s, v3.4h, v15.h[0]
    
        xtn         v4.4h, v4.4s
        smlal        v28.4s, v4.4h, v15.h[0]
    
        xtn         v5.4h, v5.4s
        smlal        v29.4s, v5.4h, v15.h[0]
    
        xtn         v6.4h, v6.4s
        smlal        v30.4s, v6.4h, v15.h[0]
    
        xtn         v7.4h, v7.4s
        smlal        v31.4s, v7.4h, v15.h[0]
    

    b           .non_linear_loop

// vim: ft=arm




.scalar_min:
    add         x2, x0, #8
    ld1         {v0.s}[0], [ x2 ]
    dup         v0.4s, v0.s[0]
    
        
            smin       v16.4s, v0.4s, v16.4s
        
            smin       v17.4s, v0.4s, v17.4s
        
            smin       v18.4s, v0.4s, v18.4s
        
            smin       v19.4s, v0.4s, v19.4s
        
            smin       v20.4s, v0.4s, v20.4s
        
            smin       v21.4s, v0.4s, v21.4s
        
            smin       v22.4s, v0.4s, v22.4s
        
            smin       v23.4s, v0.4s, v23.4s
        
            smin       v24.4s, v0.4s, v24.4s
        
            smin       v25.4s, v0.4s, v25.4s
        
            smin       v26.4s, v0.4s, v26.4s
        
            smin       v27.4s, v0.4s, v27.4s
        
            smin       v28.4s, v0.4s, v28.4s
        
            smin       v29.4s, v0.4s, v29.4s
        
            smin       v30.4s, v0.4s, v30.4s
        
            smin       v31.4s, v0.4s, v31.4s
        
    

    b           .non_linear_loop


.scalar_max:
    add         x2, x0, #8
    ld1         {v0.s}[0], [ x2 ]
    dup         v0.4s, v0.s[0]
    
        
            smax       v16.4s, v0.4s, v16.4s
        
            smax       v17.4s, v0.4s, v17.4s
        
            smax       v18.4s, v0.4s, v18.4s
        
            smax       v19.4s, v0.4s, v19.4s
        
            smax       v20.4s, v0.4s, v20.4s
        
            smax       v21.4s, v0.4s, v21.4s
        
            smax       v22.4s, v0.4s, v22.4s
        
            smax       v23.4s, v0.4s, v23.4s
        
            smax       v24.4s, v0.4s, v24.4s
        
            smax       v25.4s, v0.4s, v25.4s
        
            smax       v26.4s, v0.4s, v26.4s
        
            smax       v27.4s, v0.4s, v27.4s
        
            smax       v28.4s, v0.4s, v28.4s
        
            smax       v29.4s, v0.4s, v29.4s
        
            smax       v30.4s, v0.4s, v30.4s
        
            smax       v31.4s, v0.4s, v31.4s
        
    

    b           .non_linear_loop


.scalar_mul:
    add         x2, x0, #8
    ld1         {v0.s}[0], [ x2 ]
    dup         v0.4s, v0.s[0]
    
        
            mul       v16.4s, v0.4s, v16.4s
        
            mul       v17.4s, v0.4s, v17.4s
        
            mul       v18.4s, v0.4s, v18.4s
        
            mul       v19.4s, v0.4s, v19.4s
        
            mul       v20.4s, v0.4s, v20.4s
        
            mul       v21.4s, v0.4s, v21.4s
        
            mul       v22.4s, v0.4s, v22.4s
        
            mul       v23.4s, v0.4s, v23.4s
        
            mul       v24.4s, v0.4s, v24.4s
        
            mul       v25.4s, v0.4s, v25.4s
        
            mul       v26.4s, v0.4s, v26.4s
        
            mul       v27.4s, v0.4s, v27.4s
        
            mul       v28.4s, v0.4s, v28.4s
        
            mul       v29.4s, v0.4s, v29.4s
        
            mul       v30.4s, v0.4s, v30.4s
        
            mul       v31.4s, v0.4s, v31.4s
        
    

    b           .non_linear_loop


.scalar_add:
    add         x2, x0, #8
    ld1         {v0.s}[0], [ x2 ]
    dup         v0.4s, v0.s[0]
    
        
            add       v16.4s, v0.4s, v16.4s
        
            add       v17.4s, v0.4s, v17.4s
        
            add       v18.4s, v0.4s, v18.4s
        
            add       v19.4s, v0.4s, v19.4s
        
            add       v20.4s, v0.4s, v20.4s
        
            add       v21.4s, v0.4s, v21.4s
        
            add       v22.4s, v0.4s, v22.4s
        
            add       v23.4s, v0.4s, v23.4s
        
            add       v24.4s, v0.4s, v24.4s
        
            add       v25.4s, v0.4s, v25.4s
        
            add       v26.4s, v0.4s, v26.4s
        
            add       v27.4s, v0.4s, v27.4s
        
            add       v28.4s, v0.4s, v28.4s
        
            add       v29.4s, v0.4s, v29.4s
        
            add       v30.4s, v0.4s, v30.4s
        
            add       v31.4s, v0.4s, v31.4s
        
    

    b           .non_linear_loop


.scalar_sub:
    add         x2, x0, #8
    ld1         {v0.s}[0], [ x2 ]
    dup         v0.4s, v0.s[0]
    
        
            sub       v16.4s, v0.4s, v16.4s
        
            sub       v17.4s, v0.4s, v17.4s
        
            sub       v18.4s, v0.4s, v18.4s
        
            sub       v19.4s, v0.4s, v19.4s
        
            sub       v20.4s, v0.4s, v20.4s
        
            sub       v21.4s, v0.4s, v21.4s
        
            sub       v22.4s, v0.4s, v22.4s
        
            sub       v23.4s, v0.4s, v23.4s
        
            sub       v24.4s, v0.4s, v24.4s
        
            sub       v25.4s, v0.4s, v25.4s
        
            sub       v26.4s, v0.4s, v26.4s
        
            sub       v27.4s, v0.4s, v27.4s
        
            sub       v28.4s, v0.4s, v28.4s
        
            sub       v29.4s, v0.4s, v29.4s
        
            sub       v30.4s, v0.4s, v30.4s
        
            sub       v31.4s, v0.4s, v31.4s
        
    

    b           .non_linear_loop


.scalar_sub_flipped:
    add         x2, x0, #8
    ld1         {v0.s}[0], [ x2 ]
    dup         v0.4s, v0.s[0]
    
        
            sub       v16.4s, v16.4s, v0.4s
        
            sub       v17.4s, v17.4s, v0.4s
        
            sub       v18.4s, v18.4s, v0.4s
        
            sub       v19.4s, v19.4s, v0.4s
        
            sub       v20.4s, v20.4s, v0.4s
        
            sub       v21.4s, v21.4s, v0.4s
        
            sub       v22.4s, v22.4s, v0.4s
        
            sub       v23.4s, v23.4s, v0.4s
        
            sub       v24.4s, v24.4s, v0.4s
        
            sub       v25.4s, v25.4s, v0.4s
        
            sub       v26.4s, v26.4s, v0.4s
        
            sub       v27.4s, v27.4s, v0.4s
        
            sub       v28.4s, v28.4s, v0.4s
        
            sub       v29.4s, v29.4s, v0.4s
        
            sub       v30.4s, v30.4s, v0.4s
        
            sub       v31.4s, v31.4s, v0.4s
        
    

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
    
        mul   v0.4s, v16.4s, v4.4s
        cmge  v1.4s, v16.4s, #0
        bsl   v1.16b, v16.16b, v0.16b
        and   v16.16b, v1.16b, v1.16b
    
        mul   v0.4s, v17.4s, v4.4s
        cmge  v1.4s, v17.4s, #0
        bsl   v1.16b, v17.16b, v0.16b
        and   v17.16b, v1.16b, v1.16b
    
        mul   v0.4s, v18.4s, v4.4s
        cmge  v1.4s, v18.4s, #0
        bsl   v1.16b, v18.16b, v0.16b
        and   v18.16b, v1.16b, v1.16b
    
        mul   v0.4s, v19.4s, v4.4s
        cmge  v1.4s, v19.4s, #0
        bsl   v1.16b, v19.16b, v0.16b
        and   v19.16b, v1.16b, v1.16b
    
        mul   v0.4s, v20.4s, v4.4s
        cmge  v1.4s, v20.4s, #0
        bsl   v1.16b, v20.16b, v0.16b
        and   v20.16b, v1.16b, v1.16b
    
        mul   v0.4s, v21.4s, v4.4s
        cmge  v1.4s, v21.4s, #0
        bsl   v1.16b, v21.16b, v0.16b
        and   v21.16b, v1.16b, v1.16b
    
        mul   v0.4s, v22.4s, v4.4s
        cmge  v1.4s, v22.4s, #0
        bsl   v1.16b, v22.16b, v0.16b
        and   v22.16b, v1.16b, v1.16b
    
        mul   v0.4s, v23.4s, v4.4s
        cmge  v1.4s, v23.4s, #0
        bsl   v1.16b, v23.16b, v0.16b
        and   v23.16b, v1.16b, v1.16b
    
        mul   v0.4s, v24.4s, v4.4s
        cmge  v1.4s, v24.4s, #0
        bsl   v1.16b, v24.16b, v0.16b
        and   v24.16b, v1.16b, v1.16b
    
        mul   v0.4s, v25.4s, v4.4s
        cmge  v1.4s, v25.4s, #0
        bsl   v1.16b, v25.16b, v0.16b
        and   v25.16b, v1.16b, v1.16b
    
        mul   v0.4s, v26.4s, v4.4s
        cmge  v1.4s, v26.4s, #0
        bsl   v1.16b, v26.16b, v0.16b
        and   v26.16b, v1.16b, v1.16b
    
        mul   v0.4s, v27.4s, v4.4s
        cmge  v1.4s, v27.4s, #0
        bsl   v1.16b, v27.16b, v0.16b
        and   v27.16b, v1.16b, v1.16b
    
        mul   v0.4s, v28.4s, v4.4s
        cmge  v1.4s, v28.4s, #0
        bsl   v1.16b, v28.16b, v0.16b
        and   v28.16b, v1.16b, v1.16b
    
        mul   v0.4s, v29.4s, v4.4s
        cmge  v1.4s, v29.4s, #0
        bsl   v1.16b, v29.16b, v0.16b
        and   v29.16b, v1.16b, v1.16b
    
        mul   v0.4s, v30.4s, v4.4s
        cmge  v1.4s, v30.4s, #0
        bsl   v1.16b, v30.16b, v0.16b
        and   v30.16b, v1.16b, v1.16b
    
        mul   v0.4s, v31.4s, v4.4s
        cmge  v1.4s, v31.4s, #0
        bsl   v1.16b, v31.16b, v0.16b
        and   v31.16b, v1.16b, v1.16b
    

    b .non_linear_loop
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



    
        
        smin v16.4s, v0.4s, v16.4s
    
        
        smin v17.4s, v1.4s, v17.4s
    
        
        smin v18.4s, v2.4s, v18.4s
    
        
        smin v19.4s, v3.4s, v19.4s
    
        
        smin v20.4s, v4.4s, v20.4s
    
        
        smin v21.4s, v5.4s, v21.4s
    
        
        smin v22.4s, v6.4s, v22.4s
    
        
        smin v23.4s, v7.4s, v23.4s
    
        
        smin v24.4s, v8.4s, v24.4s
    
        
        smin v25.4s, v9.4s, v25.4s
    
        
        smin v26.4s, v10.4s, v26.4s
    
        
        smin v27.4s, v11.4s, v27.4s
    
        
        smin v28.4s, v12.4s, v28.4s
    
        
        smin v29.4s, v13.4s, v29.4s
    
        
        smin v30.4s, v14.4s, v30.4s
    
        
        smin v31.4s, v15.4s, v31.4s
    


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



    
        
        smax v16.4s, v0.4s, v16.4s
    
        
        smax v17.4s, v1.4s, v17.4s
    
        
        smax v18.4s, v2.4s, v18.4s
    
        
        smax v19.4s, v3.4s, v19.4s
    
        
        smax v20.4s, v4.4s, v20.4s
    
        
        smax v21.4s, v5.4s, v21.4s
    
        
        smax v22.4s, v6.4s, v22.4s
    
        
        smax v23.4s, v7.4s, v23.4s
    
        
        smax v24.4s, v8.4s, v24.4s
    
        
        smax v25.4s, v9.4s, v25.4s
    
        
        smax v26.4s, v10.4s, v26.4s
    
        
        smax v27.4s, v11.4s, v27.4s
    
        
        smax v28.4s, v12.4s, v28.4s
    
        
        smax v29.4s, v13.4s, v29.4s
    
        
        smax v30.4s, v14.4s, v30.4s
    
        
        smax v31.4s, v15.4s, v31.4s
    


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



    
        
        mul v16.4s, v0.4s, v16.4s
    
        
        mul v17.4s, v1.4s, v17.4s
    
        
        mul v18.4s, v2.4s, v18.4s
    
        
        mul v19.4s, v3.4s, v19.4s
    
        
        mul v20.4s, v4.4s, v20.4s
    
        
        mul v21.4s, v5.4s, v21.4s
    
        
        mul v22.4s, v6.4s, v22.4s
    
        
        mul v23.4s, v7.4s, v23.4s
    
        
        mul v24.4s, v8.4s, v24.4s
    
        
        mul v25.4s, v9.4s, v25.4s
    
        
        mul v26.4s, v10.4s, v26.4s
    
        
        mul v27.4s, v11.4s, v27.4s
    
        
        mul v28.4s, v12.4s, v28.4s
    
        
        mul v29.4s, v13.4s, v29.4s
    
        
        mul v30.4s, v14.4s, v30.4s
    
        
        mul v31.4s, v15.4s, v31.4s
    


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



    
        
        add v16.4s, v0.4s, v16.4s
    
        
        add v17.4s, v1.4s, v17.4s
    
        
        add v18.4s, v2.4s, v18.4s
    
        
        add v19.4s, v3.4s, v19.4s
    
        
        add v20.4s, v4.4s, v20.4s
    
        
        add v21.4s, v5.4s, v21.4s
    
        
        add v22.4s, v6.4s, v22.4s
    
        
        add v23.4s, v7.4s, v23.4s
    
        
        add v24.4s, v8.4s, v24.4s
    
        
        add v25.4s, v9.4s, v25.4s
    
        
        add v26.4s, v10.4s, v26.4s
    
        
        add v27.4s, v11.4s, v27.4s
    
        
        add v28.4s, v12.4s, v28.4s
    
        
        add v29.4s, v13.4s, v29.4s
    
        
        add v30.4s, v14.4s, v30.4s
    
        
        add v31.4s, v15.4s, v31.4s
    


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



    
        
        sub v16.4s, v0.4s, v16.4s
    
        
        sub v17.4s, v1.4s, v17.4s
    
        
        sub v18.4s, v2.4s, v18.4s
    
        
        sub v19.4s, v3.4s, v19.4s
    
        
        sub v20.4s, v4.4s, v20.4s
    
        
        sub v21.4s, v5.4s, v21.4s
    
        
        sub v22.4s, v6.4s, v22.4s
    
        
        sub v23.4s, v7.4s, v23.4s
    
        
        sub v24.4s, v8.4s, v24.4s
    
        
        sub v25.4s, v9.4s, v25.4s
    
        
        sub v26.4s, v10.4s, v26.4s
    
        
        sub v27.4s, v11.4s, v27.4s
    
        
        sub v28.4s, v12.4s, v28.4s
    
        
        sub v29.4s, v13.4s, v29.4s
    
        
        sub v30.4s, v14.4s, v30.4s
    
        
        sub v31.4s, v15.4s, v31.4s
    


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



    
        
        sub v16.4s, v16.4s, v0.4s
    
        
        sub v17.4s, v17.4s, v1.4s
    
        
        sub v18.4s, v18.4s, v2.4s
    
        
        sub v19.4s, v19.4s, v3.4s
    
        
        sub v20.4s, v20.4s, v4.4s
    
        
        sub v21.4s, v21.4s, v5.4s
    
        
        sub v22.4s, v22.4s, v6.4s
    
        
        sub v23.4s, v23.4s, v7.4s
    
        
        sub v24.4s, v24.4s, v8.4s
    
        
        sub v25.4s, v25.4s, v9.4s
    
        
        sub v26.4s, v26.4s, v10.4s
    
        
        sub v27.4s, v27.4s, v11.4s
    
        
        sub v28.4s, v28.4s, v12.4s
    
        
        sub v29.4s, v29.4s, v13.4s
    
        
        sub v30.4s, v30.4s, v14.4s
    
        
        sub v31.4s, v31.4s, v15.4s
    


b           .non_linear_loop

// vim: ft=arm




.per_col_min:
    ldr         x2, [x0, #8]







        ld1         {v0.s}[0], [ x2 ]


// 64 1


    dup v3.4s, v0.s[0]
    
        
        
            smin v16.4s, v3.4s, v16.4s
        
    
        
        
            smin v17.4s, v3.4s, v17.4s
        
    
        
        
            smin v18.4s, v3.4s, v18.4s
        
    
        
        
            smin v19.4s, v3.4s, v19.4s
        
    
        
        
            smin v20.4s, v3.4s, v20.4s
        
    
        
        
            smin v21.4s, v3.4s, v21.4s
        
    
        
        
            smin v22.4s, v3.4s, v22.4s
        
    
        
        
            smin v23.4s, v3.4s, v23.4s
        
    
        
        
            smin v24.4s, v3.4s, v24.4s
        
    
        
        
            smin v25.4s, v3.4s, v25.4s
        
    
        
        
            smin v26.4s, v3.4s, v26.4s
        
    
        
        
            smin v27.4s, v3.4s, v27.4s
        
    
        
        
            smin v28.4s, v3.4s, v28.4s
        
    
        
        
            smin v29.4s, v3.4s, v29.4s
        
    
        
        
            smin v30.4s, v3.4s, v30.4s
        
    
        
        
            smin v31.4s, v3.4s, v31.4s
        
    


b           .non_linear_loop


.per_col_max:
    ldr         x2, [x0, #8]







        ld1         {v0.s}[0], [ x2 ]


// 64 1


    dup v3.4s, v0.s[0]
    
        
        
            smax v16.4s, v3.4s, v16.4s
        
    
        
        
            smax v17.4s, v3.4s, v17.4s
        
    
        
        
            smax v18.4s, v3.4s, v18.4s
        
    
        
        
            smax v19.4s, v3.4s, v19.4s
        
    
        
        
            smax v20.4s, v3.4s, v20.4s
        
    
        
        
            smax v21.4s, v3.4s, v21.4s
        
    
        
        
            smax v22.4s, v3.4s, v22.4s
        
    
        
        
            smax v23.4s, v3.4s, v23.4s
        
    
        
        
            smax v24.4s, v3.4s, v24.4s
        
    
        
        
            smax v25.4s, v3.4s, v25.4s
        
    
        
        
            smax v26.4s, v3.4s, v26.4s
        
    
        
        
            smax v27.4s, v3.4s, v27.4s
        
    
        
        
            smax v28.4s, v3.4s, v28.4s
        
    
        
        
            smax v29.4s, v3.4s, v29.4s
        
    
        
        
            smax v30.4s, v3.4s, v30.4s
        
    
        
        
            smax v31.4s, v3.4s, v31.4s
        
    


b           .non_linear_loop


.per_col_mul:
    ldr         x2, [x0, #8]







        ld1         {v0.s}[0], [ x2 ]


// 64 1


    dup v3.4s, v0.s[0]
    
        
        
            mul v16.4s, v3.4s, v16.4s
        
    
        
        
            mul v17.4s, v3.4s, v17.4s
        
    
        
        
            mul v18.4s, v3.4s, v18.4s
        
    
        
        
            mul v19.4s, v3.4s, v19.4s
        
    
        
        
            mul v20.4s, v3.4s, v20.4s
        
    
        
        
            mul v21.4s, v3.4s, v21.4s
        
    
        
        
            mul v22.4s, v3.4s, v22.4s
        
    
        
        
            mul v23.4s, v3.4s, v23.4s
        
    
        
        
            mul v24.4s, v3.4s, v24.4s
        
    
        
        
            mul v25.4s, v3.4s, v25.4s
        
    
        
        
            mul v26.4s, v3.4s, v26.4s
        
    
        
        
            mul v27.4s, v3.4s, v27.4s
        
    
        
        
            mul v28.4s, v3.4s, v28.4s
        
    
        
        
            mul v29.4s, v3.4s, v29.4s
        
    
        
        
            mul v30.4s, v3.4s, v30.4s
        
    
        
        
            mul v31.4s, v3.4s, v31.4s
        
    


b           .non_linear_loop


.per_col_add:
    ldr         x2, [x0, #8]







        ld1         {v0.s}[0], [ x2 ]


// 64 1


    dup v3.4s, v0.s[0]
    
        
        
            add v16.4s, v3.4s, v16.4s
        
    
        
        
            add v17.4s, v3.4s, v17.4s
        
    
        
        
            add v18.4s, v3.4s, v18.4s
        
    
        
        
            add v19.4s, v3.4s, v19.4s
        
    
        
        
            add v20.4s, v3.4s, v20.4s
        
    
        
        
            add v21.4s, v3.4s, v21.4s
        
    
        
        
            add v22.4s, v3.4s, v22.4s
        
    
        
        
            add v23.4s, v3.4s, v23.4s
        
    
        
        
            add v24.4s, v3.4s, v24.4s
        
    
        
        
            add v25.4s, v3.4s, v25.4s
        
    
        
        
            add v26.4s, v3.4s, v26.4s
        
    
        
        
            add v27.4s, v3.4s, v27.4s
        
    
        
        
            add v28.4s, v3.4s, v28.4s
        
    
        
        
            add v29.4s, v3.4s, v29.4s
        
    
        
        
            add v30.4s, v3.4s, v30.4s
        
    
        
        
            add v31.4s, v3.4s, v31.4s
        
    


b           .non_linear_loop


.per_col_sub:
    ldr         x2, [x0, #8]







        ld1         {v0.s}[0], [ x2 ]


// 64 1


    dup v3.4s, v0.s[0]
    
        
        
            sub v16.4s, v3.4s, v16.4s
        
    
        
        
            sub v17.4s, v3.4s, v17.4s
        
    
        
        
            sub v18.4s, v3.4s, v18.4s
        
    
        
        
            sub v19.4s, v3.4s, v19.4s
        
    
        
        
            sub v20.4s, v3.4s, v20.4s
        
    
        
        
            sub v21.4s, v3.4s, v21.4s
        
    
        
        
            sub v22.4s, v3.4s, v22.4s
        
    
        
        
            sub v23.4s, v3.4s, v23.4s
        
    
        
        
            sub v24.4s, v3.4s, v24.4s
        
    
        
        
            sub v25.4s, v3.4s, v25.4s
        
    
        
        
            sub v26.4s, v3.4s, v26.4s
        
    
        
        
            sub v27.4s, v3.4s, v27.4s
        
    
        
        
            sub v28.4s, v3.4s, v28.4s
        
    
        
        
            sub v29.4s, v3.4s, v29.4s
        
    
        
        
            sub v30.4s, v3.4s, v30.4s
        
    
        
        
            sub v31.4s, v3.4s, v31.4s
        
    


b           .non_linear_loop


.per_col_sub_flipped:
    ldr         x2, [x0, #8]







        ld1         {v0.s}[0], [ x2 ]


// 64 1


    dup v3.4s, v0.s[0]
    
        
        
            sub v16.4s, v16.4s, v3.4s
        
    
        
        
            sub v17.4s, v17.4s, v3.4s
        
    
        
        
            sub v18.4s, v18.4s, v3.4s
        
    
        
        
            sub v19.4s, v19.4s, v3.4s
        
    
        
        
            sub v20.4s, v20.4s, v3.4s
        
    
        
        
            sub v21.4s, v21.4s, v3.4s
        
    
        
        
            sub v22.4s, v22.4s, v3.4s
        
    
        
        
            sub v23.4s, v23.4s, v3.4s
        
    
        
        
            sub v24.4s, v24.4s, v3.4s
        
    
        
        
            sub v25.4s, v25.4s, v3.4s
        
    
        
        
            sub v26.4s, v26.4s, v3.4s
        
    
        
        
            sub v27.4s, v27.4s, v3.4s
        
    
        
        
            sub v28.4s, v28.4s, v3.4s
        
    
        
        
            sub v29.4s, v29.4s, v3.4s
        
    
        
        
            sub v30.4s, v30.4s, v3.4s
        
    
        
        
            sub v31.4s, v31.4s, v3.4s
        
    


b           .non_linear_loop


// vim: ft=arm

.q_scale:
    ldp     x5, x6, [x0, #8]            // x5: shift, x6: policy
    add     x2, x0, #24
    ld1r    { v2.4s }, [x2]             // v2.4s <- multiplier

    mov     w3, #1
    ins     v4.d[0], x3
    dup     v4.2d, v4.d[0]              // v4.2d <- 1

    add     x5, x5, #32                 // add 32 to shift
    neg     x5, x5                      // broadcast shift
    ins     v1.d[0], x5
    dup     v1.2d, v1.d[0]              // v1.2s <- -(shift + 32)

    cmp     x6, 1
    beq     .q_scale_rounding_zero
    cmp     x6, 2
    beq     .q_scale_rounding_away
    cmp     x6, 3
    beq     .q_scale_rounding_minus_inf
    cmp     x6, 4
    beq     .q_scale_rounding_plus_inf
    cmp     x6, 5
    beq     .q_scale_rounding_even
    cmp     x6, 6
    beq     .q_scale_rounding_odd

    b .unsupported

.q_scale_rounding_zero:
        // rust: signum * ((abs + nudge2) >> shift
        // asm: signum * (2*abs - 1) >>r (shift + 1)

    
        cmlt        v0.4s, v16.4s, #0
        abs         v16.4s, v16.4s
        sqdmull     v8.2d, v16.2s, v2.2s
        sqdmull2    v9.2d, v16.4s, v2.4s     //mul without shift and store results in v8 and v9

        sub         v8.2d, v8.2d, v4.2d
        sqrshl      v8.2d, v8.2d, v1.2d

        sub         v9.2d, v9.2d, v4.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v16.4s, v8.4s, v9.4s    //combine back

        neg         v3.4s, v16.4s
        bit         v16.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v17.4s, #0
        abs         v17.4s, v17.4s
        sqdmull     v8.2d, v17.2s, v2.2s
        sqdmull2    v9.2d, v17.4s, v2.4s     //mul without shift and store results in v8 and v9

        sub         v8.2d, v8.2d, v4.2d
        sqrshl      v8.2d, v8.2d, v1.2d

        sub         v9.2d, v9.2d, v4.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v17.4s, v8.4s, v9.4s    //combine back

        neg         v3.4s, v17.4s
        bit         v17.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v18.4s, #0
        abs         v18.4s, v18.4s
        sqdmull     v8.2d, v18.2s, v2.2s
        sqdmull2    v9.2d, v18.4s, v2.4s     //mul without shift and store results in v8 and v9

        sub         v8.2d, v8.2d, v4.2d
        sqrshl      v8.2d, v8.2d, v1.2d

        sub         v9.2d, v9.2d, v4.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v18.4s, v8.4s, v9.4s    //combine back

        neg         v3.4s, v18.4s
        bit         v18.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v19.4s, #0
        abs         v19.4s, v19.4s
        sqdmull     v8.2d, v19.2s, v2.2s
        sqdmull2    v9.2d, v19.4s, v2.4s     //mul without shift and store results in v8 and v9

        sub         v8.2d, v8.2d, v4.2d
        sqrshl      v8.2d, v8.2d, v1.2d

        sub         v9.2d, v9.2d, v4.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v19.4s, v8.4s, v9.4s    //combine back

        neg         v3.4s, v19.4s
        bit         v19.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v20.4s, #0
        abs         v20.4s, v20.4s
        sqdmull     v8.2d, v20.2s, v2.2s
        sqdmull2    v9.2d, v20.4s, v2.4s     //mul without shift and store results in v8 and v9

        sub         v8.2d, v8.2d, v4.2d
        sqrshl      v8.2d, v8.2d, v1.2d

        sub         v9.2d, v9.2d, v4.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v20.4s, v8.4s, v9.4s    //combine back

        neg         v3.4s, v20.4s
        bit         v20.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v21.4s, #0
        abs         v21.4s, v21.4s
        sqdmull     v8.2d, v21.2s, v2.2s
        sqdmull2    v9.2d, v21.4s, v2.4s     //mul without shift and store results in v8 and v9

        sub         v8.2d, v8.2d, v4.2d
        sqrshl      v8.2d, v8.2d, v1.2d

        sub         v9.2d, v9.2d, v4.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v21.4s, v8.4s, v9.4s    //combine back

        neg         v3.4s, v21.4s
        bit         v21.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v22.4s, #0
        abs         v22.4s, v22.4s
        sqdmull     v8.2d, v22.2s, v2.2s
        sqdmull2    v9.2d, v22.4s, v2.4s     //mul without shift and store results in v8 and v9

        sub         v8.2d, v8.2d, v4.2d
        sqrshl      v8.2d, v8.2d, v1.2d

        sub         v9.2d, v9.2d, v4.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v22.4s, v8.4s, v9.4s    //combine back

        neg         v3.4s, v22.4s
        bit         v22.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v23.4s, #0
        abs         v23.4s, v23.4s
        sqdmull     v8.2d, v23.2s, v2.2s
        sqdmull2    v9.2d, v23.4s, v2.4s     //mul without shift and store results in v8 and v9

        sub         v8.2d, v8.2d, v4.2d
        sqrshl      v8.2d, v8.2d, v1.2d

        sub         v9.2d, v9.2d, v4.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v23.4s, v8.4s, v9.4s    //combine back

        neg         v3.4s, v23.4s
        bit         v23.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v24.4s, #0
        abs         v24.4s, v24.4s
        sqdmull     v8.2d, v24.2s, v2.2s
        sqdmull2    v9.2d, v24.4s, v2.4s     //mul without shift and store results in v8 and v9

        sub         v8.2d, v8.2d, v4.2d
        sqrshl      v8.2d, v8.2d, v1.2d

        sub         v9.2d, v9.2d, v4.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v24.4s, v8.4s, v9.4s    //combine back

        neg         v3.4s, v24.4s
        bit         v24.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v25.4s, #0
        abs         v25.4s, v25.4s
        sqdmull     v8.2d, v25.2s, v2.2s
        sqdmull2    v9.2d, v25.4s, v2.4s     //mul without shift and store results in v8 and v9

        sub         v8.2d, v8.2d, v4.2d
        sqrshl      v8.2d, v8.2d, v1.2d

        sub         v9.2d, v9.2d, v4.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v25.4s, v8.4s, v9.4s    //combine back

        neg         v3.4s, v25.4s
        bit         v25.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v26.4s, #0
        abs         v26.4s, v26.4s
        sqdmull     v8.2d, v26.2s, v2.2s
        sqdmull2    v9.2d, v26.4s, v2.4s     //mul without shift and store results in v8 and v9

        sub         v8.2d, v8.2d, v4.2d
        sqrshl      v8.2d, v8.2d, v1.2d

        sub         v9.2d, v9.2d, v4.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v26.4s, v8.4s, v9.4s    //combine back

        neg         v3.4s, v26.4s
        bit         v26.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v27.4s, #0
        abs         v27.4s, v27.4s
        sqdmull     v8.2d, v27.2s, v2.2s
        sqdmull2    v9.2d, v27.4s, v2.4s     //mul without shift and store results in v8 and v9

        sub         v8.2d, v8.2d, v4.2d
        sqrshl      v8.2d, v8.2d, v1.2d

        sub         v9.2d, v9.2d, v4.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v27.4s, v8.4s, v9.4s    //combine back

        neg         v3.4s, v27.4s
        bit         v27.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v28.4s, #0
        abs         v28.4s, v28.4s
        sqdmull     v8.2d, v28.2s, v2.2s
        sqdmull2    v9.2d, v28.4s, v2.4s     //mul without shift and store results in v8 and v9

        sub         v8.2d, v8.2d, v4.2d
        sqrshl      v8.2d, v8.2d, v1.2d

        sub         v9.2d, v9.2d, v4.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v28.4s, v8.4s, v9.4s    //combine back

        neg         v3.4s, v28.4s
        bit         v28.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v29.4s, #0
        abs         v29.4s, v29.4s
        sqdmull     v8.2d, v29.2s, v2.2s
        sqdmull2    v9.2d, v29.4s, v2.4s     //mul without shift and store results in v8 and v9

        sub         v8.2d, v8.2d, v4.2d
        sqrshl      v8.2d, v8.2d, v1.2d

        sub         v9.2d, v9.2d, v4.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v29.4s, v8.4s, v9.4s    //combine back

        neg         v3.4s, v29.4s
        bit         v29.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v30.4s, #0
        abs         v30.4s, v30.4s
        sqdmull     v8.2d, v30.2s, v2.2s
        sqdmull2    v9.2d, v30.4s, v2.4s     //mul without shift and store results in v8 and v9

        sub         v8.2d, v8.2d, v4.2d
        sqrshl      v8.2d, v8.2d, v1.2d

        sub         v9.2d, v9.2d, v4.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v30.4s, v8.4s, v9.4s    //combine back

        neg         v3.4s, v30.4s
        bit         v30.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v31.4s, #0
        abs         v31.4s, v31.4s
        sqdmull     v8.2d, v31.2s, v2.2s
        sqdmull2    v9.2d, v31.4s, v2.4s     //mul without shift and store results in v8 and v9

        sub         v8.2d, v8.2d, v4.2d
        sqrshl      v8.2d, v8.2d, v1.2d

        sub         v9.2d, v9.2d, v4.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v31.4s, v8.4s, v9.4s    //combine back

        neg         v3.4s, v31.4s
        bit         v31.16b, v3.16b, v0.16b
    

    b .non_linear_loop

.q_scale_rounding_away: // signum * (abs >> (shift-1) + 1 >> 1)

    
        cmlt        v0.4s, v16.4s, #0
        abs         v16.4s, v16.4s
        sqdmull     v8.2d, v16.2s, v2.2s
        sqdmull2    v9.2d, v16.4s, v2.4s     //mul without shift and store results in v8 and v9

        sqrshl      v8.2d, v8.2d, v1.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v16.4s, v8.4s, v9.4s    //combine back

        neg         v3.4s, v16.4s
        bit         v16.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v17.4s, #0
        abs         v17.4s, v17.4s
        sqdmull     v8.2d, v17.2s, v2.2s
        sqdmull2    v9.2d, v17.4s, v2.4s     //mul without shift and store results in v8 and v9

        sqrshl      v8.2d, v8.2d, v1.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v17.4s, v8.4s, v9.4s    //combine back

        neg         v3.4s, v17.4s
        bit         v17.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v18.4s, #0
        abs         v18.4s, v18.4s
        sqdmull     v8.2d, v18.2s, v2.2s
        sqdmull2    v9.2d, v18.4s, v2.4s     //mul without shift and store results in v8 and v9

        sqrshl      v8.2d, v8.2d, v1.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v18.4s, v8.4s, v9.4s    //combine back

        neg         v3.4s, v18.4s
        bit         v18.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v19.4s, #0
        abs         v19.4s, v19.4s
        sqdmull     v8.2d, v19.2s, v2.2s
        sqdmull2    v9.2d, v19.4s, v2.4s     //mul without shift and store results in v8 and v9

        sqrshl      v8.2d, v8.2d, v1.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v19.4s, v8.4s, v9.4s    //combine back

        neg         v3.4s, v19.4s
        bit         v19.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v20.4s, #0
        abs         v20.4s, v20.4s
        sqdmull     v8.2d, v20.2s, v2.2s
        sqdmull2    v9.2d, v20.4s, v2.4s     //mul without shift and store results in v8 and v9

        sqrshl      v8.2d, v8.2d, v1.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v20.4s, v8.4s, v9.4s    //combine back

        neg         v3.4s, v20.4s
        bit         v20.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v21.4s, #0
        abs         v21.4s, v21.4s
        sqdmull     v8.2d, v21.2s, v2.2s
        sqdmull2    v9.2d, v21.4s, v2.4s     //mul without shift and store results in v8 and v9

        sqrshl      v8.2d, v8.2d, v1.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v21.4s, v8.4s, v9.4s    //combine back

        neg         v3.4s, v21.4s
        bit         v21.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v22.4s, #0
        abs         v22.4s, v22.4s
        sqdmull     v8.2d, v22.2s, v2.2s
        sqdmull2    v9.2d, v22.4s, v2.4s     //mul without shift and store results in v8 and v9

        sqrshl      v8.2d, v8.2d, v1.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v22.4s, v8.4s, v9.4s    //combine back

        neg         v3.4s, v22.4s
        bit         v22.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v23.4s, #0
        abs         v23.4s, v23.4s
        sqdmull     v8.2d, v23.2s, v2.2s
        sqdmull2    v9.2d, v23.4s, v2.4s     //mul without shift and store results in v8 and v9

        sqrshl      v8.2d, v8.2d, v1.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v23.4s, v8.4s, v9.4s    //combine back

        neg         v3.4s, v23.4s
        bit         v23.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v24.4s, #0
        abs         v24.4s, v24.4s
        sqdmull     v8.2d, v24.2s, v2.2s
        sqdmull2    v9.2d, v24.4s, v2.4s     //mul without shift and store results in v8 and v9

        sqrshl      v8.2d, v8.2d, v1.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v24.4s, v8.4s, v9.4s    //combine back

        neg         v3.4s, v24.4s
        bit         v24.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v25.4s, #0
        abs         v25.4s, v25.4s
        sqdmull     v8.2d, v25.2s, v2.2s
        sqdmull2    v9.2d, v25.4s, v2.4s     //mul without shift and store results in v8 and v9

        sqrshl      v8.2d, v8.2d, v1.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v25.4s, v8.4s, v9.4s    //combine back

        neg         v3.4s, v25.4s
        bit         v25.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v26.4s, #0
        abs         v26.4s, v26.4s
        sqdmull     v8.2d, v26.2s, v2.2s
        sqdmull2    v9.2d, v26.4s, v2.4s     //mul without shift and store results in v8 and v9

        sqrshl      v8.2d, v8.2d, v1.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v26.4s, v8.4s, v9.4s    //combine back

        neg         v3.4s, v26.4s
        bit         v26.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v27.4s, #0
        abs         v27.4s, v27.4s
        sqdmull     v8.2d, v27.2s, v2.2s
        sqdmull2    v9.2d, v27.4s, v2.4s     //mul without shift and store results in v8 and v9

        sqrshl      v8.2d, v8.2d, v1.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v27.4s, v8.4s, v9.4s    //combine back

        neg         v3.4s, v27.4s
        bit         v27.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v28.4s, #0
        abs         v28.4s, v28.4s
        sqdmull     v8.2d, v28.2s, v2.2s
        sqdmull2    v9.2d, v28.4s, v2.4s     //mul without shift and store results in v8 and v9

        sqrshl      v8.2d, v8.2d, v1.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v28.4s, v8.4s, v9.4s    //combine back

        neg         v3.4s, v28.4s
        bit         v28.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v29.4s, #0
        abs         v29.4s, v29.4s
        sqdmull     v8.2d, v29.2s, v2.2s
        sqdmull2    v9.2d, v29.4s, v2.4s     //mul without shift and store results in v8 and v9

        sqrshl      v8.2d, v8.2d, v1.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v29.4s, v8.4s, v9.4s    //combine back

        neg         v3.4s, v29.4s
        bit         v29.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v30.4s, #0
        abs         v30.4s, v30.4s
        sqdmull     v8.2d, v30.2s, v2.2s
        sqdmull2    v9.2d, v30.4s, v2.4s     //mul without shift and store results in v8 and v9

        sqrshl      v8.2d, v8.2d, v1.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v30.4s, v8.4s, v9.4s    //combine back

        neg         v3.4s, v30.4s
        bit         v30.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v31.4s, #0
        abs         v31.4s, v31.4s
        sqdmull     v8.2d, v31.2s, v2.2s
        sqdmull2    v9.2d, v31.4s, v2.4s     //mul without shift and store results in v8 and v9

        sqrshl      v8.2d, v8.2d, v1.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v31.4s, v8.4s, v9.4s    //combine back

        neg         v3.4s, v31.4s
        bit         v31.16b, v3.16b, v0.16b
    

    b .non_linear_loop

.q_scale_rounding_minus_inf: // val >> shift

    
        sqdmull     v8.2d, v16.2s, v2.2s
        sqdmull2    v9.2d, v16.4s, v2.4s     //mul without shift and store results in v8 and v9

        sub         v8.2d, v8.2d, v4.2d
        sqrshl      v8.2d, v8.2d, v1.2d

        sub         v9.2d, v9.2d, v4.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v16.4s, v8.4s, v9.4s    //combine back
    
        sqdmull     v8.2d, v17.2s, v2.2s
        sqdmull2    v9.2d, v17.4s, v2.4s     //mul without shift and store results in v8 and v9

        sub         v8.2d, v8.2d, v4.2d
        sqrshl      v8.2d, v8.2d, v1.2d

        sub         v9.2d, v9.2d, v4.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v17.4s, v8.4s, v9.4s    //combine back
    
        sqdmull     v8.2d, v18.2s, v2.2s
        sqdmull2    v9.2d, v18.4s, v2.4s     //mul without shift and store results in v8 and v9

        sub         v8.2d, v8.2d, v4.2d
        sqrshl      v8.2d, v8.2d, v1.2d

        sub         v9.2d, v9.2d, v4.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v18.4s, v8.4s, v9.4s    //combine back
    
        sqdmull     v8.2d, v19.2s, v2.2s
        sqdmull2    v9.2d, v19.4s, v2.4s     //mul without shift and store results in v8 and v9

        sub         v8.2d, v8.2d, v4.2d
        sqrshl      v8.2d, v8.2d, v1.2d

        sub         v9.2d, v9.2d, v4.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v19.4s, v8.4s, v9.4s    //combine back
    
        sqdmull     v8.2d, v20.2s, v2.2s
        sqdmull2    v9.2d, v20.4s, v2.4s     //mul without shift and store results in v8 and v9

        sub         v8.2d, v8.2d, v4.2d
        sqrshl      v8.2d, v8.2d, v1.2d

        sub         v9.2d, v9.2d, v4.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v20.4s, v8.4s, v9.4s    //combine back
    
        sqdmull     v8.2d, v21.2s, v2.2s
        sqdmull2    v9.2d, v21.4s, v2.4s     //mul without shift and store results in v8 and v9

        sub         v8.2d, v8.2d, v4.2d
        sqrshl      v8.2d, v8.2d, v1.2d

        sub         v9.2d, v9.2d, v4.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v21.4s, v8.4s, v9.4s    //combine back
    
        sqdmull     v8.2d, v22.2s, v2.2s
        sqdmull2    v9.2d, v22.4s, v2.4s     //mul without shift and store results in v8 and v9

        sub         v8.2d, v8.2d, v4.2d
        sqrshl      v8.2d, v8.2d, v1.2d

        sub         v9.2d, v9.2d, v4.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v22.4s, v8.4s, v9.4s    //combine back
    
        sqdmull     v8.2d, v23.2s, v2.2s
        sqdmull2    v9.2d, v23.4s, v2.4s     //mul without shift and store results in v8 and v9

        sub         v8.2d, v8.2d, v4.2d
        sqrshl      v8.2d, v8.2d, v1.2d

        sub         v9.2d, v9.2d, v4.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v23.4s, v8.4s, v9.4s    //combine back
    
        sqdmull     v8.2d, v24.2s, v2.2s
        sqdmull2    v9.2d, v24.4s, v2.4s     //mul without shift and store results in v8 and v9

        sub         v8.2d, v8.2d, v4.2d
        sqrshl      v8.2d, v8.2d, v1.2d

        sub         v9.2d, v9.2d, v4.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v24.4s, v8.4s, v9.4s    //combine back
    
        sqdmull     v8.2d, v25.2s, v2.2s
        sqdmull2    v9.2d, v25.4s, v2.4s     //mul without shift and store results in v8 and v9

        sub         v8.2d, v8.2d, v4.2d
        sqrshl      v8.2d, v8.2d, v1.2d

        sub         v9.2d, v9.2d, v4.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v25.4s, v8.4s, v9.4s    //combine back
    
        sqdmull     v8.2d, v26.2s, v2.2s
        sqdmull2    v9.2d, v26.4s, v2.4s     //mul without shift and store results in v8 and v9

        sub         v8.2d, v8.2d, v4.2d
        sqrshl      v8.2d, v8.2d, v1.2d

        sub         v9.2d, v9.2d, v4.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v26.4s, v8.4s, v9.4s    //combine back
    
        sqdmull     v8.2d, v27.2s, v2.2s
        sqdmull2    v9.2d, v27.4s, v2.4s     //mul without shift and store results in v8 and v9

        sub         v8.2d, v8.2d, v4.2d
        sqrshl      v8.2d, v8.2d, v1.2d

        sub         v9.2d, v9.2d, v4.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v27.4s, v8.4s, v9.4s    //combine back
    
        sqdmull     v8.2d, v28.2s, v2.2s
        sqdmull2    v9.2d, v28.4s, v2.4s     //mul without shift and store results in v8 and v9

        sub         v8.2d, v8.2d, v4.2d
        sqrshl      v8.2d, v8.2d, v1.2d

        sub         v9.2d, v9.2d, v4.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v28.4s, v8.4s, v9.4s    //combine back
    
        sqdmull     v8.2d, v29.2s, v2.2s
        sqdmull2    v9.2d, v29.4s, v2.4s     //mul without shift and store results in v8 and v9

        sub         v8.2d, v8.2d, v4.2d
        sqrshl      v8.2d, v8.2d, v1.2d

        sub         v9.2d, v9.2d, v4.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v29.4s, v8.4s, v9.4s    //combine back
    
        sqdmull     v8.2d, v30.2s, v2.2s
        sqdmull2    v9.2d, v30.4s, v2.4s     //mul without shift and store results in v8 and v9

        sub         v8.2d, v8.2d, v4.2d
        sqrshl      v8.2d, v8.2d, v1.2d

        sub         v9.2d, v9.2d, v4.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v30.4s, v8.4s, v9.4s    //combine back
    
        sqdmull     v8.2d, v31.2s, v2.2s
        sqdmull2    v9.2d, v31.4s, v2.4s     //mul without shift and store results in v8 and v9

        sub         v8.2d, v8.2d, v4.2d
        sqrshl      v8.2d, v8.2d, v1.2d

        sub         v9.2d, v9.2d, v4.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v31.4s, v8.4s, v9.4s    //combine back
    

    b .non_linear_loop

.q_scale_rounding_plus_inf: // (val >> shift-1)+1 >>1

    
        sqdmull     v8.2d, v16.2s, v2.2s
        sqdmull2    v9.2d, v16.4s, v2.4s     //mul without shift and store results in v8 and v9

        sqrshl      v8.2d, v8.2d, v1.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v16.4s, v8.4s, v9.4s    //combine back
    
        sqdmull     v8.2d, v17.2s, v2.2s
        sqdmull2    v9.2d, v17.4s, v2.4s     //mul without shift and store results in v8 and v9

        sqrshl      v8.2d, v8.2d, v1.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v17.4s, v8.4s, v9.4s    //combine back
    
        sqdmull     v8.2d, v18.2s, v2.2s
        sqdmull2    v9.2d, v18.4s, v2.4s     //mul without shift and store results in v8 and v9

        sqrshl      v8.2d, v8.2d, v1.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v18.4s, v8.4s, v9.4s    //combine back
    
        sqdmull     v8.2d, v19.2s, v2.2s
        sqdmull2    v9.2d, v19.4s, v2.4s     //mul without shift and store results in v8 and v9

        sqrshl      v8.2d, v8.2d, v1.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v19.4s, v8.4s, v9.4s    //combine back
    
        sqdmull     v8.2d, v20.2s, v2.2s
        sqdmull2    v9.2d, v20.4s, v2.4s     //mul without shift and store results in v8 and v9

        sqrshl      v8.2d, v8.2d, v1.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v20.4s, v8.4s, v9.4s    //combine back
    
        sqdmull     v8.2d, v21.2s, v2.2s
        sqdmull2    v9.2d, v21.4s, v2.4s     //mul without shift and store results in v8 and v9

        sqrshl      v8.2d, v8.2d, v1.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v21.4s, v8.4s, v9.4s    //combine back
    
        sqdmull     v8.2d, v22.2s, v2.2s
        sqdmull2    v9.2d, v22.4s, v2.4s     //mul without shift and store results in v8 and v9

        sqrshl      v8.2d, v8.2d, v1.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v22.4s, v8.4s, v9.4s    //combine back
    
        sqdmull     v8.2d, v23.2s, v2.2s
        sqdmull2    v9.2d, v23.4s, v2.4s     //mul without shift and store results in v8 and v9

        sqrshl      v8.2d, v8.2d, v1.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v23.4s, v8.4s, v9.4s    //combine back
    
        sqdmull     v8.2d, v24.2s, v2.2s
        sqdmull2    v9.2d, v24.4s, v2.4s     //mul without shift and store results in v8 and v9

        sqrshl      v8.2d, v8.2d, v1.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v24.4s, v8.4s, v9.4s    //combine back
    
        sqdmull     v8.2d, v25.2s, v2.2s
        sqdmull2    v9.2d, v25.4s, v2.4s     //mul without shift and store results in v8 and v9

        sqrshl      v8.2d, v8.2d, v1.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v25.4s, v8.4s, v9.4s    //combine back
    
        sqdmull     v8.2d, v26.2s, v2.2s
        sqdmull2    v9.2d, v26.4s, v2.4s     //mul without shift and store results in v8 and v9

        sqrshl      v8.2d, v8.2d, v1.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v26.4s, v8.4s, v9.4s    //combine back
    
        sqdmull     v8.2d, v27.2s, v2.2s
        sqdmull2    v9.2d, v27.4s, v2.4s     //mul without shift and store results in v8 and v9

        sqrshl      v8.2d, v8.2d, v1.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v27.4s, v8.4s, v9.4s    //combine back
    
        sqdmull     v8.2d, v28.2s, v2.2s
        sqdmull2    v9.2d, v28.4s, v2.4s     //mul without shift and store results in v8 and v9

        sqrshl      v8.2d, v8.2d, v1.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v28.4s, v8.4s, v9.4s    //combine back
    
        sqdmull     v8.2d, v29.2s, v2.2s
        sqdmull2    v9.2d, v29.4s, v2.4s     //mul without shift and store results in v8 and v9

        sqrshl      v8.2d, v8.2d, v1.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v29.4s, v8.4s, v9.4s    //combine back
    
        sqdmull     v8.2d, v30.2s, v2.2s
        sqdmull2    v9.2d, v30.4s, v2.4s     //mul without shift and store results in v8 and v9

        sqrshl      v8.2d, v8.2d, v1.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v30.4s, v8.4s, v9.4s    //combine back
    
        sqdmull     v8.2d, v31.2s, v2.2s
        sqdmull2    v9.2d, v31.4s, v2.4s     //mul without shift and store results in v8 and v9

        sqrshl      v8.2d, v8.2d, v1.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v31.4s, v8.4s, v9.4s    //combine back
    

    b .non_linear_loop

.q_scale_rounding_even: // signum * ((abs >> shift-1) + (abs & 0x1) - 1 >> 1)

    
        cmlt        v0.4s, v16.4s, #0
        abs         v16.4s, v16.4s
        sqdmull     v8.2d, v16.2s, v2.2s
        sqdmull2    v9.2d, v16.4s, v2.4s     //mul without shift and store results in v8 and v9

        sqshl       v3.2d, v8.2d, v1.2d         // abs >> shift - 1
        and         v3.16b, v3.16b, v4.16b      // abs & 0x1
        sub         v3.2d, v3.2d, v4.2d         //nudge : -1 if we want to round down, 0 if up

        add         v8.2d, v8.2d, v3.2d
        sqrshl      v8.2d, v8.2d, v1.2d

        sqshl       v3.2d, v9.2d, v1.2d
        and         v3.16b, v3.16b, v4.16b
        sub         v3.2d, v3.2d, v4.2d         //nudge : -1 if we want to round down, 0 if up

        add         v9.2d, v9.2d, v3.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v16.4s, v8.4s, v9.4s    //combine back

        neg         v3.4s, v16.4s
        bit         v16.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v17.4s, #0
        abs         v17.4s, v17.4s
        sqdmull     v8.2d, v17.2s, v2.2s
        sqdmull2    v9.2d, v17.4s, v2.4s     //mul without shift and store results in v8 and v9

        sqshl       v3.2d, v8.2d, v1.2d         // abs >> shift - 1
        and         v3.16b, v3.16b, v4.16b      // abs & 0x1
        sub         v3.2d, v3.2d, v4.2d         //nudge : -1 if we want to round down, 0 if up

        add         v8.2d, v8.2d, v3.2d
        sqrshl      v8.2d, v8.2d, v1.2d

        sqshl       v3.2d, v9.2d, v1.2d
        and         v3.16b, v3.16b, v4.16b
        sub         v3.2d, v3.2d, v4.2d         //nudge : -1 if we want to round down, 0 if up

        add         v9.2d, v9.2d, v3.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v17.4s, v8.4s, v9.4s    //combine back

        neg         v3.4s, v17.4s
        bit         v17.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v18.4s, #0
        abs         v18.4s, v18.4s
        sqdmull     v8.2d, v18.2s, v2.2s
        sqdmull2    v9.2d, v18.4s, v2.4s     //mul without shift and store results in v8 and v9

        sqshl       v3.2d, v8.2d, v1.2d         // abs >> shift - 1
        and         v3.16b, v3.16b, v4.16b      // abs & 0x1
        sub         v3.2d, v3.2d, v4.2d         //nudge : -1 if we want to round down, 0 if up

        add         v8.2d, v8.2d, v3.2d
        sqrshl      v8.2d, v8.2d, v1.2d

        sqshl       v3.2d, v9.2d, v1.2d
        and         v3.16b, v3.16b, v4.16b
        sub         v3.2d, v3.2d, v4.2d         //nudge : -1 if we want to round down, 0 if up

        add         v9.2d, v9.2d, v3.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v18.4s, v8.4s, v9.4s    //combine back

        neg         v3.4s, v18.4s
        bit         v18.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v19.4s, #0
        abs         v19.4s, v19.4s
        sqdmull     v8.2d, v19.2s, v2.2s
        sqdmull2    v9.2d, v19.4s, v2.4s     //mul without shift and store results in v8 and v9

        sqshl       v3.2d, v8.2d, v1.2d         // abs >> shift - 1
        and         v3.16b, v3.16b, v4.16b      // abs & 0x1
        sub         v3.2d, v3.2d, v4.2d         //nudge : -1 if we want to round down, 0 if up

        add         v8.2d, v8.2d, v3.2d
        sqrshl      v8.2d, v8.2d, v1.2d

        sqshl       v3.2d, v9.2d, v1.2d
        and         v3.16b, v3.16b, v4.16b
        sub         v3.2d, v3.2d, v4.2d         //nudge : -1 if we want to round down, 0 if up

        add         v9.2d, v9.2d, v3.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v19.4s, v8.4s, v9.4s    //combine back

        neg         v3.4s, v19.4s
        bit         v19.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v20.4s, #0
        abs         v20.4s, v20.4s
        sqdmull     v8.2d, v20.2s, v2.2s
        sqdmull2    v9.2d, v20.4s, v2.4s     //mul without shift and store results in v8 and v9

        sqshl       v3.2d, v8.2d, v1.2d         // abs >> shift - 1
        and         v3.16b, v3.16b, v4.16b      // abs & 0x1
        sub         v3.2d, v3.2d, v4.2d         //nudge : -1 if we want to round down, 0 if up

        add         v8.2d, v8.2d, v3.2d
        sqrshl      v8.2d, v8.2d, v1.2d

        sqshl       v3.2d, v9.2d, v1.2d
        and         v3.16b, v3.16b, v4.16b
        sub         v3.2d, v3.2d, v4.2d         //nudge : -1 if we want to round down, 0 if up

        add         v9.2d, v9.2d, v3.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v20.4s, v8.4s, v9.4s    //combine back

        neg         v3.4s, v20.4s
        bit         v20.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v21.4s, #0
        abs         v21.4s, v21.4s
        sqdmull     v8.2d, v21.2s, v2.2s
        sqdmull2    v9.2d, v21.4s, v2.4s     //mul without shift and store results in v8 and v9

        sqshl       v3.2d, v8.2d, v1.2d         // abs >> shift - 1
        and         v3.16b, v3.16b, v4.16b      // abs & 0x1
        sub         v3.2d, v3.2d, v4.2d         //nudge : -1 if we want to round down, 0 if up

        add         v8.2d, v8.2d, v3.2d
        sqrshl      v8.2d, v8.2d, v1.2d

        sqshl       v3.2d, v9.2d, v1.2d
        and         v3.16b, v3.16b, v4.16b
        sub         v3.2d, v3.2d, v4.2d         //nudge : -1 if we want to round down, 0 if up

        add         v9.2d, v9.2d, v3.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v21.4s, v8.4s, v9.4s    //combine back

        neg         v3.4s, v21.4s
        bit         v21.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v22.4s, #0
        abs         v22.4s, v22.4s
        sqdmull     v8.2d, v22.2s, v2.2s
        sqdmull2    v9.2d, v22.4s, v2.4s     //mul without shift and store results in v8 and v9

        sqshl       v3.2d, v8.2d, v1.2d         // abs >> shift - 1
        and         v3.16b, v3.16b, v4.16b      // abs & 0x1
        sub         v3.2d, v3.2d, v4.2d         //nudge : -1 if we want to round down, 0 if up

        add         v8.2d, v8.2d, v3.2d
        sqrshl      v8.2d, v8.2d, v1.2d

        sqshl       v3.2d, v9.2d, v1.2d
        and         v3.16b, v3.16b, v4.16b
        sub         v3.2d, v3.2d, v4.2d         //nudge : -1 if we want to round down, 0 if up

        add         v9.2d, v9.2d, v3.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v22.4s, v8.4s, v9.4s    //combine back

        neg         v3.4s, v22.4s
        bit         v22.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v23.4s, #0
        abs         v23.4s, v23.4s
        sqdmull     v8.2d, v23.2s, v2.2s
        sqdmull2    v9.2d, v23.4s, v2.4s     //mul without shift and store results in v8 and v9

        sqshl       v3.2d, v8.2d, v1.2d         // abs >> shift - 1
        and         v3.16b, v3.16b, v4.16b      // abs & 0x1
        sub         v3.2d, v3.2d, v4.2d         //nudge : -1 if we want to round down, 0 if up

        add         v8.2d, v8.2d, v3.2d
        sqrshl      v8.2d, v8.2d, v1.2d

        sqshl       v3.2d, v9.2d, v1.2d
        and         v3.16b, v3.16b, v4.16b
        sub         v3.2d, v3.2d, v4.2d         //nudge : -1 if we want to round down, 0 if up

        add         v9.2d, v9.2d, v3.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v23.4s, v8.4s, v9.4s    //combine back

        neg         v3.4s, v23.4s
        bit         v23.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v24.4s, #0
        abs         v24.4s, v24.4s
        sqdmull     v8.2d, v24.2s, v2.2s
        sqdmull2    v9.2d, v24.4s, v2.4s     //mul without shift and store results in v8 and v9

        sqshl       v3.2d, v8.2d, v1.2d         // abs >> shift - 1
        and         v3.16b, v3.16b, v4.16b      // abs & 0x1
        sub         v3.2d, v3.2d, v4.2d         //nudge : -1 if we want to round down, 0 if up

        add         v8.2d, v8.2d, v3.2d
        sqrshl      v8.2d, v8.2d, v1.2d

        sqshl       v3.2d, v9.2d, v1.2d
        and         v3.16b, v3.16b, v4.16b
        sub         v3.2d, v3.2d, v4.2d         //nudge : -1 if we want to round down, 0 if up

        add         v9.2d, v9.2d, v3.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v24.4s, v8.4s, v9.4s    //combine back

        neg         v3.4s, v24.4s
        bit         v24.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v25.4s, #0
        abs         v25.4s, v25.4s
        sqdmull     v8.2d, v25.2s, v2.2s
        sqdmull2    v9.2d, v25.4s, v2.4s     //mul without shift and store results in v8 and v9

        sqshl       v3.2d, v8.2d, v1.2d         // abs >> shift - 1
        and         v3.16b, v3.16b, v4.16b      // abs & 0x1
        sub         v3.2d, v3.2d, v4.2d         //nudge : -1 if we want to round down, 0 if up

        add         v8.2d, v8.2d, v3.2d
        sqrshl      v8.2d, v8.2d, v1.2d

        sqshl       v3.2d, v9.2d, v1.2d
        and         v3.16b, v3.16b, v4.16b
        sub         v3.2d, v3.2d, v4.2d         //nudge : -1 if we want to round down, 0 if up

        add         v9.2d, v9.2d, v3.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v25.4s, v8.4s, v9.4s    //combine back

        neg         v3.4s, v25.4s
        bit         v25.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v26.4s, #0
        abs         v26.4s, v26.4s
        sqdmull     v8.2d, v26.2s, v2.2s
        sqdmull2    v9.2d, v26.4s, v2.4s     //mul without shift and store results in v8 and v9

        sqshl       v3.2d, v8.2d, v1.2d         // abs >> shift - 1
        and         v3.16b, v3.16b, v4.16b      // abs & 0x1
        sub         v3.2d, v3.2d, v4.2d         //nudge : -1 if we want to round down, 0 if up

        add         v8.2d, v8.2d, v3.2d
        sqrshl      v8.2d, v8.2d, v1.2d

        sqshl       v3.2d, v9.2d, v1.2d
        and         v3.16b, v3.16b, v4.16b
        sub         v3.2d, v3.2d, v4.2d         //nudge : -1 if we want to round down, 0 if up

        add         v9.2d, v9.2d, v3.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v26.4s, v8.4s, v9.4s    //combine back

        neg         v3.4s, v26.4s
        bit         v26.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v27.4s, #0
        abs         v27.4s, v27.4s
        sqdmull     v8.2d, v27.2s, v2.2s
        sqdmull2    v9.2d, v27.4s, v2.4s     //mul without shift and store results in v8 and v9

        sqshl       v3.2d, v8.2d, v1.2d         // abs >> shift - 1
        and         v3.16b, v3.16b, v4.16b      // abs & 0x1
        sub         v3.2d, v3.2d, v4.2d         //nudge : -1 if we want to round down, 0 if up

        add         v8.2d, v8.2d, v3.2d
        sqrshl      v8.2d, v8.2d, v1.2d

        sqshl       v3.2d, v9.2d, v1.2d
        and         v3.16b, v3.16b, v4.16b
        sub         v3.2d, v3.2d, v4.2d         //nudge : -1 if we want to round down, 0 if up

        add         v9.2d, v9.2d, v3.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v27.4s, v8.4s, v9.4s    //combine back

        neg         v3.4s, v27.4s
        bit         v27.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v28.4s, #0
        abs         v28.4s, v28.4s
        sqdmull     v8.2d, v28.2s, v2.2s
        sqdmull2    v9.2d, v28.4s, v2.4s     //mul without shift and store results in v8 and v9

        sqshl       v3.2d, v8.2d, v1.2d         // abs >> shift - 1
        and         v3.16b, v3.16b, v4.16b      // abs & 0x1
        sub         v3.2d, v3.2d, v4.2d         //nudge : -1 if we want to round down, 0 if up

        add         v8.2d, v8.2d, v3.2d
        sqrshl      v8.2d, v8.2d, v1.2d

        sqshl       v3.2d, v9.2d, v1.2d
        and         v3.16b, v3.16b, v4.16b
        sub         v3.2d, v3.2d, v4.2d         //nudge : -1 if we want to round down, 0 if up

        add         v9.2d, v9.2d, v3.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v28.4s, v8.4s, v9.4s    //combine back

        neg         v3.4s, v28.4s
        bit         v28.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v29.4s, #0
        abs         v29.4s, v29.4s
        sqdmull     v8.2d, v29.2s, v2.2s
        sqdmull2    v9.2d, v29.4s, v2.4s     //mul without shift and store results in v8 and v9

        sqshl       v3.2d, v8.2d, v1.2d         // abs >> shift - 1
        and         v3.16b, v3.16b, v4.16b      // abs & 0x1
        sub         v3.2d, v3.2d, v4.2d         //nudge : -1 if we want to round down, 0 if up

        add         v8.2d, v8.2d, v3.2d
        sqrshl      v8.2d, v8.2d, v1.2d

        sqshl       v3.2d, v9.2d, v1.2d
        and         v3.16b, v3.16b, v4.16b
        sub         v3.2d, v3.2d, v4.2d         //nudge : -1 if we want to round down, 0 if up

        add         v9.2d, v9.2d, v3.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v29.4s, v8.4s, v9.4s    //combine back

        neg         v3.4s, v29.4s
        bit         v29.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v30.4s, #0
        abs         v30.4s, v30.4s
        sqdmull     v8.2d, v30.2s, v2.2s
        sqdmull2    v9.2d, v30.4s, v2.4s     //mul without shift and store results in v8 and v9

        sqshl       v3.2d, v8.2d, v1.2d         // abs >> shift - 1
        and         v3.16b, v3.16b, v4.16b      // abs & 0x1
        sub         v3.2d, v3.2d, v4.2d         //nudge : -1 if we want to round down, 0 if up

        add         v8.2d, v8.2d, v3.2d
        sqrshl      v8.2d, v8.2d, v1.2d

        sqshl       v3.2d, v9.2d, v1.2d
        and         v3.16b, v3.16b, v4.16b
        sub         v3.2d, v3.2d, v4.2d         //nudge : -1 if we want to round down, 0 if up

        add         v9.2d, v9.2d, v3.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v30.4s, v8.4s, v9.4s    //combine back

        neg         v3.4s, v30.4s
        bit         v30.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v31.4s, #0
        abs         v31.4s, v31.4s
        sqdmull     v8.2d, v31.2s, v2.2s
        sqdmull2    v9.2d, v31.4s, v2.4s     //mul without shift and store results in v8 and v9

        sqshl       v3.2d, v8.2d, v1.2d         // abs >> shift - 1
        and         v3.16b, v3.16b, v4.16b      // abs & 0x1
        sub         v3.2d, v3.2d, v4.2d         //nudge : -1 if we want to round down, 0 if up

        add         v8.2d, v8.2d, v3.2d
        sqrshl      v8.2d, v8.2d, v1.2d

        sqshl       v3.2d, v9.2d, v1.2d
        and         v3.16b, v3.16b, v4.16b
        sub         v3.2d, v3.2d, v4.2d         //nudge : -1 if we want to round down, 0 if up

        add         v9.2d, v9.2d, v3.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1         v31.4s, v8.4s, v9.4s    //combine back

        neg         v3.4s, v31.4s
        bit         v31.16b, v3.16b, v0.16b
    

    b .non_linear_loop

.q_scale_rounding_odd: // signum * ((abs >> shift-1) - (abs & 0x1) >> 1)

    
        cmlt        v0.4s, v16.4s, #0
        abs         v16.4s, v16.4s
        sqdmull     v8.2d, v16.2s, v2.2s
        sqdmull2    v9.2d, v16.4s, v2.4s     //mul without shift and store results in v8 and v9

        sqshl       v3.2d, v8.2d, v1.2d
        and         v3.16b, v3.16b, v4.16b      //nudge : -1 if we want to round down, 0 if up

        sub         v8.2d, v8.2d, v3.2d
        sqrshl      v8.2d, v8.2d, v1.2d

        sqshl       v3.2d, v9.2d, v1.2d
        and         v3.16b, v3.16b, v4.16b      //nudge : -1 if we want to round down, 0 if up

        sub         v9.2d, v9.2d, v3.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1        v16.4s, v8.4s, v9.4s    //combine back

        neg         v3.4s, v16.4s
        bit         v16.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v17.4s, #0
        abs         v17.4s, v17.4s
        sqdmull     v8.2d, v17.2s, v2.2s
        sqdmull2    v9.2d, v17.4s, v2.4s     //mul without shift and store results in v8 and v9

        sqshl       v3.2d, v8.2d, v1.2d
        and         v3.16b, v3.16b, v4.16b      //nudge : -1 if we want to round down, 0 if up

        sub         v8.2d, v8.2d, v3.2d
        sqrshl      v8.2d, v8.2d, v1.2d

        sqshl       v3.2d, v9.2d, v1.2d
        and         v3.16b, v3.16b, v4.16b      //nudge : -1 if we want to round down, 0 if up

        sub         v9.2d, v9.2d, v3.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1        v17.4s, v8.4s, v9.4s    //combine back

        neg         v3.4s, v17.4s
        bit         v17.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v18.4s, #0
        abs         v18.4s, v18.4s
        sqdmull     v8.2d, v18.2s, v2.2s
        sqdmull2    v9.2d, v18.4s, v2.4s     //mul without shift and store results in v8 and v9

        sqshl       v3.2d, v8.2d, v1.2d
        and         v3.16b, v3.16b, v4.16b      //nudge : -1 if we want to round down, 0 if up

        sub         v8.2d, v8.2d, v3.2d
        sqrshl      v8.2d, v8.2d, v1.2d

        sqshl       v3.2d, v9.2d, v1.2d
        and         v3.16b, v3.16b, v4.16b      //nudge : -1 if we want to round down, 0 if up

        sub         v9.2d, v9.2d, v3.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1        v18.4s, v8.4s, v9.4s    //combine back

        neg         v3.4s, v18.4s
        bit         v18.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v19.4s, #0
        abs         v19.4s, v19.4s
        sqdmull     v8.2d, v19.2s, v2.2s
        sqdmull2    v9.2d, v19.4s, v2.4s     //mul without shift and store results in v8 and v9

        sqshl       v3.2d, v8.2d, v1.2d
        and         v3.16b, v3.16b, v4.16b      //nudge : -1 if we want to round down, 0 if up

        sub         v8.2d, v8.2d, v3.2d
        sqrshl      v8.2d, v8.2d, v1.2d

        sqshl       v3.2d, v9.2d, v1.2d
        and         v3.16b, v3.16b, v4.16b      //nudge : -1 if we want to round down, 0 if up

        sub         v9.2d, v9.2d, v3.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1        v19.4s, v8.4s, v9.4s    //combine back

        neg         v3.4s, v19.4s
        bit         v19.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v20.4s, #0
        abs         v20.4s, v20.4s
        sqdmull     v8.2d, v20.2s, v2.2s
        sqdmull2    v9.2d, v20.4s, v2.4s     //mul without shift and store results in v8 and v9

        sqshl       v3.2d, v8.2d, v1.2d
        and         v3.16b, v3.16b, v4.16b      //nudge : -1 if we want to round down, 0 if up

        sub         v8.2d, v8.2d, v3.2d
        sqrshl      v8.2d, v8.2d, v1.2d

        sqshl       v3.2d, v9.2d, v1.2d
        and         v3.16b, v3.16b, v4.16b      //nudge : -1 if we want to round down, 0 if up

        sub         v9.2d, v9.2d, v3.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1        v20.4s, v8.4s, v9.4s    //combine back

        neg         v3.4s, v20.4s
        bit         v20.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v21.4s, #0
        abs         v21.4s, v21.4s
        sqdmull     v8.2d, v21.2s, v2.2s
        sqdmull2    v9.2d, v21.4s, v2.4s     //mul without shift and store results in v8 and v9

        sqshl       v3.2d, v8.2d, v1.2d
        and         v3.16b, v3.16b, v4.16b      //nudge : -1 if we want to round down, 0 if up

        sub         v8.2d, v8.2d, v3.2d
        sqrshl      v8.2d, v8.2d, v1.2d

        sqshl       v3.2d, v9.2d, v1.2d
        and         v3.16b, v3.16b, v4.16b      //nudge : -1 if we want to round down, 0 if up

        sub         v9.2d, v9.2d, v3.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1        v21.4s, v8.4s, v9.4s    //combine back

        neg         v3.4s, v21.4s
        bit         v21.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v22.4s, #0
        abs         v22.4s, v22.4s
        sqdmull     v8.2d, v22.2s, v2.2s
        sqdmull2    v9.2d, v22.4s, v2.4s     //mul without shift and store results in v8 and v9

        sqshl       v3.2d, v8.2d, v1.2d
        and         v3.16b, v3.16b, v4.16b      //nudge : -1 if we want to round down, 0 if up

        sub         v8.2d, v8.2d, v3.2d
        sqrshl      v8.2d, v8.2d, v1.2d

        sqshl       v3.2d, v9.2d, v1.2d
        and         v3.16b, v3.16b, v4.16b      //nudge : -1 if we want to round down, 0 if up

        sub         v9.2d, v9.2d, v3.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1        v22.4s, v8.4s, v9.4s    //combine back

        neg         v3.4s, v22.4s
        bit         v22.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v23.4s, #0
        abs         v23.4s, v23.4s
        sqdmull     v8.2d, v23.2s, v2.2s
        sqdmull2    v9.2d, v23.4s, v2.4s     //mul without shift and store results in v8 and v9

        sqshl       v3.2d, v8.2d, v1.2d
        and         v3.16b, v3.16b, v4.16b      //nudge : -1 if we want to round down, 0 if up

        sub         v8.2d, v8.2d, v3.2d
        sqrshl      v8.2d, v8.2d, v1.2d

        sqshl       v3.2d, v9.2d, v1.2d
        and         v3.16b, v3.16b, v4.16b      //nudge : -1 if we want to round down, 0 if up

        sub         v9.2d, v9.2d, v3.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1        v23.4s, v8.4s, v9.4s    //combine back

        neg         v3.4s, v23.4s
        bit         v23.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v24.4s, #0
        abs         v24.4s, v24.4s
        sqdmull     v8.2d, v24.2s, v2.2s
        sqdmull2    v9.2d, v24.4s, v2.4s     //mul without shift and store results in v8 and v9

        sqshl       v3.2d, v8.2d, v1.2d
        and         v3.16b, v3.16b, v4.16b      //nudge : -1 if we want to round down, 0 if up

        sub         v8.2d, v8.2d, v3.2d
        sqrshl      v8.2d, v8.2d, v1.2d

        sqshl       v3.2d, v9.2d, v1.2d
        and         v3.16b, v3.16b, v4.16b      //nudge : -1 if we want to round down, 0 if up

        sub         v9.2d, v9.2d, v3.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1        v24.4s, v8.4s, v9.4s    //combine back

        neg         v3.4s, v24.4s
        bit         v24.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v25.4s, #0
        abs         v25.4s, v25.4s
        sqdmull     v8.2d, v25.2s, v2.2s
        sqdmull2    v9.2d, v25.4s, v2.4s     //mul without shift and store results in v8 and v9

        sqshl       v3.2d, v8.2d, v1.2d
        and         v3.16b, v3.16b, v4.16b      //nudge : -1 if we want to round down, 0 if up

        sub         v8.2d, v8.2d, v3.2d
        sqrshl      v8.2d, v8.2d, v1.2d

        sqshl       v3.2d, v9.2d, v1.2d
        and         v3.16b, v3.16b, v4.16b      //nudge : -1 if we want to round down, 0 if up

        sub         v9.2d, v9.2d, v3.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1        v25.4s, v8.4s, v9.4s    //combine back

        neg         v3.4s, v25.4s
        bit         v25.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v26.4s, #0
        abs         v26.4s, v26.4s
        sqdmull     v8.2d, v26.2s, v2.2s
        sqdmull2    v9.2d, v26.4s, v2.4s     //mul without shift and store results in v8 and v9

        sqshl       v3.2d, v8.2d, v1.2d
        and         v3.16b, v3.16b, v4.16b      //nudge : -1 if we want to round down, 0 if up

        sub         v8.2d, v8.2d, v3.2d
        sqrshl      v8.2d, v8.2d, v1.2d

        sqshl       v3.2d, v9.2d, v1.2d
        and         v3.16b, v3.16b, v4.16b      //nudge : -1 if we want to round down, 0 if up

        sub         v9.2d, v9.2d, v3.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1        v26.4s, v8.4s, v9.4s    //combine back

        neg         v3.4s, v26.4s
        bit         v26.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v27.4s, #0
        abs         v27.4s, v27.4s
        sqdmull     v8.2d, v27.2s, v2.2s
        sqdmull2    v9.2d, v27.4s, v2.4s     //mul without shift and store results in v8 and v9

        sqshl       v3.2d, v8.2d, v1.2d
        and         v3.16b, v3.16b, v4.16b      //nudge : -1 if we want to round down, 0 if up

        sub         v8.2d, v8.2d, v3.2d
        sqrshl      v8.2d, v8.2d, v1.2d

        sqshl       v3.2d, v9.2d, v1.2d
        and         v3.16b, v3.16b, v4.16b      //nudge : -1 if we want to round down, 0 if up

        sub         v9.2d, v9.2d, v3.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1        v27.4s, v8.4s, v9.4s    //combine back

        neg         v3.4s, v27.4s
        bit         v27.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v28.4s, #0
        abs         v28.4s, v28.4s
        sqdmull     v8.2d, v28.2s, v2.2s
        sqdmull2    v9.2d, v28.4s, v2.4s     //mul without shift and store results in v8 and v9

        sqshl       v3.2d, v8.2d, v1.2d
        and         v3.16b, v3.16b, v4.16b      //nudge : -1 if we want to round down, 0 if up

        sub         v8.2d, v8.2d, v3.2d
        sqrshl      v8.2d, v8.2d, v1.2d

        sqshl       v3.2d, v9.2d, v1.2d
        and         v3.16b, v3.16b, v4.16b      //nudge : -1 if we want to round down, 0 if up

        sub         v9.2d, v9.2d, v3.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1        v28.4s, v8.4s, v9.4s    //combine back

        neg         v3.4s, v28.4s
        bit         v28.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v29.4s, #0
        abs         v29.4s, v29.4s
        sqdmull     v8.2d, v29.2s, v2.2s
        sqdmull2    v9.2d, v29.4s, v2.4s     //mul without shift and store results in v8 and v9

        sqshl       v3.2d, v8.2d, v1.2d
        and         v3.16b, v3.16b, v4.16b      //nudge : -1 if we want to round down, 0 if up

        sub         v8.2d, v8.2d, v3.2d
        sqrshl      v8.2d, v8.2d, v1.2d

        sqshl       v3.2d, v9.2d, v1.2d
        and         v3.16b, v3.16b, v4.16b      //nudge : -1 if we want to round down, 0 if up

        sub         v9.2d, v9.2d, v3.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1        v29.4s, v8.4s, v9.4s    //combine back

        neg         v3.4s, v29.4s
        bit         v29.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v30.4s, #0
        abs         v30.4s, v30.4s
        sqdmull     v8.2d, v30.2s, v2.2s
        sqdmull2    v9.2d, v30.4s, v2.4s     //mul without shift and store results in v8 and v9

        sqshl       v3.2d, v8.2d, v1.2d
        and         v3.16b, v3.16b, v4.16b      //nudge : -1 if we want to round down, 0 if up

        sub         v8.2d, v8.2d, v3.2d
        sqrshl      v8.2d, v8.2d, v1.2d

        sqshl       v3.2d, v9.2d, v1.2d
        and         v3.16b, v3.16b, v4.16b      //nudge : -1 if we want to round down, 0 if up

        sub         v9.2d, v9.2d, v3.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1        v30.4s, v8.4s, v9.4s    //combine back

        neg         v3.4s, v30.4s
        bit         v30.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v31.4s, #0
        abs         v31.4s, v31.4s
        sqdmull     v8.2d, v31.2s, v2.2s
        sqdmull2    v9.2d, v31.4s, v2.4s     //mul without shift and store results in v8 and v9

        sqshl       v3.2d, v8.2d, v1.2d
        and         v3.16b, v3.16b, v4.16b      //nudge : -1 if we want to round down, 0 if up

        sub         v8.2d, v8.2d, v3.2d
        sqrshl      v8.2d, v8.2d, v1.2d

        sqshl       v3.2d, v9.2d, v1.2d
        and         v3.16b, v3.16b, v4.16b      //nudge : -1 if we want to round down, 0 if up

        sub         v9.2d, v9.2d, v3.2d
        sqrshl      v9.2d, v9.2d, v1.2d

        uzp1        v31.4s, v8.4s, v9.4s    //combine back

        neg         v3.4s, v31.4s
        bit         v31.16b, v3.16b, v0.16b
    

    b .non_linear_loop

.q_shl:
    ldr     x5, [x0, #8]                // x5: shift
    ins     v1.s[0], w5
    dup     v1.4s, v1.s[0]              // v1.4s <- shift

    
        sqrshl      v16.4s, v16.4s, v1.4s
    
        sqrshl      v17.4s, v17.4s, v1.4s
    
        sqrshl      v18.4s, v18.4s, v1.4s
    
        sqrshl      v19.4s, v19.4s, v1.4s
    
        sqrshl      v20.4s, v20.4s, v1.4s
    
        sqrshl      v21.4s, v21.4s, v1.4s
    
        sqrshl      v22.4s, v22.4s, v1.4s
    
        sqrshl      v23.4s, v23.4s, v1.4s
    
        sqrshl      v24.4s, v24.4s, v1.4s
    
        sqrshl      v25.4s, v25.4s, v1.4s
    
        sqrshl      v26.4s, v26.4s, v1.4s
    
        sqrshl      v27.4s, v27.4s, v1.4s
    
        sqrshl      v28.4s, v28.4s, v1.4s
    
        sqrshl      v29.4s, v29.4s, v1.4s
    
        sqrshl      v30.4s, v30.4s, v1.4s
    
        sqrshl      v31.4s, v31.4s, v1.4s
    
    b .non_linear_loop

.q_shr:
    ldp     x5, x6, [x0, #8]            // x5: shift, x6: policy

    mov     w3, #1
    ins     v4.s[0], w3
    dup     v4.4s, v4.s[0]              // v4.4d <- 1

    neg     w5, w5                      // broadcast shift
    ins     v1.s[0], w5
    dup     v1.4s, v1.s[0]              // v1.4s <- -shift

    cmp     x6, 1
    beq     .q_shr_rounding_zero
    cmp     x6, 2
    beq     .q_shr_rounding_away
    cmp     x6, 3
    beq     .q_shr_rounding_minus_inf
    cmp     x6, 4
    beq     .q_shr_rounding_plus_inf
    cmp     x6, 5
    beq     .q_shr_rounding_even
    cmp     x6, 6
    beq     .q_shr_rounding_odd

    b .unsupported

.q_shr_rounding_zero:
    // asm: signum * (abs >>r shift)
    
        cmlt        v0.4s, v16.4s, #0
        abs         v16.4s, v16.4s

        sub         v16.4s, v16.4s, v4.4s
        sqrshl      v16.4s, v16.4s, v1.4s

        neg         v3.4s, v16.4s
        bit         v16.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v17.4s, #0
        abs         v17.4s, v17.4s

        sub         v17.4s, v17.4s, v4.4s
        sqrshl      v17.4s, v17.4s, v1.4s

        neg         v3.4s, v17.4s
        bit         v17.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v18.4s, #0
        abs         v18.4s, v18.4s

        sub         v18.4s, v18.4s, v4.4s
        sqrshl      v18.4s, v18.4s, v1.4s

        neg         v3.4s, v18.4s
        bit         v18.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v19.4s, #0
        abs         v19.4s, v19.4s

        sub         v19.4s, v19.4s, v4.4s
        sqrshl      v19.4s, v19.4s, v1.4s

        neg         v3.4s, v19.4s
        bit         v19.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v20.4s, #0
        abs         v20.4s, v20.4s

        sub         v20.4s, v20.4s, v4.4s
        sqrshl      v20.4s, v20.4s, v1.4s

        neg         v3.4s, v20.4s
        bit         v20.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v21.4s, #0
        abs         v21.4s, v21.4s

        sub         v21.4s, v21.4s, v4.4s
        sqrshl      v21.4s, v21.4s, v1.4s

        neg         v3.4s, v21.4s
        bit         v21.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v22.4s, #0
        abs         v22.4s, v22.4s

        sub         v22.4s, v22.4s, v4.4s
        sqrshl      v22.4s, v22.4s, v1.4s

        neg         v3.4s, v22.4s
        bit         v22.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v23.4s, #0
        abs         v23.4s, v23.4s

        sub         v23.4s, v23.4s, v4.4s
        sqrshl      v23.4s, v23.4s, v1.4s

        neg         v3.4s, v23.4s
        bit         v23.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v24.4s, #0
        abs         v24.4s, v24.4s

        sub         v24.4s, v24.4s, v4.4s
        sqrshl      v24.4s, v24.4s, v1.4s

        neg         v3.4s, v24.4s
        bit         v24.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v25.4s, #0
        abs         v25.4s, v25.4s

        sub         v25.4s, v25.4s, v4.4s
        sqrshl      v25.4s, v25.4s, v1.4s

        neg         v3.4s, v25.4s
        bit         v25.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v26.4s, #0
        abs         v26.4s, v26.4s

        sub         v26.4s, v26.4s, v4.4s
        sqrshl      v26.4s, v26.4s, v1.4s

        neg         v3.4s, v26.4s
        bit         v26.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v27.4s, #0
        abs         v27.4s, v27.4s

        sub         v27.4s, v27.4s, v4.4s
        sqrshl      v27.4s, v27.4s, v1.4s

        neg         v3.4s, v27.4s
        bit         v27.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v28.4s, #0
        abs         v28.4s, v28.4s

        sub         v28.4s, v28.4s, v4.4s
        sqrshl      v28.4s, v28.4s, v1.4s

        neg         v3.4s, v28.4s
        bit         v28.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v29.4s, #0
        abs         v29.4s, v29.4s

        sub         v29.4s, v29.4s, v4.4s
        sqrshl      v29.4s, v29.4s, v1.4s

        neg         v3.4s, v29.4s
        bit         v29.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v30.4s, #0
        abs         v30.4s, v30.4s

        sub         v30.4s, v30.4s, v4.4s
        sqrshl      v30.4s, v30.4s, v1.4s

        neg         v3.4s, v30.4s
        bit         v30.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v31.4s, #0
        abs         v31.4s, v31.4s

        sub         v31.4s, v31.4s, v4.4s
        sqrshl      v31.4s, v31.4s, v1.4s

        neg         v3.4s, v31.4s
        bit         v31.16b, v3.16b, v0.16b
    
    b .non_linear_loop

.q_shr_rounding_away:
    
        cmlt        v0.4s, v16.4s, #0
        abs         v16.4s, v16.4s

        sqrshl      v16.4s, v16.4s, v1.4s

        neg         v3.4s, v16.4s
        bit         v16.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v17.4s, #0
        abs         v17.4s, v17.4s

        sqrshl      v17.4s, v17.4s, v1.4s

        neg         v3.4s, v17.4s
        bit         v17.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v18.4s, #0
        abs         v18.4s, v18.4s

        sqrshl      v18.4s, v18.4s, v1.4s

        neg         v3.4s, v18.4s
        bit         v18.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v19.4s, #0
        abs         v19.4s, v19.4s

        sqrshl      v19.4s, v19.4s, v1.4s

        neg         v3.4s, v19.4s
        bit         v19.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v20.4s, #0
        abs         v20.4s, v20.4s

        sqrshl      v20.4s, v20.4s, v1.4s

        neg         v3.4s, v20.4s
        bit         v20.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v21.4s, #0
        abs         v21.4s, v21.4s

        sqrshl      v21.4s, v21.4s, v1.4s

        neg         v3.4s, v21.4s
        bit         v21.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v22.4s, #0
        abs         v22.4s, v22.4s

        sqrshl      v22.4s, v22.4s, v1.4s

        neg         v3.4s, v22.4s
        bit         v22.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v23.4s, #0
        abs         v23.4s, v23.4s

        sqrshl      v23.4s, v23.4s, v1.4s

        neg         v3.4s, v23.4s
        bit         v23.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v24.4s, #0
        abs         v24.4s, v24.4s

        sqrshl      v24.4s, v24.4s, v1.4s

        neg         v3.4s, v24.4s
        bit         v24.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v25.4s, #0
        abs         v25.4s, v25.4s

        sqrshl      v25.4s, v25.4s, v1.4s

        neg         v3.4s, v25.4s
        bit         v25.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v26.4s, #0
        abs         v26.4s, v26.4s

        sqrshl      v26.4s, v26.4s, v1.4s

        neg         v3.4s, v26.4s
        bit         v26.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v27.4s, #0
        abs         v27.4s, v27.4s

        sqrshl      v27.4s, v27.4s, v1.4s

        neg         v3.4s, v27.4s
        bit         v27.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v28.4s, #0
        abs         v28.4s, v28.4s

        sqrshl      v28.4s, v28.4s, v1.4s

        neg         v3.4s, v28.4s
        bit         v28.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v29.4s, #0
        abs         v29.4s, v29.4s

        sqrshl      v29.4s, v29.4s, v1.4s

        neg         v3.4s, v29.4s
        bit         v29.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v30.4s, #0
        abs         v30.4s, v30.4s

        sqrshl      v30.4s, v30.4s, v1.4s

        neg         v3.4s, v30.4s
        bit         v30.16b, v3.16b, v0.16b
    
        cmlt        v0.4s, v31.4s, #0
        abs         v31.4s, v31.4s

        sqrshl      v31.4s, v31.4s, v1.4s

        neg         v3.4s, v31.4s
        bit         v31.16b, v3.16b, v0.16b
    
    b .non_linear_loop

.q_shr_rounding_minus_inf:
    
        sqneg       v16.4s, v16.4s
        sqrshl      v16.4s, v16.4s, v1.4s
        sqneg       v16.4s, v16.4s
    
        sqneg       v17.4s, v17.4s
        sqrshl      v17.4s, v17.4s, v1.4s
        sqneg       v17.4s, v17.4s
    
        sqneg       v18.4s, v18.4s
        sqrshl      v18.4s, v18.4s, v1.4s
        sqneg       v18.4s, v18.4s
    
        sqneg       v19.4s, v19.4s
        sqrshl      v19.4s, v19.4s, v1.4s
        sqneg       v19.4s, v19.4s
    
        sqneg       v20.4s, v20.4s
        sqrshl      v20.4s, v20.4s, v1.4s
        sqneg       v20.4s, v20.4s
    
        sqneg       v21.4s, v21.4s
        sqrshl      v21.4s, v21.4s, v1.4s
        sqneg       v21.4s, v21.4s
    
        sqneg       v22.4s, v22.4s
        sqrshl      v22.4s, v22.4s, v1.4s
        sqneg       v22.4s, v22.4s
    
        sqneg       v23.4s, v23.4s
        sqrshl      v23.4s, v23.4s, v1.4s
        sqneg       v23.4s, v23.4s
    
        sqneg       v24.4s, v24.4s
        sqrshl      v24.4s, v24.4s, v1.4s
        sqneg       v24.4s, v24.4s
    
        sqneg       v25.4s, v25.4s
        sqrshl      v25.4s, v25.4s, v1.4s
        sqneg       v25.4s, v25.4s
    
        sqneg       v26.4s, v26.4s
        sqrshl      v26.4s, v26.4s, v1.4s
        sqneg       v26.4s, v26.4s
    
        sqneg       v27.4s, v27.4s
        sqrshl      v27.4s, v27.4s, v1.4s
        sqneg       v27.4s, v27.4s
    
        sqneg       v28.4s, v28.4s
        sqrshl      v28.4s, v28.4s, v1.4s
        sqneg       v28.4s, v28.4s
    
        sqneg       v29.4s, v29.4s
        sqrshl      v29.4s, v29.4s, v1.4s
        sqneg       v29.4s, v29.4s
    
        sqneg       v30.4s, v30.4s
        sqrshl      v30.4s, v30.4s, v1.4s
        sqneg       v30.4s, v30.4s
    
        sqneg       v31.4s, v31.4s
        sqrshl      v31.4s, v31.4s, v1.4s
        sqneg       v31.4s, v31.4s
    
    b .non_linear_loop

.q_shr_rounding_plus_inf:
    
        sqrshl      v16.4s, v16.4s, v1.4s
    
        sqrshl      v17.4s, v17.4s, v1.4s
    
        sqrshl      v18.4s, v18.4s, v1.4s
    
        sqrshl      v19.4s, v19.4s, v1.4s
    
        sqrshl      v20.4s, v20.4s, v1.4s
    
        sqrshl      v21.4s, v21.4s, v1.4s
    
        sqrshl      v22.4s, v22.4s, v1.4s
    
        sqrshl      v23.4s, v23.4s, v1.4s
    
        sqrshl      v24.4s, v24.4s, v1.4s
    
        sqrshl      v25.4s, v25.4s, v1.4s
    
        sqrshl      v26.4s, v26.4s, v1.4s
    
        sqrshl      v27.4s, v27.4s, v1.4s
    
        sqrshl      v28.4s, v28.4s, v1.4s
    
        sqrshl      v29.4s, v29.4s, v1.4s
    
        sqrshl      v30.4s, v30.4s, v1.4s
    
        sqrshl      v31.4s, v31.4s, v1.4s
    
    b .non_linear_loop

.q_shr_rounding_even:
    // sqrshl is round(+inf), sqshl trauncates
    // we look at parity of result by truncation: if it's odd, we have nothing more to do, we go towards +inf
    // if it's even, we need to nudge towards 0 by adding -1
    // => nudge = (x >>l shift) & 0x1 - 1 (>>l is sqshl)
    // => result is (x + nudge) >>r shift (with sqrshl)
    
        sqshl       v3.4s, v16.4s, v1.4s // trunc
        and         v3.16b, v3.16b, v4.16b
        sub         v3.4s, v3.4s, v4.4s
        add         v16.4s, v16.4s, v3.4s

        sqrshl      v16.4s, v16.4s, v1.4s
    
        sqshl       v3.4s, v17.4s, v1.4s // trunc
        and         v3.16b, v3.16b, v4.16b
        sub         v3.4s, v3.4s, v4.4s
        add         v17.4s, v17.4s, v3.4s

        sqrshl      v17.4s, v17.4s, v1.4s
    
        sqshl       v3.4s, v18.4s, v1.4s // trunc
        and         v3.16b, v3.16b, v4.16b
        sub         v3.4s, v3.4s, v4.4s
        add         v18.4s, v18.4s, v3.4s

        sqrshl      v18.4s, v18.4s, v1.4s
    
        sqshl       v3.4s, v19.4s, v1.4s // trunc
        and         v3.16b, v3.16b, v4.16b
        sub         v3.4s, v3.4s, v4.4s
        add         v19.4s, v19.4s, v3.4s

        sqrshl      v19.4s, v19.4s, v1.4s
    
        sqshl       v3.4s, v20.4s, v1.4s // trunc
        and         v3.16b, v3.16b, v4.16b
        sub         v3.4s, v3.4s, v4.4s
        add         v20.4s, v20.4s, v3.4s

        sqrshl      v20.4s, v20.4s, v1.4s
    
        sqshl       v3.4s, v21.4s, v1.4s // trunc
        and         v3.16b, v3.16b, v4.16b
        sub         v3.4s, v3.4s, v4.4s
        add         v21.4s, v21.4s, v3.4s

        sqrshl      v21.4s, v21.4s, v1.4s
    
        sqshl       v3.4s, v22.4s, v1.4s // trunc
        and         v3.16b, v3.16b, v4.16b
        sub         v3.4s, v3.4s, v4.4s
        add         v22.4s, v22.4s, v3.4s

        sqrshl      v22.4s, v22.4s, v1.4s
    
        sqshl       v3.4s, v23.4s, v1.4s // trunc
        and         v3.16b, v3.16b, v4.16b
        sub         v3.4s, v3.4s, v4.4s
        add         v23.4s, v23.4s, v3.4s

        sqrshl      v23.4s, v23.4s, v1.4s
    
        sqshl       v3.4s, v24.4s, v1.4s // trunc
        and         v3.16b, v3.16b, v4.16b
        sub         v3.4s, v3.4s, v4.4s
        add         v24.4s, v24.4s, v3.4s

        sqrshl      v24.4s, v24.4s, v1.4s
    
        sqshl       v3.4s, v25.4s, v1.4s // trunc
        and         v3.16b, v3.16b, v4.16b
        sub         v3.4s, v3.4s, v4.4s
        add         v25.4s, v25.4s, v3.4s

        sqrshl      v25.4s, v25.4s, v1.4s
    
        sqshl       v3.4s, v26.4s, v1.4s // trunc
        and         v3.16b, v3.16b, v4.16b
        sub         v3.4s, v3.4s, v4.4s
        add         v26.4s, v26.4s, v3.4s

        sqrshl      v26.4s, v26.4s, v1.4s
    
        sqshl       v3.4s, v27.4s, v1.4s // trunc
        and         v3.16b, v3.16b, v4.16b
        sub         v3.4s, v3.4s, v4.4s
        add         v27.4s, v27.4s, v3.4s

        sqrshl      v27.4s, v27.4s, v1.4s
    
        sqshl       v3.4s, v28.4s, v1.4s // trunc
        and         v3.16b, v3.16b, v4.16b
        sub         v3.4s, v3.4s, v4.4s
        add         v28.4s, v28.4s, v3.4s

        sqrshl      v28.4s, v28.4s, v1.4s
    
        sqshl       v3.4s, v29.4s, v1.4s // trunc
        and         v3.16b, v3.16b, v4.16b
        sub         v3.4s, v3.4s, v4.4s
        add         v29.4s, v29.4s, v3.4s

        sqrshl      v29.4s, v29.4s, v1.4s
    
        sqshl       v3.4s, v30.4s, v1.4s // trunc
        and         v3.16b, v3.16b, v4.16b
        sub         v3.4s, v3.4s, v4.4s
        add         v30.4s, v30.4s, v3.4s

        sqrshl      v30.4s, v30.4s, v1.4s
    
        sqshl       v3.4s, v31.4s, v1.4s // trunc
        and         v3.16b, v3.16b, v4.16b
        sub         v3.4s, v3.4s, v4.4s
        add         v31.4s, v31.4s, v3.4s

        sqrshl      v31.4s, v31.4s, v1.4s
    
    b .non_linear_loop

.q_shr_rounding_odd:
    // here: nudge is -((x >>l shift) & 0x1)
    
        sqshl       v3.4s, v16.4s, v1.4s // trunc
        and         v3.16b, v3.16b, v4.16b
        neg         v3.4s, v3.4s
        add         v16.4s, v16.4s, v3.4s

        sqrshl      v16.4s, v16.4s, v1.4s
    
        sqshl       v3.4s, v17.4s, v1.4s // trunc
        and         v3.16b, v3.16b, v4.16b
        neg         v3.4s, v3.4s
        add         v17.4s, v17.4s, v3.4s

        sqrshl      v17.4s, v17.4s, v1.4s
    
        sqshl       v3.4s, v18.4s, v1.4s // trunc
        and         v3.16b, v3.16b, v4.16b
        neg         v3.4s, v3.4s
        add         v18.4s, v18.4s, v3.4s

        sqrshl      v18.4s, v18.4s, v1.4s
    
        sqshl       v3.4s, v19.4s, v1.4s // trunc
        and         v3.16b, v3.16b, v4.16b
        neg         v3.4s, v3.4s
        add         v19.4s, v19.4s, v3.4s

        sqrshl      v19.4s, v19.4s, v1.4s
    
        sqshl       v3.4s, v20.4s, v1.4s // trunc
        and         v3.16b, v3.16b, v4.16b
        neg         v3.4s, v3.4s
        add         v20.4s, v20.4s, v3.4s

        sqrshl      v20.4s, v20.4s, v1.4s
    
        sqshl       v3.4s, v21.4s, v1.4s // trunc
        and         v3.16b, v3.16b, v4.16b
        neg         v3.4s, v3.4s
        add         v21.4s, v21.4s, v3.4s

        sqrshl      v21.4s, v21.4s, v1.4s
    
        sqshl       v3.4s, v22.4s, v1.4s // trunc
        and         v3.16b, v3.16b, v4.16b
        neg         v3.4s, v3.4s
        add         v22.4s, v22.4s, v3.4s

        sqrshl      v22.4s, v22.4s, v1.4s
    
        sqshl       v3.4s, v23.4s, v1.4s // trunc
        and         v3.16b, v3.16b, v4.16b
        neg         v3.4s, v3.4s
        add         v23.4s, v23.4s, v3.4s

        sqrshl      v23.4s, v23.4s, v1.4s
    
        sqshl       v3.4s, v24.4s, v1.4s // trunc
        and         v3.16b, v3.16b, v4.16b
        neg         v3.4s, v3.4s
        add         v24.4s, v24.4s, v3.4s

        sqrshl      v24.4s, v24.4s, v1.4s
    
        sqshl       v3.4s, v25.4s, v1.4s // trunc
        and         v3.16b, v3.16b, v4.16b
        neg         v3.4s, v3.4s
        add         v25.4s, v25.4s, v3.4s

        sqrshl      v25.4s, v25.4s, v1.4s
    
        sqshl       v3.4s, v26.4s, v1.4s // trunc
        and         v3.16b, v3.16b, v4.16b
        neg         v3.4s, v3.4s
        add         v26.4s, v26.4s, v3.4s

        sqrshl      v26.4s, v26.4s, v1.4s
    
        sqshl       v3.4s, v27.4s, v1.4s // trunc
        and         v3.16b, v3.16b, v4.16b
        neg         v3.4s, v3.4s
        add         v27.4s, v27.4s, v3.4s

        sqrshl      v27.4s, v27.4s, v1.4s
    
        sqshl       v3.4s, v28.4s, v1.4s // trunc
        and         v3.16b, v3.16b, v4.16b
        neg         v3.4s, v3.4s
        add         v28.4s, v28.4s, v3.4s

        sqrshl      v28.4s, v28.4s, v1.4s
    
        sqshl       v3.4s, v29.4s, v1.4s // trunc
        and         v3.16b, v3.16b, v4.16b
        neg         v3.4s, v3.4s
        add         v29.4s, v29.4s, v3.4s

        sqrshl      v29.4s, v29.4s, v1.4s
    
        sqshl       v3.4s, v30.4s, v1.4s // trunc
        and         v3.16b, v3.16b, v4.16b
        neg         v3.4s, v3.4s
        add         v30.4s, v30.4s, v3.4s

        sqrshl      v30.4s, v30.4s, v1.4s
    
        sqshl       v3.4s, v31.4s, v1.4s // trunc
        and         v3.16b, v3.16b, v4.16b
        neg         v3.4s, v3.4s
        add         v31.4s, v31.4s, v3.4s

        sqrshl      v31.4s, v31.4s, v1.4s
    
    b .non_linear_loop
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


.store:
    ldp         x5, x6, [x0, #8]            // c base ptr, rsc
    ldp         x7, x8, [x0, #24]           // csc, item_size

    cmp         x8, #4
    beq         .store_strides_i32

    
        
            st1 { v16.b }[0], [ x5 ], x6
        
            st1 { v16.b }[4], [ x5 ], x6
        
            st1 { v16.b }[8], [ x5 ], x6
        
            st1 { v16.b }[12], [ x5 ], x6
        
    
        
            st1 { v17.b }[0], [ x5 ], x6
        
            st1 { v17.b }[4], [ x5 ], x6
        
            st1 { v17.b }[8], [ x5 ], x6
        
            st1 { v17.b }[12], [ x5 ], x6
        
    
        
            st1 { v18.b }[0], [ x5 ], x6
        
            st1 { v18.b }[4], [ x5 ], x6
        
            st1 { v18.b }[8], [ x5 ], x6
        
            st1 { v18.b }[12], [ x5 ], x6
        
    
        
            st1 { v19.b }[0], [ x5 ], x6
        
            st1 { v19.b }[4], [ x5 ], x6
        
            st1 { v19.b }[8], [ x5 ], x6
        
            st1 { v19.b }[12], [ x5 ], x6
        
    
        
            st1 { v20.b }[0], [ x5 ], x6
        
            st1 { v20.b }[4], [ x5 ], x6
        
            st1 { v20.b }[8], [ x5 ], x6
        
            st1 { v20.b }[12], [ x5 ], x6
        
    
        
            st1 { v21.b }[0], [ x5 ], x6
        
            st1 { v21.b }[4], [ x5 ], x6
        
            st1 { v21.b }[8], [ x5 ], x6
        
            st1 { v21.b }[12], [ x5 ], x6
        
    
        
            st1 { v22.b }[0], [ x5 ], x6
        
            st1 { v22.b }[4], [ x5 ], x6
        
            st1 { v22.b }[8], [ x5 ], x6
        
            st1 { v22.b }[12], [ x5 ], x6
        
    
        
            st1 { v23.b }[0], [ x5 ], x6
        
            st1 { v23.b }[4], [ x5 ], x6
        
            st1 { v23.b }[8], [ x5 ], x6
        
            st1 { v23.b }[12], [ x5 ], x6
        
    
        
            st1 { v24.b }[0], [ x5 ], x6
        
            st1 { v24.b }[4], [ x5 ], x6
        
            st1 { v24.b }[8], [ x5 ], x6
        
            st1 { v24.b }[12], [ x5 ], x6
        
    
        
            st1 { v25.b }[0], [ x5 ], x6
        
            st1 { v25.b }[4], [ x5 ], x6
        
            st1 { v25.b }[8], [ x5 ], x6
        
            st1 { v25.b }[12], [ x5 ], x6
        
    
        
            st1 { v26.b }[0], [ x5 ], x6
        
            st1 { v26.b }[4], [ x5 ], x6
        
            st1 { v26.b }[8], [ x5 ], x6
        
            st1 { v26.b }[12], [ x5 ], x6
        
    
        
            st1 { v27.b }[0], [ x5 ], x6
        
            st1 { v27.b }[4], [ x5 ], x6
        
            st1 { v27.b }[8], [ x5 ], x6
        
            st1 { v27.b }[12], [ x5 ], x6
        
    
        
            st1 { v28.b }[0], [ x5 ], x6
        
            st1 { v28.b }[4], [ x5 ], x6
        
            st1 { v28.b }[8], [ x5 ], x6
        
            st1 { v28.b }[12], [ x5 ], x6
        
    
        
            st1 { v29.b }[0], [ x5 ], x6
        
            st1 { v29.b }[4], [ x5 ], x6
        
            st1 { v29.b }[8], [ x5 ], x6
        
            st1 { v29.b }[12], [ x5 ], x6
        
    
        
            st1 { v30.b }[0], [ x5 ], x6
        
            st1 { v30.b }[4], [ x5 ], x6
        
            st1 { v30.b }[8], [ x5 ], x6
        
            st1 { v30.b }[12], [ x5 ], x6
        
    
        
            st1 { v31.b }[0], [ x5 ], x6
        
            st1 { v31.b }[4], [ x5 ], x6
        
            st1 { v31.b }[8], [ x5 ], x6
        
            st1 { v31.b }[12], [ x5 ], x6
        
    

    b   .non_linear_loop

.store_strides_i32:
    
        
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
        
    

    b   .non_linear_loop

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
