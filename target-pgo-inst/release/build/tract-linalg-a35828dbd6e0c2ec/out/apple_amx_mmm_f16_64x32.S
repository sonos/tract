// vim: ft=arm
.text
.align 4

/* Z: 64x32 tile. each Z reg is f16x32
    Z0
    Z2
    ...
    Z62
    
    Z1
    Z3
    S63
*/
    

.global _apple_amx_mmm_f16_64x32_0_23_4_pre
_apple_amx_mmm_f16_64x32_0_23_4_pre:

nop
nop
nop
.word 0x201220


    // set x1 to a 128 bytes aligned block for loads
    mov x1, sp
    lsr x1, x1, #7
    lsl x1, x1, #7
    sub x1, x1, 128

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


.leaky_relu:
.q_scale:
.q_shl:
.q_shr:
    b .unsupported

.add_mat_mul:

    ldr         x2, [x0, #24]       // b
    ldp         x3, x4, [x0, #8]    // k, a

    cmp         x3, #0
    beq         .non_linear_loop

    orr         x4, x4, 4611686018427387904    // load pairs (A)

    eor         x5, x5, x5                  // top left

    orr         x7, x5, 1048576
    orr         x7, x7, 64   // bottom left

    .packed_packed_loop_1:
    .word 0x201002 				// AMX ldx x2

    .word 0x201024 				// AMX ldy x4

    add x2, x2, 64
    add x4, x4, 128

    .word 0x2011e5 				// AMX fma16 x5

    .word 0x2011e7 				// AMX fma16 x7


    subs x3, x3, #1
    bne .packed_packed_loop_1

    b .non_linear_loop

.clear:
    // top left
    eor x2, x2, x2
    orr x2, x2, 134217728
    orr x2, x2, 268435456
    orr x2, x2, 536870912  // Z = 0
    .word 0x201182 				// AMX fma32 x2


    // top right
    orr x2, x2, 1048576  // Z row = 1
    .word 0x201182 				// AMX fma32 x2


    // bottom right
    orr x2, x2, 2097152  // Z row = 3
    .word 0x201182 				// AMX fma32 x2


    // bottom left
    eor x2, x2, 1048576  // Z row = 2
    .word 0x201182 				// AMX fma32 x2


    mov     x3, #16
    str     x3, [x1]

    b .non_linear_loop

.per_col_sub:

    // performs a unary neg on Z
    eor x2, x2, x2                      // X[0] = Z[0]

    mov x4, -9223372036854775808          // vector mode
    orr x4, x4, 268435456
    orr x4, x4, 134217728      // Z=-X

    mov x6, 64
    .per_col_sub_loop:
        .word 0x201102 				// AMX extrx x2

        .word 0x201204 				// AMX fms16 x4

        add x2, x2, 1048576    // next Z row
        add x4, x4, 1048576    // next Z row
    subs x6, x6, 1
    bne .per_col_sub_loop

    // continue

.per_col_add:
    ldr         x2, [x0, #8]

    ld1         { v0.4s, v1.4s, v2.4s, v3.4s }, [x2]
    st1         { v0.4s, v1.4s, v2.4s, v3.4s }, [x1]
    .word 0x201001 				// AMX ldx x1


    mov x2, 268435456      // z += y

    // top left
    .word 0x2011e2 				// AMX fma16 x2


    // bottom left
    orr x2, x2, 1048576  // Z row = 2
    .word 0x2011e2 				// AMX fma16 x2


    b .non_linear_loop

.per_col_sub_flipped:
    ldr         x2, [x0, #8]

    ld1         { v0.8h, v1.8h, v2.8h, v3.8h }, [x2]
    st1         { v0.8h, v1.8h, v2.8h, v3.8h }, [x1]

    .word 0x201001 				// AMX ldx x1


    mov x2, 268435456      // z += y

    .word 0x201202 				// AMX fms16 x2

    orr x2, x2, 1048576  // Z row = 1
    .word 0x201202 				// AMX fms16 x2


    b .non_linear_loop

.per_row_sub_flipped:
    ldr         x2, [x0, #8]

    ld1         { v0.8h, v1.8h, v2.8h, v3.8h }, [x2], #64
    st1         { v0.8h, v1.8h, v2.8h, v3.8h }, [x1], #64
    ld1         { v0.8h, v1.8h, v2.8h, v3.8h }, [x2]
    st1         { v0.8h, v1.8h, v2.8h, v3.8h }, [x1]
    sub x1, x1, #64

    orr x2, x1, 4611686018427387904  // load a pair
    .word 0x201022 				// AMX ldy x2


    mov x2, 536870912      // z += y

    // top left
    .word 0x201202 				// AMX fms16 x2


    // bottom right
    orr x2, x2, 1048576  // Z row = 3
    orr x2, x2, 64   // Y offset
    .word 0x201202 				// AMX fms16 x2


    b .non_linear_loop

.per_row_sub:
    // performs a unary neg on Z
    eor x2, x2, x2                      // X[0] = Z[0]

    mov x4, -9223372036854775808          // vector mode
    orr x4, x4, 268435456
    orr x4, x4, 134217728      // Z=-X

    mov x6, 64
    .per_row_sub_loop:
        .word 0x201102 				// AMX extrx x2

        .word 0x201204 				// AMX fms16 x4

        add x2, x2, 1048576    // next Z row
        add x4, x4, 1048576    // next Z row
    subs x6, x6, 1
    bne .per_row_sub_loop

    // continue

.per_row_add:
    ldr         x2, [x0, #8]

    ld1         { v0.8h, v1.8h, v2.8h, v3.8h }, [x2], #64
    st1         { v0.8h, v1.8h, v2.8h, v3.8h }, [x1], #64
    ld1         { v0.8h, v1.8h, v2.8h, v3.8h }, [x2]
    st1         { v0.8h, v1.8h, v2.8h, v3.8h }, [x1]
    sub x1, x1, #64

    orr x2, x1, 4611686018427387904  // load a pair
    .word 0x201022 				// AMX ldy x2


    mov x2, 536870912      // z += y

    // top left
    .word 0x2011e2 				// AMX fma16 x2


    // bottom right
    orr x2, x2, 1048576  // Z row = 1
    orr x2, x2, 64   // Y offset
    .word 0x2011e2 				// AMX fma16 x2


    b .non_linear_loop

.per_row_min:
    mov x2, 5
    b .per_row_min_max
.per_row_max:
    mov x2, 7
.per_row_min_max:
    ldr         x5, [x0, #8]

    add x6, x5, 64

    lsl x2, x2, 47                  // max(x,z) (or min)
    orr x2, x2, 8796093022208  // f16

    orr x8, x2, 1048576  // bottom left

    mov x4, 32
    .loop_per_row_max:
        // top half
        ld1         { v0.h }[0], [x5], #2
        dup         v0.8h, v0.h[0]
        dup         v1.8h, v0.h[0]
        dup         v2.8h, v0.h[0]
        dup         v3.8h, v0.h[0]
        st1         { v0.8h, v1.8h, v2.8h, v3.8h }, [x1]

        .word 0x201001 				// AMX ldx x1

        .word 0x201262 				// AMX vecfp x2


        add x2, x2, 2097152

        // bottom half
        ld1         { v0.h }[0], [x6], #2
        dup         v0.8h, v0.h[0]
        dup         v1.8h, v0.h[0]
        dup         v2.8h, v0.h[0]
        dup         v3.8h, v0.h[0]
        st1         { v0.8h, v1.8h, v2.8h, v3.8h }, [x1]

        .word 0x201001 				// AMX ldx x1

        .word 0x201268 				// AMX vecfp x8


        add x8, x8, 2097152

    subs x4, x4, 1
    bne .loop_per_row_max

    b .non_linear_loop

.per_col_min:
    mov x2, 5
    b .per_col_min_max
.per_col_max:
    mov x2, 7
.per_col_min_max:
    ldr         x4, [x0, #8]

    ld1         { v0.8h, v1.8h, v2.8h, v3.8h }, [x4]
    st1         { v0.8h, v1.8h, v2.8h, v3.8h }, [x1]
    .word 0x201001 				// AMX ldx x1


    lsl x2, x2, 47                  // max(x,z) (or min)
    orr x2, x2, 8796093022208  // f16

    mov x4, 64
    .loop_per_col_max:
        .word 0x201262 				// AMX vecfp x2

        add x2, x2, 1048576
    subs x4, x4, 1
    bne .loop_per_col_max

    b .non_linear_loop

.per_col_mul:
    ldr         x4, [x0, #8]

    ld1         { v0.4s, v1.4s, v2.4s, v3.4s }, [x4]
    st1         { v0.4s, v1.4s, v2.4s, v3.4s }, [x1]
    .word 0x201021 				// AMX ldy x1


    eor x2, x2, x2                      // X[0] = Z[0]

    mov x4, -9223372036854775808          // vector mode
    orr x4, x4, 134217728      // Z=X*Y

    mov x6, 64
    .loop_per_col_mul:
        .word 0x201102 				// AMX extrx x2

        .word 0x2011e4 				// AMX fma16 x4

        add x2, x2, 1048576
        add x4, x4, 1048576
    subs x6, x6, 1
    bne .loop_per_col_mul

    b .non_linear_loop

.per_row_mul:
    ldr         x14, [x0, #8]
    add         x15, x14, 64

    // extrx
    eor x2, x2, x2                      // X[0] = Z[0] (top left)

    eor x4, x4, x4
    orr x4, x4, 1048576        // X[0] = Z[1] (bottom left)

    // fma16
    eor x6, x6, x6
    orr x6, x6, -9223372036854775808        // vector mode
    orr x6, x6, 134217728        // Z=X*Y       Z[0]=X[0]*Y[0]

    orr x8, x6, 1048576        // Z[1]

    mov x10, 32
    .loop_per_row_mul:
        // top
        ld1         { v0.h }[0], [x14], #2
        dup         v0.8h, v0.h[0]
        dup         v1.8h, v0.h[0]
        dup         v2.8h, v0.h[0]
        dup         v3.8h, v0.h[0]
        st1         { v0.8h, v1.8h, v2.8h, v3.8h }, [x1]

        .word 0x201021 				// AMX ldy x1

        .word 0x201102 				// AMX extrx x2

        .word 0x2011e6 				// AMX fma16 x6


        add x2, x2, 2097152
        add x6, x6, 2097152

        // bottom
        ld1         { v0.h }[0], [x15], #2
        dup         v0.8h, v0.h[0]
        dup         v1.8h, v0.h[0]
        dup         v2.8h, v0.h[0]
        dup         v3.8h, v0.h[0]
        st1         { v0.8h, v1.8h, v2.8h, v3.8h }, [x1]

        .word 0x201021 				// AMX ldy x1

        .word 0x201104 				// AMX extrx x4

        .word 0x2011e8 				// AMX fma16 x8


        add x4, x4, 2097152
        add x8, x8, 2097152

    subs x10, x10, 1
    bne .loop_per_row_mul

    b .non_linear_loop

.scalar_sub:
    // performs a unary neg on Z, then go to scalar_add
    eor x2, x2, x2                      // X[0] = Z[0]

    mov x4, -9223372036854775808          // vector mode
    orr x4, x4, 268435456
    orr x4, x4, 134217728      // Z=-X

    mov x6, 64
    .scalar_sub_loop:
        .word 0x201102 				// AMX extrx x2

        .word 0x201204 				// AMX fms16 x4

        add x2, x2, 1048576    // next Z row
        add x4, x4, 1048576    // next Z row
    subs x6, x6, 1
    bne .scalar_sub_loop

    // continue on purpose

.scalar_add:
    ldr         w5, [x0, #8]

    fmov        h0, w5
    dup         v0.8h, v0.h[0]
    dup         v1.8h, v0.h[0]
    dup         v2.8h, v0.h[0]
    dup         v3.8h, v0.h[0]

    st1         { v0.8h, v1.8h, v2.8h, v3.8h }, [x1]
    .word 0x201001 				// AMX ldx x1
    // load 16 values

    mov x2, 268435456          // Z+=X 
    .word 0x2011e2 				// AMX fma16 x2

    add x2, x2, 1048576    // Z1
    .word 0x2011e2 				// AMX fma16 x2

    b .non_linear_loop

.scalar_sub_flipped:
    ldr         w5, [x0, #8]

    fmov        h0, w5
    dup         v0.8h, v0.h[0]
    dup         v1.8h, v0.h[0]
    dup         v2.8h, v0.h[0]
    dup         v3.8h, v0.h[0]

    st1         { v0.8h, v1.8h, v2.8h, v3.8h }, [x1]
    .word 0x201001 				// AMX ldx x1
    // load 32 values

    mov x2, 268435456          // Z-=X 
    .word 0x201202 				// AMX fms16 x2

    add x2, x2, 1048576    // next Z row
    .word 0x201202 				// AMX fms16 x2

    b .non_linear_loop

.scalar_mul:
    ldr         w5, [x0, #8]

    fmov        h0, w5
    dup         v0.8h, v0.h[0]
    dup         v1.8h, v0.h[0]
    dup         v2.8h, v0.h[0]
    dup         v3.8h, v0.h[0]

    st1         { v0.8h, v1.8h, v2.8h, v3.8h }, [x1]
    .word 0x201021 				// AMX ldy x1
    // load 32 values

    eor x2, x2, x2                      // X[0] = Z[0]

    mov x4, -9223372036854775808          // vector mode
    orr x4, x4, 134217728      // Z=X*Y

    mov x6, 64
    .scalar_mul_loop:
        .word 0x201102 				// AMX extrx x2

        .word 0x2011e4 				// AMX fma16 x4

        add x2, x2, 1048576    // next Z row
        add x4, x4, 1048576    // next Z row
    subs x6, x6, 1
    bne .scalar_mul_loop

    b .non_linear_loop

.scalar_min:
    mov x2, 5
    b .scalar_min_max
.scalar_max:
    mov x2, 7
.scalar_min_max:
    ldr         w5, [x0, #8]

    fmov        h0, w5
    dup         v0.8h, v0.h[0]
    dup         v1.8h, v0.h[0]
    dup         v2.8h, v0.h[0]
    dup         v3.8h, v0.h[0]

    st1         { v0.8h, v1.8h, v2.8h, v3.8h }, [x1]
    .word 0x201001 				// AMX ldx x1
    // load 16 values

    lsl x2, x2, 47
    orr x2, x2, 8796093022208 // f32

    mov x3, 64
    .loop_scalar_max:
        add x2, x2, 1048576 // next Z
        .word 0x201262 				// AMX vecfp x2

        subs x3, x3, 1
        bne .loop_scalar_max

    b .non_linear_loop

.add_unicast:
    ldp         x5, x6, [x0, #8]            // c base ptr, rsc
    ldp         x7, x8, [x0, #24]           // csc, item_size

    mov x3, 0                               // x3 is the row
    .loop_load:
        // z reg is (row % 32) * 2 + (row / 32)
        and x9, x3, 0x1f
        lsl x9, x9, 1
        lsr x10, x3, 5
        add x9, x9, x10

        mov x4, x5
        
            
                ld1 { v0.h }[0], [x4], x7
            
                ld1 { v0.h }[1], [x4], x7
            
                ld1 { v0.h }[2], [x4], x7
            
                ld1 { v0.h }[3], [x4], x7
            
                ld1 { v0.h }[4], [x4], x7
            
                ld1 { v0.h }[5], [x4], x7
            
                ld1 { v0.h }[6], [x4], x7
            
                ld1 { v0.h }[7], [x4], x7
            
        
            
                ld1 { v1.h }[0], [x4], x7
            
                ld1 { v1.h }[1], [x4], x7
            
                ld1 { v1.h }[2], [x4], x7
            
                ld1 { v1.h }[3], [x4], x7
            
                ld1 { v1.h }[4], [x4], x7
            
                ld1 { v1.h }[5], [x4], x7
            
                ld1 { v1.h }[6], [x4], x7
            
                ld1 { v1.h }[7], [x4], x7
            
        
            
                ld1 { v2.h }[0], [x4], x7
            
                ld1 { v2.h }[1], [x4], x7
            
                ld1 { v2.h }[2], [x4], x7
            
                ld1 { v2.h }[3], [x4], x7
            
                ld1 { v2.h }[4], [x4], x7
            
                ld1 { v2.h }[5], [x4], x7
            
                ld1 { v2.h }[6], [x4], x7
            
                ld1 { v2.h }[7], [x4], x7
            
        
            
                ld1 { v3.h }[0], [x4], x7
            
                ld1 { v3.h }[1], [x4], x7
            
                ld1 { v3.h }[2], [x4], x7
            
                ld1 { v3.h }[3], [x4], x7
            
                ld1 { v3.h }[4], [x4], x7
            
                ld1 { v3.h }[5], [x4], x7
            
                ld1 { v3.h }[6], [x4], x7
            
                ld1 { v3.h }[7], [x4], x7
            
        

        st1 { v0.8h, v1.8h, v2.8h, v3.8h }, [x1]
        .word 0x201021 				// AMX ldy x1


        lsl x2, x9, 20                  // Z register to update
        orr x2, x2, -9223372036854775808  // vector mode
        orr x2, x2, 536870912  // perform Z+=Y
        .word 0x2011e2 				// AMX fma16 x2


        add x5, x5, x6
        add x3, x3, 1
        cmp x3, 64
    bne .loop_load

    /*
    mov x3, 0                               // x3 is the row
    .loop_load:
        and x9, x3, 0xf                     // x9 = row % 16
        lsl x9, x9, 2                       // x9 = (row % 16) * 4
        lsr x10, x3, 4                      // x10 = row / 16 
        lsl x10, x10, 1                     // x10 = (row / 16) * 2
        add x9, x9, x10                     // x9 = x9 + x10

        mov x4, x5
        
            
                ld1 { v0.s }[0], [x4], x7
            
                ld1 { v0.s }[1], [x4], x7
            
                ld1 { v0.s }[2], [x4], x7
            
                ld1 { v0.s }[3], [x4], x7
            
        
            
                ld1 { v1.s }[0], [x4], x7
            
                ld1 { v1.s }[1], [x4], x7
            
                ld1 { v1.s }[2], [x4], x7
            
                ld1 { v1.s }[3], [x4], x7
            
        
            
                ld1 { v2.s }[0], [x4], x7
            
                ld1 { v2.s }[1], [x4], x7
            
                ld1 { v2.s }[2], [x4], x7
            
                ld1 { v2.s }[3], [x4], x7
            
        
            
                ld1 { v3.s }[0], [x4], x7
            
                ld1 { v3.s }[1], [x4], x7
            
                ld1 { v3.s }[2], [x4], x7
            
                ld1 { v3.s }[3], [x4], x7
            
        
        st1 { v0.4s, v1.4s, v2.4s, v3.4s }, [x1]
        
            
                ld1 { v0.s }[0], [x4], x7
            
                ld1 { v0.s }[1], [x4], x7
            
                ld1 { v0.s }[2], [x4], x7
            
                ld1 { v0.s }[3], [x4], x7
            
        
            
                ld1 { v1.s }[0], [x4], x7
            
                ld1 { v1.s }[1], [x4], x7
            
                ld1 { v1.s }[2], [x4], x7
            
                ld1 { v1.s }[3], [x4], x7
            
        
            
                ld1 { v2.s }[0], [x4], x7
            
                ld1 { v2.s }[1], [x4], x7
            
                ld1 { v2.s }[2], [x4], x7
            
                ld1 { v2.s }[3], [x4], x7
            
        
            
                ld1 { v3.s }[0], [x4], x7
            
                ld1 { v3.s }[1], [x4], x7
            
                ld1 { v3.s }[2], [x4], x7
            
                ld1 { v3.s }[3], [x4], x7
            
        
        st1 { v0.4s, v1.4s, v2.4s, v3.4s }, [x8]

        mov x2, x1
        orr x2, x2, 4611686018427387904 // load 32 values
        .word 0x201022 				// AMX ldy x2


        lsl x2, x9, 20                  // left Z register to update
        orr x2, x2, -9223372036854775808  // vector mode
        orr x2, x2, 536870912  // perform Z+=Y
        .word 0x201182 				// AMX fma32 x2


        add x2, x2, 1048576
        orr x2, x2, 64                  // offset Y by 16 values
        .word 0x201182 				// AMX fma32 x2


        add x5, x5, x6
    add x3, x3, 1
    cmp x3, 32
    bne .loop_load

*/

    b .non_linear_loop

.add_row_col_products:
    ldp         x5, x6, [x0, #8]            // a base ptr, b base ptr

    add x8, x1, 64

    ld1         { v0.4s, v1.4s, v2.4s, v3.4s }, [x5], #64
    st1         { v0.4s, v1.4s, v2.4s, v3.4s }, [x1], #64
    ld1         { v0.4s, v1.4s, v2.4s, v3.4s }, [x5]
    st1         { v0.4s, v1.4s, v2.4s, v3.4s }, [x1]
    sub x1, x1, #64

    orr x2, x1, 4611686018427387904  // load a pair
    .word 0x201022 				// AMX ldy x2


    ld1         { v0.4s, v1.4s, v2.4s, v3.4s }, [x6]
    st1         { v0.4s, v1.4s, v2.4s, v3.4s }, [x1]

    .word 0x201001 				// AMX ldx x1


    // top
    eor x2, x2, x2
    .word 0x2011e2 				// AMX fma16 x2


    // bottom right
    orr x2, x2, 1048576  // Z row = 1
    orr x2, x2, 64   // Y offset
    .word 0x2011e2 				// AMX fma16 x2


    b .non_linear_loop

.store:
    ldp         x5, x6, [x0, #8]            // c base ptr, rsc
    ldp         x7, x8, [x0, #24]           // csc, item_size

    cmp         x7, 2
    bne         .store_generic
    ands        x8, x5, 0x7f
    bne         .store_generic
    ands        x8, x6, 0x7f
    bne         .store_generic

    lsl x8, x6, 5
    add x8, x8, x5                          // x8 = 32*rsc
    orr x8, x8, 72057594037927936          // first to x8 is z1

    mov x4, 144115188075855872                // Zreg += 2
    add x4, x4, x6                          // +rsc

    mov x3, 32
    .loop_store_direct:
        .word 0x2010a5 				// AMX stz x5

        .word 0x2010a8 				// AMX stz x8

        add x5, x5, x4
        add x8, x8, x4
    subs x3, x3, 1
    bne .loop_store_direct

    b .non_linear_loop

.store_generic:

    mov x3, 0                               // row id
    .loop_store:
        // z reg is (row % 32) * 2 + (row / 32)
        and x9, x3, 0x1f
        lsl x9, x9, 1
        lsr x10, x3, 5
        add x9, x9, x10

        lsl x2, x9, 56
        orr x2, x2, x1
        .word 0x2010a2 				// AMX stz x2
                            // f16 x 32

        ld1 { v0.8h, v1.8h, v2.8h, v3.8h }, [x1]

        mov x4, x5
        
            
                st1 { v0.h }[0], [x4], x7
            
                st1 { v0.h }[1], [x4], x7
            
                st1 { v0.h }[2], [x4], x7
            
                st1 { v0.h }[3], [x4], x7
            
                st1 { v0.h }[4], [x4], x7
            
                st1 { v0.h }[5], [x4], x7
            
                st1 { v0.h }[6], [x4], x7
            
                st1 { v0.h }[7], [x4], x7
            
        
            
                st1 { v1.h }[0], [x4], x7
            
                st1 { v1.h }[1], [x4], x7
            
                st1 { v1.h }[2], [x4], x7
            
                st1 { v1.h }[3], [x4], x7
            
                st1 { v1.h }[4], [x4], x7
            
                st1 { v1.h }[5], [x4], x7
            
                st1 { v1.h }[6], [x4], x7
            
                st1 { v1.h }[7], [x4], x7
            
        
            
                st1 { v2.h }[0], [x4], x7
            
                st1 { v2.h }[1], [x4], x7
            
                st1 { v2.h }[2], [x4], x7
            
                st1 { v2.h }[3], [x4], x7
            
                st1 { v2.h }[4], [x4], x7
            
                st1 { v2.h }[5], [x4], x7
            
                st1 { v2.h }[6], [x4], x7
            
                st1 { v2.h }[7], [x4], x7
            
        
            
                st1 { v3.h }[0], [x4], x7
            
                st1 { v3.h }[1], [x4], x7
            
                st1 { v3.h }[2], [x4], x7
            
                st1 { v3.h }[3], [x4], x7
            
                st1 { v3.h }[4], [x4], x7
            
                st1 { v3.h }[5], [x4], x7
            
                st1 { v3.h }[6], [x4], x7
            
                st1 { v3.h }[7], [x4], x7
            
        
        add x5, x5, x6

        add x3, x3, 1
        cmp x3, 64
    bne .loop_store
    b .non_linear_loop

.load_tile:
    ldr  x2, [x0, #16]                      // row major ptr
    orr  x3, x2, 72057594037927936
    add  x3, x3, #2048
    
    mov  x4, 144115188075855872               // z+=2
    add  x4, x4, #64

    mov x8, 32
    .loop_load_tile:
        .word 0x201082 				// AMX ldz x2

        .word 0x201083 				// AMX ldz x3

        add x2, x2, x4
        add x3, x3, x4
    subs x8, x8, 1
    bne .loop_load_tile

    b .non_linear_loop
   
.return:
nop
nop
nop
.word 0x201221

ret