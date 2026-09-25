// vim: ft=arm
.text
.align 4

/* Z: 32x1
 z0[0] ..  z0[15] z1[0] .. z1[15]
*/
    

.global _apple_amx_mmm_f16_64x1_0_23_4_pre
_apple_amx_mmm_f16_64x1_0_23_4_pre:

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

    orr x4, x4, 4611686018427387904  // load a pair of A

    mov x5, 8796093022208      // f16
    orr x5, x5, 274877906944  // Broadcast Y

    orr x6, x5, 1048576  // z offset
    orr x6, x6, 65536  // x offset

    cmp         x3, #32
    blt         .packed_packed_loop_1

    mov x9, 4294967296        // Y broadcast offset += 1

    .packed_packed_loop_32:
        mov x7, x5
        mov x8, x6
        .word 0x201022 				// AMX ldy x2

        
            .word 0x201004 				// AMX ldx x4

            add x4, x4, 128
            .word 0x201267 				// AMX vecfp x7

            .word 0x201268 				// AMX vecfp x8

            add x7, x7, x9
            add x8, x8, x9
        
            .word 0x201004 				// AMX ldx x4

            add x4, x4, 128
            .word 0x201267 				// AMX vecfp x7

            .word 0x201268 				// AMX vecfp x8

            add x7, x7, x9
            add x8, x8, x9
        
            .word 0x201004 				// AMX ldx x4

            add x4, x4, 128
            .word 0x201267 				// AMX vecfp x7

            .word 0x201268 				// AMX vecfp x8

            add x7, x7, x9
            add x8, x8, x9
        
            .word 0x201004 				// AMX ldx x4

            add x4, x4, 128
            .word 0x201267 				// AMX vecfp x7

            .word 0x201268 				// AMX vecfp x8

            add x7, x7, x9
            add x8, x8, x9
        
            .word 0x201004 				// AMX ldx x4

            add x4, x4, 128
            .word 0x201267 				// AMX vecfp x7

            .word 0x201268 				// AMX vecfp x8

            add x7, x7, x9
            add x8, x8, x9
        
            .word 0x201004 				// AMX ldx x4

            add x4, x4, 128
            .word 0x201267 				// AMX vecfp x7

            .word 0x201268 				// AMX vecfp x8

            add x7, x7, x9
            add x8, x8, x9
        
            .word 0x201004 				// AMX ldx x4

            add x4, x4, 128
            .word 0x201267 				// AMX vecfp x7

            .word 0x201268 				// AMX vecfp x8

            add x7, x7, x9
            add x8, x8, x9
        
            .word 0x201004 				// AMX ldx x4

            add x4, x4, 128
            .word 0x201267 				// AMX vecfp x7

            .word 0x201268 				// AMX vecfp x8

            add x7, x7, x9
            add x8, x8, x9
        
            .word 0x201004 				// AMX ldx x4

            add x4, x4, 128
            .word 0x201267 				// AMX vecfp x7

            .word 0x201268 				// AMX vecfp x8

            add x7, x7, x9
            add x8, x8, x9
        
            .word 0x201004 				// AMX ldx x4

            add x4, x4, 128
            .word 0x201267 				// AMX vecfp x7

            .word 0x201268 				// AMX vecfp x8

            add x7, x7, x9
            add x8, x8, x9
        
            .word 0x201004 				// AMX ldx x4

            add x4, x4, 128
            .word 0x201267 				// AMX vecfp x7

            .word 0x201268 				// AMX vecfp x8

            add x7, x7, x9
            add x8, x8, x9
        
            .word 0x201004 				// AMX ldx x4

            add x4, x4, 128
            .word 0x201267 				// AMX vecfp x7

            .word 0x201268 				// AMX vecfp x8

            add x7, x7, x9
            add x8, x8, x9
        
            .word 0x201004 				// AMX ldx x4

            add x4, x4, 128
            .word 0x201267 				// AMX vecfp x7

            .word 0x201268 				// AMX vecfp x8

            add x7, x7, x9
            add x8, x8, x9
        
            .word 0x201004 				// AMX ldx x4

            add x4, x4, 128
            .word 0x201267 				// AMX vecfp x7

            .word 0x201268 				// AMX vecfp x8

            add x7, x7, x9
            add x8, x8, x9
        
            .word 0x201004 				// AMX ldx x4

            add x4, x4, 128
            .word 0x201267 				// AMX vecfp x7

            .word 0x201268 				// AMX vecfp x8

            add x7, x7, x9
            add x8, x8, x9
        
            .word 0x201004 				// AMX ldx x4

            add x4, x4, 128
            .word 0x201267 				// AMX vecfp x7

            .word 0x201268 				// AMX vecfp x8

            add x7, x7, x9
            add x8, x8, x9
        
            .word 0x201004 				// AMX ldx x4

            add x4, x4, 128
            .word 0x201267 				// AMX vecfp x7

            .word 0x201268 				// AMX vecfp x8

            add x7, x7, x9
            add x8, x8, x9
        
            .word 0x201004 				// AMX ldx x4

            add x4, x4, 128
            .word 0x201267 				// AMX vecfp x7

            .word 0x201268 				// AMX vecfp x8

            add x7, x7, x9
            add x8, x8, x9
        
            .word 0x201004 				// AMX ldx x4

            add x4, x4, 128
            .word 0x201267 				// AMX vecfp x7

            .word 0x201268 				// AMX vecfp x8

            add x7, x7, x9
            add x8, x8, x9
        
            .word 0x201004 				// AMX ldx x4

            add x4, x4, 128
            .word 0x201267 				// AMX vecfp x7

            .word 0x201268 				// AMX vecfp x8

            add x7, x7, x9
            add x8, x8, x9
        
            .word 0x201004 				// AMX ldx x4

            add x4, x4, 128
            .word 0x201267 				// AMX vecfp x7

            .word 0x201268 				// AMX vecfp x8

            add x7, x7, x9
            add x8, x8, x9
        
            .word 0x201004 				// AMX ldx x4

            add x4, x4, 128
            .word 0x201267 				// AMX vecfp x7

            .word 0x201268 				// AMX vecfp x8

            add x7, x7, x9
            add x8, x8, x9
        
            .word 0x201004 				// AMX ldx x4

            add x4, x4, 128
            .word 0x201267 				// AMX vecfp x7

            .word 0x201268 				// AMX vecfp x8

            add x7, x7, x9
            add x8, x8, x9
        
            .word 0x201004 				// AMX ldx x4

            add x4, x4, 128
            .word 0x201267 				// AMX vecfp x7

            .word 0x201268 				// AMX vecfp x8

            add x7, x7, x9
            add x8, x8, x9
        
            .word 0x201004 				// AMX ldx x4

            add x4, x4, 128
            .word 0x201267 				// AMX vecfp x7

            .word 0x201268 				// AMX vecfp x8

            add x7, x7, x9
            add x8, x8, x9
        
            .word 0x201004 				// AMX ldx x4

            add x4, x4, 128
            .word 0x201267 				// AMX vecfp x7

            .word 0x201268 				// AMX vecfp x8

            add x7, x7, x9
            add x8, x8, x9
        
            .word 0x201004 				// AMX ldx x4

            add x4, x4, 128
            .word 0x201267 				// AMX vecfp x7

            .word 0x201268 				// AMX vecfp x8

            add x7, x7, x9
            add x8, x8, x9
        
            .word 0x201004 				// AMX ldx x4

            add x4, x4, 128
            .word 0x201267 				// AMX vecfp x7

            .word 0x201268 				// AMX vecfp x8

            add x7, x7, x9
            add x8, x8, x9
        
            .word 0x201004 				// AMX ldx x4

            add x4, x4, 128
            .word 0x201267 				// AMX vecfp x7

            .word 0x201268 				// AMX vecfp x8

            add x7, x7, x9
            add x8, x8, x9
        
            .word 0x201004 				// AMX ldx x4

            add x4, x4, 128
            .word 0x201267 				// AMX vecfp x7

            .word 0x201268 				// AMX vecfp x8

            add x7, x7, x9
            add x8, x8, x9
        
            .word 0x201004 				// AMX ldx x4

            add x4, x4, 128
            .word 0x201267 				// AMX vecfp x7

            .word 0x201268 				// AMX vecfp x8

            add x7, x7, x9
            add x8, x8, x9
        
            .word 0x201004 				// AMX ldx x4

            add x4, x4, 128
            .word 0x201267 				// AMX vecfp x7

            .word 0x201268 				// AMX vecfp x8

            add x7, x7, x9
            add x8, x8, x9
        
        add x2, x2, #64
        sub x3, x3, #32
        cmp x3, #32
    bge .packed_packed_loop_32

    cmp x3, #0
    beq .non_linear_loop

    .packed_packed_loop_1:
        ldr w7, [x2], #2
        str w7, [x1]
        .word 0x201004 				// AMX ldx x4

        .word 0x201021 				// AMX ldy x1

        .word 0x201265 				// AMX vecfp x5

        .word 0x201266 				// AMX vecfp x6

        add x4, x4, 128
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


    b .non_linear_loop

.per_col_sub:

    // performs a unary neg on Z
    eor x2, x2, x2                      // X[0] = Z[0]
    // extr[hxyz] is suport confusing

    mov x4, -9223372036854775808          // vector mode
    orr x4, x4, 268435456
    orr x4, x4, 134217728      // Z=-X

    .word 0x201102 				// AMX extrx x2

    .word 0x201204 				// AMX fms16 x4

    add x2, x2, 1048576    // next Z row
    add x4, x4, 1048576    // next Z row
    .word 0x201102 				// AMX extrx x2
              // extr[hxyz] is confusing
    .word 0x201204 				// AMX fms16 x4


    // continue

.per_col_add:
    ldr         x2, [x0, #8]

    // broadcast value to x0
    ld1         { v0.h }[0], [x2]
    dup         v0.8h, v0.h[0]
    st1         { v0.8h }, [x1], #16
    st1         { v0.8h }, [x1], #16
    st1         { v0.8h }, [x1], #16
    st1         { v0.8h }, [x1], #16
    sub         x1, x1, #64

    .word 0x201001 				// AMX ldx x1
 // load into x0 by default

    mov x2, 268435456      // z += y
    .word 0x2011e2 				// AMX fma16 x2


    orr x2, x2, 1048576  // target is now z1
    .word 0x2011e2 				// AMX fma16 x2


    b .non_linear_loop

.per_col_sub_flipped:
    ldr         x2, [x0, #8]

    // broadcast value to x0
    ld1         { v0.h }[0], [x2]
    dup         v0.8h, v0.h[0]
    st1         { v0.8h }, [x1], #16
    st1         { v0.8h }, [x1], #16
    st1         { v0.8h }, [x1], #16
    st1         { v0.8h }, [x1], #16
    sub x1, x1, #64

    .word 0x201001 				// AMX ldx x1
 // load into x0 by default

    mov x2, 268435456      // z += y
    .word 0x201202 				// AMX fms16 x2


    orr x2, x2, 1048576  // target is now z1
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


    mov x2, -9223372036854775808      // vector mode
    orr x2, x2, 536870912  // z -= y

    // top left
    .word 0x201202 				// AMX fms16 x2


    // bottom left
    orr x2, x2, 1048576  // Z row = 1
    orr x2, x2, 64   // Y offset
    .word 0x201202 				// AMX fms16 x2


    b .non_linear_loop

.per_row_sub:
    // performs a unary neg on Z
    eor x2, x2, x2                      // X[0] = Z[0]

    mov x4, -9223372036854775808          // vector mode
    orr x4, x4, 268435456
    orr x4, x4, 134217728      // Z=-X

    .word 0x201102 				// AMX extrx x2

    .word 0x201204 				// AMX fms16 x4

    add x2, x2, 1048576    // next Z row
    add x4, x4, 1048576    // next Z row
    .word 0x201102 				// AMX extrx x2

    .word 0x201204 				// AMX fms16 x4


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


    mov x2, -9223372036854775808      // vector mode
    orr x2, x2, 536870912  // z += y

    // top left
    .word 0x2011e2 				// AMX fma16 x2


    // bottom left
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

    ld1         { v0.8h, v1.8h, v2.8h, v3.8h }, [x5], #64
    st1         { v0.8h, v1.8h, v2.8h, v3.8h }, [x1], #64
    ld1         { v0.8h, v1.8h, v2.8h, v3.8h }, [x5]
    st1         { v0.8h, v1.8h, v2.8h, v3.8h }, [x1]
    sub x1, x1, #64

    orr x5, x1, 4611686018427387904  // load a pair
    .word 0x201005 				// AMX ldx x5


    lsl x2, x2, 47                  // max(x,z) (or min)
    orr x2, x2, 17592186044416  // f32
    .word 0x201262 				// AMX vecfp x2


    orr x2, x2, 65536  // x1
    orr x2, x2, 1048576  // z1
    .word 0x201262 				// AMX vecfp x2


    b .non_linear_loop

.per_col_min:
    mov x2, 5
    b .per_col_min_max
.per_col_max:
    mov x2, 7
.per_col_min_max:
    ldr         x4, [x0, #8]

    // broadcast value to x0
    ld1         { v0.h }[0], [x4]
    dup         v0.8h, v0.h[0]
    st1         { v0.8h }, [x1], #16
    st1         { v0.8h }, [x1], #16
    st1         { v0.8h }, [x1], #16
    st1         { v0.8h }, [x1], #16
    sub         x1, x1, #64

    .word 0x201001 				// AMX ldx x1


    lsl x2, x2, 47                  // max(x,z) (or min)
    orr x2, x2, 8796093022208  // f32

    .word 0x201262 				// AMX vecfp x2

    orr x2, x2, 1048576  // z offset
    .word 0x201262 				// AMX vecfp x2


    b .non_linear_loop

.per_col_mul:
    ldr         x4, [x0, #8]

    // broadcast value to y0
    ld1         { v0.h }[0], [x4]
    dup         v0.8h, v0.h[0]
    st1         { v0.8h }, [x1], #16
    st1         { v0.8h }, [x1], #16
    st1         { v0.8h }, [x1], #16
    st1         { v0.8h }, [x1], #16
    sub         x1, x1, #64

    .word 0x201021 				// AMX ldy x1


    eor x2, x2, x2                      // X[0] = Z[0]
    .word 0x201102 				// AMX extrx x2

    mov x4, -9223372036854775808          // vector mode
    orr x4, x4, 134217728      // Z=X*Y
    .word 0x2011e4 				// AMX fma16 x4

    orr x2, x2, 1048576      // Z1
    .word 0x201102 				// AMX extrx x2

    orr x4, x4, 1048576      // Z1
    .word 0x2011e4 				// AMX fma16 x4


    b .non_linear_loop

.per_row_mul:
    ldr         x2, [x0, #8]

    ld1         { v0.8h, v1.8h, v2.8h, v3.8h }, [x2], #64
    st1         { v0.8h, v1.8h, v2.8h, v3.8h }, [x1], #64
    ld1         { v0.8h, v1.8h, v2.8h, v3.8h }, [x2]
    st1         { v0.8h, v1.8h, v2.8h, v3.8h }, [x1]
    sub x1, x1, #64

    orr x2, x1, 4611686018427387904      // pair
    .word 0x201022 				// AMX ldy x2


    eor x2, x2, x2                      // X[0] = Z[0]
    .word 0x201102 				// AMX extrx x2

    mov x4, -9223372036854775808          // vector mode
    orr x4, x4, 134217728      // Z=X*Y
    .word 0x2011e4 				// AMX fma16 x4

    orr x2, x2, 1048576      // Z1
    .word 0x201102 				// AMX extrx x2

    orr x4, x4, 1048576      // Z1
    orr x4, x4, 64       // Y1
    .word 0x2011e4 				// AMX fma16 x4


    b .non_linear_loop

.scalar_sub:
    // performs a unary neg on Z, then go to scalar_add
    eor x2, x2, x2                      // X[0] = Z[0]

    mov x4, -9223372036854775808          // vector mode
    orr x4, x4, 268435456
    orr x4, x4, 134217728      // Z=-X
    .word 0x201102 				// AMX extrx x2

    .word 0x201204 				// AMX fms16 x4

    add x2, x2, 1048576    // next Z row
    add x4, x4, 1048576    // next Z row
    .word 0x201102 				// AMX extrx x2

    .word 0x201204 				// AMX fms16 x4


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

    add x2, x2, 1048576    // next Z row
    .word 0x2011e2 				// AMX fma16 x2

    b .non_linear_loop

.scalar_sub_flipped:
    ldr         w5, [x0, #8]
    fmov        s0, w5
    dup         v0.8h, v0.h[0]
    dup         v1.8h, v0.h[0]
    dup         v2.8h, v0.h[0]
    dup         v3.8h, v0.h[0]

    st1         { v0.8h, v1.8h, v2.8h, v3.8h }, [x1]
    .word 0x201001 				// AMX ldx x1
    // load 16 values

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


    eor x2, x2, x2                      // X[0] = Z[0]

    mov x4, -9223372036854775808          // vector mode
    orr x4, x4, 134217728      // Z=X*Y

    .word 0x201102 				// AMX extrx x2

    .word 0x2011e4 				// AMX fma16 x4

    add x2, x2, 1048576    // next Z row
    add x4, x4, 1048576    // next Z row
    .word 0x201102 				// AMX extrx x2

    .word 0x2011e4 				// AMX fma16 x4


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
    orr x2, x2, 8796093022208 // f16

    .word 0x201262 				// AMX vecfp x2

    add x2, x2, 1048576 // next Z
    .word 0x201262 				// AMX vecfp x2


    b .non_linear_loop

.add_unicast:
    ldp         x5, x6, [x0, #8]            // c base ptr, rsc
    ldp         x7, x8, [x0, #24]           // csc, item_size

    
       
           ld1 { v0.h }[0], [x5], x6
       
           ld1 { v0.h }[1], [x5], x6
       
           ld1 { v0.h }[2], [x5], x6
       
           ld1 { v0.h }[3], [x5], x6
       
           ld1 { v0.h }[4], [x5], x6
       
           ld1 { v0.h }[5], [x5], x6
       
           ld1 { v0.h }[6], [x5], x6
       
           ld1 { v0.h }[7], [x5], x6
       
    
       
           ld1 { v1.h }[0], [x5], x6
       
           ld1 { v1.h }[1], [x5], x6
       
           ld1 { v1.h }[2], [x5], x6
       
           ld1 { v1.h }[3], [x5], x6
       
           ld1 { v1.h }[4], [x5], x6
       
           ld1 { v1.h }[5], [x5], x6
       
           ld1 { v1.h }[6], [x5], x6
       
           ld1 { v1.h }[7], [x5], x6
       
    
       
           ld1 { v2.h }[0], [x5], x6
       
           ld1 { v2.h }[1], [x5], x6
       
           ld1 { v2.h }[2], [x5], x6
       
           ld1 { v2.h }[3], [x5], x6
       
           ld1 { v2.h }[4], [x5], x6
       
           ld1 { v2.h }[5], [x5], x6
       
           ld1 { v2.h }[6], [x5], x6
       
           ld1 { v2.h }[7], [x5], x6
       
    
       
           ld1 { v3.h }[0], [x5], x6
       
           ld1 { v3.h }[1], [x5], x6
       
           ld1 { v3.h }[2], [x5], x6
       
           ld1 { v3.h }[3], [x5], x6
       
           ld1 { v3.h }[4], [x5], x6
       
           ld1 { v3.h }[5], [x5], x6
       
           ld1 { v3.h }[6], [x5], x6
       
           ld1 { v3.h }[7], [x5], x6
       
    
       
           ld1 { v4.h }[0], [x5], x6
       
           ld1 { v4.h }[1], [x5], x6
       
           ld1 { v4.h }[2], [x5], x6
       
           ld1 { v4.h }[3], [x5], x6
       
           ld1 { v4.h }[4], [x5], x6
       
           ld1 { v4.h }[5], [x5], x6
       
           ld1 { v4.h }[6], [x5], x6
       
           ld1 { v4.h }[7], [x5], x6
       
    
       
           ld1 { v5.h }[0], [x5], x6
       
           ld1 { v5.h }[1], [x5], x6
       
           ld1 { v5.h }[2], [x5], x6
       
           ld1 { v5.h }[3], [x5], x6
       
           ld1 { v5.h }[4], [x5], x6
       
           ld1 { v5.h }[5], [x5], x6
       
           ld1 { v5.h }[6], [x5], x6
       
           ld1 { v5.h }[7], [x5], x6
       
    
       
           ld1 { v6.h }[0], [x5], x6
       
           ld1 { v6.h }[1], [x5], x6
       
           ld1 { v6.h }[2], [x5], x6
       
           ld1 { v6.h }[3], [x5], x6
       
           ld1 { v6.h }[4], [x5], x6
       
           ld1 { v6.h }[5], [x5], x6
       
           ld1 { v6.h }[6], [x5], x6
       
           ld1 { v6.h }[7], [x5], x6
       
    
       
           ld1 { v7.h }[0], [x5], x6
       
           ld1 { v7.h }[1], [x5], x6
       
           ld1 { v7.h }[2], [x5], x6
       
           ld1 { v7.h }[3], [x5], x6
       
           ld1 { v7.h }[4], [x5], x6
       
           ld1 { v7.h }[5], [x5], x6
       
           ld1 { v7.h }[6], [x5], x6
       
           ld1 { v7.h }[7], [x5], x6
       
    
    mov x8, x1
    st1 { v0.8h, v1.8h, v2.8h, v3.8h }, [x8], #64
    st1 { v4.8h, v5.8h, v6.8h, v7.8h }, [x8], #64

    orr x8, x1, 4611686018427387904          // pair
    .word 0x201028 				// AMX ldy x8


    eor x2, x2, x2
    orr x2, x2, -9223372036854775808  // vector mode
    orr x2, x2, 536870912  // perform Z0+=Y0
    .word 0x2011e2 				// AMX fma16 x2

    orr x2, x2, 1048576  // Z1
    orr x2, x2, 64                  // offset Y
    .word 0x2011e2 				// AMX fma16 x2

    
    b .non_linear_loop

.add_row_col_products:
    ldp         x5, x6, [x0, #8]            // a base ptr, b base ptr

    ld1         { v0.h }[0], [x6]
    st1         { v0.h }[0], [x1]
    .word 0x201021 				// AMX ldy x1


    ld1         { v0.4s, v1.4s, v2.4s, v3.4s }, [x5], #64
    st1         { v0.4s, v1.4s, v2.4s, v3.4s }, [x1], #64
    ld1         { v0.4s, v1.4s, v2.4s, v3.4s }, [x5]
    st1         { v0.4s, v1.4s, v2.4s, v3.4s }, [x1]
    sub x1, x1, #64

    orr x2, x1, 4611686018427387904  // load a pair
    .word 0x201002 				// AMX ldx x2


    mov x2, 8796093022208      // f16
    orr x2, x2, 274877906944  // Broadcast Y
    .word 0x201262 				// AMX vecfp x2


    orr x2, x2, 1048576  // Z row = 1
    orr x2, x2, 65536  // X offset
    .word 0x201262 				// AMX vecfp x2


    b .non_linear_loop

.store:
    ldp         x5, x6, [x0, #8]            // c base ptr, rsc
    ldp         x7, x8, [x0, #24]           // csc, item_size

    ands        x8, x5, 0x7f
    bne         .store_generic
    cmp         x6, 4
    bne         .store_generic
    cmp         x7, 4
    bne         .store_generic
 
    orr x5, x5, 4611686018427387904          // pair
    .word 0x2010a5 				// AMX stz x5

    b .non_linear_loop

 .store_generic:

    orr x8, x1, 4611686018427387904          // pair
    .word 0x2010a8 				// AMX stz x8


    mov x8, x1
    ld1 { v0.8h, v1.8h, v2.8h, v3.8h }, [x8], #64
    ld1 { v4.8h, v5.8h, v6.8h, v7.8h }, [x8], #64
    
       
           st1 { v0.h }[0], [x5], x6
       
           st1 { v0.h }[1], [x5], x6
       
           st1 { v0.h }[2], [x5], x6
       
           st1 { v0.h }[3], [x5], x6
       
           st1 { v0.h }[4], [x5], x6
       
           st1 { v0.h }[5], [x5], x6
       
           st1 { v0.h }[6], [x5], x6
       
           st1 { v0.h }[7], [x5], x6
       
    
       
           st1 { v1.h }[0], [x5], x6
       
           st1 { v1.h }[1], [x5], x6
       
           st1 { v1.h }[2], [x5], x6
       
           st1 { v1.h }[3], [x5], x6
       
           st1 { v1.h }[4], [x5], x6
       
           st1 { v1.h }[5], [x5], x6
       
           st1 { v1.h }[6], [x5], x6
       
           st1 { v1.h }[7], [x5], x6
       
    
       
           st1 { v2.h }[0], [x5], x6
       
           st1 { v2.h }[1], [x5], x6
       
           st1 { v2.h }[2], [x5], x6
       
           st1 { v2.h }[3], [x5], x6
       
           st1 { v2.h }[4], [x5], x6
       
           st1 { v2.h }[5], [x5], x6
       
           st1 { v2.h }[6], [x5], x6
       
           st1 { v2.h }[7], [x5], x6
       
    
       
           st1 { v3.h }[0], [x5], x6
       
           st1 { v3.h }[1], [x5], x6
       
           st1 { v3.h }[2], [x5], x6
       
           st1 { v3.h }[3], [x5], x6
       
           st1 { v3.h }[4], [x5], x6
       
           st1 { v3.h }[5], [x5], x6
       
           st1 { v3.h }[6], [x5], x6
       
           st1 { v3.h }[7], [x5], x6
       
    
       
           st1 { v4.h }[0], [x5], x6
       
           st1 { v4.h }[1], [x5], x6
       
           st1 { v4.h }[2], [x5], x6
       
           st1 { v4.h }[3], [x5], x6
       
           st1 { v4.h }[4], [x5], x6
       
           st1 { v4.h }[5], [x5], x6
       
           st1 { v4.h }[6], [x5], x6
       
           st1 { v4.h }[7], [x5], x6
       
    
       
           st1 { v5.h }[0], [x5], x6
       
           st1 { v5.h }[1], [x5], x6
       
           st1 { v5.h }[2], [x5], x6
       
           st1 { v5.h }[3], [x5], x6
       
           st1 { v5.h }[4], [x5], x6
       
           st1 { v5.h }[5], [x5], x6
       
           st1 { v5.h }[6], [x5], x6
       
           st1 { v5.h }[7], [x5], x6
       
    
       
           st1 { v6.h }[0], [x5], x6
       
           st1 { v6.h }[1], [x5], x6
       
           st1 { v6.h }[2], [x5], x6
       
           st1 { v6.h }[3], [x5], x6
       
           st1 { v6.h }[4], [x5], x6
       
           st1 { v6.h }[5], [x5], x6
       
           st1 { v6.h }[6], [x5], x6
       
           st1 { v6.h }[7], [x5], x6
       
    
       
           st1 { v7.h }[0], [x5], x6
       
           st1 { v7.h }[1], [x5], x6
       
           st1 { v7.h }[2], [x5], x6
       
           st1 { v7.h }[3], [x5], x6
       
           st1 { v7.h }[4], [x5], x6
       
           st1 { v7.h }[5], [x5], x6
       
           st1 { v7.h }[6], [x5], x6
       
           st1 { v7.h }[7], [x5], x6
       
    
    
    b .non_linear_loop

.load_tile:
    ldr  x2, [x0, #16]                      // row major ptr
    orr  x2, x2, 4611686018427387904           // load pairs
    .word 0x201082 				// AMX ldz x2

    b .non_linear_loop

.return:
nop
nop
nop
.word 0x201221

ret