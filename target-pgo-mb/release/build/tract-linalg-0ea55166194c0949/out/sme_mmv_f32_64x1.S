// vim: ft=arm
//
// SME2 f32 64x1 GEMV kernel.
//
// Accumulator layout: ZA tile slot rows 0..3 (one vgx4 group starting at
// w8=0). The 64-element output column maps to these 4 slots × 16 f32 each.
//
// Inner K-step: load 64 f32 of A's column into {z0.s-z3.s} via one SME2
// multi-vec LD1W, broadcast B[k] into z4 with LD1RW, issue ONE multi-vec
// vgx4 FMLA-into-ZA. Measured peak on M4: ~125 GFLOPS (ZA-write port).
// Plain SME single-vec predicated FMLA into Z regs caps at ~31 GFLOPS so
// is not used. Detection therefore gates on FEAT_SME2, not just FEAT_SME.
//
// Calling convention (extern "C", AAPCS64):
//   x0 = const *FusedKerSpec<f32>, advanced 40 B per dispatcher iteration.
//   x1 = stack-resident 256 B scratch buffer for the strided-store path.
//   w8 = 0 throughout (vgx-group base index, set in prologue).
//
// Streaming-mode rules: PSTATE.SM=1 from prologue smstart to epilogue
// smstop. v8..v15 saved/restored across the streaming region.

.arch armv9-a+sme2
.text
.align 4

.global _sme_mmv_f32_64x1_0_23_4_pre
_sme_mmv_f32_64x1_0_23_4_pre:

    stp     q8,  q9,  [sp, #-128]!
    stp     q10, q11, [sp, #32]
    stp     q12, q13, [sp, #64]
    stp     q14, q15, [sp, #96]

    // 256 B = 64 f32 spill buffer for the strided-store / AddUnicast paths.
    sub     sp, sp, #256
    mov     x1, sp

    smstart
    ptrue   p0.b
    ptrue   pn8.b
    mov     w8, #0

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


// -------- supported fuse ops -----------------------------------------------

.add_mat_mul:
    ldr     x2, [x0, #24]       // b ptr
    ldp     x3, x4, [x0, #8]    // k, a ptr
    cmp     x3, #0
    b.eq    .non_linear_loop

.Lmmv_loop:
    ld1w    {z0.s-z3.s}, pn8/z, [x4]
    add     x4, x4, #256
    ld1rw   {z4.s}, p0/z, [x2]
    add     x2, x2, #4
    fmla    za.s[w8, 0, vgx4], {z0.s-z3.s}, z4.s[0]
    subs    x3, x3, #1
    b.ne    .Lmmv_loop
    b       .non_linear_loop

.clear:
    zero    {za}
    b       .non_linear_loop

.store:
    // FusedKerSpec::Store(OutputStoreKer { ptr, row_byte_stride,
    //                                      col_byte_stride, item_size })
    //   [x0, #8]  = ptr           [x0, #16] = row_byte_stride
    //   [x0, #24] = col_byte_stride  [x0, #32] = item_size
    // x8 must NOT be touched (it's the vgx-base index, set to 0 in prologue).
    ldp     x5, x6, [x0, #8]    // ptr, row_byte_stride
    ldp     x7, x9, [x0, #24]   // col_byte_stride, item_size

    // At NR=1 the output column is one element per row; the fast path
    // triggers when (row_byte_stride==4 AND item_size==4) i.e. the 64
    // outputs are contiguous in memory.
    cmp     x6, #4
    b.ne    .Lstore_generic
    cmp     x9, #4
    b.ne    .Lstore_generic

    mov     {z0.s-z3.s}, za.s[w8, 0, vgx4]
    st1w    {z0.s}, p0, [x5]
    st1w    {z1.s}, p0, [x5, #1, mul vl]
    st1w    {z2.s}, p0, [x5, #2, mul vl]
    st1w    {z3.s}, p0, [x5, #3, mul vl]
    b       .non_linear_loop

.Lstore_generic:
    // Spill ZA → 256 B scratch buffer x1, then per-element strided write.
    mov     {z0.s-z3.s}, za.s[w8, 0, vgx4]
    st1w    {z0.s}, p0, [x1]
    st1w    {z1.s}, p0, [x1, #1, mul vl]
    st1w    {z2.s}, p0, [x1, #2, mul vl]
    st1w    {z3.s}, p0, [x1, #3, mul vl]

    mov     x3, #0
    mov     x9, x1
.Lstore_scatter:
    ldr     w10, [x9], #4
    str     w10, [x5]
    add     x5, x5, x6
    add     x3, x3, #1
    cmp     x3, #64
    b.lt    .Lstore_scatter
    b       .non_linear_loop

// -------- LoadTile: ZA := tile from row-major source -----------------------
//
// FusedKerSpec::LoadTile(col_major_ptr, row_major_ptr) — same as Phase 1's
// 32x32 LoadTile. NR=1 collapses both pointers to the same 64-element vec;
// we use the row-major form at [x0, #16].

.load_tile:
    ldr     x2, [x0, #16]
    ld1w    {z0.s-z3.s}, pn8/z, [x2]
    mov     za.s[w8, 0, vgx4], {z0.s-z3.s}
    b       .non_linear_loop

// -------- AddRowColProducts: ZA += rows ⊗ cols (rank-1 K=1) ---------------
//
// NR=1: cols is a single f32, rows is a 64-element vector. Effectively one
// K-step of add_mat_mul with K=1.

.add_row_col_products:
    ldp     x2, x3, [x0, #8]           // rows ptr, cols ptr
    ld1w    {z0.s-z3.s}, pn8/z, [x2]
    ld1rw   {z4.s}, p0/z, [x3]
    fmla    za.s[w8, 0, vgx4], {z0.s-z3.s}, z4.s[0]
    b       .non_linear_loop

// -------- AddUnicast: ZA += C from strided buffer --------------------------
//
// NR=1 implies a 64-element column vec layout. Fast path = contiguous f32
// rows (row_stride == 4); generic path gathers strided.

.add_unicast:
    ldp     x5, x6, [x0, #8]    // ptr, row_byte_stride
    ldp     x7, x9, [x0, #24]   // col_byte_stride, item_size

    cmp     x6, #4
    b.ne    .Laddu_generic
    cmp     x9, #4
    b.ne    .Laddu_generic

    // Fast path: contiguous load via 4-vec LD1W.
    ld1w    {z16.s-z19.s}, pn8/z, [x5]
    mov     {z0.s-z3.s}, za.s[w8, 0, vgx4]
    fadd    z0.s, p0/m, z0.s, z16.s
    fadd    z1.s, p0/m, z1.s, z17.s
    fadd    z2.s, p0/m, z2.s, z18.s
    fadd    z3.s, p0/m, z3.s, z19.s
    mov     za.s[w8, 0, vgx4], {z0.s-z3.s}
    b       .non_linear_loop

.Laddu_generic:
    // Per-element strided gather into scratch, then contiguous accumulate.
    mov     x3, #0
    mov     x9, x1
.Laddu_gather:
    ldr     w10, [x5]
    str     w10, [x9], #4
    add     x5, x5, x6
    add     x3, x3, #1
    cmp     x3, #64
    b.lt    .Laddu_gather
    ld1w    {z16.s-z19.s}, pn8/z, [x1]
    mov     {z0.s-z3.s}, za.s[w8, 0, vgx4]
    fadd    z0.s, p0/m, z0.s, z16.s
    fadd    z1.s, p0/m, z1.s, z17.s
    fadd    z2.s, p0/m, z2.s, z18.s
    fadd    z3.s, p0/m, z3.s, z19.s
    mov     za.s[w8, 0, vgx4], {z0.s-z3.s}
    b       .non_linear_loop

// -------- scalar / per_col ops (degenerate at NR=1; per_col == scalar) -----
//
// Per Phase 1's mapping:
//   ScalarSub   → result = scalar - z   (fsubr)
//   ScalarSubF  → result = z - scalar   (fsub)
// Same convention applies to PerCol*.




.scalar_add:
    ldr     w2, [x0, #8]
    dup     z4.s, w2
    mov     {z0.s-z3.s}, za.s[w8, 0, vgx4]
    fadd  z0.s, p0/m, z0.s, z4.s
    fadd  z1.s, p0/m, z1.s, z4.s
    fadd  z2.s, p0/m, z2.s, z4.s
    fadd  z3.s, p0/m, z3.s, z4.s
    mov     za.s[w8, 0, vgx4], {z0.s-z3.s}
    b       .non_linear_loop


.scalar_mul:
    ldr     w2, [x0, #8]
    dup     z4.s, w2
    mov     {z0.s-z3.s}, za.s[w8, 0, vgx4]
    fmul  z0.s, p0/m, z0.s, z4.s
    fmul  z1.s, p0/m, z1.s, z4.s
    fmul  z2.s, p0/m, z2.s, z4.s
    fmul  z3.s, p0/m, z3.s, z4.s
    mov     za.s[w8, 0, vgx4], {z0.s-z3.s}
    b       .non_linear_loop


.scalar_sub:
    ldr     w2, [x0, #8]
    dup     z4.s, w2
    mov     {z0.s-z3.s}, za.s[w8, 0, vgx4]
    fsubr  z0.s, p0/m, z0.s, z4.s
    fsubr  z1.s, p0/m, z1.s, z4.s
    fsubr  z2.s, p0/m, z2.s, z4.s
    fsubr  z3.s, p0/m, z3.s, z4.s
    mov     za.s[w8, 0, vgx4], {z0.s-z3.s}
    b       .non_linear_loop


.scalar_sub_flipped:
    ldr     w2, [x0, #8]
    dup     z4.s, w2
    mov     {z0.s-z3.s}, za.s[w8, 0, vgx4]
    fsub  z0.s, p0/m, z0.s, z4.s
    fsub  z1.s, p0/m, z1.s, z4.s
    fsub  z2.s, p0/m, z2.s, z4.s
    fsub  z3.s, p0/m, z3.s, z4.s
    mov     za.s[w8, 0, vgx4], {z0.s-z3.s}
    b       .non_linear_loop


.scalar_min:
    ldr     w2, [x0, #8]
    dup     z4.s, w2
    mov     {z0.s-z3.s}, za.s[w8, 0, vgx4]
    fmin  z0.s, p0/m, z0.s, z4.s
    fmin  z1.s, p0/m, z1.s, z4.s
    fmin  z2.s, p0/m, z2.s, z4.s
    fmin  z3.s, p0/m, z3.s, z4.s
    mov     za.s[w8, 0, vgx4], {z0.s-z3.s}
    b       .non_linear_loop


.scalar_max:
    ldr     w2, [x0, #8]
    dup     z4.s, w2
    mov     {z0.s-z3.s}, za.s[w8, 0, vgx4]
    fmax  z0.s, p0/m, z0.s, z4.s
    fmax  z1.s, p0/m, z1.s, z4.s
    fmax  z2.s, p0/m, z2.s, z4.s
    fmax  z3.s, p0/m, z3.s, z4.s
    mov     za.s[w8, 0, vgx4], {z0.s-z3.s}
    b       .non_linear_loop


// per_col at NR=1 takes a *pointer* to 1 f32 at [x0, #8]; dereference
// and broadcast. Result is functionally identical to scalar but the
// load path differs.




.per_col_add:
    ldr     x2, [x0, #8]
    ld1rw   {z4.s}, p0/z, [x2]
    mov     {z0.s-z3.s}, za.s[w8, 0, vgx4]
    fadd  z0.s, p0/m, z0.s, z4.s
    fadd  z1.s, p0/m, z1.s, z4.s
    fadd  z2.s, p0/m, z2.s, z4.s
    fadd  z3.s, p0/m, z3.s, z4.s
    mov     za.s[w8, 0, vgx4], {z0.s-z3.s}
    b       .non_linear_loop


.per_col_mul:
    ldr     x2, [x0, #8]
    ld1rw   {z4.s}, p0/z, [x2]
    mov     {z0.s-z3.s}, za.s[w8, 0, vgx4]
    fmul  z0.s, p0/m, z0.s, z4.s
    fmul  z1.s, p0/m, z1.s, z4.s
    fmul  z2.s, p0/m, z2.s, z4.s
    fmul  z3.s, p0/m, z3.s, z4.s
    mov     za.s[w8, 0, vgx4], {z0.s-z3.s}
    b       .non_linear_loop


.per_col_sub:
    ldr     x2, [x0, #8]
    ld1rw   {z4.s}, p0/z, [x2]
    mov     {z0.s-z3.s}, za.s[w8, 0, vgx4]
    fsubr  z0.s, p0/m, z0.s, z4.s
    fsubr  z1.s, p0/m, z1.s, z4.s
    fsubr  z2.s, p0/m, z2.s, z4.s
    fsubr  z3.s, p0/m, z3.s, z4.s
    mov     za.s[w8, 0, vgx4], {z0.s-z3.s}
    b       .non_linear_loop


.per_col_sub_flipped:
    ldr     x2, [x0, #8]
    ld1rw   {z4.s}, p0/z, [x2]
    mov     {z0.s-z3.s}, za.s[w8, 0, vgx4]
    fsub  z0.s, p0/m, z0.s, z4.s
    fsub  z1.s, p0/m, z1.s, z4.s
    fsub  z2.s, p0/m, z2.s, z4.s
    fsub  z3.s, p0/m, z3.s, z4.s
    mov     za.s[w8, 0, vgx4], {z0.s-z3.s}
    b       .non_linear_loop


.per_col_min:
    ldr     x2, [x0, #8]
    ld1rw   {z4.s}, p0/z, [x2]
    mov     {z0.s-z3.s}, za.s[w8, 0, vgx4]
    fmin  z0.s, p0/m, z0.s, z4.s
    fmin  z1.s, p0/m, z1.s, z4.s
    fmin  z2.s, p0/m, z2.s, z4.s
    fmin  z3.s, p0/m, z3.s, z4.s
    mov     za.s[w8, 0, vgx4], {z0.s-z3.s}
    b       .non_linear_loop


.per_col_max:
    ldr     x2, [x0, #8]
    ld1rw   {z4.s}, p0/z, [x2]
    mov     {z0.s-z3.s}, za.s[w8, 0, vgx4]
    fmax  z0.s, p0/m, z0.s, z4.s
    fmax  z1.s, p0/m, z1.s, z4.s
    fmax  z2.s, p0/m, z2.s, z4.s
    fmax  z3.s, p0/m, z3.s, z4.s
    mov     za.s[w8, 0, vgx4], {z0.s-z3.s}
    b       .non_linear_loop


// -------- per_row ops: 64-element bias, lane-wise op against accumulator --




.per_row_add:
    ldr     x2, [x0, #8]
    ld1w    {z16.s-z19.s}, pn8/z, [x2]
    mov     {z0.s-z3.s}, za.s[w8, 0, vgx4]
    fadd  z0.s, p0/m, z0.s, z16.s
    fadd  z1.s, p0/m, z1.s, z17.s
    fadd  z2.s, p0/m, z2.s, z18.s
    fadd  z3.s, p0/m, z3.s, z19.s
    mov     za.s[w8, 0, vgx4], {z0.s-z3.s}
    b       .non_linear_loop


.per_row_mul:
    ldr     x2, [x0, #8]
    ld1w    {z16.s-z19.s}, pn8/z, [x2]
    mov     {z0.s-z3.s}, za.s[w8, 0, vgx4]
    fmul  z0.s, p0/m, z0.s, z16.s
    fmul  z1.s, p0/m, z1.s, z17.s
    fmul  z2.s, p0/m, z2.s, z18.s
    fmul  z3.s, p0/m, z3.s, z19.s
    mov     za.s[w8, 0, vgx4], {z0.s-z3.s}
    b       .non_linear_loop


.per_row_sub:
    ldr     x2, [x0, #8]
    ld1w    {z16.s-z19.s}, pn8/z, [x2]
    mov     {z0.s-z3.s}, za.s[w8, 0, vgx4]
    fsubr  z0.s, p0/m, z0.s, z16.s
    fsubr  z1.s, p0/m, z1.s, z17.s
    fsubr  z2.s, p0/m, z2.s, z18.s
    fsubr  z3.s, p0/m, z3.s, z19.s
    mov     za.s[w8, 0, vgx4], {z0.s-z3.s}
    b       .non_linear_loop


.per_row_sub_flipped:
    ldr     x2, [x0, #8]
    ld1w    {z16.s-z19.s}, pn8/z, [x2]
    mov     {z0.s-z3.s}, za.s[w8, 0, vgx4]
    fsub  z0.s, p0/m, z0.s, z16.s
    fsub  z1.s, p0/m, z1.s, z17.s
    fsub  z2.s, p0/m, z2.s, z18.s
    fsub  z3.s, p0/m, z3.s, z19.s
    mov     za.s[w8, 0, vgx4], {z0.s-z3.s}
    b       .non_linear_loop


.per_row_min:
    ldr     x2, [x0, #8]
    ld1w    {z16.s-z19.s}, pn8/z, [x2]
    mov     {z0.s-z3.s}, za.s[w8, 0, vgx4]
    fmin  z0.s, p0/m, z0.s, z16.s
    fmin  z1.s, p0/m, z1.s, z17.s
    fmin  z2.s, p0/m, z2.s, z18.s
    fmin  z3.s, p0/m, z3.s, z19.s
    mov     za.s[w8, 0, vgx4], {z0.s-z3.s}
    b       .non_linear_loop


.per_row_max:
    ldr     x2, [x0, #8]
    ld1w    {z16.s-z19.s}, pn8/z, [x2]
    mov     {z0.s-z3.s}, za.s[w8, 0, vgx4]
    fmax  z0.s, p0/m, z0.s, z16.s
    fmax  z1.s, p0/m, z1.s, z17.s
    fmax  z2.s, p0/m, z2.s, z18.s
    fmax  z3.s, p0/m, z3.s, z19.s
    mov     za.s[w8, 0, vgx4], {z0.s-z3.s}
    b       .non_linear_loop


// -------- not yet implemented ----------------------------------------------

.leaky_relu:
.q_scale:
.q_shl:
.q_shr:
    b       .unsupported

// -------- epilogue ---------------------------------------------------------

.return:
    smstop
    add     sp, sp, #256
    ldp     q14, q15, [sp, #96]
    ldp     q12, q13, [sp, #64]
    ldp     q10, q11, [sp, #32]
    ldp     q8,  q9,  [sp], #128
    ret