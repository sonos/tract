#include "unistd.h"
#include "stdio.h"

void arm_vfpv2_mm_s4x4(size_t k, float *a, float *b, float *c, size_t rsc, size_t csc) {
    float zero = 0.0;
    float *pzero = &zero;

    __asm__ volatile (
    " vmrs    r0,FPSCR                 \n\t"
    " bic     r0,r0,#0x00370000        \n\t"
    " vmsr    FPSCR,r0                 \n\t"

    " ldr r0,%[pzero]                  \n\t"
    " ldr r1,%[k]                      \n\t"
    " ldr r2,%[a]                      \n\t"
    " ldr r3,%[b]                      \n\t"
    " ldr r4,%[c]                      \n\t"
    " ldr r8,%[csc]                    \n\t"
    " ldr r9,%[rsc]                    \n\t"

    " flds            s16, [ r0 ]                 \n\t"
    " vmov.f32        s17, s16                    \n\t"
    " vmov.f32        s18, s16                    \n\t"
    " vmov.f32        s19, s16                    \n\t"
    " vmov.f32        s20, s16                    \n\t"
    " vmov.f32        s21, s16                    \n\t"
    " vmov.f32        s22, s16                    \n\t"
    " vmov.f32        s23, s16                    \n\t"
    " vmov.f32        s24, s16                    \n\t"
    " vmov.f32        s25, s16                    \n\t"
    " vmov.f32        s26, s16                    \n\t"
    " vmov.f32        s27, s16                    \n\t"
    " vmov.f32        s28, s16                    \n\t"
    " vmov.f32        s29, s16                    \n\t"
    " vmov.f32        s30, s16                    \n\t"
    " vmov.f32        s31, s16                    \n\t"

    " cmp r1, #0                                  \n\t"
    " beq .STORE                                  \n\t"

    ".LOOP:                                       \n\t"

    " vldr            s0, [r2]                    \n\t"
    " vldr            s1, [r2, #4]                \n\t"
    " vldr            s2, [r2, #8]                \n\t"
    " vldr            s3, [r2, #12]               \n\t"

    " vldr            s4, [r3]                    \n\t"
    " vldr            s5, [r3, #4]                \n\t"
    " vldr            s6, [r3, #8]                \n\t"
    " vldr            s7, [r3, #12]               \n\t"

    " vmla.f32        s16, s0, s4                 \n\t"
    " vmla.f32        s17, s1, s4                 \n\t"
    " vmla.f32        s18, s2, s4                 \n\t"
    " vmla.f32        s19, s3, s4                 \n\t"

    " vmla.f32        s20, s0, s5                 \n\t"
    " vmla.f32        s21, s1, s5                 \n\t"
    " vmla.f32        s22, s2, s5                 \n\t"
    " vmla.f32        s23, s3, s5                 \n\t"

    " vmla.f32        s24, s0, s6                 \n\t"
    " vmla.f32        s25, s1, s6                 \n\t"
    " vmla.f32        s26, s2, s6                 \n\t"
    " vmla.f32        s27, s3, s6                 \n\t"

    " vmla.f32        s28, s0, s7                 \n\t"
    " vmla.f32        s29, s1, s7                 \n\t"
    " vmla.f32        s30, s2, s7                 \n\t"
    " vmla.f32        s31, s3, s7                 \n\t"

    " add r2, r2, #16                             \n\t"
    " add r3, r3, #16                             \n\t"
    " subs r1, r1, #1                             \n\t"
    " bne .LOOP                                   \n\t"

    ".STORE:                                      \n\t"

    " lsl r8, r8, #2                              \n\t" // r8 *= sizeof(float) // csc
    " lsl r9, r9, #2                              \n\t" // r9 *= sizeof(float) // rsc

    " add r5, r4, r8                              \n\t"
    " add r6, r5, r8                              \n\t"
    " add r7, r6, r8                              \n\t" // r4,r5,r6,r7 are now addr for cols of C

    " fsts    s16, [ r4 ]                         \n\t"
    " add r4 , r4, r9                             \n\t"
    " fsts    s17, [ r4 ]                         \n\t"
    " add r4 , r4, r9                             \n\t"
    " fsts    s18, [ r4 ]                         \n\t"
    " add r4 , r4, r9                             \n\t"
    " fsts    s19, [ r4 ]                         \n\t"

    " fsts    s20, [ r5 ]                         \n\t"
    " add r5 , r5, r9                             \n\t"
    " fsts    s21, [ r5 ]                         \n\t"
    " add r5 , r5, r9                             \n\t"
    " fsts    s22, [ r5 ]                         \n\t"
    " add r5 , r5, r9                             \n\t"
    " fsts    s23, [ r5 ]                         \n\t"

    " fsts    s24, [ r6 ]                         \n\t"
    " add r6 , r6, r9                             \n\t"
    " fsts    s25, [ r6 ]                         \n\t"
    " add r6 , r6, r9                             \n\t"
    " fsts    s26, [ r6 ]                         \n\t"
    " add r6 , r6, r9                             \n\t"
    " fsts    s27, [ r6 ]                         \n\t"

    " fsts    s28, [ r7 ]                         \n\t"
    " add r7 , r7, r9                             \n\t"
    " fsts    s29, [ r7 ]                         \n\t"
    " add r7 , r7, r9                             \n\t"
    " fsts    s30, [ r7 ]                         \n\t"
    " add r7 , r7, r9                             \n\t"
    " fsts    s31, [ r7 ]                         \n\t"

    : // Outputs
    : // Inputs
    [pzero]  "m" (pzero),
    [k]      "m" (k),
    [a]      "m" (a),
    [b]      "m" (b),
    [c]      "m" (c),
    [csc]    "m" (csc),
    [rsc]    "m" (rsc)
    : // Clobber
      "memory",
      "r0", "r1", "r2", "r3",
      "r4", "r5", "r6", "r7",
      "r8", "r9",

      "s0",   "s1",  "s2",  "s3",
      "s4",   "s5",  "s6",  "s7",
      "s8",   "s9", "s10", "s11",
      "s12", "s13", "s14", "s15",
      "s16", "s17", "s18", "s19",
      "s20", "s21", "s22", "s23",
      "s24", "s25", "s26", "s27",
      "s28", "s29", "s30", "s31"
);
}
