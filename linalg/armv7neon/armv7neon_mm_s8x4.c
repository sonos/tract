#include "unistd.h"
#include "stdio.h"

void armv7neon_mm_s4x4(size_t k, float *a, float *b, float *c, size_t rsc, size_t csc) {
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

    " pld [r2]                        \n\t"
    " pld [r2, #8]                    \n\t"
    " pld [r3]                        \n\t"
    " pld [r3, #8]                    \n\t"

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

    " cmp r1, #4                                  \n\t"
    " blt .LOOP                                   \n\t"

    ".LOOP4:                                      \n\t"

    // 1
    " vldmia          r2!, { s0, s1 }             \n\t"
    " vldmia          r3!, { s8, s9 }             \n\t"

    " vmla.f32        s16, s0, s8                 \n\t"
    " vldmia          r2!, { s2, s3 }             \n\t"
    " vmla.f32        s17, s1, s8                 \n\t"
    " vldmia          r3!, { s10, s11 }             \n\t"
    " vmla.f32        s18, s2, s8                 \n\t"
    " vmla.f32        s19, s3, s8                 \n\t"

    " vmla.f32        s20, s0, s9                 \n\t"
    " vmla.f32        s21, s1, s9                 \n\t"
    " vmla.f32        s22, s2, s9                 \n\t"
    " vmla.f32        s23, s3, s9                 \n\t"

    " vldmia          r2!, { s4-s7 }              \n\t"
    " vmla.f32        s24, s0, s10                 \n\t"
    " vmla.f32        s25, s1, s10                 \n\t"
    " vmla.f32        s26, s2, s10                 \n\t"
    " vmla.f32        s27, s3, s10                 \n\t"

    " vldmia          r3!, { s12-s15 }            \n\t"
    " vmla.f32        s28, s0, s11                 \n\t"
    " vmla.f32        s29, s1, s11                 \n\t"
    " vmla.f32        s30, s2, s11                 \n\t"
    " vmla.f32        s31, s3, s11                 \n\t"

    // 2
    " vmla.f32        s16, s4, s12                 \n\t"
    " vmla.f32        s17, s5, s12                 \n\t"
    " vmla.f32        s18, s6, s12                 \n\t"
    " vmla.f32        s19, s7, s12                 \n\t"

    " vldmia          r2!, { s0-s3 }               \n\t"

    " vmla.f32        s20, s4, s13                 \n\t"
    " vmla.f32        s21, s5, s13                 \n\t"
    " vmla.f32        s22, s6, s13                 \n\t"
    " vmla.f32        s23, s7, s13                 \n\t"

    " vldmia          r3!, { s8-s11 }              \n\t"

    " vmla.f32        s24, s4, s14                 \n\t"
    " vmla.f32        s25, s5, s14                 \n\t"
    " vmla.f32        s26, s6, s14                 \n\t"
    " vmla.f32        s27, s7, s14                 \n\t"

    " vmla.f32        s28, s4, s15                 \n\t"
    " vmla.f32        s29, s5, s15                 \n\t"
    " vmla.f32        s30, s6, s15                 \n\t"
    " vmla.f32        s31, s7, s15                 \n\t"

    // 3
    " vmla.f32        s16, s0, s8                 \n\t"
    " vmla.f32        s17, s1, s8                 \n\t"
    " vmla.f32        s18, s2, s8                 \n\t"
    " vmla.f32        s19, s3, s8                 \n\t"

    " vldmia          r2!, { s4-s7 }              \n\t"

    " vmla.f32        s20, s0, s9                 \n\t"
    " vmla.f32        s21, s1, s9                 \n\t"
    " vmla.f32        s22, s2, s9                 \n\t"
    " vmla.f32        s23, s3, s9                 \n\t"

    " vldmia          r3!, { s8-s11 }             \n\t"

    " vmla.f32        s24, s0, s10                 \n\t"
    " vmla.f32        s25, s1, s10                 \n\t"
    " vmla.f32        s26, s2, s10                 \n\t"
    " vmla.f32        s27, s3, s10                 \n\t"

    " pld [r2]                        \n\t"

    " vmla.f32        s28, s0, s11                 \n\t"
    " vmla.f32        s29, s1, s11                 \n\t"
    " vmla.f32        s30, s2, s11                 \n\t"
    " vmla.f32        s31, s3, s11                 \n\t"

    " pld [r3]                        \n\t"

    // 4
    " vmla.f32        s16, s4, s8                 \n\t"
    " vmla.f32        s17, s5, s8                 \n\t"
    " vmla.f32        s18, s6, s8                 \n\t"
    " vmla.f32        s19, s7, s8                 \n\t"

    " vmla.f32        s20, s4, s9                 \n\t"
    " vmla.f32        s21, s5, s9                 \n\t"
    " vmla.f32        s22, s6, s9                 \n\t"
    " vmla.f32        s23, s7, s9                 \n\t"

    " vmla.f32        s24, s4, s10                 \n\t"
    " vmla.f32        s25, s5, s10                 \n\t"
    " vmla.f32        s26, s6, s10                 \n\t"
    " vmla.f32        s27, s7, s10                 \n\t"

    " vmla.f32        s28, s4, s11                 \n\t"
    " vmla.f32        s29, s5, s11                 \n\t"
    " vmla.f32        s30, s6, s11                 \n\t"
    " vmla.f32        s31, s7, s11                 \n\t"

    " sub r1, r1, #4                              \n\t"
    " cmp r1, #4                                  \n\t"
    " bge .LOOP4                                  \n\t"

    " cmp r1, #0                                  \n\t"
    " beq .STORE                                  \n\t"

    ".LOOP:                                       \n\t"

    " vldmia          r2!, { s0, s1 }             \n\t"
    " vldmia          r3!, { s8, s9 }             \n\t"

    " vmla.f32        s16, s0, s8                 \n\t"
    " vldmia          r2!, { s2, s3 }             \n\t"
    " vmla.f32        s17, s1, s8                 \n\t"
    " vldmia          r3!, { s10, s11 }             \n\t"
    " vmla.f32        s18, s2, s8                 \n\t"
    " vmla.f32        s19, s3, s8                 \n\t"

    " vmla.f32        s20, s0, s9                 \n\t"
    " vmla.f32        s21, s1, s9                 \n\t"
    " vmla.f32        s22, s2, s9                 \n\t"
    " vmla.f32        s23, s3, s9                 \n\t"

    " vmla.f32        s24, s0, s10                 \n\t"
    " vmla.f32        s25, s1, s10                 \n\t"
    " vmla.f32        s26, s2, s10                 \n\t"
    " vmla.f32        s27, s3, s10                 \n\t"

    " vmla.f32        s28, s0, s11                 \n\t"
    " vmla.f32        s29, s1, s11                 \n\t"
    " vmla.f32        s30, s2, s11                 \n\t"
    " vmla.f32        s31, s3, s11                 \n\t"

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
