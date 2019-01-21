#include "unistd.h"
#include "stdio.h"

//
// q8[0]    q10[0]   q12[0]    q14[0]
// q8[1]    q10[1]   q12[1]    q14[1]
// q8[2]    q10[2]   q12[2]    q14[2]
// q8[3]    q10[3]   q12[3]    q14[3]
//
// q9[0]    q11[0]   q13[0]    q15[0]
// q9[1]    q11[1]   q13[1]    q15[1]
// q9[2]    q11[2]   q13[2]    q15[2]
// q9[3]    q11[3]   q13[3]    q15[3]

void armv7neon_mm_s8x4(size_t k, float *a, float *b, float *c, size_t rsc, size_t csc) {

    float zero = 0.0;
    float *pzero = &zero;

    __asm__ volatile (

    " ldr r0,%[pzero]                  \n\t"
    " ldr r1,%[k]                      \n\t"
    " ldr r2,%[a]                      \n\t"
    " ldr r3,%[b]                      \n\t"
    " ldr r4,%[c]                      \n\t"
    " ldr r8,%[csc]                    \n\t"
    " ldr r9,%[rsc]                    \n\t"

    " veor      q8, q8 ,q8         \n\t"
    " veor      q9, q9 ,q9         \n\t"
    " veor      q10, q10 ,q10         \n\t"
    " veor      q11, q11 ,q11         \n\t"
    " veor      q12, q12 ,q12         \n\t"
    " veor      q13, q13 ,q13         \n\t"
    " veor      q14, q14 ,q14         \n\t"
    " veor      q15, q15 ,q15         \n\t"

    " cmp r1, #0                                  \n\t"
    " beq .STORE                                  \n\t"

/*

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

    " vldmia          r3!, { s12-s15 }            \n\t"

    " vmla.f32        s24, s0, s10                 \n\t"
    " vmla.f32        s25, s1, s10                 \n\t"
    " vmla.f32        s26, s2, s10                 \n\t"
    " vmla.f32        s27, s3, s10                 \n\t"

    " vmla.f32        s28, s0, s11                 \n\t"
    " vmla.f32        s29, s1, s11                 \n\t"
    " vmla.f32        s30, s2, s11                 \n\t"
    " vmla.f32        s31, s3, s11                 \n\t"

    // 4
    " vmla.f32        s16, s4, s12                \n\t"
    " vmla.f32        s17, s5, s12                \n\t"
    " vmla.f32        s18, s6, s12                \n\t"
    " vmla.f32        s19, s7, s12                \n\t"

    " vmla.f32        s20, s4, s13                 \n\t"
    " vmla.f32        s21, s5, s13                 \n\t"
    " vmla.f32        s22, s6, s13                 \n\t"
    " vmla.f32        s23, s7, s13                 \n\t"

    " vmla.f32        s24, s4, s14                 \n\t"
    " vmla.f32        s25, s5, s14                 \n\t"
    " vmla.f32        s26, s6, s14                 \n\t"
    " vmla.f32        s27, s7, s14                 \n\t"

    " vmla.f32        s28, s4, s15                 \n\t"
    " vmla.f32        s29, s5, s15                 \n\t"
    " vmla.f32        s30, s6, s15                 \n\t"
    " vmla.f32        s31, s7, s15                 \n\t"

    " sub r1, r1, #4                              \n\t"
    " cmp r1, #4                                  \n\t"
    " bge .LOOP4                                  \n\t"

    " cmp r1, #0                                  \n\t"
    " beq .STORE                                  \n\t"

*/

    ".LOOP:                                       \n\t"

    " vldmia          r2!, { q0, q1 }            \n\t"
    " vldmia          r3!, { q4 }                \n\t"

    " vmla.f32        q8, q0, d8[0]              \n\t"
    " vmla.f32        q9, q1, d8[0]              \n\t"

    " vmla.f32        q10, q0, d8[1]              \n\t"
    " vmla.f32        q11, q1, d8[1]              \n\t"

    " vmla.f32        q12, q0, d9[0]              \n\t"
    " vmla.f32        q13, q1, d9[0]              \n\t"

    " vmla.f32        q14, q0, d9[1]              \n\t"
    " vmla.f32        q15, q1, d9[1]              \n\t"

    " subs r1, r1, #1                             \n\t"
    " bne .LOOP                                   \n\t"

    ".STORE:                                      \n\t"

    " lsl r8, r8, #2                              \n\t" // r8 *= sizeof(float) // csc
    " lsl r9, r9, #2                              \n\t" // r9 *= sizeof(float) // rsc

    " add r5, r4, r8                              \n\t"
    " add r6, r5, r8                              \n\t"
    " add r7, r6, r8                              \n\t" // r4,r5,r6,r7 are now addr for cols of C

    " vst1.f32    d16[0], [ r4 ]                       \n\t"
    " add r4 , r4, r9                             \n\t"
    " vst1.f32    d16[1], [ r4 ]                       \n\t"
    " add r4 , r4, r9                             \n\t"
    " vst1.f32    d17[0], [ r4 ]                       \n\t"
    " add r4 , r4, r9                             \n\t"
    " vst1.f32    d17[1], [ r4 ]                       \n\t"
    " add r4 , r4, r9                             \n\t"

    " vst1.f32   d18[0], [ r4 ]                       \n\t"
    " add r4 , r4, r9                             \n\t"
    " vst1.f32   d18[1], [ r4 ]                       \n\t"
    " add r4 , r4, r9                             \n\t"
    " vst1.f32   d19[0], [ r4 ]                       \n\t"
    " add r4 , r4, r9                             \n\t"
    " vst1.f32   d19[1], [ r4 ]                       \n\t"

    " vst1.f32   d20[0], [ r5 ]                       \n\t"
    " add r5 , r5, r9                             \n\t"
    " vst1.f32   d20[1], [ r5 ]                       \n\t"
    " add r5 , r5, r9                             \n\t"
    " vst1.f32   d21[0], [ r5 ]                       \n\t"
    " add r5 , r5, r9                             \n\t"
    " vst1.f32   d21[1], [ r5 ]                       \n\t"
    " add r5 , r5, r9                             \n\t"

    " vst1.f32   d22[0], [ r5 ]                       \n\t"
    " add r5 , r5, r9                             \n\t"
    " vst1.f32   d22[1], [ r5 ]                       \n\t"
    " add r5 , r5, r9                             \n\t"
    " vst1.f32   d23[0], [ r5 ]                       \n\t"
    " add r5 , r5, r9                             \n\t"
    " vst1.f32   d23[1], [ r5 ]                       \n\t"

    " vst1.f32   d24[0], [ r6 ]                       \n\t"
    " add r6 , r6, r9                             \n\t"
    " vst1.f32   d24[1], [ r6 ]                       \n\t"
    " add r6 , r6, r9                             \n\t"
    " vst1.f32   d25[0], [ r6 ]                       \n\t"
    " add r6 , r6, r9                             \n\t"
    " vst1.f32   d25[1], [ r6 ]                       \n\t"
    " add r6 , r6, r9                             \n\t"

    " vst1.f32   d26[0], [ r6 ]                       \n\t"
    " add r6 , r6, r9                             \n\t"
    " vst1.f32   d26[1], [ r6 ]                       \n\t"
    " add r6 , r6, r9                             \n\t"
    " vst1.f32   d27[0], [ r6 ]                       \n\t"
    " add r6 , r6, r9                             \n\t"
    " vst1.f32   d27[1], [ r6 ]                       \n\t"

    " vst1.f32   d28[0], [ r7 ]                       \n\t"
    " add r7 , r7, r9                             \n\t"
    " vst1.f32   d28[1], [ r7 ]                       \n\t"
    " add r7 , r7, r9                             \n\t"
    " vst1.f32   d29[0], [ r7 ]                       \n\t"
    " add r7 , r7, r9                             \n\t"
    " vst1.f32   d29[1], [ r7 ]                       \n\t"
    " add r7 , r7, r9                             \n\t"

    " vst1.f32   d30[0], [ r7 ]                       \n\t"
    " add r7 , r7, r9                             \n\t"
    " vst1.f32   d30[1], [ r7 ]                       \n\t"
    " add r7 , r7, r9                             \n\t"
    " vst1.f32   d31[0], [ r7 ]                       \n\t"
    " add r7 , r7, r9                             \n\t"
    " vst1.f32   d31[1], [ r7 ]                       \n\t"


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

      "q0",   "q1",  "q2",  "q3",
      "q4",   "q5",  "q6",  "q7",
      "q8",   "q9", "q10", "q11",
      "q12", "q13", "q14", "q15"
);
}
