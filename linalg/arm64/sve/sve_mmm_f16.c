// SVE f16 GEMM kernel for tract's MMM framework (the mmm_f16 slot).
//
// Tile MR=8 x NR=8 with native f16 accumulation, the f16 sibling of
// sve_mmm_f32.c. The hot AddMatMul is the same vector-length-agnostic
// broadcast-A rank-1 update, but over f16 lanes: the NR columns are walked in
// svcnth() chunks with whilelt predication and folded with svmla_n_f16 (native
// f16 fused multiply-add, as the NEON arm64fp16 kernels do), so one binary is
// correct and full-width at any SVE VL 128..2048-bit.
//
// Gated on FEAT_SVE2 AND FEAT_FP16 (Rust side). Built with +fp16. Consumes
// tract's native f16 K-major packing. Fuse ops act on the MRxNR tile in memory
// (scalar; not the hot path); as with the f32 kernel, LeakyRelu and the i32
// quantization ops are excluded by CAN_FUSE. Returns 0 on success, 1 otherwise.

#include <arm_sve.h>
#include <stdint.h>
#include <string.h>

#define MR 8
#define NR 8

enum {
    DONE = 0, CLEAR, LOAD_TILE,
    SCALAR_MIN, SCALAR_MAX, SCALAR_ADD, SCALAR_MUL, SCALAR_SUB, SCALAR_SUBF,
    LEAKY_RELU,
    PER_ROW_MIN, PER_ROW_MAX, PER_ROW_ADD, PER_ROW_MUL, PER_ROW_SUB, PER_ROW_SUBF,
    PER_COL_MIN, PER_COL_MAX, PER_COL_ADD, PER_COL_MUL, PER_COL_SUB, PER_COL_SUBF,
    Q_SCALE, Q_SHR, Q_SHL,
    ADD_UNICAST, ADD_ROW_COL_PRODUCTS, STORE, ADD_MAT_MUL
};

typedef struct {
    uint64_t disc;
    uint64_t f0, f1, f2, f3;
} spec_t;

static inline __fp16 f16_of(uint64_t bits) {
    __fp16 f;
    uint16_t lo = (uint16_t)bits;
    memcpy(&f, &lo, 2);
    return f;
}

// AddMatMul: ab[m][n] += sum_k pa[k*MR+m] * pb[k*NR+n]. VLA over NR (f16 lanes).
static void add_mat_mul(__fp16 ab[MR][NR], const __fp16 *pa, const __fp16 *pb, long k) {
    for (long n0 = 0; n0 < NR; n0 += svcnth()) {
        svbool_t pg = svwhilelt_b16((uint64_t)n0, (uint64_t)NR);
        svfloat16_t a0 = svld1_f16(pg, &ab[0][n0]), a1 = svld1_f16(pg, &ab[1][n0]);
        svfloat16_t a2 = svld1_f16(pg, &ab[2][n0]), a3 = svld1_f16(pg, &ab[3][n0]);
        svfloat16_t a4 = svld1_f16(pg, &ab[4][n0]), a5 = svld1_f16(pg, &ab[5][n0]);
        svfloat16_t a6 = svld1_f16(pg, &ab[6][n0]), a7 = svld1_f16(pg, &ab[7][n0]);
        for (long kk = 0; kk < k; kk++) {
            svfloat16_t b = svld1_f16(pg, &pb[kk * NR + n0]);
            const __fp16 *arow = &pa[kk * MR];
            a0 = svmla_n_f16_x(pg, a0, b, arow[0]);
            a1 = svmla_n_f16_x(pg, a1, b, arow[1]);
            a2 = svmla_n_f16_x(pg, a2, b, arow[2]);
            a3 = svmla_n_f16_x(pg, a3, b, arow[3]);
            a4 = svmla_n_f16_x(pg, a4, b, arow[4]);
            a5 = svmla_n_f16_x(pg, a5, b, arow[5]);
            a6 = svmla_n_f16_x(pg, a6, b, arow[6]);
            a7 = svmla_n_f16_x(pg, a7, b, arow[7]);
        }
        svst1_f16(pg, &ab[0][n0], a0); svst1_f16(pg, &ab[1][n0], a1);
        svst1_f16(pg, &ab[2][n0], a2); svst1_f16(pg, &ab[3][n0], a3);
        svst1_f16(pg, &ab[4][n0], a4); svst1_f16(pg, &ab[5][n0], a5);
        svst1_f16(pg, &ab[6][n0], a6); svst1_f16(pg, &ab[7][n0], a7);
    }
}

// Store the MRxNR f16 tile with arbitrary row/col byte strides.
static void store_tile(__fp16 ab[MR][NR], const spec_t *s) {
    uint8_t *ptr = (uint8_t *)s->f0;
    long rstride = (long)s->f1, cstride = (long)s->f2, isz = (long)s->f3;
    for (long i = 0; i < MR; i++)
        for (long j = 0; j < NR; j++) {
            uint8_t *p = ptr + i * rstride + j * cstride;
            if (isz == 2)
                *(__fp16 *)p = ab[i][j];
            else if (isz == 4)
                *(float *)p = (float)ab[i][j];
            else
                memcpy(p, &ab[i][j], isz);
        }
}

intptr_t sve_mmm_f16_kernel(const spec_t *ops) {
    __fp16 ab[MR][NR];
    memset(ab, 0, sizeof(ab));
    for (const spec_t *s = ops;; s++) {
        switch (s->disc) {
            case DONE:
                return 0;
            case CLEAR:
                memset(ab, 0, sizeof(ab));
                break;
            case ADD_MAT_MUL: {
                long k = (long)s->f0;
                add_mat_mul(ab, (const __fp16 *)s->f1, (const __fp16 *)s->f2, k);
                break;
            }
            case STORE:
                store_tile(ab, s);
                break;
            case LOAD_TILE: {
                const __fp16 *src = (const __fp16 *)s->f1;
                for (long i = 0; i < MR; i++)
                    for (long j = 0; j < NR; j++) ab[i][j] = src[i * NR + j];
                break;
            }
            case ADD_UNICAST: {
                uint8_t *ptr = (uint8_t *)s->f0;
                long rstride = (long)s->f1, cstride = (long)s->f2, isz = (long)s->f3;
                for (long i = 0; i < MR; i++)
                    for (long j = 0; j < NR; j++) {
                        const uint8_t *p = ptr + i * rstride + j * cstride;
                        if (isz == 2)
                            ab[i][j] += *(const __fp16 *)p;
                        else
                            ab[i][j] += (__fp16) * (const float *)p;
                    }
                break;
            }
            case ADD_ROW_COL_PRODUCTS: {
                const __fp16 *rows = (const __fp16 *)s->f0;
                const __fp16 *cols = (const __fp16 *)s->f1;
                for (long i = 0; i < MR; i++)
                    for (long j = 0; j < NR; j++) ab[i][j] += rows[i] * cols[j];
                break;
            }
            case SCALAR_MIN: { __fp16 v=f16_of(s->f0); for(long i=0;i<MR;i++)for(long j=0;j<NR;j++) ab[i][j]=ab[i][j]<v?ab[i][j]:v; break; }
            case SCALAR_MAX: { __fp16 v=f16_of(s->f0); for(long i=0;i<MR;i++)for(long j=0;j<NR;j++) ab[i][j]=ab[i][j]>v?ab[i][j]:v; break; }
            case SCALAR_ADD: { __fp16 v=f16_of(s->f0); for(long i=0;i<MR;i++)for(long j=0;j<NR;j++) ab[i][j]+=v; break; }
            case SCALAR_MUL: { __fp16 v=f16_of(s->f0); for(long i=0;i<MR;i++)for(long j=0;j<NR;j++) ab[i][j]*=v; break; }
            case SCALAR_SUB: { __fp16 v=f16_of(s->f0); for(long i=0;i<MR;i++)for(long j=0;j<NR;j++) ab[i][j]=v-ab[i][j]; break; }
            case SCALAR_SUBF:{ __fp16 v=f16_of(s->f0); for(long i=0;i<MR;i++)for(long j=0;j<NR;j++) ab[i][j]=ab[i][j]-v; break; }
            case PER_ROW_MIN: { const __fp16*m=(const __fp16*)s->f0; for(long i=0;i<MR;i++)for(long j=0;j<NR;j++) ab[i][j]=ab[i][j]<m[i]?ab[i][j]:m[i]; break; }
            case PER_ROW_MAX: { const __fp16*m=(const __fp16*)s->f0; for(long i=0;i<MR;i++)for(long j=0;j<NR;j++) ab[i][j]=ab[i][j]>m[i]?ab[i][j]:m[i]; break; }
            case PER_ROW_ADD: { const __fp16*m=(const __fp16*)s->f0; for(long i=0;i<MR;i++)for(long j=0;j<NR;j++) ab[i][j]+=m[i]; break; }
            case PER_ROW_MUL: { const __fp16*m=(const __fp16*)s->f0; for(long i=0;i<MR;i++)for(long j=0;j<NR;j++) ab[i][j]*=m[i]; break; }
            case PER_ROW_SUB: { const __fp16*m=(const __fp16*)s->f0; for(long i=0;i<MR;i++)for(long j=0;j<NR;j++) ab[i][j]=m[i]-ab[i][j]; break; }
            case PER_ROW_SUBF:{ const __fp16*m=(const __fp16*)s->f0; for(long i=0;i<MR;i++)for(long j=0;j<NR;j++) ab[i][j]=ab[i][j]-m[i]; break; }
            case PER_COL_MIN: { const __fp16*m=(const __fp16*)s->f0; for(long i=0;i<MR;i++)for(long j=0;j<NR;j++) ab[i][j]=ab[i][j]<m[j]?ab[i][j]:m[j]; break; }
            case PER_COL_MAX: { const __fp16*m=(const __fp16*)s->f0; for(long i=0;i<MR;i++)for(long j=0;j<NR;j++) ab[i][j]=ab[i][j]>m[j]?ab[i][j]:m[j]; break; }
            case PER_COL_ADD: { const __fp16*m=(const __fp16*)s->f0; for(long i=0;i<MR;i++)for(long j=0;j<NR;j++) ab[i][j]+=m[j]; break; }
            case PER_COL_MUL: { const __fp16*m=(const __fp16*)s->f0; for(long i=0;i<MR;i++)for(long j=0;j<NR;j++) ab[i][j]*=m[j]; break; }
            case PER_COL_SUB: { const __fp16*m=(const __fp16*)s->f0; for(long i=0;i<MR;i++)for(long j=0;j<NR;j++) ab[i][j]=m[j]-ab[i][j]; break; }
            case PER_COL_SUBF:{ const __fp16*m=(const __fp16*)s->f0; for(long i=0;i<MR;i++)for(long j=0;j<NR;j++) ab[i][j]=ab[i][j]-m[j]; break; }
            default:
                return 1;
        }
    }
}
