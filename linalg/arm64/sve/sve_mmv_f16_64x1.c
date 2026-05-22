// SVE f16 GEMV kernel for tract's MMM framework (the mmv_f16 slot, dispatched
// when N == 1: matrix x f16 column vector).
//
// Tile MR=64 x NR=1 with native f16 accumulation, the f16 sibling of
// sve_mmv_f32_64x1.c. The hot AddMatMul is vectorized over M: per K-step it
// loads MR f16 of the A-panel column and folds a single svmla_n_f16 (native f16
// fused multiply-add) with the scalar B[k]. The MR rows are walked in svcnth()
// chunks with whilelt predication, so one binary is correct and full-width at
// any SVE VL 128..2048-bit.
//
// Gated on FEAT_SVE2 AND FEAT_FP16 (Rust side). Built with +fp16. At NR=1 the
// per_col/scalar fuse ops degenerate to a broadcast and per_row is element-wise
// over the MR outputs. LeakyRelu and the i32 quantization ops are excluded by
// CAN_FUSE. Returns 0 on success, 1 otherwise.

#include <arm_sve.h>
#include <stdint.h>
#include <string.h>

#define MR 64
#define NR 1

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

// AddMatMul: ab[m] += sum_k pa[k*MR+m] * pb[k]. VLA over MR (f16 lanes).
static void add_mat_mul(__fp16 ab[MR], const __fp16 *pa, const __fp16 *pb, long k) {
    for (long m0 = 0; m0 < MR; m0 += svcnth()) {
        svbool_t pg = svwhilelt_b16((uint64_t)m0, (uint64_t)MR);
        svfloat16_t acc = svld1_f16(pg, &ab[m0]);
        for (long kk = 0; kk < k; kk++) {
            svfloat16_t a = svld1_f16(pg, &pa[kk * MR + m0]);
            acc = svmla_n_f16_x(pg, acc, a, pb[kk]);
        }
        svst1_f16(pg, &ab[m0], acc);
    }
}

intptr_t sve_mmv_f16_64x1_kernel(const spec_t *ops) {
    __fp16 ab[MR];
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
            case STORE: {
                uint8_t *ptr = (uint8_t *)s->f0;
                long rstride = (long)s->f1, isz = (long)s->f3;
                for (long m = 0; m < MR; m++) {
                    uint8_t *p = ptr + m * rstride;
                    if (isz == 2)
                        *(__fp16 *)p = ab[m];
                    else if (isz == 4)
                        *(float *)p = (float)ab[m];
                    else
                        memcpy(p, &ab[m], isz);
                }
                break;
            }
            case LOAD_TILE: {
                const __fp16 *src = (const __fp16 *)s->f1;
                for (long m = 0; m < MR; m++) ab[m] = src[m];
                break;
            }
            case ADD_UNICAST: {
                uint8_t *ptr = (uint8_t *)s->f0;
                long rstride = (long)s->f1, isz = (long)s->f3;
                for (long m = 0; m < MR; m++) {
                    const uint8_t *p = ptr + m * rstride;
                    if (isz == 2)
                        ab[m] += *(const __fp16 *)p;
                    else
                        ab[m] += (__fp16) * (const float *)p;
                }
                break;
            }
            case ADD_ROW_COL_PRODUCTS: {
                const __fp16 *rows = (const __fp16 *)s->f0;
                const __fp16 *cols = (const __fp16 *)s->f1;
                for (long m = 0; m < MR; m++) ab[m] += rows[m] * cols[0];
                break;
            }
            case SCALAR_MIN: { __fp16 v=f16_of(s->f0); for(long m=0;m<MR;m++) ab[m]=ab[m]<v?ab[m]:v; break; }
            case SCALAR_MAX: { __fp16 v=f16_of(s->f0); for(long m=0;m<MR;m++) ab[m]=ab[m]>v?ab[m]:v; break; }
            case SCALAR_ADD: { __fp16 v=f16_of(s->f0); for(long m=0;m<MR;m++) ab[m]+=v; break; }
            case SCALAR_MUL: { __fp16 v=f16_of(s->f0); for(long m=0;m<MR;m++) ab[m]*=v; break; }
            case SCALAR_SUB: { __fp16 v=f16_of(s->f0); for(long m=0;m<MR;m++) ab[m]=v-ab[m]; break; }
            case SCALAR_SUBF:{ __fp16 v=f16_of(s->f0); for(long m=0;m<MR;m++) ab[m]=ab[m]-v; break; }
            case PER_ROW_MIN: { const __fp16*m_=(const __fp16*)s->f0; for(long m=0;m<MR;m++) ab[m]=ab[m]<m_[m]?ab[m]:m_[m]; break; }
            case PER_ROW_MAX: { const __fp16*m_=(const __fp16*)s->f0; for(long m=0;m<MR;m++) ab[m]=ab[m]>m_[m]?ab[m]:m_[m]; break; }
            case PER_ROW_ADD: { const __fp16*m_=(const __fp16*)s->f0; for(long m=0;m<MR;m++) ab[m]+=m_[m]; break; }
            case PER_ROW_MUL: { const __fp16*m_=(const __fp16*)s->f0; for(long m=0;m<MR;m++) ab[m]*=m_[m]; break; }
            case PER_ROW_SUB: { const __fp16*m_=(const __fp16*)s->f0; for(long m=0;m<MR;m++) ab[m]=m_[m]-ab[m]; break; }
            case PER_ROW_SUBF:{ const __fp16*m_=(const __fp16*)s->f0; for(long m=0;m<MR;m++) ab[m]=ab[m]-m_[m]; break; }
            case PER_COL_MIN: { __fp16 v=*(const __fp16*)s->f0; for(long m=0;m<MR;m++) ab[m]=ab[m]<v?ab[m]:v; break; }
            case PER_COL_MAX: { __fp16 v=*(const __fp16*)s->f0; for(long m=0;m<MR;m++) ab[m]=ab[m]>v?ab[m]:v; break; }
            case PER_COL_ADD: { __fp16 v=*(const __fp16*)s->f0; for(long m=0;m<MR;m++) ab[m]+=v; break; }
            case PER_COL_MUL: { __fp16 v=*(const __fp16*)s->f0; for(long m=0;m<MR;m++) ab[m]*=v; break; }
            case PER_COL_SUB: { __fp16 v=*(const __fp16*)s->f0; for(long m=0;m<MR;m++) ab[m]=v-ab[m]; break; }
            case PER_COL_SUBF:{ __fp16 v=*(const __fp16*)s->f0; for(long m=0;m<MR;m++) ab[m]=ab[m]-v; break; }
            default:
                return 1;
        }
    }
}
