// SVE f32 GEMV kernel for tract's MMM framework (the mmv_f32 slot, dispatched
// when N == 1: matrix x f32 column vector).
//
// Tile MR=64 x NR=1. The hot AddMatMul is the vector-length-agnostic update
// vectorized over M: per K-step it loads MR f32 of the A-panel column and folds
// a single svmla_n_f32 with the scalar B[k]. The MR rows are walked in svcntw()
// chunks with whilelt predication, so the SAME code is correct and full-width at
// any SVE vector length (128..2048-bit).
//
// Sibling of sve_mmm_f32.c (the 8x8 GEMM); shares the same FusedKerSpec<f32> ABI
// and fuse-op surface. At NR=1, per_col / scalar fuse ops degenerate to a single
// broadcast value and per_row is element-wise over the MR outputs. As with the
// f32 GEMM kernel, LeakyRelu and the i32 quantization ops are excluded by
// CAN_FUSE. Returns 0 on success, 1 on an unsupported fused op.

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

static inline float f32_of(uint64_t bits) {
    float f;
    uint32_t lo = (uint32_t)bits;
    memcpy(&f, &lo, 4);
    return f;
}

// AddMatMul: ab[m] += sum_k pa[k*MR+m] * pb[k]. VLA over MR.
static void add_mat_mul(float ab[MR], const float *pa, const float *pb, long k) {
    for (long m0 = 0; m0 < MR; m0 += svcntw()) {
        svbool_t pg = svwhilelt_b32((uint64_t)m0, (uint64_t)MR);
        svfloat32_t acc = svld1_f32(pg, &ab[m0]);
        for (long kk = 0; kk < k; kk++) {
            svfloat32_t a = svld1_f32(pg, &pa[kk * MR + m0]);
            acc = svmla_n_f32_x(pg, acc, a, pb[kk]);
        }
        svst1_f32(pg, &ab[m0], acc);
    }
}

intptr_t sve_mmv_f32_64x1_kernel(const spec_t *ops) {
    float ab[MR];
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
                add_mat_mul(ab, (const float *)s->f1, (const float *)s->f2, k);
                break;
            }
            case STORE: {
                uint8_t *ptr = (uint8_t *)s->f0;
                long rstride = (long)s->f1, isz = (long)s->f3;
                for (long m = 0; m < MR; m++) {
                    uint8_t *p = ptr + m * rstride;
                    if (isz == 4)
                        *(float *)p = ab[m];
                    else
                        memcpy(p, &ab[m], isz);
                }
                break;
            }
            case LOAD_TILE: {
                const float *src = (const float *)s->f1; // row-major MR values
                for (long m = 0; m < MR; m++) ab[m] = src[m];
                break;
            }
            case ADD_UNICAST: {
                uint8_t *ptr = (uint8_t *)s->f0;
                long rstride = (long)s->f1;
                for (long m = 0; m < MR; m++) ab[m] += *(const float *)(ptr + m * rstride);
                break;
            }
            case ADD_ROW_COL_PRODUCTS: {
                const float *rows = (const float *)s->f0;
                const float *cols = (const float *)s->f1;
                for (long m = 0; m < MR; m++) ab[m] += rows[m] * cols[0];
                break;
            }
            // scalar fuse ops (f32 bits in low 32 bits of f0)
            case SCALAR_MIN: { float v=f32_of(s->f0); for(long m=0;m<MR;m++) ab[m]=ab[m]<v?ab[m]:v; break; }
            case SCALAR_MAX: { float v=f32_of(s->f0); for(long m=0;m<MR;m++) ab[m]=ab[m]>v?ab[m]:v; break; }
            case SCALAR_ADD: { float v=f32_of(s->f0); for(long m=0;m<MR;m++) ab[m]+=v; break; }
            case SCALAR_MUL: { float v=f32_of(s->f0); for(long m=0;m<MR;m++) ab[m]*=v; break; }
            case SCALAR_SUB: { float v=f32_of(s->f0); for(long m=0;m<MR;m++) ab[m]=v-ab[m]; break; }
            case SCALAR_SUBF:{ float v=f32_of(s->f0); for(long m=0;m<MR;m++) ab[m]=ab[m]-v; break; }
            // per-row fuse ops (one f32 per m-row)
            case PER_ROW_MIN: { const float*m_=(const float*)s->f0; for(long m=0;m<MR;m++) ab[m]=ab[m]<m_[m]?ab[m]:m_[m]; break; }
            case PER_ROW_MAX: { const float*m_=(const float*)s->f0; for(long m=0;m<MR;m++) ab[m]=ab[m]>m_[m]?ab[m]:m_[m]; break; }
            case PER_ROW_ADD: { const float*m_=(const float*)s->f0; for(long m=0;m<MR;m++) ab[m]+=m_[m]; break; }
            case PER_ROW_MUL: { const float*m_=(const float*)s->f0; for(long m=0;m<MR;m++) ab[m]*=m_[m]; break; }
            case PER_ROW_SUB: { const float*m_=(const float*)s->f0; for(long m=0;m<MR;m++) ab[m]=m_[m]-ab[m]; break; }
            case PER_ROW_SUBF:{ const float*m_=(const float*)s->f0; for(long m=0;m<MR;m++) ab[m]=ab[m]-m_[m]; break; }
            // per-col fuse ops degenerate to a single broadcast value at NR=1
            case PER_COL_MIN: { float v=*(const float*)s->f0; for(long m=0;m<MR;m++) ab[m]=ab[m]<v?ab[m]:v; break; }
            case PER_COL_MAX: { float v=*(const float*)s->f0; for(long m=0;m<MR;m++) ab[m]=ab[m]>v?ab[m]:v; break; }
            case PER_COL_ADD: { float v=*(const float*)s->f0; for(long m=0;m<MR;m++) ab[m]+=v; break; }
            case PER_COL_MUL: { float v=*(const float*)s->f0; for(long m=0;m<MR;m++) ab[m]*=v; break; }
            case PER_COL_SUB: { float v=*(const float*)s->f0; for(long m=0;m<MR;m++) ab[m]=v-ab[m]; break; }
            case PER_COL_SUBF:{ float v=*(const float*)s->f0; for(long m=0;m<MR;m++) ab[m]=ab[m]-v; break; }
            default:
                // LeakyRelu / QScale / RoundingShiftRight / ShiftLeft excluded by CAN_FUSE.
                return 1;
        }
    }
}
