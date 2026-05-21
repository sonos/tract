// SVE int8 -> int32 GEMV kernel for tract's MMM framework (the qmmv_i32 slot,
// dispatched when N == 1: matrix x int8 column vector).
//
// Tile MR=64 x NR=1, i32 accumulator. The hot AddMatMul is the vector-length-
// agnostic widening update vectorized over M: per K-step it loads MR signed
// bytes of the A-panel column, sign-extends to i32 (svld1sb_s32), and folds a
// single svmla_n_s32 with the (sign-extended) scalar B[k]. The MR rows are
// walked in svcntw() chunks with whilelt predication, so the SAME code is
// correct and full-width at any SVE vector length (128..2048-bit).
//
// Same rationale as the 8x8 kernel: widening MLA (not SDOT) consumes tract's
// native K-major i8i8 packing directly. int8 inputs arrive via that packing
// (AddMatMul packing == 1); the default i32i32 packing (packing == 0) is handled
// scalar for the auto-test surface.
//
// ABI: identical 40-byte FusedKerSpec<i32> walk. At NR=1, per_col / scalar fuse
// ops degenerate to a single broadcast value and per_row is element-wise over
// the MR outputs. Quantization ops q_scale / q_shr / q_shl are ported bit-exact
// from linalg/src/generic/rounding.rs. Returns 0 on success, 1 on an
// unsupported fused op / packing.

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

enum { RP_NATIVE = 0, RP_ZERO, RP_AWAY, RP_MINUSINF, RP_PLUSINF, RP_EVEN, RP_ODD };

typedef struct {
    uint64_t disc;
    uint64_t f0, f1, f2, f3;
} spec_t;

// AddMatMul, i8 x i8 -> i32 (packing 1): ab[m] += sum_k pa[k*MR+m]*pb[k].
// VLA widening update over MR.
static void add_mat_mul_i8(int32_t ab[MR], const int8_t *pa, const int8_t *pb, long k) {
    for (long m0 = 0; m0 < MR; m0 += svcntw()) {
        svbool_t pg = svwhilelt_b32((uint64_t)m0, (uint64_t)MR);
        svint32_t acc = svld1_s32(pg, &ab[m0]);
        for (long kk = 0; kk < k; kk++) {
            svint32_t a = svld1sb_s32(pg, &pa[kk * MR + m0]); // load i8 col, sign-extend
            acc = svmla_n_s32_x(pg, acc, a, (int32_t)pb[kk]);
        }
        svst1_s32(pg, &ab[m0], acc);
    }
}

// AddMatMul, i32 x i32 -> i32 (packing 0, default): auto-test surface only.
static void add_mat_mul_i32(int32_t ab[MR], const int32_t *pa, const int32_t *pb, long k) {
    for (long kk = 0; kk < k; kk++) {
        int32_t b = pb[kk];
        const int32_t *acol = &pa[kk * MR];
        for (long m = 0; m < MR; m++) ab[m] += acol[m] * b;
    }
}

// ---- quantization helpers, ported bit-exact from generic/rounding.rs ----

static int32_t q_shr_i32(int32_t v, long shift, int rp) {
    int32_t half = (int32_t)1 << (shift - 1);
    int32_t a = v < 0 ? -v : v;
    int32_t nudge;
    switch (rp) {
        case RP_ZERO:     nudge = -1; break;
        case RP_MINUSINF: nudge = -(int32_t)(v >= 0); break;
        case RP_PLUSINF:  nudge = -(int32_t)(v <= 0); break;
        case RP_AWAY:     nudge = 0; break;
        case RP_EVEN:     nudge = ((a >> shift) & 0x1) - 1; break;
        case RP_ODD:      nudge = -((a >> shift) & 0x1); break;
        default:          nudge = 0; break;
    }
    int32_t sign = (v > 0) - (v < 0);
    return sign * ((a + half + nudge) >> shift);
}

static int32_t q_scale_i32(int32_t v, long shift_in, int policy, int32_t mult) {
    int64_t val = (int64_t)mult * (int64_t)v;
    long shift = shift_in + 31;
    if (shift > 0) {
        int64_t half = (int64_t)1 << (shift - 1);
        int64_t a = val < 0 ? -val : val;
        int64_t nudge;
        switch (policy) {
            case RP_ZERO:     nudge = -1; break;
            case RP_MINUSINF: nudge = -(int64_t)(val >= 0); break;
            case RP_PLUSINF:  nudge = -(int64_t)(val <= 0); break;
            case RP_AWAY:     nudge = 0; break;
            case RP_EVEN:     nudge = ((a >> shift) & 0x1) - 1; break;
            case RP_ODD:      nudge = -((a >> shift) & 0x1); break;
            default:          nudge = 0; break;
        }
        int64_t sign = (val > 0) - (val < 0);
        return (int32_t)(sign * ((a + half + nudge) >> shift));
    } else {
        return (int32_t)(val << (-shift));
    }
}

intptr_t sve_mmm_i32_64x1_kernel(const spec_t *ops) {
    int32_t ab[MR];
    memset(ab, 0, sizeof(ab));
    for (const spec_t *s = ops;; s++) {
        switch (s->disc) {
            case DONE:
                return 0;
            case CLEAR:
                memset(ab, 0, sizeof(ab));
                break;
            case ADD_MAT_MUL: {
                long k = (long)s->f0, packing = (long)s->f3;
                if (packing == 1)
                    add_mat_mul_i8(ab, (const int8_t *)s->f1, (const int8_t *)s->f2, k);
                else if (packing == 0)
                    add_mat_mul_i32(ab, (const int32_t *)s->f1, (const int32_t *)s->f2, k);
                else
                    return 1;
                break;
            }
            case STORE: {
                uint8_t *ptr = (uint8_t *)s->f0;
                long rstride = (long)s->f1, isz = (long)s->f3;
                for (long m = 0; m < MR; m++) {
                    uint8_t *p = ptr + m * rstride;
                    int32_t v = ab[m];
                    switch (isz) {
                        case 1: *(uint8_t *)p = (uint8_t)v; break;
                        case 2: *(uint16_t *)p = (uint16_t)v; break;
                        case 4: *(int32_t *)p = v; break;
                        case 8: { int64_t w = v; memcpy(p, &w, 8); break; }
                        default: memcpy(p, &v, isz < 4 ? (size_t)isz : 4); break;
                    }
                }
                break;
            }
            case LOAD_TILE: {
                const int32_t *src = (const int32_t *)s->f1; // row-major MR values
                for (long m = 0; m < MR; m++) ab[m] = src[m];
                break;
            }
            case ADD_UNICAST: {
                uint8_t *ptr = (uint8_t *)s->f0;
                long rstride = (long)s->f1, isz = (long)s->f3;
                for (long m = 0; m < MR; m++) {
                    const uint8_t *p = ptr + m * rstride;
                    if (isz == 1)
                        ab[m] += *(const int8_t *)p;
                    else
                        ab[m] += *(const int32_t *)p;
                }
                break;
            }
            case ADD_ROW_COL_PRODUCTS: {
                const int32_t *rows = (const int32_t *)s->f0;
                const int32_t *cols = (const int32_t *)s->f1;
                for (long m = 0; m < MR; m++) ab[m] += rows[m] * cols[0];
                break;
            }
            case Q_SCALE: {
                long shift = (long)s->f0; int policy = (int)s->f1; int32_t mult = (int32_t)s->f2;
                for (long m = 0; m < MR; m++) ab[m] = q_scale_i32(ab[m], shift, policy, mult);
                break;
            }
            case Q_SHR: {
                long shift = (long)s->f0; int policy = (int)s->f1;
                for (long m = 0; m < MR; m++) ab[m] = q_shr_i32(ab[m], shift, policy);
                break;
            }
            case Q_SHL: {
                long shift = (long)s->f0;
                for (long m = 0; m < MR; m++) ab[m] = ab[m] << shift;
                break;
            }
            // scalar fuse ops (single i32 in low 32 bits of f0)
            case SCALAR_MIN: { int32_t v=(int32_t)s->f0; for(long m=0;m<MR;m++) ab[m]=ab[m]<v?ab[m]:v; break; }
            case SCALAR_MAX: { int32_t v=(int32_t)s->f0; for(long m=0;m<MR;m++) ab[m]=ab[m]>v?ab[m]:v; break; }
            case SCALAR_ADD: { int32_t v=(int32_t)s->f0; for(long m=0;m<MR;m++) ab[m]+=v; break; }
            case SCALAR_MUL: { int32_t v=(int32_t)s->f0; for(long m=0;m<MR;m++) ab[m]*=v; break; }
            case SCALAR_SUB: { int32_t v=(int32_t)s->f0; for(long m=0;m<MR;m++) ab[m]=v-ab[m]; break; }
            case SCALAR_SUBF:{ int32_t v=(int32_t)s->f0; for(long m=0;m<MR;m++) ab[m]=ab[m]-v; break; }
            // per-row fuse ops (one i32 per m-row)
            case PER_ROW_MIN: { const int32_t*m_=(const int32_t*)s->f0; for(long m=0;m<MR;m++) ab[m]=ab[m]<m_[m]?ab[m]:m_[m]; break; }
            case PER_ROW_MAX: { const int32_t*m_=(const int32_t*)s->f0; for(long m=0;m<MR;m++) ab[m]=ab[m]>m_[m]?ab[m]:m_[m]; break; }
            case PER_ROW_ADD: { const int32_t*m_=(const int32_t*)s->f0; for(long m=0;m<MR;m++) ab[m]+=m_[m]; break; }
            case PER_ROW_MUL: { const int32_t*m_=(const int32_t*)s->f0; for(long m=0;m<MR;m++) ab[m]*=m_[m]; break; }
            case PER_ROW_SUB: { const int32_t*m_=(const int32_t*)s->f0; for(long m=0;m<MR;m++) ab[m]=m_[m]-ab[m]; break; }
            case PER_ROW_SUBF:{ const int32_t*m_=(const int32_t*)s->f0; for(long m=0;m<MR;m++) ab[m]=ab[m]-m_[m]; break; }
            // per-col fuse ops degenerate to a single broadcast value at NR=1
            case PER_COL_MIN: { int32_t v=*(const int32_t*)s->f0; for(long m=0;m<MR;m++) ab[m]=ab[m]<v?ab[m]:v; break; }
            case PER_COL_MAX: { int32_t v=*(const int32_t*)s->f0; for(long m=0;m<MR;m++) ab[m]=ab[m]>v?ab[m]:v; break; }
            case PER_COL_ADD: { int32_t v=*(const int32_t*)s->f0; for(long m=0;m<MR;m++) ab[m]+=v; break; }
            case PER_COL_MUL: { int32_t v=*(const int32_t*)s->f0; for(long m=0;m<MR;m++) ab[m]*=v; break; }
            case PER_COL_SUB: { int32_t v=*(const int32_t*)s->f0; for(long m=0;m<MR;m++) ab[m]=v-ab[m]; break; }
            case PER_COL_SUBF:{ int32_t v=*(const int32_t*)s->f0; for(long m=0;m<MR;m++) ab[m]=ab[m]-v; break; }
            default:
                return 1;
        }
    }
}
