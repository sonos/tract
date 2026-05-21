// SVE f32 GEMM kernel for tract's MMM framework.
//
// Tile MR=8 x NR=8. The hot AddMatMul uses the vector-length-agnostic
// broadcast-A rank-1 update: the NR columns are walked in svcntw() chunks with
// whilelt predication, so the SAME code runs correctly (and uses the full
// vector) at any SVE vector length — 128 to 2048-bit. The MR=8 accumulators are
// held in SVE registers across the K loop. Fuse ops operate on an MRxNR tile in
// memory (scalar C — they are not the hot path), mirroring the generic kernel.
//
// ABI: the kernel walks a *const FusedKerSpec<f32> array (40 bytes / entry,
// discriminant u64 at offset 0, fields at 8/16/24/32) until Done, exactly like
// the asm dispatcher. f32 GEMM consumes tract's native K-major packing
// (pa[k*MR+m], pb[k*NR+n]) so no custom packing format is required.
//
// Returns 0 on success, 1 if asked to do an unsupported fused op.

#include <arm_sve.h>
#include <stdint.h>
#include <string.h>

#define MR 8
#define NR 8

// FusedKerSpec discriminants (must match frame/mmm/fuse.rs enum order).
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
    uint64_t f0, f1, f2, f3; // fields at byte offsets 8, 16, 24, 32
} spec_t;

// AddMatMul: ab[m][n] += sum_k pa[k*MR+m] * pb[k*NR+n]. VLA over NR.
static void add_mat_mul(float ab[MR][NR], const float *pa, const float *pb, long k) {
    for (long n0 = 0; n0 < NR; n0 += svcntw()) {
        svbool_t pg = svwhilelt_b32((uint64_t)n0, (uint64_t)NR);
        svfloat32_t a0 = svld1_f32(pg, &ab[0][n0]), a1 = svld1_f32(pg, &ab[1][n0]);
        svfloat32_t a2 = svld1_f32(pg, &ab[2][n0]), a3 = svld1_f32(pg, &ab[3][n0]);
        svfloat32_t a4 = svld1_f32(pg, &ab[4][n0]), a5 = svld1_f32(pg, &ab[5][n0]);
        svfloat32_t a6 = svld1_f32(pg, &ab[6][n0]), a7 = svld1_f32(pg, &ab[7][n0]);
        for (long kk = 0; kk < k; kk++) {
            svfloat32_t b = svld1_f32(pg, &pb[kk * NR + n0]);
            const float *arow = &pa[kk * MR];
            a0 = svmla_n_f32_x(pg, a0, b, arow[0]);
            a1 = svmla_n_f32_x(pg, a1, b, arow[1]);
            a2 = svmla_n_f32_x(pg, a2, b, arow[2]);
            a3 = svmla_n_f32_x(pg, a3, b, arow[3]);
            a4 = svmla_n_f32_x(pg, a4, b, arow[4]);
            a5 = svmla_n_f32_x(pg, a5, b, arow[5]);
            a6 = svmla_n_f32_x(pg, a6, b, arow[6]);
            a7 = svmla_n_f32_x(pg, a7, b, arow[7]);
        }
        svst1_f32(pg, &ab[0][n0], a0); svst1_f32(pg, &ab[1][n0], a1);
        svst1_f32(pg, &ab[2][n0], a2); svst1_f32(pg, &ab[3][n0], a3);
        svst1_f32(pg, &ab[4][n0], a4); svst1_f32(pg, &ab[5][n0], a5);
        svst1_f32(pg, &ab[6][n0], a6); svst1_f32(pg, &ab[7][n0], a7);
    }
}

static inline float f32_of(uint64_t bits) {
    float f;
    uint32_t lo = (uint32_t)bits;
    memcpy(&f, &lo, 4);
    return f;
}

// Store the MRxNR tile to memory with arbitrary row/col byte strides.
static void store_tile(float ab[MR][NR], const spec_t *s) {
    uint8_t *ptr = (uint8_t *)s->f0;
    long rstride = (long)s->f1, cstride = (long)s->f2, isz = (long)s->f3;
    for (long i = 0; i < MR; i++)
        for (long j = 0; j < NR; j++) {
            uint8_t *p = ptr + i * rstride + j * cstride;
            if (isz == 4)
                *(float *)p = ab[i][j];
            else
                memcpy(p, &ab[i][j], isz);
        }
}

// Returns isize (64-bit) to match tract's kernel ABI — NOT int (would leave the
// upper 32 bits of x0 undefined).
intptr_t sve_mmm_f32_kernel(const spec_t *ops) {
    float ab[MR][NR];
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
                const float *pa = (const float *)s->f1;
                const float *pb = (const float *)s->f2;
                add_mat_mul(ab, pa, pb, k);
                break;
            }
            case STORE:
                store_tile(ab, s);
                break;
            case LOAD_TILE: {
                // LoadTile(col_major_ptr, row_major_ptr); use the row-major one.
                const float *src = (const float *)s->f1;
                for (long i = 0; i < MR; i++)
                    for (long j = 0; j < NR; j++) ab[i][j] = src[i * NR + j];
                break;
            }
            case ADD_UNICAST: {
                uint8_t *ptr = (uint8_t *)s->f0;
                long rstride = (long)s->f1, cstride = (long)s->f2;
                for (long i = 0; i < MR; i++)
                    for (long j = 0; j < NR; j++)
                        ab[i][j] += *(const float *)(ptr + i * rstride + j * cstride);
                break;
            }
            case ADD_ROW_COL_PRODUCTS: {
                const float *rows = (const float *)s->f0;
                const float *cols = (const float *)s->f1;
                for (long i = 0; i < MR; i++)
                    for (long j = 0; j < NR; j++) ab[i][j] += rows[i] * cols[j];
                break;
            }
            // ---- scalar fuse ops ----
            case SCALAR_MIN: { float v = f32_of(s->f0); for (long i=0;i<MR;i++) for(long j=0;j<NR;j++) ab[i][j]=ab[i][j]<v?ab[i][j]:v; break; }
            case SCALAR_MAX: { float v = f32_of(s->f0); for (long i=0;i<MR;i++) for(long j=0;j<NR;j++) ab[i][j]=ab[i][j]>v?ab[i][j]:v; break; }
            case SCALAR_ADD: { float v = f32_of(s->f0); for (long i=0;i<MR;i++) for(long j=0;j<NR;j++) ab[i][j]+=v; break; }
            case SCALAR_MUL: { float v = f32_of(s->f0); for (long i=0;i<MR;i++) for(long j=0;j<NR;j++) ab[i][j]*=v; break; }
            case SCALAR_SUB: { float v = f32_of(s->f0); for (long i=0;i<MR;i++) for(long j=0;j<NR;j++) ab[i][j]=v-ab[i][j]; break; }
            case SCALAR_SUBF:{ float v = f32_of(s->f0); for (long i=0;i<MR;i++) for(long j=0;j<NR;j++) ab[i][j]=ab[i][j]-v; break; }
            // ---- per-row fuse ops (one value per m-row) ----
            case PER_ROW_MIN: { const float*m=(const float*)s->f0; for(long i=0;i<MR;i++) for(long j=0;j<NR;j++) ab[i][j]=ab[i][j]<m[i]?ab[i][j]:m[i]; break; }
            case PER_ROW_MAX: { const float*m=(const float*)s->f0; for(long i=0;i<MR;i++) for(long j=0;j<NR;j++) ab[i][j]=ab[i][j]>m[i]?ab[i][j]:m[i]; break; }
            case PER_ROW_ADD: { const float*m=(const float*)s->f0; for(long i=0;i<MR;i++) for(long j=0;j<NR;j++) ab[i][j]+=m[i]; break; }
            case PER_ROW_MUL: { const float*m=(const float*)s->f0; for(long i=0;i<MR;i++) for(long j=0;j<NR;j++) ab[i][j]*=m[i]; break; }
            case PER_ROW_SUB: { const float*m=(const float*)s->f0; for(long i=0;i<MR;i++) for(long j=0;j<NR;j++) ab[i][j]=m[i]-ab[i][j]; break; }
            case PER_ROW_SUBF:{ const float*m=(const float*)s->f0; for(long i=0;i<MR;i++) for(long j=0;j<NR;j++) ab[i][j]=ab[i][j]-m[i]; break; }
            // ---- per-col fuse ops (one value per n-col) ----
            case PER_COL_MIN: { const float*m=(const float*)s->f0; for(long i=0;i<MR;i++) for(long j=0;j<NR;j++) ab[i][j]=ab[i][j]<m[j]?ab[i][j]:m[j]; break; }
            case PER_COL_MAX: { const float*m=(const float*)s->f0; for(long i=0;i<MR;i++) for(long j=0;j<NR;j++) ab[i][j]=ab[i][j]>m[j]?ab[i][j]:m[j]; break; }
            case PER_COL_ADD: { const float*m=(const float*)s->f0; for(long i=0;i<MR;i++) for(long j=0;j<NR;j++) ab[i][j]+=m[j]; break; }
            case PER_COL_MUL: { const float*m=(const float*)s->f0; for(long i=0;i<MR;i++) for(long j=0;j<NR;j++) ab[i][j]*=m[j]; break; }
            case PER_COL_SUB: { const float*m=(const float*)s->f0; for(long i=0;i<MR;i++) for(long j=0;j<NR;j++) ab[i][j]=m[j]-ab[i][j]; break; }
            case PER_COL_SUBF:{ const float*m=(const float*)s->f0; for(long i=0;i<MR;i++) for(long j=0;j<NR;j++) ab[i][j]=ab[i][j]-m[j]; break; }
            default:
                // LeakyRelu / QScale / RoundingShiftRight / ShiftLeft: excluded
                // by CAN_FUSE for f32 — should never arrive. Anything else: error.
                return 1;
        }
    }
}
