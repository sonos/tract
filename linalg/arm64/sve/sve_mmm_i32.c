// SVE int8 -> int32 GEMM kernel for tract's MMM framework (the qmmm_i32 slot).
//
// Tile MR=8 x NR=8, i32 accumulator. The hot AddMatMul is the vector-length-
// agnostic *widening* broadcast-A rank-1 update: per K-step it loads NR signed
// bytes of a B row and sign-extends them to i32 with `svld1sb_s32`, then folds
// MR `svmla_n_s32` updates with the (sign-extended) scalar from the A column.
// The NR columns are walked in svcntw() chunks with whilelt predication, so the
// SAME code is correct and full-width at any SVE vector length (128..2048-bit).
//
// Why widening MLA and not SDOT: SDOT reduces 4 K-contiguous i8 per i32 lane, so
// it needs A and B packed K-contiguous within a lane. tract's PackedFormat is
// K-major (mn-inner: for each k, r contiguous mn values), which is exactly what
// the per-k widening update consumes — and what arm64simd's i32 kernel uses via
// NEON SMLAL. A SDOT/SMMLA path would need a custom interleaved packer; that is
// a separate (max-throughput) kernel, not this one.
//
// int8 inputs arrive via tract's native i8i8 packing (AddMatMul packing == 1).
// The default i32i32 packing (packing == 0) is also handled (scalar) so the
// generic auto-test surface (mmm_packed_packed_tests i32i32:0) passes.
//
// ABI: identical 40-byte FusedKerSpec<i32> walk as the f32 kernel (discriminant
// u64 at offset 0, fields at 8/16/24/32). Fuse ops act on the MRxNR i32 tile in
// memory (scalar C — not the hot path), including the quantization ops
// q_scale / q_shr (rounding) / q_shl, ported bit-exact from
// linalg/src/generic/rounding.rs.
//
// Returns 0 on success, 1 if asked to do an unsupported fused op / packing.

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

// RoundingPolicy is #[repr(usize)] in fuse.rs: Native=0, Zero=1, Away=2,
// MinusInf=3, PlusInf=4, Even=5, Odd=6.
enum { RP_NATIVE = 0, RP_ZERO, RP_AWAY, RP_MINUSINF, RP_PLUSINF, RP_EVEN, RP_ODD };

typedef struct {
    uint64_t disc;
    uint64_t f0, f1, f2, f3; // fields at byte offsets 8, 16, 24, 32
} spec_t;

// AddMatMul, i8 x i8 -> i32 (packing 1): ab[m][n] += sum_k pa[k*MR+m]*pb[k*NR+n].
// VLA widening rank-1 update over NR.
static void add_mat_mul_i8(int32_t ab[MR][NR], const int8_t *pa, const int8_t *pb, long k) {
    for (long n0 = 0; n0 < NR; n0 += svcntw()) {
        svbool_t pg = svwhilelt_b32((uint64_t)n0, (uint64_t)NR);
        svint32_t a0 = svld1_s32(pg, &ab[0][n0]), a1 = svld1_s32(pg, &ab[1][n0]);
        svint32_t a2 = svld1_s32(pg, &ab[2][n0]), a3 = svld1_s32(pg, &ab[3][n0]);
        svint32_t a4 = svld1_s32(pg, &ab[4][n0]), a5 = svld1_s32(pg, &ab[5][n0]);
        svint32_t a6 = svld1_s32(pg, &ab[6][n0]), a7 = svld1_s32(pg, &ab[7][n0]);
        for (long kk = 0; kk < k; kk++) {
            // Load NR int8 of B row kk, sign-extending each lane to i32.
            svint32_t b = svld1sb_s32(pg, &pb[kk * NR + n0]);
            const int8_t *arow = &pa[kk * MR];
            a0 = svmla_n_s32_x(pg, a0, b, (int32_t)arow[0]);
            a1 = svmla_n_s32_x(pg, a1, b, (int32_t)arow[1]);
            a2 = svmla_n_s32_x(pg, a2, b, (int32_t)arow[2]);
            a3 = svmla_n_s32_x(pg, a3, b, (int32_t)arow[3]);
            a4 = svmla_n_s32_x(pg, a4, b, (int32_t)arow[4]);
            a5 = svmla_n_s32_x(pg, a5, b, (int32_t)arow[5]);
            a6 = svmla_n_s32_x(pg, a6, b, (int32_t)arow[6]);
            a7 = svmla_n_s32_x(pg, a7, b, (int32_t)arow[7]);
        }
        svst1_s32(pg, &ab[0][n0], a0); svst1_s32(pg, &ab[1][n0], a1);
        svst1_s32(pg, &ab[2][n0], a2); svst1_s32(pg, &ab[3][n0], a3);
        svst1_s32(pg, &ab[4][n0], a4); svst1_s32(pg, &ab[5][n0], a5);
        svst1_s32(pg, &ab[6][n0], a6); svst1_s32(pg, &ab[7][n0], a7);
    }
}

// AddMatMul, i32 x i32 -> i32 (packing 0, default): only used by the auto-test
// surface, never in production (quantized matmul uses the i8i8 packing). Scalar.
static void add_mat_mul_i32(int32_t ab[MR][NR], const int32_t *pa, const int32_t *pb, long k) {
    for (long kk = 0; kk < k; kk++) {
        const int32_t *arow = &pa[kk * MR], *brow = &pb[kk * NR];
        for (long i = 0; i < MR; i++)
            for (long j = 0; j < NR; j++) ab[i][j] += arow[i] * brow[j];
    }
}

// ---- Quantization helpers, ported bit-exact from generic/rounding.rs ----

// i32::q_shr(shift, rp): rounding arithmetic shift right.
static int32_t q_shr_i32(int32_t v, long shift, int rp) {
    int32_t half = (int32_t)1 << (shift - 1);
    int32_t a = v < 0 ? -v : v; // abs (test inputs are small; matches Rust .abs())
    int32_t nudge;
    switch (rp) {
        case RP_ZERO:     nudge = -1; break;
        case RP_MINUSINF: nudge = -(int32_t)(v >= 0); break;
        case RP_PLUSINF:  nudge = -(int32_t)(v <= 0); break;
        case RP_AWAY:     nudge = 0; break;
        case RP_EVEN:     nudge = ((a >> shift) & 0x1) - 1; break;
        case RP_ODD:      nudge = -((a >> shift) & 0x1); break;
        default:          nudge = 0; break; // Native: unreachable for q ops
    }
    int32_t sign = (v > 0) - (v < 0); // signum: -1 / 0 / 1
    return sign * ((a + half + nudge) >> shift);
}

// i32::q_scale(Scaler{mult, shift, policy}) with mult always present (the QScale
// fused op carries an explicit multiplier). Mirrors `Mul<i32> for Scaler`.
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

// Store the MRxNR i32 tile to memory with arbitrary row/col byte strides,
// truncating to the destination item size (matches generic store_t semantics
// for the tested widths 1 and 4).
static void store_tile(int32_t ab[MR][NR], const spec_t *s) {
    uint8_t *ptr = (uint8_t *)s->f0;
    long rstride = (long)s->f1, cstride = (long)s->f2, isz = (long)s->f3;
    for (long i = 0; i < MR; i++)
        for (long j = 0; j < NR; j++) {
            uint8_t *p = ptr + i * rstride + j * cstride;
            int32_t v = ab[i][j];
            switch (isz) {
                case 1: *(uint8_t *)p = (uint8_t)v; break;
                case 2: *(uint16_t *)p = (uint16_t)v; break;
                case 4: *(int32_t *)p = v; break;
                case 8: { int64_t w = v; memcpy(p, &w, 8); break; }
                default: memcpy(p, &v, isz < 4 ? (size_t)isz : 4); break;
            }
        }
}

// Returns isize (64-bit) to match tract's kernel ABI.
intptr_t sve_mmm_i32_kernel(const spec_t *ops) {
    int32_t ab[MR][NR];
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
                long packing = (long)s->f3;
                if (packing == 1) {
                    add_mat_mul_i8(ab, (const int8_t *)s->f1, (const int8_t *)s->f2, k);
                } else if (packing == 0) {
                    add_mat_mul_i32(ab, (const int32_t *)s->f1, (const int32_t *)s->f2, k);
                } else {
                    return 1;
                }
                break;
            }
            case STORE:
                store_tile(ab, s);
                break;
            case LOAD_TILE: {
                // LoadTile(col_major_ptr, row_major_ptr); use the row-major one.
                const int32_t *src = (const int32_t *)s->f1;
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
                        if (isz == 1)
                            ab[i][j] += *(const int8_t *)p; // sign-extend
                        else
                            ab[i][j] += *(const int32_t *)p;
                    }
                break;
            }
            case ADD_ROW_COL_PRODUCTS: {
                const int32_t *rows = (const int32_t *)s->f0;
                const int32_t *cols = (const int32_t *)s->f1;
                for (long i = 0; i < MR; i++)
                    for (long j = 0; j < NR; j++) ab[i][j] += rows[i] * cols[j];
                break;
            }
            // ---- quantization fuse ops ----
            case Q_SCALE: {
                long shift = (long)s->f0;
                int policy = (int)s->f1;
                int32_t mult = (int32_t)s->f2;
                for (long i = 0; i < MR; i++)
                    for (long j = 0; j < NR; j++) ab[i][j] = q_scale_i32(ab[i][j], shift, policy, mult);
                break;
            }
            case Q_SHR: {
                long shift = (long)s->f0;
                int policy = (int)s->f1;
                for (long i = 0; i < MR; i++)
                    for (long j = 0; j < NR; j++) ab[i][j] = q_shr_i32(ab[i][j], shift, policy);
                break;
            }
            case Q_SHL: {
                long shift = (long)s->f0;
                for (long i = 0; i < MR; i++)
                    for (long j = 0; j < NR; j++) ab[i][j] = ab[i][j] << shift;
                break;
            }
            // ---- scalar fuse ops (value is an i32 in the low 32 bits of f0) ----
            case SCALAR_MIN: { int32_t v=(int32_t)s->f0; for(long i=0;i<MR;i++)for(long j=0;j<NR;j++) ab[i][j]=ab[i][j]<v?ab[i][j]:v; break; }
            case SCALAR_MAX: { int32_t v=(int32_t)s->f0; for(long i=0;i<MR;i++)for(long j=0;j<NR;j++) ab[i][j]=ab[i][j]>v?ab[i][j]:v; break; }
            case SCALAR_ADD: { int32_t v=(int32_t)s->f0; for(long i=0;i<MR;i++)for(long j=0;j<NR;j++) ab[i][j]+=v; break; }
            case SCALAR_MUL: { int32_t v=(int32_t)s->f0; for(long i=0;i<MR;i++)for(long j=0;j<NR;j++) ab[i][j]*=v; break; }
            case SCALAR_SUB: { int32_t v=(int32_t)s->f0; for(long i=0;i<MR;i++)for(long j=0;j<NR;j++) ab[i][j]=v-ab[i][j]; break; }
            case SCALAR_SUBF:{ int32_t v=(int32_t)s->f0; for(long i=0;i<MR;i++)for(long j=0;j<NR;j++) ab[i][j]=ab[i][j]-v; break; }
            // ---- per-row fuse ops (one i32 per m-row) ----
            case PER_ROW_MIN: { const int32_t*m=(const int32_t*)s->f0; for(long i=0;i<MR;i++)for(long j=0;j<NR;j++) ab[i][j]=ab[i][j]<m[i]?ab[i][j]:m[i]; break; }
            case PER_ROW_MAX: { const int32_t*m=(const int32_t*)s->f0; for(long i=0;i<MR;i++)for(long j=0;j<NR;j++) ab[i][j]=ab[i][j]>m[i]?ab[i][j]:m[i]; break; }
            case PER_ROW_ADD: { const int32_t*m=(const int32_t*)s->f0; for(long i=0;i<MR;i++)for(long j=0;j<NR;j++) ab[i][j]+=m[i]; break; }
            case PER_ROW_MUL: { const int32_t*m=(const int32_t*)s->f0; for(long i=0;i<MR;i++)for(long j=0;j<NR;j++) ab[i][j]*=m[i]; break; }
            case PER_ROW_SUB: { const int32_t*m=(const int32_t*)s->f0; for(long i=0;i<MR;i++)for(long j=0;j<NR;j++) ab[i][j]=m[i]-ab[i][j]; break; }
            case PER_ROW_SUBF:{ const int32_t*m=(const int32_t*)s->f0; for(long i=0;i<MR;i++)for(long j=0;j<NR;j++) ab[i][j]=ab[i][j]-m[i]; break; }
            // ---- per-col fuse ops (one i32 per n-col) ----
            case PER_COL_MIN: { const int32_t*m=(const int32_t*)s->f0; for(long i=0;i<MR;i++)for(long j=0;j<NR;j++) ab[i][j]=ab[i][j]<m[j]?ab[i][j]:m[j]; break; }
            case PER_COL_MAX: { const int32_t*m=(const int32_t*)s->f0; for(long i=0;i<MR;i++)for(long j=0;j<NR;j++) ab[i][j]=ab[i][j]>m[j]?ab[i][j]:m[j]; break; }
            case PER_COL_ADD: { const int32_t*m=(const int32_t*)s->f0; for(long i=0;i<MR;i++)for(long j=0;j<NR;j++) ab[i][j]+=m[j]; break; }
            case PER_COL_MUL: { const int32_t*m=(const int32_t*)s->f0; for(long i=0;i<MR;i++)for(long j=0;j<NR;j++) ab[i][j]*=m[j]; break; }
            case PER_COL_SUB: { const int32_t*m=(const int32_t*)s->f0; for(long i=0;i<MR;i++)for(long j=0;j<NR;j++) ab[i][j]=m[j]-ab[i][j]; break; }
            case PER_COL_SUBF:{ const int32_t*m=(const int32_t*)s->f0; for(long i=0;i<MR;i++)for(long j=0;j<NR;j++) ab[i][j]=ab[i][j]-m[j]; break; }
            default:
                // LeakyRelu excluded by CAN_FUSE_I32; anything else is an error.
                return 1;
        }
    }
}
