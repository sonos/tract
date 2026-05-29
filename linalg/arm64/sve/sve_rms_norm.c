// VLA SVE2 f32 fused row-wise RmsNorm. Mirrors the NEON kernel in
// arm64/arm64simd/rms_norm.rs and the AVX-512 kernel in
// x86_64_fma/rms_norm.rs.
//
//   Pass 1 (sum of squares):
//     4 svfloat32_t accumulators (s0..s3), 4*svcntw() lanes per inner
//     iteration. Tail handled by a predicated loop over the residue
//     (svwhilelt_b32) — no scalar tail.
//   Pass 2 (multiply-back):
//     broadcast inv_std into inv_v, fmla/store each 4-vec chunk in place;
//     same predicated tail.
//
// Width-agnostic by construction: identical correct output at any FEAT_SVE
// streaming vector length (128 → 2048 bits). Wider VL = wider lanes, fewer
// loop iterations, real perf scaling.
//
// ABI: void sve_rms_norm_f32_kernel(float *buf, int64_t n, float eps).
// Called from sve.rs::sve_rms_norm_f32 when FEAT_SVE2 is present on Linux
// aarch64. Plugs into Ops::rms_norm_f32; the core/nn::RmsNorm::eval fast
// path dispatches here automatically for trailing-axis F32 RmsNorm.

#include <arm_sve.h>
#include <math.h>
#include <stdint.h>

void sve_rms_norm_f32_kernel(float *buf, int64_t n, float eps) {
    if (n <= 0) return;

    const int64_t vl = (int64_t)svcntw();
    const int64_t step = 4 * vl;
    const svbool_t ptrue = svptrue_b32();

    // --- Pass 1: sum of squares ---
    svfloat32_t s0 = svdup_n_f32(0.0f);
    svfloat32_t s1 = svdup_n_f32(0.0f);
    svfloat32_t s2 = svdup_n_f32(0.0f);
    svfloat32_t s3 = svdup_n_f32(0.0f);

    int64_t i = 0;
    for (; i + step <= n; i += step) {
        svfloat32_t x0 = svld1_f32(ptrue, buf + i + 0 * vl);
        svfloat32_t x1 = svld1_f32(ptrue, buf + i + 1 * vl);
        svfloat32_t x2 = svld1_f32(ptrue, buf + i + 2 * vl);
        svfloat32_t x3 = svld1_f32(ptrue, buf + i + 3 * vl);
        s0 = svmla_f32_x(ptrue, s0, x0, x0);
        s1 = svmla_f32_x(ptrue, s1, x1, x1);
        s2 = svmla_f32_x(ptrue, s2, x2, x2);
        s3 = svmla_f32_x(ptrue, s3, x3, x3);
    }
    // Predicated tail: handles the (n % step) remainder, possibly distributed
    // across up to 4 partial vl-chunks. No scalar epilogue.
    for (; i < n; i += vl) {
        svbool_t pg = svwhilelt_b32((uint64_t)i, (uint64_t)n);
        svfloat32_t x = svld1_f32(pg, buf + i);
        s0 = svmla_f32_x(pg, s0, x, x);
    }

    // Reduce 4 accumulators → scalar via tree-add + horizontal reduce.
    s0 = svadd_f32_x(ptrue, s0, s1);
    s2 = svadd_f32_x(ptrue, s2, s3);
    s0 = svadd_f32_x(ptrue, s0, s2);
    float sum_sq = svaddv_f32(ptrue, s0);

    float mean_sq = sum_sq / (float)n;
    float inv_std = 1.0f / sqrtf(mean_sq + eps);

    // --- Pass 2: multiply by inv_std ---
    svfloat32_t inv_v = svdup_n_f32(inv_std);

    i = 0;
    for (; i + step <= n; i += step) {
        svfloat32_t x0 = svld1_f32(ptrue, buf + i + 0 * vl);
        svfloat32_t x1 = svld1_f32(ptrue, buf + i + 1 * vl);
        svfloat32_t x2 = svld1_f32(ptrue, buf + i + 2 * vl);
        svfloat32_t x3 = svld1_f32(ptrue, buf + i + 3 * vl);
        svst1_f32(ptrue, buf + i + 0 * vl, svmul_f32_x(ptrue, x0, inv_v));
        svst1_f32(ptrue, buf + i + 1 * vl, svmul_f32_x(ptrue, x1, inv_v));
        svst1_f32(ptrue, buf + i + 2 * vl, svmul_f32_x(ptrue, x2, inv_v));
        svst1_f32(ptrue, buf + i + 3 * vl, svmul_f32_x(ptrue, x3, inv_v));
    }
    for (; i < n; i += vl) {
        svbool_t pg = svwhilelt_b32((uint64_t)i, (uint64_t)n);
        svfloat32_t x = svld1_f32(pg, buf + i);
        svst1_f32(pg, buf + i, svmul_f32_x(pg, x, inv_v));
    }
}
