#include "common.cuh"
#include <math_constants.h>

#define ROUTE_LANES 32
#define ROUTE_REGS 8

// Order-preserving image of a score: a larger key is a larger score, and
// -0.0 maps with 0.0.
static __device__ __forceinline__ unsigned int route_key(float v) {
  const unsigned int bits = __float_as_uint(v == 0.0f ? 0.0f : v);
  return (bits & 0x80000000u) ? ~bits : (bits | 0x80000000u);
}

static __device__ __forceinline__ float route_score(unsigned int key) {
  return __uint_as_float((key & 0x80000000u) ? (key ^ 0x80000000u) : ~key);
}

// One warp per token. Expert e lives in lane e % 32, register e / 32; each
// slot takes the largest key, then the lowest expert id among equal keys.
extern "C" __global__ void route_select_f32(
    const float *scores, const float *bias, long long *token_ids,
    long long *expert_ids, float *weights, int tokens, int experts, int k,
    int gate, int has_bias) {
  const int token = blockIdx.x;
  const int lane = threadIdx.x;
  if (token >= tokens) {
    return;
  }
  const float *row = scores + (long long)token * experts;
  float s[ROUTE_REGS];
  unsigned int valid = 0;
#pragma unroll
  for (int r = 0; r < ROUTE_REGS; r++) {
    const int e = r * ROUTE_LANES + lane;
    float v = -CUDART_INF_F;
    if (e < experts) {
      valid |= 1u << r;
      v = row[e] + (has_bias ? bias[e] : 0.0f);
    }
    s[r] = v;
  }

  float local_max = s[0];
#pragma unroll
  for (int r = 1; r < ROUTE_REGS; r++) {
    local_max = fmaxf(local_max, s[r]);
  }
  const float max_all = warp_reduce_max<ROUTE_LANES>(local_max);
  float denom_all = 0.0f;
  if (gate == 1) {
    float local_sum = 0.0f;
#pragma unroll
    for (int r = 0; r < ROUTE_REGS; r++) {
      local_sum += expf(s[r] - max_all);
    }
    denom_all = warp_reduce_sum<ROUTE_LANES>(local_sum);
  }

  float top = 0.0f;
  float denom_topk = 0.0f;
  for (int slot = 0; slot < k; slot++) {
    unsigned int best_key = 0;
    unsigned int best_expert = 0xFFFFFFFFu;
#pragma unroll
    for (int r = 0; r < ROUTE_REGS; r++) {
      if (valid & (1u << r)) {
        const unsigned int key = route_key(s[r]);
        if (best_expert == 0xFFFFFFFFu || key > best_key) {
          best_key = key;
          best_expert = r * ROUTE_LANES + lane;
        }
      }
    }
    unsigned int win_key = best_key;
#pragma unroll
    for (int offset = ROUTE_LANES / 2; offset > 0; offset >>= 1) {
      win_key = max(win_key, __shfl_xor_sync(0xffffffff, win_key, offset));
    }
    unsigned int win_expert = best_key == win_key ? best_expert : 0xFFFFFFFFu;
#pragma unroll
    for (int offset = ROUTE_LANES / 2; offset > 0; offset >>= 1) {
      win_expert = min(win_expert, __shfl_xor_sync(0xffffffff, win_expert, offset));
    }
    const float win_score = route_score(win_key);
    if (slot == 0) {
      top = win_score;
    }
    denom_topk += expf(win_score - top);
    if (lane == (int)(win_expert % ROUTE_LANES)) {
      valid &= ~(1u << (win_expert / ROUTE_LANES));
    }
    if (lane == 0) {
      const long long route = (long long)token * k + slot;
      token_ids[route] = token;
      expert_ids[route] = win_expert;
      float w;
      if (gate == 1) {
        w = expf(win_score - max_all) / denom_all;
      } else if (gate == 2) {
        w = 1.0f / (1.0f + expf(-win_score));
      } else if (gate == 3) {
        w = win_score;
      } else {
        w = expf(win_score - top);
      }
      weights[route] = w;
    }
  }
  if (gate == 0 && lane == 0) {
    for (int slot = 0; slot < k; slot++) {
      weights[(long long)token * k + slot] /= denom_topk;
    }
  }
}

template <typename T>
static __device__ void routed_combine(const float *values, const float *weights,
                                      T *out, int tokens, int d, int k) {
  const long long ix = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (ix >= (long long)tokens * d) {
    return;
  }
  const long long token = ix / d;
  const long long col = ix - token * d;
  float acc = 0.0f;
  for (int j = 0; j < k; j++) {
    const long long route = token * k + j;
    acc += weights[route] * values[route * d + col];
  }
  out[ix] = (T)acc;
}

extern "C" __global__ void routed_combine_f32(const float *values,
                                              const float *weights, float *out,
                                              int tokens, int d, int k) {
  routed_combine<float>(values, weights, out, tokens, d, k);
}

extern "C" __global__ void routed_combine_f16(const float *values,
                                              const float *weights, half *out,
                                              int tokens, int d, int k) {
  routed_combine<half>(values, weights, out, tokens, d, k);
}
