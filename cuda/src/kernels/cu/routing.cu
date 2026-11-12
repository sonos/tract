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

#define ROUTED_WARPS 4

// Sum of each thread's partial over the ROUTED_WARPS warps of the block,
// valid in thread (0, 0).
static __device__ __forceinline__ float routed_block_sum(float partial) {
  __shared__ float shared[ROUTED_WARPS - 1][WARP_SIZE];
  if (threadIdx.y > 0) {
    shared[threadIdx.y - 1][threadIdx.x] = partial;
  }
  __syncthreads();
  if (threadIdx.y == 0) {
    for (int w = 0; w < ROUTED_WARPS - 1; w++) {
      partial += shared[w][threadIdx.x];
    }
    partial = warp_reduce_sum<WARP_SIZE>(partial);
  }
  return partial;
}

static __device__ __forceinline__ float block_dot(const block_q4_0 *b,
                                                 const float *x) {
  float sum = 0.0f;
#pragma unroll
  for (int j = 0; j < 16; j++) {
    const int q = b->qs[j];
    sum += (float)((q & 0xF) - 8) * x[j] + (float)((q >> 4) - 8) * x[j + 16];
  }
  return __half2float(b->d) * sum;
}

static __device__ __forceinline__ float block_dot(const block_q2_0_t *b,
                                                 const float *x) {
  float sum = 0.0f;
#pragma unroll
  for (int j = 0; j < 8; j++) {
    const int q = b->qs[j];
#pragma unroll
    for (int c = 0; c < 4; c++) {
      sum += (float)(((q >> (2 * c)) & 3) - 1) * x[4 * j + c];
    }
  }
  return __half2float(b->d) * sum;
}

// Output row blockIdx.x of route blockIdx.y: the route's f32 input row dotted
// with row `n_row` of its expert's block-quantized weights, unpacked to f32.
template <typename block_t>
static __device__ void routed_mul_vec_q(const block_t *w, const float *x,
                                        float *dst, const long long *row_ids,
                                        const long long *expert_ids, int k,
                                        int n) {
  const int route = blockIdx.y;
  const int n_row = blockIdx.x;
  const long long in_row = row_ids ? row_ids[route] : route;
  const int blocks = k / 32;
  const block_t *wr = w + (expert_ids[route] * n + n_row) * (long long)blocks;
  const float *xr = x + in_row * k;
  float partial = 0.0f;
  for (int b = WARP_SIZE * threadIdx.y + threadIdx.x; b < blocks;
       b += ROUTED_WARPS * WARP_SIZE) {
    partial += block_dot(wr + b, xr + 32 * b);
  }
  const float sum = routed_block_sum(partial);
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    dst[(long long)route * n + n_row] = sum;
  }
}

#define INSTANTIATE_ROUTED_MUL_VEC_Q(name, block_t)                            \
  extern "C" __launch_bounds__(ROUTED_WARPS *WARP_SIZE, 1) __global__ void     \
      routed_mul_vec_##name(const block_t *w, const float *x, float *dst,      \
                            const long long *row_ids,                          \
                            const long long *expert_ids, int k, int n) {       \
    routed_mul_vec_q<block_t>(w, x, dst, row_ids, expert_ids, k, n);           \
  }

INSTANTIATE_ROUTED_MUL_VEC_Q(q40, block_q4_0)
INSTANTIATE_ROUTED_MUL_VEC_Q(q20t, block_q2_0_t)

template <typename T>
static __device__ void routed_mul_vec_plain(const T *w, const float *x,
                                            float *dst,
                                            const long long *row_ids,
                                            const long long *expert_ids, int k,
                                            int n) {
  const int route = blockIdx.y;
  const int n_row = blockIdx.x;
  const long long in_row = row_ids ? row_ids[route] : route;
  const T *wr = w + (expert_ids[route] * n + n_row) * (long long)k;
  const float *xr = x + in_row * k;
  float partial = 0.0f;
  for (int i = WARP_SIZE * threadIdx.y + threadIdx.x; i < k;
       i += ROUTED_WARPS * WARP_SIZE) {
    partial += (float)wr[i] * xr[i];
  }
  const float sum = routed_block_sum(partial);
  if (threadIdx.x == 0 && threadIdx.y == 0) {
    dst[(long long)route * n + n_row] = sum;
  }
}

#define INSTANTIATE_ROUTED_MUL_VEC_PLAIN(name, T)                              \
  extern "C" __launch_bounds__(ROUTED_WARPS *WARP_SIZE, 1) __global__ void     \
      routed_mul_vec_##name(const T *w, const float *x, float *dst,            \
                            const long long *row_ids,                          \
                            const long long *expert_ids, int k, int n) {       \
    routed_mul_vec_plain<T>(w, x, dst, row_ids, expert_ids, k, n);             \
  }

INSTANTIATE_ROUTED_MUL_VEC_PLAIN(f32, float)
INSTANTIATE_ROUTED_MUL_VEC_PLAIN(f16, half)

// activation: 0 none, 1 silu, 2 tanh gelu, 3 relu, 4 clamped swiglu (pairs).
static __device__ __forceinline__ float routed_activate(float v, int activation) {
  if (activation == 1) {
    return v / (1.0f + expf(-v));
  } else if (activation == 2) {
    return 0.5f * v * (1.0f + tanhf(0.7978845608f * (v + 0.044715f * v * v * v)));
  } else if (activation == 3) {
    return fmaxf(v, 0.0f);
  }
  return v;
}

extern "C" __global__ void routed_postamble(
    const float *gate, const float *up, const float *bias,
    const float *bias_up, const long long *expert_ids, float *out, int routes,
    int n, int activation, float alpha, float limit) {
  const long long ix = (long long)blockIdx.x * blockDim.x + threadIdx.x;
  if (ix >= (long long)routes * n) {
    return;
  }
  const long long route = ix / n;
  const long long col = ix - route * n;
  const long long bias_ix = expert_ids[route] * n + col;
  float g = gate[ix] + (bias ? bias[bias_ix] : 0.0f);
  if (!up) {
    out[ix] = routed_activate(g, activation);
    return;
  }
  float u = up[ix] + (bias_up ? bias_up[bias_ix] : 0.0f);
  if (activation == 4) {
    g = fminf(g, limit);
    u = fminf(fmaxf(u, -limit), limit);
    out[ix] = (u + 1.0f) * g / (1.0f + expf(-alpha * g));
  } else {
    out[ix] = routed_activate(g, activation) * u;
  }
}
