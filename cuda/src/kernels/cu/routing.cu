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

#define ROUTED_ROWS 4

// One struct per weight format the routed kernels read: `codes4` gives
// weights k .. k + 3 of row `row`, k a multiple of 4, as f32 before the
// scale `scale` of their block. Kernels multiply by the scale once per
// dot product where they can, as the compiler may not reassociate it.
struct w_f32 {
  static __device__ __forceinline__ float scale(const void *, long long, int,
                                                int) {
    return 1.0f;
  }
  static __device__ __forceinline__ float4 codes4(const void *w, long long row,
                                                  int row_len, int k) {
    return *(const float4 *)((const float *)w + row * row_len + k);
  }
};

struct w_f16 {
  static __device__ __forceinline__ float scale(const void *, long long, int,
                                                int) {
    return 1.0f;
  }
  static __device__ __forceinline__ float4 codes4(const void *w, long long row,
                                                  int row_len, int k) {
    const half2 *p = (const half2 *)((const half *)w + row * row_len + k);
    const float2 lo = __half22float2(p[0]);
    const float2 hi = __half22float2(p[1]);
    return make_float4(lo.x, lo.y, hi.x, hi.y);
  }
};

struct w_q40 {
  static __device__ __forceinline__ const block_q4_0 *
  block(const void *w, long long row, int row_len, int k) {
    return (const block_q4_0 *)w + row * (row_len / 32) + k / 32;
  }
  static __device__ __forceinline__ float scale(const void *w, long long row,
                                                int row_len, int k) {
    return __half2float(block(w, row, row_len, k)->d);
  }
  static __device__ __forceinline__ float4 codes4(const void *w, long long row,
                                                  int row_len, int k) {
    const int i = k % 32;
    const int shift = i < 16 ? 0 : 4;
    const uint8_t *qs = block(w, row, row_len, k)->qs + i % 16;
    return make_float4((float)(((qs[0] >> shift) & 0xF) - 8),
                       (float)(((qs[1] >> shift) & 0xF) - 8),
                       (float)(((qs[2] >> shift) & 0xF) - 8),
                       (float)(((qs[3] >> shift) & 0xF) - 8));
  }
};

struct w_q20t {
  static __device__ __forceinline__ const block_q2_0_t *
  block(const void *w, long long row, int row_len, int k) {
    return (const block_q2_0_t *)w + row * (row_len / 32) + k / 32;
  }
  static __device__ __forceinline__ float scale(const void *w, long long row,
                                                int row_len, int k) {
    return __half2float(block(w, row, row_len, k)->d);
  }
  static __device__ __forceinline__ float4 codes4(const void *w, long long row,
                                                  int row_len, int k) {
    const int q = block(w, row, row_len, k)->qs[(k % 32) / 4];
    return make_float4((float)((q & 3) - 1), (float)(((q >> 2) & 3) - 1),
                       (float)(((q >> 4) & 3) - 1), (float)(((q >> 6) & 3) - 1));
  }
};

template <typename W>
static __device__ __forceinline__ float4 load4(const void *w, long long row,
                                               int row_len, int k) {
  const float d = W::scale(w, row, row_len, k);
  const float4 c = W::codes4(w, row, row_len, k);
  return make_float4(c.x * d, c.y * d, c.z * d, c.w * d);
}

static __device__ __forceinline__ float dot4(float4 a, float4 b) {
  return a.x * b.x + a.y * b.y + a.z * b.z + a.w * b.w;
}

// One warp per ROUTED_ROWS output rows of route blockIdx.y, all reading the
// same f32 input row: eight lanes share each run of 32 weights, four each,
// so a warp reads 128 consecutive weights and inputs a step.
template <typename W>
static __device__ void routed_mul_vec_q(const void *w, const float *x,
                                        float *dst, const long long *row_ids,
                                        const long long *expert_ids, int k,
                                        int n) {
  const int route = blockIdx.y;
  const int row0 = (blockIdx.x * ROUTED_WARPS + threadIdx.y) * ROUTED_ROWS;
  if (row0 >= n) {
    return;
  }
  const long long in_row = row_ids ? row_ids[route] : route;
  const long long w_row = expert_ids[route] * n + row0;
  const float *xr = x + in_row * k;
  const int rows = min(ROUTED_ROWS, n - row0);
  float acc[ROUTED_ROWS] = {0.0f};
  for (int kk = 4 * threadIdx.x; kk < k; kk += 4 * WARP_SIZE) {
    const float4 xv = *(const float4 *)(xr + kk);
#pragma unroll
    for (int r = 0; r < ROUTED_ROWS; r++) {
      if (r < rows) {
        acc[r] += W::scale(w, w_row + r, k, kk) *
                  dot4(W::codes4(w, w_row + r, k, kk), xv);
      }
    }
  }
#pragma unroll
  for (int r = 0; r < ROUTED_ROWS; r++) {
    acc[r] = warp_reduce_sum<WARP_SIZE>(acc[r]);
    if (threadIdx.x == 0 && r < rows) {
      dst[(long long)route * n + row0 + r] = acc[r];
    }
  }
}

// One warp per output row: row blockIdx.x * ROUTED_WARPS + threadIdx.y of
// route blockIdx.y, lanes on consecutive inputs.
template <typename T>
static __device__ void routed_mul_vec_plain(const T *w, const float *x,
                                            float *dst,
                                            const long long *row_ids,
                                            const long long *expert_ids, int k,
                                            int n) {
  const int route = blockIdx.y;
  const int n_row = blockIdx.x * ROUTED_WARPS + threadIdx.y;
  if (n_row >= n) {
    return;
  }
  const long long in_row = row_ids ? row_ids[route] : route;
  const T *wr = w + (expert_ids[route] * n + n_row) * (long long)k;
  const float *xr = x + in_row * k;
  float partial = 0.0f;
  for (int i = threadIdx.x; i < k; i += WARP_SIZE) {
    partial += (float)wr[i] * xr[i];
  }
  partial = warp_reduce_sum<WARP_SIZE>(partial);
  if (threadIdx.x == 0) {
    dst[(long long)route * n + n_row] = partial;
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

#define RMM_M 64
#define RMM_N 64
#define RMM_K 32
#define RMM_MAX_EXPERTS 256

// Sorts routes by expert: `sorted` lists each expert's routes contiguously,
// and `chunks` holds [expert, base, len] for every `chunk` routes of an
// expert, then expert -1 up to `max_chunks`. One block of 256 threads.
extern "C" __global__ void route_plan(const long long *expert_ids, int routes,
                                      int experts, int *sorted, int *chunks,
                                      int max_chunks, int chunk) {
  __shared__ int count[RMM_MAX_EXPERTS];
  __shared__ int offset[RMM_MAX_EXPERTS];
  __shared__ int cursor[RMM_MAX_EXPERTS];
  __shared__ int chunk_offset[RMM_MAX_EXPERTS + 1];
  const int t = threadIdx.x;
  for (int e = t; e < experts; e += blockDim.x) {
    count[e] = 0;
    cursor[e] = 0;
  }
  __syncthreads();
  for (int r = t; r < routes; r += blockDim.x) {
    atomicAdd(&count[expert_ids[r]], 1);
  }
  __syncthreads();
  if (t == 0) {
    int o = 0;
    int c = 0;
    for (int e = 0; e < experts; e++) {
      offset[e] = o;
      chunk_offset[e] = c;
      o += count[e];
      c += (count[e] + chunk - 1) / chunk;
    }
    chunk_offset[experts] = c;
  }
  __syncthreads();
  for (int e = t; e < experts; e += blockDim.x) {
    for (int i = 0; i * chunk < count[e]; i++) {
      const int ch = chunk_offset[e] + i;
      chunks[3 * ch] = e;
      chunks[3 * ch + 1] = offset[e] + i * chunk;
      chunks[3 * ch + 2] = min(chunk, count[e] - i * chunk);
    }
  }
  for (int ch = chunk_offset[experts] + t; ch < max_chunks; ch += blockDim.x) {
    chunks[3 * ch] = -1;
  }
  for (int r = t; r < routes; r += blockDim.x) {
    const int e = (int)expert_ids[r];
    sorted[offset[e] + atomicAdd(&cursor[e], 1)] = r;
  }
}

// One chunk of an expert's sorted routes (blockIdx.y) by RMM_N output rows
// (blockIdx.x): f32 tiles of the gathered input rows and of the expert's
// weights unpacked to f32, scattered back to the routes' output rows.
template <typename W>
static __device__ void routed_mm(const void *w, const float *x, float *dst,
                                 const long long *row_ids, const int *sorted,
                                 const int *chunks, int k, int n) {
  const int expert = chunks[3 * blockIdx.y];
  if (expert < 0) {
    return;
  }
  const int base = chunks[3 * blockIdx.y + 1];
  const int len = chunks[3 * blockIdx.y + 2];
  const int n0 = blockIdx.x * RMM_N;
  __shared__ float as[RMM_K][RMM_M + 4];
  __shared__ float bs[RMM_K][RMM_N + 4];
  const int t = threadIdx.x;
  const int tx = t % 16;
  const int ty = t / 16;
  const int load_row = t / 4;
  const int part = t % 4;
  const bool a_valid = load_row < len;
  const int a_route = a_valid ? sorted[base + load_row] : 0;
  const long long a_row = row_ids ? row_ids[a_route] : a_route;
  const int b_n = n0 + load_row;
  const bool b_valid = b_n < n;
  const long long b_row = (long long)expert * n + b_n;
  float acc[4][4] = {{0.0f}};
  for (int k0 = 0; k0 < k; k0 += RMM_K) {
    float v[8];
    if (a_valid) {
      const float *p = x + a_row * k + k0 + 8 * part;
#pragma unroll
      for (int j = 0; j < 8; j++) {
        v[j] = p[j];
      }
    } else {
#pragma unroll
      for (int j = 0; j < 8; j++) {
        v[j] = 0.0f;
      }
    }
#pragma unroll
    for (int j = 0; j < 8; j++) {
      as[8 * part + j][load_row] = v[j];
    }
    if (b_valid) {
      const float d = W::scale(w, b_row, k, k0 + 8 * part);
      const float4 lo = W::codes4(w, b_row, k, k0 + 8 * part);
      const float4 hi = W::codes4(w, b_row, k, k0 + 8 * part + 4);
      v[0] = lo.x * d;
      v[1] = lo.y * d;
      v[2] = lo.z * d;
      v[3] = lo.w * d;
      v[4] = hi.x * d;
      v[5] = hi.y * d;
      v[6] = hi.z * d;
      v[7] = hi.w * d;
    } else {
#pragma unroll
      for (int j = 0; j < 8; j++) {
        v[j] = 0.0f;
      }
    }
#pragma unroll
    for (int j = 0; j < 8; j++) {
      bs[8 * part + j][load_row] = v[j];
    }
    __syncthreads();
#pragma unroll 8
    for (int kk = 0; kk < RMM_K; kk++) {
      float a[4];
      float b[4];
#pragma unroll
      for (int i = 0; i < 4; i++) {
        a[i] = as[kk][ty * 4 + i];
        b[i] = bs[kk][tx * 4 + i];
      }
#pragma unroll
      for (int i = 0; i < 4; i++) {
#pragma unroll
        for (int j = 0; j < 4; j++) {
          acc[i][j] += a[i] * b[j];
        }
      }
    }
    __syncthreads();
  }
#pragma unroll
  for (int i = 0; i < 4; i++) {
    const int m = ty * 4 + i;
    if (m < len) {
      const long long route = sorted[base + m];
#pragma unroll
      for (int j = 0; j < 4; j++) {
        const int col = n0 + tx * 4 + j;
        if (col < n) {
          dst[route * n + col] = acc[i][j];
        }
      }
    }
  }
}


#define RMV_ROUTES 8

// One chunk of up to RMV_ROUTES of an expert's sorted routes (blockIdx.y):
// each warp loads ROUTED_ROWS weight rows once and dots them with every route
// of the chunk, eight lanes per block of 32 weights, four each.
template <typename W>
static __device__ void routed_mv_grouped(const void *w, const float *x,
                                         float *dst, const long long *row_ids,
                                         const int *sorted, const int *chunks,
                                         int k, int n) {
  const int expert = chunks[3 * blockIdx.y];
  if (expert < 0) {
    return;
  }
  const int base = chunks[3 * blockIdx.y + 1];
  const int len = chunks[3 * blockIdx.y + 2];
  const int row0 = (blockIdx.x * ROUTED_WARPS + threadIdx.y) * ROUTED_ROWS;
  if (row0 >= n) {
    return;
  }
  const int rows = min(ROUTED_ROWS, n - row0);
  const int sub = threadIdx.x % 8;
  const float *xr[RMV_ROUTES];
  long long route[RMV_ROUTES];
#pragma unroll
  for (int c = 0; c < RMV_ROUTES; c++) {
    route[c] = c < len ? sorted[base + c] : 0;
    xr[c] = x + (row_ids ? row_ids[route[c]] : route[c]) * k;
  }
  float acc[ROUTED_ROWS][RMV_ROUTES] = {{0.0f}};
  const long long w_row = (long long)expert * n + row0;
  for (int b = threadIdx.x / 8; b < k / 32; b += WARP_SIZE / 8) {
    float4 wv[ROUTED_ROWS];
#pragma unroll
    for (int r = 0; r < ROUTED_ROWS; r++) {
      wv[r] = r < rows ? load4<W>(w, w_row + r, k, 32 * b + 4 * sub)
                       : make_float4(0.0f, 0.0f, 0.0f, 0.0f);
    }
#pragma unroll
    for (int c = 0; c < RMV_ROUTES; c++) {
      if (c < len) {
        const float4 xv = *(const float4 *)(xr[c] + 32 * b + 4 * sub);
#pragma unroll
        for (int r = 0; r < ROUTED_ROWS; r++) {
          acc[r][c] += wv[r].x * xv.x + wv[r].y * xv.y + wv[r].z * xv.z +
                       wv[r].w * xv.w;
        }
      }
    }
  }
#pragma unroll
  for (int r = 0; r < ROUTED_ROWS; r++) {
#pragma unroll
    for (int c = 0; c < RMV_ROUTES; c++) {
      const float sum = warp_reduce_sum<WARP_SIZE>(acc[r][c]);
      if (threadIdx.x == 0 && r < rows && c < len) {
        dst[route[c] * n + row0 + r] = sum;
      }
    }
  }
}

#define INSTANTIATE_ROUTED_TILED(name, W)                                      \
  extern "C" __launch_bounds__(256) __global__ void routed_mm_##name(          \
      const void *w, const float *x, float *dst, const long long *row_ids,     \
      const int *sorted, const int *chunks, int k, int n) {                    \
    routed_mm<W>(w, x, dst, row_ids, sorted, chunks, k, n);                    \
  }                                                                            \
  extern "C" __launch_bounds__(ROUTED_WARPS *WARP_SIZE) __global__ void        \
      routed_mv_grouped_##name(const void *w, const float *x, float *dst,      \
                               const long long *row_ids, const int *sorted,    \
                               const int *chunks, int k, int n) {              \
    routed_mv_grouped<W>(w, x, dst, row_ids, sorted, chunks, k, n);            \
  }

// Block-quantized formats also get the per-route GEMV over codes4; plain ones
// keep routed_mul_vec_plain, one warp per output row.
#define INSTANTIATE_ROUTED_BLOCK_QUANT(name, W)                                \
  INSTANTIATE_ROUTED_TILED(name, W)                                            \
  extern "C" __launch_bounds__(ROUTED_WARPS *WARP_SIZE, 1) __global__ void     \
      routed_mul_vec_##name(const void *w, const float *x, float *dst,         \
                            const long long *row_ids,                          \
                            const long long *expert_ids, int k, int n) {       \
    routed_mul_vec_q<W>(w, x, dst, row_ids, expert_ids, k, n);                 \
  }

INSTANTIATE_ROUTED_TILED(f32, w_f32)
INSTANTIATE_ROUTED_TILED(f16, w_f16)
INSTANTIATE_ROUTED_BLOCK_QUANT(q40, w_q40)
INSTANTIATE_ROUTED_BLOCK_QUANT(q20t, w_q20t)
