#include "common.cuh"

// Check CC Version
#define CUDA_CC_TURING 750
#if defined(__CUDA_ARCH__) && (__CUDA_ARCH__ < CUDA_CC_TURING)
#error "Requires GPU with compute capability 7.5 or higher"
#endif

#define VDR_Q4_0_Q8_1_MMVQ 2
#define VDR_Q4_0_Q8_1_MMQ 4

#define N_WARPS 8

#define MMQ_Y 128

#define MMQ_ITER_K 256
#define MMQ_TILE_Y_K (WARP_SIZE + WARP_SIZE / QI8_1)

#define PAD(x, n) (((x) + (n) - 1) & ~((n) - 1))
#define MMQ_MMA_TILE_X_K_Q8_0 (2 * WARP_SIZE + 2 * WARP_SIZE / QI8_0 + 4)


static __device__ __forceinline__ int get_int_b2(const void *x,
                                                 const int &i32) {
  const uint16_t *x16 = (const uint16_t *)x; // assume at least 2 byte alignment

  int x32 = x16[2 * i32 + 0] << 0;
  x32 |= x16[2 * i32 + 1] << 16;

  return x32;
}

static __device__ __forceinline__ int get_int_b4(const void *x,
                                                 const int &i32) {
  return ((const int *)x)[i32]; // assume at least 4 byte alignment
}

using namespace cuda_mma;

typedef void (*load_tiles_mmq_t)(const char *__restrict__ x, int *x_tile,
                                 const int kbx0, const int i_max,
                                 const int stride);
typedef void (*vec_dot_mmq_t)(const int *__restrict__ x,
                              const int *__restrict__ y,
                              float *__restrict__ sum, const int k00);
typedef void (*mmq_write_back_t)(const float *__restrict__ sum,
                                 const int32_t *__restrict__ get_rows_to_sorted,
                                 float *__restrict__ dst, const int stride,
                                 const int i_max, const int j_max);

static constexpr __device__ int mmq_get_granularity_device(const int mmq_x) {
  return mmq_x >= 48 ? 16 : 8;
}

template <int mmq_y, int nwarps, bool need_check>
static __device__ __forceinline__ void
load_tiles_q4_0(const char *__restrict__ x, int *__restrict__ x_tile,
                const int kbx0, const int i_max, const int stride) {

  int *x_qs = (int *)x_tile;
  float *x_df = (float *)(x_qs + 2 * WARP_SIZE);

  const int kbx = threadIdx.x / QI4_0;
  const int kqsx = threadIdx.x % QI4_0;

  _Pragma("unroll") for (int i0 = 0; i0 < mmq_y; i0 += N_WARPS) {
    int i = i0 + threadIdx.y;

    if (need_check) {
      i = min(i, i_max);
    }
    const block_q4_0 *bxi = (const block_q4_0 *)x + kbx0 + i * stride + kbx;
    const int qs0 = get_int_b2(bxi->qs, kqsx);

    x_qs[i * MMQ_MMA_TILE_X_K_Q8_0 + kbx * (2 * QI4_0) + kqsx + 0] =
        __vsubss4((qs0 >> 0) & 0x0F0F0F0F, 0x08080808);
    x_qs[i * MMQ_MMA_TILE_X_K_Q8_0 + kbx * (2 * QI4_0) + kqsx + QI4_0] =
        __vsubss4((qs0 >> 4) & 0x0F0F0F0F, 0x08080808);
  }

  const int blocks_per_tile_x_row = WARP_SIZE / QI4_0;
  const int kbxd = threadIdx.x % blocks_per_tile_x_row;

  _Pragma("unroll") for (int i0 = 0; i0 < MMQ_Y; i0 += N_WARPS * QI4_0) {
    int i = i0 + threadIdx.y * QI4_0 + threadIdx.x / blocks_per_tile_x_row;

    if (need_check) {
      i = min(i, i_max);
    }

    const block_q4_0 *bxi = (const block_q4_0 *)x + kbx0 + i * stride + kbxd;

    x_df[i * MMQ_MMA_TILE_X_K_Q8_0 + kbxd] = bxi->d;
  }
}

template <int mmq_x, int mmq_y, int nwarps>
static __device__ __forceinline__ void
vec_dot_q8_0_q8_1_mma(const int *__restrict__ x, const int *__restrict__ y,
                      float *__restrict__ sum, const int k00) {

  typedef tile<16, 8, int> tile_A;
  typedef tile<8, 8, int> tile_B;
  typedef tile<16, 8, int> tile_C;

  constexpr int granularity = mmq_get_granularity_device(mmq_x);
  constexpr int rows_per_warp = 2 * granularity;
  constexpr int ntx =
      rows_per_warp / tile_C::I; // Number of x minitiles per warp.

  y += (threadIdx.y % ntx) * (tile_B::I * MMQ_TILE_Y_K);

  const int *x_qs = (const int *)x;
  const float *x_df = (const float *)x_qs + 2 * WARP_SIZE;
  const int *y_qs = (const int *)y + 4;
  const half2 *y_ds = (const half2 *)y;

  tile_A A[ntx][WARP_SIZE / QI8_0];
  float dA[ntx][tile_C::ne / 2][WARP_SIZE / QI8_0];

  const int i0 = (threadIdx.y / ntx) * rows_per_warp;

  _Pragma("unroll") for (int n = 0; n < ntx; ++n) {
    _Pragma("unroll") for (int k01 = 0; k01 < WARP_SIZE; k01 += QI8_0) {
      const int k0 = k00 + k01;

      load_ldmatrix(A[n][k01 / QI8_0],
                    x_qs + (i0 + n * tile_A::I) * MMQ_MMA_TILE_X_K_Q8_0 + k0,
                    MMQ_MMA_TILE_X_K_Q8_0);
    }

    _Pragma("unroll") for (int l = 0; l < tile_C::ne / 2; ++l) {
      const int i = i0 + n * tile_A::I + tile_C::get_i(2 * l);

      _Pragma("unroll") for (int k01 = 0; k01 < WARP_SIZE; k01 += QI8_0) {
        const int k0 = k00 + k01;

        dA[n][l][k01 / QI8_0] = x_df[i * MMQ_MMA_TILE_X_K_Q8_0 + k0 / QI8_0];
      }
    }
  }

  _Pragma("unroll") for (int j0 = 0; j0 < mmq_x; j0 += ntx * tile_C::J) {
    _Pragma("unroll") for (int k01 = 0; k01 < WARP_SIZE; k01 += QI8_0) {
      tile_B B;
      float dB[tile_C::ne / 2];

      load_generic(B, y_qs + j0 * MMQ_TILE_Y_K + k01,
                   MMQ_TILE_Y_K); // faster than load_ldmatrix

      _Pragma("unroll") for (int l = 0; l < tile_C::ne / 2; ++l) {
        const int j = j0 + tile_C::get_j(l);

        dB[l] = __low2float(y_ds[j * MMQ_TILE_Y_K + k01 / QI8_1]);
      }

      _Pragma("unroll") for (int n = 0; n < ntx; ++n) {
        tile_C C;
        mma(C, A[n][k01 / QI8_0], B);

        _Pragma("unroll") for (int l = 0; l < tile_C::ne; ++l) {
          sum[(j0 / tile_C::J + n) * tile_C::ne + l] +=
              C.x[l] * dA[n][l / 2][k01 / QI8_0] * dB[l % 2];
        }
      }
    }
  }
}

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
static __device__ __forceinline__ void
mmq_write_back_mma(const float *__restrict__ sum,
                   const int *__restrict__ ids_dst, float *__restrict__ dst,
                   const int stride, const int i_max, const int j_max) {
  typedef tile<16, 8, int> tile_C;

  constexpr int granularity = mmq_get_granularity_device(mmq_x);
  constexpr int rows_per_warp = 2 * granularity;
  constexpr int ntx =
      rows_per_warp / tile_C::I; // Number of x minitiles per warp.

  const int i0 = (threadIdx.y / ntx) * (ntx * tile_C::I);
  static_assert(nwarps * tile_C::I == mmq_y, "nwarps*tile_C::I != mmq_y");

#pragma unroll
  for (int j0 = 0; j0 < mmq_x; j0 += ntx * tile_C::J) {
#pragma unroll
    for (int n = 0; n < ntx; ++n) {
#pragma unroll
      for (int l = 0; l < tile_C::ne; ++l) {
        const int j = j0 + (threadIdx.y % ntx) * tile_C::J + tile_C::get_j(l);

        if (j > j_max) {
          continue;
        }

        const int i = i0 + n * tile_C::I + tile_C::get_i(l);

        if (need_check && i > i_max) {
          continue;
        }

        dst[ids_dst[j] * stride + i] =
            sum[(j0 / tile_C::J + n) * tile_C::ne + l];
      }
    }
  }
}

template <int mmq_x, int mmq_y, int nwarps, bool need_check>
struct mmq_type_traits {
  static constexpr int vdr = VDR_Q4_0_Q8_1_MMQ;
  static constexpr load_tiles_mmq_t load_tiles =
      load_tiles_q4_0<mmq_y, nwarps, need_check>;
  static constexpr vec_dot_mmq_t vec_dot_mma =
      vec_dot_q8_0_q8_1_mma<mmq_x, mmq_y, nwarps>;
};

template <int mmq_x, int nwarps, bool need_check, bool fixup>
static __device__ __forceinline__ void mul_mat_q_process_tile(
    const char *__restrict__ x, const int offset_x, const int *__restrict__ y,
    const int *__restrict__ ids_dst, float *__restrict__ dst,
    float *__restrict__ tmp_fixup, const int stride_row_x, const int ncols_y,
    const int stride_col_dst, const int tile_x_max_i, const int tile_y_max_j,
    const int kb0_start, const int kb0_stop) {

  constexpr int qk = QK4_0;
  constexpr load_tiles_mmq_t load_tiles =
      mmq_type_traits<mmq_x, MMQ_Y, N_WARPS, need_check>::load_tiles;

  extern __shared__ int data_mul_mat_q[];
  int *tile_y = data_mul_mat_q + mmq_x;
  int *tile_x = tile_y + PAD(mmq_x * (WARP_SIZE + WARP_SIZE / QI8_1),
                             N_WARPS * WARP_SIZE);

  constexpr vec_dot_mmq_t vec_dot =
      mmq_type_traits<mmq_x, MMQ_Y, N_WARPS, need_check>::vec_dot_mma;
  constexpr mmq_write_back_t write_back =
      mmq_write_back_mma<mmq_x, MMQ_Y, N_WARPS, need_check>;

  constexpr int blocks_per_iter = MMQ_ITER_K / qk;

  float sum[mmq_x * MMQ_Y / (N_WARPS * WARP_SIZE)] = {0.0f};

  for (int kb0 = kb0_start; kb0 < kb0_stop; kb0 += blocks_per_iter) {
    load_tiles(x, tile_x, offset_x + kb0, tile_x_max_i, stride_row_x);

    {
      const int *by0 = y + ncols_y * (kb0 * (qk * sizeof(block_q8_1_mmq) /
                                             (4 * QK8_1 * sizeof(int))) +
                                      0 * sizeof(block_q8_1_mmq) / sizeof(int));
      _Pragma("unroll") for (int l0 = 0; l0 < mmq_x * MMQ_TILE_Y_K;
                             l0 += N_WARPS * WARP_SIZE) {
        int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;

        tile_y[l] = by0[l];
      }
    }

    __syncthreads();

    vec_dot(tile_x, tile_y, sum, 0);

    __syncthreads();

    {
      const int *by0 = y + ncols_y * (kb0 * (qk * sizeof(block_q8_1_mmq) /
                                             (4 * QK8_1 * sizeof(int))) +
                                      1 * sizeof(block_q8_1_mmq) / sizeof(int));
      _Pragma("unroll") for (int l0 = 0; l0 < mmq_x * MMQ_TILE_Y_K;
                             l0 += N_WARPS * WARP_SIZE) {
        int l = l0 + threadIdx.y * WARP_SIZE + threadIdx.x;

        tile_y[l] = by0[l];
      }
    }

    __syncthreads();

    vec_dot(tile_x, tile_y, sum, WARP_SIZE);

    __syncthreads();
  }

  if (fixup) {
    write_back(sum, ids_dst, tmp_fixup + blockIdx.x * (mmq_x * MMQ_Y), MMQ_Y,
               MMQ_Y, mmq_x);
  } else {
    write_back(sum, ids_dst, dst, stride_col_dst, tile_x_max_i, tile_y_max_j);
  }
}

// The mul_mat_q kernel implements "stream-k" work partitioning as described in
// https://arxiv.org/abs/2301.03598

template <int mmq_x, int nwarps, bool need_check>
static __device__ __forceinline__ void
mul_mat_q(const char *__restrict__ x, const int *__restrict__ y,
          float *__restrict__ dst, float *__restrict__ tmp_fixup,
          const int ncols_x, const int nrows_x, const int ncols_dst,
          const int stride_row_x, const int ncols_y, const int stride_col_dst,
          const int channel_ratio, const int nchannels_y,
          const int stride_channel_x, const int stride_channel_y,
          const int stride_channel_dst) {
  constexpr int qk = QK4_0;

  const int ntx = (ncols_dst + mmq_x - 1) / mmq_x; // Number of tiles x
  const int nty = (nrows_x + MMQ_Y - 1) / MMQ_Y;   // Number of tiles y

  // Initialize the ids for writing back data with just the index.
  // For regular matrix multiplications this is never changed.
  // For MoE the correct indices are loaded from ids_dst.
  extern __shared__ int
      ids_dst_shared[]; // Stored at beginning of shared memory.
  _Pragma("unroll") for (int j0 = 0; j0 < mmq_x; j0 += nwarps * WARP_SIZE) {
    const int j = j0 + threadIdx.y * WARP_SIZE + threadIdx.x;

    if (j0 + nwarps * WARP_SIZE > mmq_x && j >= mmq_x) {
      break;
    }

    ids_dst_shared[j] = j;
  }
  __syncthreads();

  const int64_t blocks_per_ne00 = ncols_x / qk;
  constexpr int blocks_per_iter = MMQ_ITER_K / qk;

  // kbc == k block continuous, current index in continuous ijk space.
  int64_t kbc = (int64_t)blockIdx.x * nchannels_y * ntx * nty *
                blocks_per_ne00 / gridDim.x;
  int64_t kbc_stop = (int64_t)(blockIdx.x + 1) * nchannels_y * ntx * nty *
                     blocks_per_ne00 / gridDim.x;

  kbc -= (kbc % blocks_per_ne00) % blocks_per_iter;
  kbc_stop -= (kbc_stop % blocks_per_ne00) % blocks_per_iter;

  // kb0 == k index when doing the matrix multiplication for an output tile.
  int kb0_start = kbc % blocks_per_ne00;
  int kb0_stop = min(blocks_per_ne00, kb0_start + kbc_stop - kbc);

  while (kbc < kbc_stop && kb0_stop == blocks_per_ne00) {
    int tmp = kbc;
    const int it = tmp / (nchannels_y * ntx * blocks_per_ne00);
    tmp -= it * (nchannels_y * ntx * blocks_per_ne00);
    const int zt = tmp / (ntx * blocks_per_ne00);
    tmp -= zt * (ntx * blocks_per_ne00);
    const int jt = tmp / blocks_per_ne00;

    // Defaults for regular matrix multiplication:
    int offset_y = zt * stride_channel_y;
    int offset_dst = zt * stride_channel_dst + jt * mmq_x * stride_col_dst;

    offset_y += jt * mmq_x * (sizeof(block_q8_1_mmq) / sizeof(int));
    offset_dst += it * MMQ_Y;

    const int tile_x_max_i = nrows_x - it * MMQ_Y - 1;
    const int tile_y_max_j = ncols_dst - jt * mmq_x - 1;

    const int offset_x =
        (zt / channel_ratio) * stride_channel_x + it * MMQ_Y * stride_row_x;

    constexpr bool fixup =
        false; // All but (potentially) the last iterations write their data to
               // dst rather than the fixup buffer.
    mul_mat_q_process_tile<mmq_x, nwarps, need_check, fixup>(
        x, offset_x, y + offset_y, ids_dst_shared, dst + offset_dst, tmp_fixup,
        stride_row_x, ncols_y, stride_col_dst, tile_x_max_i, tile_y_max_j,
        kb0_start, kb0_stop);

    kbc += blocks_per_ne00;
    kbc -= kbc % blocks_per_ne00;

    kb0_start = 0;
    kb0_stop = min(blocks_per_ne00, kbc_stop - kbc);
  }

  if (kbc >= kbc_stop) {
    return;
  }

  int tmp = kbc;
  const int it = tmp / (nchannels_y * ntx * blocks_per_ne00);
  tmp -= it * (nchannels_y * ntx * blocks_per_ne00);
  const int zt = tmp / (ntx * blocks_per_ne00);
  tmp -= zt * (ntx * blocks_per_ne00);
  const int jt = tmp / blocks_per_ne00;

  // Defaults for regular matrix multiplication:
  int offset_y = zt * stride_channel_y;
  int offset_dst = zt * stride_channel_dst + jt * mmq_x * stride_col_dst;

  offset_y += jt * mmq_x * (sizeof(block_q8_1_mmq) / sizeof(int));
  offset_dst += it * MMQ_Y;

  const int tile_x_max_i = nrows_x - it * MMQ_Y - 1;
  const int tile_y_max_j = ncols_dst - jt * mmq_x - 1;

  const int offset_x =
      (zt / channel_ratio) * stride_channel_x + it * MMQ_Y * stride_row_x;

  constexpr bool fixup = true; // Last index writes its data to fixup buffer to
                               // avoid data races with other blocks.
  mul_mat_q_process_tile<mmq_x, nwarps, need_check, fixup>(
      x, offset_x, y + offset_y, ids_dst_shared, dst + offset_dst, tmp_fixup,
      stride_row_x, ncols_y, stride_col_dst, tile_x_max_i, tile_y_max_j,
      kb0_start, kb0_stop);
}

template <int mmq_x, int nwarps, bool need_check>
static __device__ __forceinline__ void
mul_mat_q_stream_k_fixup(float *__restrict__ dst,
                         const float *__restrict__ tmp_last_tile,
                         const int ncols_x, const int nrows_x,
                         const int ncols_dst, const int stride_col_dst,
                         const int nchannels_y, const int stride_channel_dst) {
  constexpr int qk = QK4_0;
  constexpr int blocks_per_iter = MMQ_ITER_K / qk;
  const int64_t blocks_per_ne00 = ncols_x / qk;

  float sum[mmq_x * MMQ_Y / (nwarps * WARP_SIZE)] = {0.0f};

  const int ntx = (ncols_dst + mmq_x - 1) / mmq_x;
  const int nty = (nrows_x + MMQ_Y - 1) / MMQ_Y;

  const int bidx0 = blockIdx.x;

  // kbc == k block continuous, current index in continuous ijk space.
  int64_t kbc0 =
      (int64_t)bidx0 * nchannels_y * ntx * nty * blocks_per_ne00 / gridDim.x;
  int64_t kbc0_stop = (int64_t)(bidx0 + 1) * nchannels_y * ntx * nty *
                      blocks_per_ne00 / gridDim.x;

  kbc0 -= (kbc0 % blocks_per_ne00) % blocks_per_iter;
  kbc0_stop -= (kbc0_stop % blocks_per_ne00) % blocks_per_iter;

  const bool did_not_have_any_data = kbc0 == kbc0_stop;
  const bool wrote_beginning_of_tile = kbc0 % blocks_per_ne00 == 0;
  const bool did_not_write_last =
      kbc0 / blocks_per_ne00 == kbc0_stop / blocks_per_ne00 &&
      kbc0_stop % blocks_per_ne00 != 0;
  if (did_not_have_any_data || wrote_beginning_of_tile || did_not_write_last) {
    return;
  }

  bool any_fixup = false;

  // Iterate over previous blocks and sum up partial sums written to fixup
  // buffer. All CUDA blocks that get here must have a previous block that needs
  // a fixup.
  int64_t bidx = bidx0 - 1;
  int64_t kbc_stop = kbc0;
  while (true) {
    int64_t kbc = bidx * nchannels_y * ntx * nty * blocks_per_ne00 / gridDim.x;
    kbc -= (kbc % blocks_per_ne00) % blocks_per_iter;

    if (kbc == kbc_stop) { // Did not have any data.
      bidx--;
      kbc_stop = kbc;
      continue;
    }

    any_fixup = true;

    _Pragma("unroll") for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
      const int j = j0 + threadIdx.y;

      _Pragma("unroll") for (int i0 = 0; i0 < MMQ_Y; i0 += WARP_SIZE) {
        const int i = i0 + threadIdx.x;

        sum[(j0 / nwarps) * (MMQ_Y / WARP_SIZE) + i0 / WARP_SIZE] +=
            tmp_last_tile[bidx * (mmq_x * MMQ_Y) + j * MMQ_Y + i];
      }
    }

    // If this block started in a previous tile we are done and don't need to
    // combine additional partial results.
    if (kbc % blocks_per_ne00 == 0 ||
        kbc / blocks_per_ne00 < kbc0 / blocks_per_ne00) {
      break;
    }
    bidx--;
    kbc_stop = kbc;
  }

  if (!any_fixup) {
    return;
  }

  int tmp = kbc0;
  const int it = tmp / (nchannels_y * ntx * blocks_per_ne00);
  tmp -= it * (nchannels_y * ntx * blocks_per_ne00);
  const int zt = tmp / (ntx * blocks_per_ne00);
  tmp -= zt * (ntx * blocks_per_ne00);
  const int jt = tmp / blocks_per_ne00;

  const int offset_dst =
      zt * stride_channel_dst + jt * mmq_x * stride_col_dst + it * MMQ_Y;
  dst += offset_dst;

  const int i_max = nrows_x - it * MMQ_Y - 1;
  const int j_max = ncols_dst - jt * mmq_x - 1;

  _Pragma("unroll") for (int j0 = 0; j0 < mmq_x; j0 += nwarps) {
    const int j = j0 + threadIdx.y;

    if (j > j_max) {
      return;
    }

    _Pragma("unroll") for (int i0 = 0; i0 < MMQ_Y; i0 += WARP_SIZE) {
      const int i = i0 + threadIdx.x;

      if (need_check && i > i_max) {
        continue;
      }

      dst[j * stride_col_dst + i] +=
          sum[(j0 / nwarps) * (MMQ_Y / WARP_SIZE) + i0 / WARP_SIZE];
    }
  }
}

#define INSTANTIATE_MMQ_KERNEL(mmq_x, nwarps, need_check)                      \
  extern "C" {                                                                 \
  __launch_bounds__(WARP_SIZE *nwarps, 1) __global__ void mul_mat_q40##_       \
      ##mmq_x##_##nwarps##_                                                    \
      ##need_check(const char *__restrict__ x, const int *__restrict__ y,      \
                   float *__restrict__ dst, float *__restrict__ tmp_fixup,     \
                   const int ncols_x, const int nrows_x, const int ncols_dst,  \
                   const int stride_row_x, const int ncols_y,                  \
                   const int stride_col_dst, const int channel_ratio,          \
                   const int nchannels_y, const int stride_channel_x,          \
                   const int stride_channel_y, const int stride_channel_dst) { \
    mul_mat_q<mmq_x, nwarps, need_check>(                                      \
        x, y, dst, tmp_fixup, ncols_x, nrows_x, ncols_dst, stride_row_x,       \
        ncols_y, stride_col_dst, channel_ratio, nchannels_y, stride_channel_x, \
        stride_channel_y, stride_channel_dst);                                 \
  }                                                                            \
                                                                               \
  __global__ void                                                              \
      mul_mat_q40_stream_k_fixup##_##mmq_x##_##nwarps##_##need_check(          \
          float *__restrict__ dst, const float *__restrict__ tmp_last_tile,    \
          const int ncols_x, const int nrows_x, const int ncols_dst,           \
          const int stride_col_dst, const int nchannels_y,                     \
          const int stride_channel_dst) {                                      \
    mul_mat_q_stream_k_fixup<mmq_x, nwarps, need_check>(                       \
        dst, tmp_last_tile, ncols_x, nrows_x, ncols_dst, stride_col_dst,       \
        nchannels_y, stride_channel_dst);                                      \
  }                                                                            \
  }

#define INSTANTIATE_MMQ_KERNEL_FOR_T(n_warps, needs_check)                     \
  INSTANTIATE_MMQ_KERNEL(8, n_warps, needs_check)                              \
  INSTANTIATE_MMQ_KERNEL(16, n_warps, needs_check)                             \
  INSTANTIATE_MMQ_KERNEL(24, n_warps, needs_check)                             \
  INSTANTIATE_MMQ_KERNEL(32, n_warps, needs_check)                             \
  INSTANTIATE_MMQ_KERNEL(40, n_warps, needs_check)                             \
  INSTANTIATE_MMQ_KERNEL(48, n_warps, needs_check)                             \
  INSTANTIATE_MMQ_KERNEL(64, n_warps, needs_check)                             \
  INSTANTIATE_MMQ_KERNEL(80, n_warps, needs_check)                             \
  INSTANTIATE_MMQ_KERNEL(96, n_warps, needs_check)                             \
  INSTANTIATE_MMQ_KERNEL(112, n_warps, needs_check)                            \
  INSTANTIATE_MMQ_KERNEL(128, n_warps, needs_check)

INSTANTIATE_MMQ_KERNEL_FOR_T(N_WARPS, true)
INSTANTIATE_MMQ_KERNEL_FOR_T(N_WARPS, false)

static constexpr __host__ __device__ int calc_nwarps(int ncols_dst) {
  switch (ncols_dst) {
  case 1:
  case 2:
  case 3:
  case 4:
    return 4;
  case 5:
  case 6:
  case 7:
  case 8:
    return 2;
  default:
    return 1;
  }
}

template <int vdr>
static __device__ __forceinline__ float
vec_dot_q4_0_q8_1_impl(const int *v, const int *u, const float &d4,
                       const half2 &ds8) {

  int sumi = 0;

#pragma unroll
  for (int i = 0; i < vdr; ++i) {
    const int vi0 = (v[i] >> 0) & 0x0F0F0F0F;
    const int vi1 = (v[i] >> 4) & 0x0F0F0F0F;

    // SIMD dot product of quantized values
    sumi = __dp4a(vi0, u[2 * i + 0], sumi);
    sumi = __dp4a(vi1, u[2 * i + 1], sumi);
  }

  const float2 ds8f = __half22float2(ds8);

  // second part effectively subtracts 8 from each quant value
  return d4 * (sumi * ds8f.x - (8 * vdr / QI4_0) * ds8f.y);
}

static __device__ __forceinline__ float
vec_dot_q4_0_q8_1(const void *__restrict__ vbq,
                  const block_q8_1 *__restrict__ bq8_1, const int &kbx,
                  const int &iqs) {

  const block_q4_0 *bq4_0 = (const block_q4_0 *)vbq + kbx;

  int v[VDR_Q4_0_Q8_1_MMVQ];
  int u[2 * VDR_Q4_0_Q8_1_MMVQ];

#pragma unroll
  for (int i = 0; i < VDR_Q4_0_Q8_1_MMVQ; ++i) {
    v[i] = get_int_b2(bq4_0->qs, iqs + i);
    u[2 * i + 0] = get_int_b4(bq8_1->qs, iqs + i);
    u[2 * i + 1] = get_int_b4(bq8_1->qs, iqs + i + QI4_0);
  }

  return vec_dot_q4_0_q8_1_impl<VDR_Q4_0_Q8_1_MMVQ>(v, u, bq4_0->d, bq8_1->ds);
}

template <int ncols_dst>
static __device__ void
mul_mat_vec_q(const void *__restrict__ vx, const void *__restrict__ vy,
              float *__restrict__ dst, const int ncols_x, const int nchannels_y,
              const int stride_row_x, const int stride_col_y,
              const int stride_col_dst, const int channel_ratio,
              const int stride_channel_x, const int stride_channel_y,
              const int stride_channel_dst) {

  constexpr int qk = QK4_0;
  constexpr int qi = QI4_0;
  constexpr int vdr = VDR_Q4_0_Q8_1_MMVQ;
  constexpr int nwarps = calc_nwarps(ncols_dst);
  constexpr int rows_per_cuda_block = ncols_dst == 1 ? 1 : 2;

  const int tid = WARP_SIZE * threadIdx.y + threadIdx.x;
  const int row0 = rows_per_cuda_block * blockIdx.x;
  const int blocks_per_row_x = ncols_x / qk;
  constexpr int blocks_per_iter = vdr * nwarps * WARP_SIZE / qi;

  const int channel_dst = blockIdx.y;
  const int channel_x = channel_dst / channel_ratio;
  const int channel_y = channel_dst;

  // partial sum for each thread
  float tmp[ncols_dst][rows_per_cuda_block] = {{0.0f}};

  const block_q8_1 *y = ((const block_q8_1 *)vy) + channel_y * stride_channel_y;
  const int kbx_offset = channel_x * stride_channel_x;

  for (int kbx = tid / (qi / vdr); kbx < blocks_per_row_x;
       kbx += blocks_per_iter) {
    const int kby = kbx * (qk / QK8_1); // y block index that aligns with kbx

    // x block quant index when casting the quants to int
    const int kqs = vdr * (tid % (qi / vdr));

#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
      for (int i0 = 0; i0 < rows_per_cuda_block; ++i0) {
        int i = i0 + row0;
        if (i >= stride_col_dst) { // Avoid OOB read from Q4_0
          break;
        }
        tmp[j][i0] +=
            vec_dot_q4_0_q8_1(vx, &y[j * stride_col_y + kby],
                              kbx_offset + i * stride_row_x + kbx, kqs);
      }
    }
  }

  __shared__ float tmp_shared[nwarps - 1 > 0 ? nwarps - 1 : 1][ncols_dst]
                             [rows_per_cuda_block][WARP_SIZE];
  if (threadIdx.y > 0) {
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
      for (int i = 0; i < rows_per_cuda_block; ++i) {
        tmp_shared[threadIdx.y - 1][j][i][threadIdx.x] = tmp[j][i];
      }
    }
  }
  __syncthreads();
  if (threadIdx.y > 0) {
    return;
  }

  dst += channel_dst * stride_channel_dst + row0;

  // sum up partial sums and write back result
#pragma unroll
  for (int j = 0; j < ncols_dst; ++j) {
#pragma unroll
    for (int i = 0; i < rows_per_cuda_block; ++i) {
#pragma unroll
      for (int l = 0; l < nwarps - 1; ++l) {
        tmp[j][i] += tmp_shared[l][j][i][threadIdx.x];
      }
      tmp[j][i] = warp_reduce_sum<WARP_SIZE>(tmp[j][i]);
    }

    if (threadIdx.x < rows_per_cuda_block &&
        (rows_per_cuda_block == 1 ||
         row0 + int(threadIdx.x) < stride_col_dst)) {
      dst[j * stride_col_dst + threadIdx.x] = tmp[j][threadIdx.x];
    }
  }
}

#define INSTANTIATE_MMQV_KERNEL(ncols_dst)                                     \
  extern "C" {                                                                 \
  __launch_bounds__(calc_nwarps(ncols_dst) * WARP_SIZE, 1) __global__          \
      void mul_vec_q40_m_                                                      \
      ##ncols_dst(const void *__restrict__ vx, const void *__restrict__ vy,    \
                  float *__restrict__ dst, const int ncols_x,                  \
                  const int nchannels_y, const int stride_row_x,               \
                  const int stride_col_y, const int stride_col_dst,            \
                  const int channel_ratio, const int stride_channel_x,         \
                  const int stride_channel_y, const int stride_channel_dst) {  \
    mul_mat_vec_q<ncols_dst>(vx, vy, dst, ncols_x, nchannels_y, stride_row_x,  \
                             stride_col_y, stride_col_dst, channel_ratio,      \
                             stride_channel_x, stride_channel_y,               \
                             stride_channel_dst);                              \
  }                                                                            \
  }

INSTANTIATE_MMQV_KERNEL(1)
INSTANTIATE_MMQV_KERNEL(2)
INSTANTIATE_MMQV_KERNEL(3)
INSTANTIATE_MMQV_KERNEL(4)
INSTANTIATE_MMQV_KERNEL(5)
INSTANTIATE_MMQV_KERNEL(6)
INSTANTIATE_MMQV_KERNEL(7)
INSTANTIATE_MMQV_KERNEL(8)
