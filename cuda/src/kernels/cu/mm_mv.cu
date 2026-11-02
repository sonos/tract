
#include <cuda_runtime.h>
#include "common.cuh"

template <typename T, int ncols_dst, int block_size>
static __device__ void
mul_mat_vec(const T *__restrict__ x, const T *__restrict__ y,
            T *__restrict__ dst, const int32_t ncols2, const int32_t nchannels_y,
            const int32_t stride_row, const int32_t stride_col_y2,
            const int32_t stride_col_dst, const int32_t channel_ratio,
            const int32_t stride_channel_x, const int32_t stride_channel_y,
            const int32_t stride_channel_dst,
            const int32_t *__restrict__ x_ring,
            const int32_t ring_rotates_k) {
  const int row = blockIdx.x;
  const int channel_dst = blockIdx.y;
  const int channel_x = channel_dst / channel_ratio;
  const int channel_y = channel_dst;
  const int tid = threadIdx.x;

  // How far this channel's window has turned, in pairs of columns, when the
  // ring turns along the contracted axis rather than along the rows.
  int ring_off2 = 0;
  if (x_ring == nullptr) {
    x += channel_x * stride_channel_x + row * stride_row;
  } else if (ring_rotates_k) {
    // x holds each channel's contracted axis as a ring of whole slots, and more
    // channels than the grid covers: the table gives how far the window has
    // turned and where this channel's rows start.
    ring_off2 = x_ring[2 * channel_dst];
    x += (x_ring[2 * channel_dst + 1] + row) * stride_row;
  } else {
    // x holds each channel's gridDim.x rows as a ring, and more channels than
    // the grid covers: the table gives this channel's first row and where its
    // rows start.
    int row_x = row + x_ring[2 * channel_dst];
    if (row_x >= (int)gridDim.x) {
      row_x -= (int)gridDim.x;
    }
    x += (x_ring[2 * channel_dst + 1] + row_x) * stride_row;
  }
  y += channel_y * stride_channel_y;
  dst += channel_dst * stride_channel_dst;

  extern __shared__ char data_mmv[];
  float *buf_iw = (float *)data_mmv;

  if (block_size > WARP_SIZE) {
    if (tid < WARP_SIZE) {
      buf_iw[tid] = 0.0f;
    }
    __syncthreads();
  }

  float sumf[ncols_dst] = {0.0f};

  if constexpr (cuda::std::is_same_v<T, float>) {
    const float2 *x2 = (const float2 *)x;
    const float2 *y2 = (const float2 *)y;
    for (int col2 = tid; col2 < ncols2; col2 += block_size) {
      int col2_x = col2 + ring_off2;
      if (col2_x >= ncols2) {
        col2_x -= ncols2;
      }
      const float2 tmpx = x2[col2_x];

#pragma unroll
      for (int j = 0; j < ncols_dst; ++j) {
        const float2 tmpy = y2[j * stride_col_y2 + col2];
        sumf[j] += tmpx.x * tmpy.x;
        sumf[j] += tmpx.y * tmpy.y;
      }
    }
  } else if constexpr (cuda::std::is_same_v<T, half>) {
    const half2 *x2 = (const half2 *)x;
    const half2 *y2 = (const half2 *)y;
    half2 sumh2[ncols_dst] = {{0.0f, 0.0f}};

    for (int col2 = tid; col2 < ncols2; col2 += block_size) {
      int col2_x = col2 + ring_off2;
      if (col2_x >= ncols2) {
        col2_x -= ncols2;
      }
      const half2 tmpx = x2[col2_x];

#pragma unroll
      for (int j = 0; j < ncols_dst; ++j) {
        const half2 tmpy = y2[j * stride_col_y2 + col2];
        sumh2[j] += tmpx * make_half2(tmpy.x, tmpy.y);
      }
    }

#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
      sumf[j] = __low2float(sumh2[j]) + __high2float(sumh2[j]);
    }
  } else {
    static_assert(cuda::std::is_same_v<T, void>, "unsupported type");
  }

#pragma unroll
  for (int j = 0; j < ncols_dst; ++j) {
    sumf[j] = warp_reduce_sum<WARP_SIZE>(sumf[j]);

    if (block_size > WARP_SIZE) {
      buf_iw[tid / WARP_SIZE] = sumf[j];
      __syncthreads();
      if (tid < WARP_SIZE) {
        sumf[j] = buf_iw[tid];
        sumf[j] = warp_reduce_sum<WARP_SIZE>(sumf[j]);
      }
      if (j < ncols_dst) {
        __syncthreads();
      }
    }
  }

  if (tid >= ncols_dst) {
    return;
  }

  dst[tid * stride_col_dst + row] = sumf[tid];
}

#define INSTANTIATE_MAT_VEC(type_name, T, ncols_dst, block_size)               \
  extern "C" __global__ void                                                   \
      ggml_matvec_##type_name##_ncols_##ncols_dst##_bs_##block_size(           \
          const T *__restrict__ x, const T *__restrict__ y,                    \
          T *__restrict__ dst, const int32_t ncols2, const int32_t nchannels_y,        \
          const int32_t stride_row, const int32_t stride_col_y2,                       \
          const int32_t stride_col_dst, const int32_t channel_ratio,                   \
          const int32_t stride_channel_x, const int32_t stride_channel_y,              \
          const int32_t stride_channel_dst,                                    \
          const int32_t *__restrict__ x_ring,                                  \
          const int32_t ring_rotates_k) {                                      \
    mul_mat_vec<T, ncols_dst, block_size>(                                     \
        x, y, dst, ncols2, nchannels_y, stride_row, stride_col_y2,             \
        stride_col_dst, channel_ratio, stride_channel_x, stride_channel_y,     \
        stride_channel_dst, x_ring, ring_rotates_k);                           \
  }

#define INSTANTIATE_MAT_VEC_FOR_BS(name, T, blocksize)                         \
  INSTANTIATE_MAT_VEC(name, T, 1, blocksize)                                   \
  INSTANTIATE_MAT_VEC(name, T, 2, blocksize)                                   \
  INSTANTIATE_MAT_VEC(name, T, 3, blocksize)                                   \
  INSTANTIATE_MAT_VEC(name, T, 4, blocksize)                                   \
  INSTANTIATE_MAT_VEC(name, T, 5, blocksize)                                   \
  INSTANTIATE_MAT_VEC(name, T, 6, blocksize)                                   \
  INSTANTIATE_MAT_VEC(name, T, 7, blocksize)                                   \
  INSTANTIATE_MAT_VEC(name, T, 8, blocksize)

#define INSTANTIATE_MAT_VEC_FOR_T(name, T)                                     \
  INSTANTIATE_MAT_VEC_FOR_BS(name, T, 32)                                      \
  INSTANTIATE_MAT_VEC_FOR_BS(name, T, 64)                                      \
  INSTANTIATE_MAT_VEC_FOR_BS(name, T, 96)                                      \
  INSTANTIATE_MAT_VEC_FOR_BS(name, T, 128)                                     \
  INSTANTIATE_MAT_VEC_FOR_BS(name, T, 160)                                     \
  INSTANTIATE_MAT_VEC_FOR_BS(name, T, 192)                                     \
  INSTANTIATE_MAT_VEC_FOR_BS(name, T, 224)                                     \
  INSTANTIATE_MAT_VEC_FOR_BS(name, T, 256)

INSTANTIATE_MAT_VEC_FOR_T(f32, float)
INSTANTIATE_MAT_VEC_FOR_T(f16, half)

// dst[m, n] = y[m, k] . x[n, k]^T for a few activation rows (ncols_dst): each
// block computes rows_per_block rows of x, so every activation load serves
// all of them.
template <int ncols_dst, int rows_per_block, int block_size>
static __device__ void
mul_mat_skinny_f32(const float *__restrict__ x, const float *__restrict__ y,
                   float *__restrict__ dst, const int32_t ncols2,
                   const int32_t nrows, const int32_t stride_row,
                   const int32_t stride_col_y2, const int32_t stride_col_dst,
                   const int32_t channel_ratio, const int32_t stride_channel_x,
                   const int32_t stride_channel_y,
                   const int32_t stride_channel_dst) {
  const int row0 = blockIdx.x * rows_per_block;
  const int channel_dst = blockIdx.y;
  const int tid = threadIdx.x;
  x += (int64_t)(channel_dst / channel_ratio) * stride_channel_x;
  y += (int64_t)channel_dst * stride_channel_y;
  dst += (int64_t)channel_dst * stride_channel_dst;

  const float2 *y2 = (const float2 *)y;
  const float2 *x2[rows_per_block];
#pragma unroll
  for (int r = 0; r < rows_per_block; ++r) {
    x2[r] = (const float2 *)(x + (int64_t)min(row0 + r, nrows - 1) * stride_row);
  }

  float sumf[rows_per_block][ncols_dst] = {{0.0f}};
  for (int col2 = tid; col2 < ncols2; col2 += block_size) {
    float2 tmpy[ncols_dst];
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
      tmpy[j] = y2[j * stride_col_y2 + col2];
    }
#pragma unroll
    for (int r = 0; r < rows_per_block; ++r) {
      const float2 tmpx = x2[r][col2];
#pragma unroll
      for (int j = 0; j < ncols_dst; ++j) {
        sumf[r][j] += tmpx.x * tmpy[j].x;
        sumf[r][j] += tmpx.y * tmpy[j].y;
      }
    }
  }

  constexpr int nwarps = block_size / WARP_SIZE;
  __shared__ float partial[nwarps][rows_per_block * ncols_dst];
#pragma unroll
  for (int r = 0; r < rows_per_block; ++r) {
#pragma unroll
    for (int j = 0; j < ncols_dst; ++j) {
      const float s = warp_reduce_sum<WARP_SIZE>(sumf[r][j]);
      if (tid % WARP_SIZE == 0) {
        partial[tid / WARP_SIZE][r * ncols_dst + j] = s;
      }
    }
  }
  __syncthreads();
  for (int i = tid; i < rows_per_block * ncols_dst; i += block_size) {
    const int r = i / ncols_dst;
    const int j = i % ncols_dst;
    if (row0 + r < nrows) {
      float s = 0.0f;
#pragma unroll
      for (int w = 0; w < nwarps; ++w) {
        s += partial[w][i];
      }
      dst[j * stride_col_dst + row0 + r] = s;
    }
  }
}

#define INSTANTIATE_SKINNY(ncols_dst, rows_per_block, block_size)              \
  extern "C" __global__ void __launch_bounds__(block_size)                     \
      ggml_skinny_f32_ncols_##ncols_dst##_rows_##rows_per_block(               \
          const float *__restrict__ x, const float *__restrict__ y,            \
          float *__restrict__ dst, const int32_t ncols2, const int32_t nrows,  \
          const int32_t stride_row, const int32_t stride_col_y2,               \
          const int32_t stride_col_dst, const int32_t channel_ratio,           \
          const int32_t stride_channel_x, const int32_t stride_channel_y,      \
          const int32_t stride_channel_dst) {                                  \
    mul_mat_skinny_f32<ncols_dst, rows_per_block, block_size>(                 \
        x, y, dst, ncols2, nrows, stride_row, stride_col_y2, stride_col_dst,   \
        channel_ratio, stride_channel_x, stride_channel_y,                     \
        stride_channel_dst);                                                   \
  }

INSTANTIATE_SKINNY(9, 2, 128)
INSTANTIATE_SKINNY(10, 2, 128)
INSTANTIATE_SKINNY(11, 2, 128)
INSTANTIATE_SKINNY(12, 2, 128)
INSTANTIATE_SKINNY(13, 2, 128)
INSTANTIATE_SKINNY(14, 2, 128)
INSTANTIATE_SKINNY(15, 2, 128)
INSTANTIATE_SKINNY(16, 2, 128)
