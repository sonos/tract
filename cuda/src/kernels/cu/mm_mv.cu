
#include <cuda_runtime.h>
#include "common.cuh"

template <typename T, int ncols_dst, int block_size>
static __device__ void
mul_mat_vec(const T *__restrict__ x, const T *__restrict__ y,
            T *__restrict__ dst, const int ncols2, const int nchannels_y,
            const int stride_row, const int stride_col_y2,
            const int stride_col_dst, const int channel_ratio,
            const int stride_channel_x, const int stride_channel_y,
            const int stride_channel_dst) {
  const int row = blockIdx.x;
  const int channel_dst = blockIdx.y;
  const int channel_x = channel_dst / channel_ratio;
  const int channel_y = channel_dst;
  const int tid = threadIdx.x;

  x += channel_x * stride_channel_x + row * stride_row;
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
      const float2 tmpx = x2[col2];

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
      const half2 tmpx = x2[col2];

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
          T *__restrict__ dst, const int ncols2, const int nchannels_y,        \
          const int stride_row, const int stride_col_y2,                       \
          const int stride_col_dst, const int channel_ratio,                   \
          const int stride_channel_x, const int stride_channel_y,              \
          const int stride_channel_dst) {                                      \
    mul_mat_vec<T, ncols_dst, block_size>(                                     \
        x, y, dst, ncols2, nchannels_y, stride_row, stride_col_y2,             \
        stride_col_dst, channel_ratio, stride_channel_x, stride_channel_y,     \
        stride_channel_dst);                                                   \
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
