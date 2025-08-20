#include <cuda_fp16.h>

#define WARP_SIZE 32

template <int width = WARP_SIZE>
static __device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
  for (int offset = width / 2; offset > 0; offset >>= 1) {
    x += __shfl_xor_sync(0xffffffff, x, offset, width);
  }
  return x;
}

template <int width = WARP_SIZE>
static __device__ __forceinline__ __half warp_reduce_sum(__half x) {
#pragma unroll
  for (int offset = width / 2; offset > 0; offset >>= 1) {
    x += __shfl_xor_sync(0xffffffff, x, offset, width);
  }
  return x;
}

template <int width = WARP_SIZE>
static __device__ __forceinline__ float warp_reduce_max(float x) {
#pragma unroll
  for (int offset = width / 2; offset > 0; offset >>= 1) {
    x = fmaxf(x, __shfl_xor_sync(0xffffffff, x, offset, width));
  }
  return x;
}

template <int width = WARP_SIZE>
static __device__ __forceinline__ __half warp_reduce_max(__half x) {
#pragma unroll
  for (int offset = width / 2; offset > 0; offset >>= 1) {
    x = __hmax(x, __shfl_xor_sync(0xffffffff, x, offset, width));
  }
  return x;
}

template <int width = WARP_SIZE>
static __device__ __forceinline__ float warp_reduce_min(float x) {
#pragma unroll
  for (int offset = width / 2; offset > 0; offset >>= 1) {
    x = fminf(x, __shfl_xor_sync(0xffffffff, x, offset, width));
  }
  return x;
}

template <int width = WARP_SIZE>
static __device__ __forceinline__ __half warp_reduce_min(__half x) {
#pragma unroll
  for (int offset = width / 2; offset > 0; offset >>= 1) {
    x = __hmin(x, __shfl_xor_sync(0xffffffff, x, offset, width));
  }
  return x;
}

template <int width = WARP_SIZE>
static __device__ __forceinline__ float warp_reduce_prod(float x) {
#pragma unroll
  for (int offset = width / 2; offset > 0; offset >>= 1) {
    x *= __shfl_xor_sync(0xffffffff, x, offset, width);
  }
  return x;
}

template <int width = WARP_SIZE>
static __device__ __forceinline__ __half warp_reduce_prod(__half x) {
#pragma unroll
  for (int offset = width / 2; offset > 0; offset >>= 1) {
    x = __hmul(x, __shfl_xor_sync(0xffffffff, x, offset, width));
  }
  return x;
}