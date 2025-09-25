#include <cuda/std/cstdint>
#include <cuda_fp16.h>

#define MAX_THREADS 1024
#define WARP_SIZE 32

#define QK8_1 32
#define QI8_1 (QK8_1 / (4 * QR8_1))
#define QR8_1 1

#define QK4_0 32
#define QI4_0 (QK4_0 / (4 * QR4_0))
#define QR4_0 2

#define QK8_0 32
#define QI8_0 (QK8_0 / (4 * QR8_0))
#define QR8_0 1

typedef struct {
  half d;                // delta
  uint8_t qs[QK4_0 / 2]; // nibbles / quants
} block_q4_0;

typedef struct {
  half2 ds;
  int8_t qs[QK8_1]; // quants
} block_q8_1;
static_assert(sizeof(block_q8_1) == 2 * sizeof(half) + QK8_1,
              "wrong q8_1 block size/padding");

struct block_q8_1_mmq {
  // The y float data is converted to a data layout that can simply be copied to
  // shared memory as a contiguous block. The y float data is first grouped as
  // blocks of 128 values. These blocks are then treated as individual data
  // values and transposed.
  //
  // To avoid shared memory bank conflicts each block is padded with 16 bytes.
  // This padding is also used to store block scales/partial sums.
  // The scales multiplied with the quantized data are equal to the unquantized
  // values. The partial sums are obtained by summing up a subgroup of the
  // contained values (prior to quantization)
  //     and are only needed for performance reasons.
  half2 ds4[4]; // 1 16 bit scale + 1 16 bit partial sum per 32 values, stored
                // as d0,s0,d1,s1,d2,s2,d3,s3
  int8_t qs[4 * QK8_1]; // 128 values quantized to 8 bit each
};
static_assert(sizeof(block_q8_1_mmq) == 4 * QK8_1 + 4 * sizeof(half2),
              "Unexpected block_q8_1_mmq size");

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
