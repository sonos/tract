#include "utils.cuh"
#include <cstdint>
#include <cuda_fp16.h>

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

static __device__ __forceinline__ void
compute_mmq_q81_block(float4 xi, int64_t ib, int64_t iqs, block_q8_1_mmq *y) {
  constexpr int vals_per_scale = 32;
  constexpr int vals_per_sum = 32;

  float amax = fabsf(xi.x);
  amax = fmaxf(amax, fabsf(xi.y));
  amax = fmaxf(amax, fabsf(xi.z));
  amax = fmaxf(amax, fabsf(xi.w));

// Exchange max. abs. value between vals_per_scale/4 threads.
#pragma unroll
  for (int offset = vals_per_scale / 8; offset > 0; offset >>= 1) {
    amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFF, amax, offset, WARP_SIZE));
  }

  float sum;
  sum = xi.x + xi.y + xi.z + xi.w;

// Calculate sums across vals_per_sum/4 threads.
#pragma unroll
  for (int offset = vals_per_sum / 8; offset > 0; offset >>= 1) {
    sum += __shfl_xor_sync(0xFFFFFFFF, sum, offset, WARP_SIZE);
  }

  float d_inv = (amax > 0.f) ? 127.f / amax : 0.f;
  char4 q;
  q.x = roundf(xi.x * d_inv);
  q.y = roundf(xi.y * d_inv);
  q.z = roundf(xi.z * d_inv);
  q.w = roundf(xi.w * d_inv);

  // Write back 4 int8 values as a single 32 bit value for better memroy
  // bandwidth:
  char4 *yqs4 = (char4 *)y[ib].qs;
  yqs4[iqs / 4] = q;

  if (iqs % 32 != 0) {
    return;
  }

  const float d = (d_inv > 0.0f) ? (1.0f / d_inv) : 0.0f;
  y[ib].ds4[iqs / 32] = make_half2(d, sum);
}

extern "C" __global__ void quantize_mmq_q8_1_fast_nd2(
    const float *__restrict__ x, void *__restrict__ vy, const int64_t k,
    const int64_t in_strides_0, const int64_t in_strides_1,
    const int64_t padded_k) {

  const int64_t i0 = ((int64_t)blockDim.x * blockIdx.y + threadIdx.x) * 4;

  if (i0 >= padded_k) {
    return;
  }

  const int64_t i1 = blockIdx.x;

  const int64_t i00 = i0;
  const int64_t i01 = i1;

  const float4 *x4 = (const float4 *)x;

  const int64_t ib0 =
      blockIdx.z * ((int64_t)gridDim.x * gridDim.y * blockDim.x /
                    QK8_1); // first block of channel
  const int64_t ib = ib0 + (i0 / (4 * QK8_1)) * gridDim.x +
                     blockIdx.x;        // block index in channel
  const int64_t iqs = i0 % (4 * QK8_1); // quant index in block

  // Load 4 floats per thread and calculate max. abs. value between them:
  const float4 xi = i0 < k ? x4[(i01 * in_strides_0 + i00) / 4]
                           : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

  compute_mmq_q81_block(xi, ib, iqs, (block_q8_1_mmq *)vy);
}

extern "C" __global__ void quantize_mmq_q8_1_fast_nd3(
    const float *__restrict__ x, void *__restrict__ vy, const int64_t k,
    const int64_t in_strides_0, const int64_t in_strides_1,
    const int64_t in_strides_2,
    const int out_shape_1, const int64_t padded_k) {

  const int64_t i0 = ((int64_t)blockDim.x * blockIdx.y + threadIdx.x) * 4;

  if (i0 >= padded_k) {
    return;
  }

  const int64_t i1 = blockIdx.x;
  const int64_t i2 = blockIdx.z;

  const int64_t i00 = i0;
  const int64_t i01 = i1;
  const int64_t i02 = i2;

  const float4 *x4 = (const float4 *)x;

  const int64_t ib0 =
      blockIdx.z * ((int64_t)gridDim.x * gridDim.y * blockDim.x /
                    QK8_1); // first block of channel
  const int64_t ib = ib0 + (i0 / (4 * QK8_1)) * out_shape_1 +
                     blockIdx.x;        // block index in channel
  const int64_t iqs = i0 % (4 * QK8_1); // quant index in block

  // Load 4 floats per thread and calculate max. abs. value between them:
  const float4 xi = i0 < k ? x4[(i02 * in_strides_0 +
                                 i01 * in_strides_1 + i00) /
                                4]
                           : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

  compute_mmq_q81_block(xi, ib, iqs, (block_q8_1_mmq *)vy);
}

extern "C" __global__ void quantize_mmq_q8_1_fast_nd4(
    const float *__restrict__ x, void *__restrict__ vy, const int64_t k,
    const int64_t in_strides_0, const int64_t in_strides_1,
    const int64_t in_strides_2, const int64_t in_strides_3,
    const int out_shape_1, const int out_shape_2, const int64_t padded_k) {

  const int64_t i0 = ((int64_t)blockDim.x * blockIdx.y + threadIdx.x) * 4;

  if (i0 >= padded_k) {
    return;
  }

  const int64_t i1 = blockIdx.x;
  const int64_t i2 = blockIdx.z % out_shape_1;
  const int64_t i3 = blockIdx.z / out_shape_1;

  const int64_t i00 = i0;
  const int64_t i01 = i1;
  const int64_t i02 = i2;
  const int64_t i03 = i3;

  const float4 *x4 = (const float4 *)x;

  const int64_t ib0 =
      blockIdx.z * ((int64_t)gridDim.x * gridDim.y * blockDim.x /
                    QK8_1); // first block of channel
  const int64_t ib = ib0 + (i0 / (4 * QK8_1)) * out_shape_2 +
                     blockIdx.x;        // block index in channel
  const int64_t iqs = i0 % (4 * QK8_1); // quant index in block

  // Load 4 floats per thread and calculate max. abs. value between them:
  const float4 xi = i0 < k ? x4[(i03 * in_strides_0 + i02 * in_strides_1 +
                                 i01 * in_strides_2 + i00) /
                                4]
                           : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

  compute_mmq_q81_block(xi, ib, iqs, (block_q8_1_mmq *)vy);
}

extern "C" __global__ void quantize_mmq_q8_1_fast_nd5(
    const float *__restrict__ x, void *__restrict__ vy, const int64_t k,
    const int64_t in_strides_0, const int64_t in_strides_1,
    const int64_t in_strides_2, const int64_t in_strides_3, const int64_t in_strides_4,
    const int out_shape_1, const int out_shape_2, const int out_shape_3, const int64_t padded_k) {

  const int64_t i0 = ((int64_t)blockDim.x * blockIdx.y + threadIdx.x) * 4;

  if (i0 >= padded_k) {
    return;
  }

  const int64_t i1 = blockIdx.x;
  const int64_t i2 = blockIdx.z % out_shape_2;
  const int64_t i3 = (blockIdx.z / out_shape_2) % out_shape_1;
  const int64_t i4 = blockIdx.z / (out_shape_2 * out_shape_1);

  const int64_t i00 = i0;
  const int64_t i01 = i1;
  const int64_t i02 = i2;
  const int64_t i03 = i3;
  const int64_t i04 = i4;

  const float4 *x4 = (const float4 *)x;

  const int64_t ib0 =
      blockIdx.z * ((int64_t)gridDim.x * gridDim.y * blockDim.x /
                    QK8_1); // first block of channel
  const int64_t ib = ib0 + (i0 / (4 * QK8_1)) * out_shape_3 +
                     blockIdx.x;        // block index in channel
  const int64_t iqs = i0 % (4 * QK8_1); // quant index in block

  // Load 4 floats per thread and calculate max. abs. value between them:
  const float4 xi = i0 < k ? x4[(i04 * in_strides_4 + i03 * in_strides_1 +
                                 i02 * in_strides_2 + i01 * in_strides_3 + i00) /
                                4]
                           : make_float4(0.0f, 0.0f, 0.0f, 0.0f);

  compute_mmq_q81_block(xi, ib, iqs, (block_q8_1_mmq *)vy);
}

extern "C" __global__ void
quantize_mmq_q8_1_nd2(const float *__restrict__ x, void *__restrict__ vy,
                      const int64_t k, const int64_t in_strides_0,
                      const int64_t in_strides_1, const int64_t padded_k) {

  const int64_t i0 = ((int64_t)blockDim.x * blockIdx.y + threadIdx.x) * 4;

  if (i0 >= padded_k) {
    return;
  }

  const int64_t i1 = blockIdx.x;

  float4 xi = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  const int64_t base = i1 * in_strides_0 + i0 * in_strides_1;

  xi.x = (i0 + 0 < k) ? x[base + 0 * in_strides_1] : 0.0f;
  xi.y = (i0 + 1 < k) ? x[base + 1 * in_strides_1] : 0.0f;
  xi.z = (i0 + 2 < k) ? x[base + 2 * in_strides_1] : 0.0f;
  xi.w = (i0 + 3 < k) ? x[base + 3 * in_strides_1] : 0.0f;

  const int64_t ib0 =
      blockIdx.z * ((int64_t)gridDim.x * gridDim.y * blockDim.x /
                    QK8_1); // first block of channel
  const int64_t ib = ib0 + (i0 / (4 * QK8_1)) * gridDim.x +
                     blockIdx.x;        // block index in channel
  const int64_t iqs = i0 % (4 * QK8_1); // quant index in block

  compute_mmq_q81_block(xi, ib, iqs, (block_q8_1_mmq *)vy);
}

extern "C" __global__ void
quantize_mmq_q8_1_nd3(const float *__restrict__ x, void *__restrict__ vy,
                      const int64_t k, const int64_t in_strides_0,
                      const int64_t in_strides_1, const int64_t in_strides_2,
                      const int out_shape_1, const int64_t padded_k) {

  const int64_t i0 = ((int64_t)blockDim.x * blockIdx.y + threadIdx.x) * 4;

  if (i0 >= padded_k) {
    return;
  }

  const int64_t i1 = blockIdx.x;
  const int64_t i2 = blockIdx.z;

  float4 xi = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  const int64_t base =
      i2 * in_strides_0 + i1 * in_strides_1 + i0 * in_strides_2;

  xi.x = (i0 + 0 < k) ? x[base + 0 * in_strides_2] : 0.0f;
  xi.y = (i0 + 1 < k) ? x[base + 1 * in_strides_2] : 0.0f;
  xi.z = (i0 + 2 < k) ? x[base + 2 * in_strides_2] : 0.0f;
  xi.w = (i0 + 3 < k) ? x[base + 3 * in_strides_2] : 0.0f;

  const int64_t ib0 =
      blockIdx.z * ((int64_t)gridDim.x * gridDim.y * blockDim.x /
                    QK8_1); // first block of channel
  const int64_t ib = ib0 + (i0 / (4 * QK8_1)) * out_shape_1 +
                     blockIdx.x;        // block index in channel
  const int64_t iqs = i0 % (4 * QK8_1); // quant index in block

  compute_mmq_q81_block(xi, ib, iqs, (block_q8_1_mmq *)vy);
}

extern "C" __global__ void
quantize_mmq_q8_1_nd4(const float *__restrict__ x, void *__restrict__ vy,
                      const int64_t k, const int64_t in_strides_0,
                      const int64_t in_strides_1, const int64_t in_strides_2,
                      const int64_t in_strides_3, const int out_shape_1,
                      const int out_shape_2, const int64_t padded_k) {

  const int64_t i0 = ((int64_t)blockDim.x * blockIdx.y + threadIdx.x) * 4;

  if (i0 >= padded_k) {
    return;
  }

  const int64_t i1 = blockIdx.x;
  const int64_t i2 = blockIdx.z % out_shape_1;
  const int64_t i3 = blockIdx.z / out_shape_1;

  float4 xi = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  const int64_t base = i3 * in_strides_0 + i2 * in_strides_1 +
                       i1 * in_strides_2 + i0 * in_strides_3;

  xi.x = (i0 + 0 < k) ? x[base + 0 * in_strides_3] : 0.0f;
  xi.y = (i0 + 1 < k) ? x[base + 1 * in_strides_3] : 0.0f;
  xi.z = (i0 + 2 < k) ? x[base + 2 * in_strides_3] : 0.0f;
  xi.w = (i0 + 3 < k) ? x[base + 3 * in_strides_3] : 0.0f;

  const int64_t ib0 =
      blockIdx.z * ((int64_t)gridDim.x * gridDim.y * blockDim.x /
                    QK8_1); // first block of channel
  const int64_t ib = ib0 + (i0 / (4 * QK8_1)) * out_shape_2 +
                     blockIdx.x;        // block index in channel
  const int64_t iqs = i0 % (4 * QK8_1); // quant index in block

  compute_mmq_q81_block(xi, ib, iqs, (block_q8_1_mmq *)vy);
}

extern "C" __global__ void
quantize_mmq_q8_1_nd5(const float *__restrict__ x, void *__restrict__ vy,
                      const int64_t k, const int64_t in_strides_0,
                      const int64_t in_strides_1, const int64_t in_strides_2,
                      const int64_t in_strides_3, const int64_t in_strides_4,
                      const int out_shape_1, const int out_shape_2,
                      const int out_shape_3, const int64_t padded_k) {

  const int64_t i0 = ((int64_t)blockDim.x * blockIdx.y + threadIdx.x) * 4;

  if (i0 >= padded_k) {
    return;
  }

  const int64_t i1 = blockIdx.x;
  const int64_t i2 = blockIdx.z % out_shape_2;
  const int64_t i3 = (blockIdx.z / out_shape_2) % out_shape_1;
  const int64_t i4 = blockIdx.z / (out_shape_2 * out_shape_1);

  float4 xi = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  const int64_t base = i4 * in_strides_0 + i3 * in_strides_1 +
                       i2 * in_strides_2 + i1 * in_strides_3 +
                       i0 * in_strides_4;

  xi.x = (i0 + 0 < k) ? x[base + 0 * in_strides_4] : 0.0f;
  xi.y = (i0 + 1 < k) ? x[base + 1 * in_strides_4] : 0.0f;
  xi.z = (i0 + 2 < k) ? x[base + 2 * in_strides_4] : 0.0f;
  xi.w = (i0 + 3 < k) ? x[base + 3 * in_strides_4] : 0.0f;

  const int64_t ib0 =
      blockIdx.z * ((int64_t)gridDim.x * gridDim.y * blockDim.x /
                    QK8_1); // first block of channel
  const int64_t ib = ib0 + (i0 / (4 * QK8_1)) * out_shape_3 +
                     blockIdx.x;        // block index in channel
  const int64_t iqs = i0 % (4 * QK8_1); // quant index in block

  compute_mmq_q81_block(xi, ib, iqs, (block_q8_1_mmq *)vy);
}

static __device__ __forceinline__ void
compute_q81_block(float xi, int64_t i_cont, block_q8_1 **y_ptr) {
  block_q8_1 *y = *y_ptr;
  const int64_t ib = i_cont / QK8_1;  // block index
  const int64_t iqs = i_cont % QK8_1; // quant index
  float amax = fabsf(xi);
  float sum = xi;

  amax = warp_reduce_max(amax);
  sum = warp_reduce_sum(sum);

  const float d = amax / 127;
  const int8_t q = amax == 0.0f ? 0 : roundf(xi / d);

  y[ib].qs[iqs] = q;

  if (iqs > 0) {
    return;
  }

  reinterpret_cast<half &>(y[ib].ds.x) = d;
  reinterpret_cast<half &>(y[ib].ds.y) = sum;
}

extern "C" __global__ void
quantize_q8_1_nd2(const float *__restrict__ x, void *__restrict__ vy,
                  const int64_t k, const int64_t in_strides_0,
                  const int64_t in_strides_1, const int64_t padded_k) {
  const int64_t i0 = (int64_t)blockDim.x * blockIdx.x + threadIdx.x;

  if (i0 >= padded_k) {
    return;
  }

  const int64_t i1 = blockIdx.y;

  const int64_t &i00 = i0;
  const int64_t &i01 = i1;

  const int64_t i_cont = i1 * padded_k + i0;

  const float xi = i0 < k ? x[i01 * in_strides_0 + i00 * in_strides_1] : 0.0f;

  compute_q81_block(xi, i_cont, (block_q8_1 **)&vy);
}

extern "C" __global__ void
quantize_q8_1_nd3(const float *__restrict__ x, void *__restrict__ vy,
                  const int64_t k, const int64_t in_strides_0,
                  const int64_t in_strides_1, const int64_t in_strides_2,
                  const int64_t out_shape_1, const int64_t padded_k) {
  const int64_t i0 = (int64_t)blockDim.x * blockIdx.x + threadIdx.x;

  if (i0 >= padded_k) {
    return;
  }

  const int64_t i1 = blockIdx.y;
  const int64_t i2 = blockIdx.z;

  const int64_t &i00 = i0;
  const int64_t &i01 = i1;
  const int64_t &i02 = i2;

  const int64_t i_cont = (i2 * out_shape_1 + i1) * padded_k + i0;

  const float xi =
      i0 < k ? x[i02 * in_strides_0 + i01 * in_strides_1 + i00 * in_strides_2]
             : 0.0f;

  compute_q81_block(xi, i_cont, (block_q8_1 **)&vy);
}

extern "C" __global__ void
quantize_q8_1_nd4(const float *__restrict__ x, void *__restrict__ vy,
                  const int64_t k, const int64_t in_strides_0,
                  const int64_t in_strides_1, const int64_t in_strides_2,
                  const int64_t in_strides_3, const int64_t out_shape_1,
                  const int64_t out_shape_2, const int64_t padded_k) {
  const int64_t i0 = (int64_t)blockDim.x * blockIdx.x + threadIdx.x;

  if (i0 >= padded_k) {
    return;
  }

  const int64_t i1 = blockIdx.y;
  const int64_t i2 = blockIdx.z % out_shape_1;
  const int64_t i3 = blockIdx.z / out_shape_1;

  const int64_t &i00 = i0;
  const int64_t &i01 = i1;
  const int64_t &i02 = i2;
  const int64_t &i03 = i3;

  const int64_t i_cont =
      ((i3 * out_shape_1 + i2) * out_shape_2 + i1) * padded_k + i0;

  const float xi = i0 < k ? x[i03 * in_strides_0 + i02 * in_strides_1 +
                              i01 * in_strides_2 + i00 * in_strides_3]
                          : 0.0f;

  compute_q81_block(xi, i_cont, (block_q8_1 **)&vy);
}

extern "C" __global__ void
quantize_q8_1_nd5(const float *__restrict__ x, void *__restrict__ vy,
                  const int64_t k, const int64_t in_strides_4,
                  const int64_t in_strides_3, const int64_t in_strides_2,
                  const int64_t in_strides_1, const int64_t in_strides_0,
                  const int64_t out_shape_1, const int64_t out_shape_2,
                  const int64_t out_shape_3, const int64_t padded_k) {
  const int64_t i0 = (int64_t)blockDim.x * blockIdx.x + threadIdx.x;

  if (i0 >= padded_k) {
    return;
  }

  const int64_t i1 = blockIdx.y;
  const int64_t i2 = blockIdx.z % out_shape_2;
  const int64_t i3 = (blockIdx.z / out_shape_2) % out_shape_1;
  const int64_t i4 = blockIdx.z / (out_shape_2 * out_shape_1);

  const int64_t &i00 = i0;
  const int64_t &i01 = i1;
  const int64_t &i02 = i2;
  const int64_t &i03 = i3;
  const int64_t &i04 = i4;

  const int64_t i_cont =
      (((i4 * out_shape_1 + i3) * out_shape_2 + i2) * out_shape_3 + i1) *
          padded_k +
      i0;

  const float xi =
      i0 < k ? x[i04 * in_strides_0 + i03 * in_strides_1 + i02 * in_strides_2 +
                 i01 * in_strides_3 + i00 * in_strides_4]
             : 0.0f;

  compute_q81_block(xi, i_cont, (block_q8_1 **)&vy);
}