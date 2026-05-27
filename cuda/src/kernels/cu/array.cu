#include "common.cuh"

template <typename T>
static __device__ void pad_constant(
    const T* __restrict__ in_ptr,
    T* __restrict__ out_ptr,
    int32_t in_shape0,  int32_t in_shape1,  int32_t in_shape2,  int32_t in_shape3,  int32_t in_shape4,
    int32_t out_shape0, int32_t out_shape1, int32_t out_shape2, int32_t out_shape3, int32_t out_shape4,
    int32_t in_stride0, int32_t in_stride1, int32_t in_stride2, int32_t in_stride3, int32_t in_stride4,
    int32_t pad_before0, int32_t pad_before1, int32_t pad_before2, int32_t pad_before3, int32_t pad_before4,
    T fill,
    int32_t total_out_elems
) {
    int o = blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= total_out_elems) return;

    int idx = o;

    int c4 = idx % out_shape4; idx /= out_shape4;
    int c3 = idx % out_shape3; idx /= out_shape3;
    int c2 = idx % out_shape2; idx /= out_shape2;
    int c1 = idx % out_shape1; idx /= out_shape1;
    int c0 = idx % out_shape0;

    int i4 = c4 - pad_before4;
    int i3 = c3 - pad_before3;
    int i2 = c2 - pad_before2;
    int i1 = c1 - pad_before1;
    int i0 = c0 - pad_before0;

    bool in_bounds =
        (i4 >= 0 && i4 < in_shape4) &&
        (i3 >= 0 && i3 < in_shape3) &&
        (i2 >= 0 && i2 < in_shape2) &&
        (i1 >= 0 && i1 < in_shape1) &&
        (i0 >= 0 && i0 < in_shape0);

    int in_off = 0;
    if (in_bounds) {
        in_off += i4 * in_stride4;
        in_off += i3 * in_stride3;
        in_off += i2 * in_stride2;
        in_off += i1 * in_stride1;
        in_off += i0 * in_stride0;
        out_ptr[o] = in_ptr[in_off];
    } else {
        out_ptr[o] = fill;
    }
}

#define INSTANTIATE_PAD_CONSTANT(name, T)                                              \
   extern "C" __global__ void pad_constant_##name(                                     \
        const T* __restrict__ in_ptr, \
        T* __restrict__ out_ptr, \
        int32_t in_shape0, int32_t in_shape1, int32_t in_shape2, int32_t in_shape3, int32_t in_shape4, \
        int32_t out_shape0, int32_t out_shape1, int32_t out_shape2, int32_t out_shape3, int32_t out_shape4, \
        int32_t in_stride0, int32_t in_stride1, int32_t in_stride2, int32_t in_stride3, int32_t in_stride4, \
        int32_t pad_before0, int32_t pad_before1, int32_t pad_before2, int32_t pad_before3, int32_t pad_before4, \
        T fill,                                                                              \
        int32_t total_out_elems) {                                    \
      pad_constant<T>(in_ptr, out_ptr, in_shape0, in_shape1, in_shape2, in_shape3, in_shape4, \
        out_shape0, out_shape1, out_shape2, out_shape3, out_shape4, \
        in_stride0, in_stride1, in_stride2, in_stride3, in_stride4, \
        pad_before0, pad_before1, pad_before2, pad_before3, pad_before4, \
        fill, total_out_elems);                        \
    }

#define INSTANTIATE_ROTATE_HALF(name, T)                                       \
  extern "C" __global__ void rotate_half_nd2_##name(                           \
      const T *input, T *output, int32_t shape_0, int32_t shape_1, int32_t strides_0,      \
      int32_t strides_1) {                                                         \
    int thread_idx_x = blockIdx.x * blockDim.x + threadIdx.x;                  \
    int thread_idx_y = blockIdx.y * blockDim.y + threadIdx.y;                  \
                                                                               \
    int rotated_idx =                                                          \
        (thread_idx_x + shape_1 / 2) * strides_1 + thread_idx_y * strides_0;   \
    int idx = thread_idx_x * strides_1 + thread_idx_y * strides_0;             \
                                                                               \
    output[idx] = -input[rotated_idx];                                         \
    output[rotated_idx] = input[idx];                                          \
  }

#define INSTANTIATE_CAST_OP(name, T_in, T_out)                                 \
  extern "C" __global__ void cast_##name(const T_in *input, T_out *output,     \
                                         int32_t len) {                            \
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                           \
    if (idx < len) {                                                           \
      output[idx] = (T_out)input[idx];                                         \
    }                                                                          \
  }

#define INSTANTIATE_COPY(name, T)                                              \
  extern "C" __global__ void copy_nd1_##name(                                  \
      const T *input, T *output, int32_t in_strides_0, int32_t out_shape_0,            \
      int32_t out_strides_0) {                                                     \
    for (int i = threadIdx.x; i < out_shape_0; i += MAX_THREADS) {  \
      output[i * out_strides_0] = input[i * in_strides_0];                     \
    }                                                                          \
  }                                                                            \
                                                                               \
  extern "C" __global__ void copy_nd2_##name(                                  \
      const T *input, T *output, int32_t in_strides_0, int32_t in_strides_1,           \
      int32_t out_shape_0, int32_t out_shape_1, int32_t out_strides_0,                     \
      int32_t out_strides_1) {                                                     \
    int in_offset = blockIdx.x * in_strides_0;                                 \
    int out_offset = blockIdx.x * out_strides_0;                               \
    for (int i = threadIdx.x; i < out_shape_1; i += MAX_THREADS) {  \
      output[out_offset + i * out_strides_1] =                                 \
          input[in_offset + i * in_strides_1];                                 \
    }                                                                          \
  }                                                                            \
                                                                               \
  extern "C" __global__ void copy_nd3_##name(                                  \
      const T *input, T *output, int32_t in_strides_0, int32_t in_strides_1,           \
      int32_t in_strides_2, int32_t out_shape_0, int32_t out_shape_1, int32_t out_shape_2,     \
      int32_t out_strides_0, int32_t out_strides_1, int32_t out_strides_2) {               \
    int in_offset = blockIdx.x * in_strides_1 + blockIdx.y * in_strides_0;     \
    int out_offset = blockIdx.x * out_strides_1 + blockIdx.y * out_strides_0;  \
    for (int i = threadIdx.x; i < out_shape_2; i += MAX_THREADS) {  \
      output[out_offset + i * out_strides_2] =                                 \
          input[in_offset + i * in_strides_2];                                 \
    }                                                                          \
  }                                                                            \
                                                                               \
  extern "C" __global__ void copy_nd4_##name(                                  \
      const T *input, T *output, int32_t in_strides_0, int32_t in_strides_1,           \
      int32_t in_strides_2, int32_t in_strides_3, int32_t out_shape_0, int32_t out_shape_1,    \
      int32_t out_shape_2, int32_t out_shape_3, int32_t out_strides_0, int32_t out_strides_1,  \
      int32_t out_strides_2, int32_t out_strides_3) {                                  \
    int in_offset = blockIdx.x * in_strides_2 + blockIdx.y * in_strides_1 +    \
                    blockIdx.z * in_strides_0;                                 \
    int out_offset = blockIdx.x * out_strides_2 + blockIdx.y * out_strides_1 + \
                     blockIdx.z * out_strides_0;                               \
    for (int i = threadIdx.x; i < out_shape_3; i += MAX_THREADS) {             \
      output[out_offset + i * out_strides_3] =                                 \
          input[in_offset + i * in_strides_3];                                 \
    }                                                                          \
  }                                                                            \
                                                                               \
  /* nd5: z=d0, y=d1, x=d2*d3 (packed), threads=d4 */                         \
  extern "C" __global__ void copy_nd5_##name(                                  \
      const T *input, T *output, int32_t in_strides_0, int32_t in_strides_1,           \
      int32_t in_strides_2, int32_t in_strides_3, int32_t in_strides_4, int32_t out_shape_0,   \
      int32_t out_shape_1, int32_t out_shape_2, int32_t out_shape_3, int32_t out_shape_4,      \
      int32_t out_strides_0, int32_t out_strides_1, int32_t out_strides_2,                 \
      int32_t out_strides_3, int32_t out_strides_4) {                                  \
    int block_idx_x = blockIdx.x;                                              \
    int idx_3 = block_idx_x % out_shape_3;                                     \
    block_idx_x /= out_shape_3;                                                \
    int idx_2 = block_idx_x;                                                   \
    int in_offset = blockIdx.z * in_strides_0 + blockIdx.y * in_strides_1 +    \
                    idx_2 * in_strides_2 + idx_3 * in_strides_3;               \
    int out_offset = blockIdx.z * out_strides_0 + blockIdx.y * out_strides_1 + \
                     idx_2 * out_strides_2 + idx_3 * out_strides_3;            \
    for (int i = threadIdx.x; i < out_shape_4; i += MAX_THREADS) {             \
      output[out_offset + i * out_strides_4] =                                 \
          input[in_offset + i * in_strides_4];                                 \
    }                                                                          \
  }                                                                            \
                                                                               \
  /* nd6: z=d0, y=d1, x=d2*d3*d4 (packed), threads=d5 */                      \
  extern "C" __global__ void copy_nd6_##name(                                  \
      const T *input, T *output, int32_t in_strides_0, int32_t in_strides_1,           \
      int32_t in_strides_2, int32_t in_strides_3, int32_t in_strides_4, int32_t in_strides_5,  \
      int32_t out_shape_0, int32_t out_shape_1, int32_t out_shape_2, int32_t out_shape_3,      \
      int32_t out_shape_4, int32_t out_shape_5, int32_t out_strides_0, int32_t out_strides_1,  \
      int32_t out_strides_2, int32_t out_strides_3, int32_t out_strides_4,                 \
      int32_t out_strides_5) {                                                     \
    int block_idx_x = blockIdx.x;                                              \
    int idx_4 = block_idx_x % out_shape_4;                                     \
    block_idx_x /= out_shape_4;                                                \
    int idx_3 = block_idx_x % out_shape_3;                                     \
    block_idx_x /= out_shape_3;                                                \
    int idx_2 = block_idx_x;                                                   \
    int in_offset = blockIdx.z * in_strides_0 + blockIdx.y * in_strides_1 +    \
                    idx_2 * in_strides_2 + idx_3 * in_strides_3 +              \
                    idx_4 * in_strides_4;                                      \
    int out_offset = blockIdx.z * out_strides_0 + blockIdx.y * out_strides_1 + \
                     idx_2 * out_strides_2 + idx_3 * out_strides_3 +           \
                     idx_4 * out_strides_4;                                    \
    for (int i = threadIdx.x; i < out_shape_5; i += MAX_THREADS) {             \
      output[out_offset + i * out_strides_5] =                                 \
          input[in_offset + i * in_strides_5];                                 \
    }                                                                          \
  }

#define INSTANTIATE_CAST_FROM(tname, type)                                     \
  INSTANTIATE_CAST_OP(tname##_bool, type, bool)                                \
  INSTANTIATE_CAST_OP(tname##_f32, type, float)                                \
  INSTANTIATE_CAST_OP(tname##_f16, type, __half)                               \
  INSTANTIATE_CAST_OP(tname##_u8, type, uint8_t)                               \
  INSTANTIATE_CAST_OP(tname##_u16, type, uint16_t)                             \
  INSTANTIATE_CAST_OP(tname##_u32, type, uint32_t)                             \
  INSTANTIATE_CAST_OP(tname##_u64, type, uint64_t)                             \
  INSTANTIATE_CAST_OP(tname##_i8, type, int8_t)                                \
  INSTANTIATE_CAST_OP(tname##_i16, type, int16_t)                              \
  INSTANTIATE_CAST_OP(tname##_i32, type, int32_t)                              \
  INSTANTIATE_CAST_OP(tname##_i64, type, int64_t)

// Copy kernels: only u8/u16/u32/u64 (copy is type-size based)
INSTANTIATE_COPY(u8, uint8_t)
INSTANTIATE_COPY(u16, uint16_t)
INSTANTIATE_COPY(u32, uint32_t)
INSTANTIATE_COPY(u64, uint64_t)

// Cast kernels: all types
INSTANTIATE_CAST_FROM(bool, bool)
INSTANTIATE_CAST_FROM(f32, float)
INSTANTIATE_CAST_FROM(f16, __half)
INSTANTIATE_CAST_FROM(i8, int8_t)
INSTANTIATE_CAST_FROM(i16, int16_t)
INSTANTIATE_CAST_FROM(i32, int32_t)
INSTANTIATE_CAST_FROM(i64, int64_t)
INSTANTIATE_CAST_FROM(u8, uint8_t)
INSTANTIATE_CAST_FROM(u16, uint16_t)
INSTANTIATE_CAST_FROM(u32, uint32_t)
INSTANTIATE_CAST_FROM(u64, uint64_t)

// Rotate half: only float types
INSTANTIATE_ROTATE_HALF(f32, float)
INSTANTIATE_ROTATE_HALF(f16, __half)

// Diagonal gather (Transformer-XL rel-pos skew, folded):
//   out[..., i, k] = in[..., i, offset + k - i], 0 on out-of-bounds.
// Leading axes are flattened by the host into one batch axis.  Each thread
// owns one (b, i, k) output element — bandwidth-bound, no shared memory.
#define INSTANTIATE_DIAG_GATHER(name, T)                                       \
    extern "C" __global__ void diag_gather_##name(                             \
        const T *input, T *output, const int32_t offset,                       \
        const int32_t batch, const int32_t t_q, const int32_t r_in,            \
        const int32_t out_len, const int32_t in_stride_b,                      \
        const int32_t in_stride_i, const int32_t in_stride_r,                  \
        const int32_t out_stride_b, const int32_t out_stride_i,                \
        const int32_t out_stride_k) {                                          \
        const int32_t k = blockIdx.x * blockDim.x + threadIdx.x;               \
        if (k >= out_len)                                                      \
            return;                                                            \
        const int32_t i = blockIdx.y;                                          \
        const int32_t b = blockIdx.z;                                          \
        const int32_t r = offset + k - i;                                      \
        T *out_ptr = output + b * out_stride_b + i * out_stride_i +            \
                     k * out_stride_k;                                         \
        if (r >= 0 && r < r_in) {                                              \
            const T *in_ptr =                                                  \
                input + b * in_stride_b + i * in_stride_i + r * in_stride_r;   \
            *out_ptr = *in_ptr;                                                \
        } else {                                                               \
            *out_ptr = (T)0;                                                   \
        }                                                                      \
    }

INSTANTIATE_DIAG_GATHER(f32, float)
INSTANTIATE_DIAG_GATHER(f16, __half)

// Gather along one axis:
//   out[i_pre, i_n, i_post] = data[i_pre, indices[i_n], i_post]
// where the host flattens to (pre × a_size × post) for data and
// (pre × n_indices × post) for output.  `n_indices` here is the *flat*
// indices count (product of the indices tensor's shape).  Negative indices
// wrap with `a_size`, matching the CPU contract.
#define INSTANTIATE_GATHER(name, T)                                            \
    extern "C" __global__ void gather_##name(                                  \
        const T *data, const int64_t *indices, T *output, const int32_t pre,   \
        const int32_t a_size, const int32_t post, const int32_t n_indices) {   \
        const int32_t i_post = blockIdx.x * blockDim.x + threadIdx.x;          \
        if (i_post >= post)                                                    \
            return;                                                            \
        const int32_t i_n = blockIdx.y;                                        \
        const int32_t i_pre = blockIdx.z;                                      \
        int64_t k = indices[i_n];                                              \
        if (k < 0)                                                             \
            k += a_size;                                                       \
        const int64_t in_off =                                                 \
            ((int64_t)i_pre * a_size + k) * post + i_post;                     \
        const int64_t out_off =                                                \
            ((int64_t)i_pre * n_indices + i_n) * post + i_post;                \
        output[out_off] = data[in_off];                                        \
    }

INSTANTIATE_GATHER(f32, float)
INSTANTIATE_GATHER(f16, __half)
