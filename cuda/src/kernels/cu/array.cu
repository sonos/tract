#include "common.cuh"

template <typename T>
static __device__ void pad_constant(
    const T* __restrict__ in_ptr,
    T* __restrict__ out_ptr,
    int in_shape0,  int in_shape1,  int in_shape2,  int in_shape3,  int in_shape4,
    int out_shape0, int out_shape1, int out_shape2, int out_shape3, int out_shape4,
    int in_stride0, int in_stride1, int in_stride2, int in_stride3, int in_stride4,
    int pad_before0, int pad_before1, int pad_before2, int pad_before3, int pad_before4,
    T fill,
    int total_out_elems
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
        int in_shape0, int in_shape1, int in_shape2, int in_shape3, int in_shape4, \
        int out_shape0, int out_shape1, int out_shape2, int out_shape3, int out_shape4, \
        int in_stride0, int in_stride1, int in_stride2, int in_stride3, int in_stride4, \
        int pad_before0, int pad_before1, int pad_before2, int pad_before3, int pad_before4, \
        T fill,                                                                              \
        int total_out_elems) {                                    \
      pad_constant<T>(in_ptr, out_ptr, in_shape0, in_shape1, in_shape2, in_shape3, in_shape4, \
        out_shape0, out_shape1, out_shape2, out_shape3, out_shape4, \
        in_stride0, in_stride1, in_stride2, in_stride3, in_stride4, \
        pad_before0, pad_before1, pad_before2, pad_before3, pad_before4, \
        fill, total_out_elems);                        \
    }

#define INSTANTIATE_ROTATE_HALF(name, T)                                       \
  extern "C" __global__ void rotate_half_nd2_##name(                           \
      const T *input, T *output, int shape_0, int shape_1, int strides_0,      \
      int strides_1) {                                                         \
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
                                         int len) {                            \
    int idx = blockIdx.x * blockDim.x + threadIdx.x;                           \
    if (idx < len) {                                                           \
      output[idx] = (T_out)input[idx];                                         \
    }                                                                          \
  }

#define INSTANTIATE_COPY(name, T)                                              \
  extern "C" __global__ void copy_nd1_##name(                                  \
      const T *input, T *output, int in_strides_0, int out_shape_0,            \
      int out_strides_0) {                                                     \
    for (int i = threadIdx.x; i < out_shape_0; i += MAX_THREADS) {  \
      output[i * out_strides_0] = input[i * in_strides_0];                     \
    }                                                                          \
  }                                                                            \
                                                                               \
  extern "C" __global__ void copy_nd2_##name(                                  \
      const T *input, T *output, int in_strides_0, int in_strides_1,           \
      int out_shape_0, int out_shape_1, int out_strides_0,                     \
      int out_strides_1) {                                                     \
    int in_offset = blockIdx.x * in_strides_0;                                 \
    int out_offset = blockIdx.x * out_strides_0;                               \
    for (int i = threadIdx.x; i < out_shape_1; i += MAX_THREADS) {  \
      output[out_offset + i * out_strides_1] =                                 \
          input[in_offset + i * in_strides_1];                                 \
    }                                                                          \
  }                                                                            \
                                                                               \
  extern "C" __global__ void copy_nd3_##name(                                  \
      const T *input, T *output, int in_strides_0, int in_strides_1,           \
      int in_strides_2, int out_shape_0, int out_shape_1, int out_shape_2,     \
      int out_strides_0, int out_strides_1, int out_strides_2) {               \
    int in_offset = blockIdx.x * in_strides_1 + blockIdx.y * in_strides_0;     \
    int out_offset = blockIdx.x * out_strides_1 + blockIdx.y * out_strides_0;  \
    for (int i = threadIdx.x; i < out_shape_2; i += MAX_THREADS) {  \
      output[out_offset + i * out_strides_2] =                                 \
          input[in_offset + i * in_strides_2];                                 \
    }                                                                          \
  }                                                                            \
                                                                               \
  extern "C" __global__ void copy_nd4_##name(                                  \
      const T *input, T *output, int in_strides_0, int in_strides_1,           \
      int in_strides_2, int in_strides_3, int out_shape_0, int out_shape_1,    \
      int out_shape_2, int out_shape_3, int out_strides_0, int out_strides_1,  \
      int out_strides_2, int out_strides_3) {                                  \
    int in_offset = blockIdx.x * in_strides_2 + blockIdx.y * in_strides_1 +    \
                    blockIdx.z * in_strides_0;                                 \
    int out_offset = blockIdx.x * out_strides_2 + blockIdx.y * out_strides_1 + \
                     blockIdx.z * out_strides_0;                               \
    for (int i = threadIdx.x; i < out_shape_3; i += MAX_THREADS) {  \
      output[out_offset + i * out_strides_3] =                                 \
          input[in_offset + i * in_strides_3];                                 \
    }                                                                          \
  }                                                                            \
                                                                               \
  extern "C" __global__ void copy_nd5_##name(                                  \
      const T *input, T *output, int in_strides_0, int in_strides_1,           \
      int in_strides_2, int in_strides_3, int in_strides_4, int out_shape_0,   \
      int out_shape_1, int out_shape_2, int out_shape_3, int out_shape_4,      \
      int out_strides_0, int out_strides_1, int out_strides_2,                 \
      int out_strides_3, int out_strides_4) {                                  \
    int in_offset = blockIdx.x * in_strides_3 + blockIdx.y * in_strides_2;     \
    int out_offset = blockIdx.x * out_strides_3 + blockIdx.y * out_strides_2;  \
    int block_idx_z = blockIdx.z;                                              \
    in_offset += (block_idx_z % out_shape_1) * in_strides_1;                   \
    out_offset += (block_idx_z % out_shape_1) * out_strides_1;                 \
    block_idx_z /= out_shape_1;                                                \
    in_offset += (block_idx_z % out_shape_0) * in_strides_0;                   \
    out_offset += (block_idx_z % out_shape_0) * out_strides_0;                 \
                                                                               \
    for (int i = threadIdx.x; i < out_shape_4; i += MAX_THREADS) {  \
      output[out_offset + i * out_strides_4] =                                 \
          input[in_offset + i * in_strides_4];                                 \
    }                                                                          \
  }                                                                            \
                                                                               \
  extern "C" __global__ void copy_nd6_##name(                                  \
      const T *input, T *output, int in_strides_0, int in_strides_1,           \
      int in_strides_2, int in_strides_3, int in_strides_4, int in_strides_5,  \
      int out_shape_0, int out_shape_1, int out_shape_2, int out_shape_3,      \
      int out_shape_4, int out_shape_5, int out_strides_0, int out_strides_1,  \
      int out_strides_2, int out_strides_3, int out_strides_4,                 \
      int out_strides_5) {                                                     \
    int in_offset = blockIdx.x * in_strides_4 + blockIdx.y * in_strides_3;     \
    int out_offset = blockIdx.x * out_strides_4 + blockIdx.y * out_strides_3;  \
    int block_idx_z = blockIdx.z;                                              \
    in_offset += (block_idx_z % out_shape_2) * in_strides_2;                   \
    out_offset += (block_idx_z % out_shape_2) * out_strides_2;                 \
    block_idx_z /= out_shape_2;                                                \
    in_offset += (block_idx_z % out_shape_1) * in_strides_1;                   \
    out_offset += (block_idx_z % out_shape_1) * out_strides_1;                 \
    block_idx_z /= out_shape_1;                                                \
    in_offset += (block_idx_z % out_shape_0) * in_strides_0;                   \
    out_offset += (block_idx_z % out_shape_0) * out_strides_0;                 \
                                                                               \
    for (int i = threadIdx.x; i < out_shape_5; i += MAX_THREADS) {  \
      output[out_offset + i * out_strides_5] =                                 \
          input[in_offset + i * in_strides_5];                                 \
    }                                                                          \
  }

#define INSTANTIATE_CAST_PAD_AND_COPY(tname, type)                             \
  INSTANTIATE_COPY(tname, type)                                                \
  INSTANTIATE_PAD_CONSTANT(tname, type)                                        \
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

#define INSTANTIATE_ALL(tname, type)                                           \
  INSTANTIATE_CAST_PAD_AND_COPY(tname, type)                                   \
  INSTANTIATE_ROTATE_HALF(tname, type)

INSTANTIATE_CAST_PAD_AND_COPY(bool, bool)
INSTANTIATE_ALL(f32, float)
INSTANTIATE_ALL(f16, __half)
INSTANTIATE_ALL(i8, int8_t)
INSTANTIATE_ALL(i16, int16_t)
INSTANTIATE_ALL(i32, int32_t)
INSTANTIATE_ALL(i64, int64_t)
INSTANTIATE_CAST_PAD_AND_COPY(u8, uint8_t)
INSTANTIATE_CAST_PAD_AND_COPY(u16, uint16_t)
INSTANTIATE_CAST_PAD_AND_COPY(u32, uint32_t)
INSTANTIATE_CAST_PAD_AND_COPY(u64, uint64_t)
