#include <stdint.h>
#include <cuda_fp16.h>

#define INSTANTIATE_ROTATE_HALF(name, T) \
extern "C" __global__ void rotate_half_nd2_##name(const T *input, T *output, \
      int shape_0, int shape_1, int strides_0, int strides_1) { \
  int thread_idx_x = blockIdx.x * blockDim.x + threadIdx.x; \
  int thread_idx_y = blockIdx.y * blockDim.y + threadIdx.y; \
\
  int rotated_idx = (thread_idx_x + shape_1 / 2) * strides_1 + thread_idx_y * strides_0; \
  int idx = thread_idx_x * strides_1 + thread_idx_y * strides_0; \
\
  output[idx] = -input[rotated_idx]; \
  output[rotated_idx] = input[idx]; \
} \

#define INSTANTIATE_CAST_OP(name, T_in, T_out) \
extern "C" __global__ void cast_##name(const T_in* input, T_out* output) { \
    int idx = blockIdx.x * blockDim.x + threadIdx.x; \
    output[idx] = (T_out) input[idx]; \
} \


#define INSTANTIATE_COPY(name, T) \
extern "C" __global__ void copy_unicast_##name(const T* input, T* output) { \
    int idx = blockIdx.x * blockDim.x + threadIdx.x; \
    output[idx] = input[idx]; \
} \
\
extern "C" __global__ void copy_nd1_##name(const T* input, T* output, int in_strides_0, int out_shape_0, int out_strides_0) { \
    for (int i = threadIdx.x; i < out_shape_0; i += blockDim.x * gridDim.x) { \
        output[i * out_strides_0] = input[i * in_strides_0]; \
    } \
} \
\
extern "C" __global__ void copy_nd2_##name(const T* input, T* output, int in_strides_0, int in_strides_1, \
                                           int out_shape_0, int out_shape_1, \
                                           int out_strides_0, int out_strides_1) { \
    int in_offset = blockIdx.x * in_strides_0; \
    int out_offset = blockIdx.x * out_strides_0; \
    for (int i = threadIdx.x; i < out_shape_1; i += blockDim.x * gridDim.x) { \
        output[out_offset + i * out_strides_1] = input[in_offset + i * in_strides_1]; \
    } \
} \
\
extern "C" __global__ void copy_nd3_##name(const T* input, T* output, int in_strides_0, int in_strides_1, int in_strides_2, \
                                           int out_shape_0, int out_shape_1, int out_shape_2, \
                                           int out_strides_0, int out_strides_1, int out_strides_2) { \
    int in_offset = blockIdx.x * in_strides_1 + blockIdx.y * in_strides_0; \
    int out_offset = blockIdx.x * out_strides_1 + blockIdx.y * out_strides_0; \
    for (int i = threadIdx.x; i < out_shape_2; i += blockDim.x * gridDim.x) { \
        output[out_offset + i * out_strides_2] = input[in_offset + i * in_strides_2]; \
    } \
} \
\
extern "C" __global__ void copy_nd4_##name(const T* input, T* output, int in_strides_0, int in_strides_1, int in_strides_2, int in_strides_3,\
                                           int out_shape_0, int out_shape_1, int out_shape_2, int out_shape_3,\
                                           int out_strides_0, int out_strides_1, int out_strides_2, int out_strides_3) { \
    int in_offset = blockIdx.x * in_strides_2 + blockIdx.y * in_strides_1 + blockIdx.z * in_strides_0; \
    int out_offset = blockIdx.x * out_strides_2 + blockIdx.y * out_strides_1 + blockIdx.z * out_strides_0; \
    for (int i = threadIdx.x; i < out_shape_3; i += blockDim.x * gridDim.x) { \
        output[out_offset + i * out_strides_3] = input[in_offset + i * in_strides_3]; \
    } \
} \
\
extern "C" __global__ void copy_nd5_##name(const T* input, T* output, \
                                           int in_strides_0, int in_strides_1, int in_strides_2, int in_strides_3, int in_strides_4, \
                                           int out_shape_0, int out_shape_1, int out_shape_2, int out_shape_3, int out_shape_4, \
                                           int out_strides_0, int out_strides_1, int out_strides_2, int out_strides_3, int out_strides_4) { \
    int in_offset = blockIdx.x * in_strides_3 + blockIdx.y * in_strides_2; \
    int out_offset = blockIdx.x * out_strides_3 + blockIdx.y * out_strides_2; \
    int block_idx_z = blockIdx.z; \
    in_offset += (block_idx_z % out_shape_1) * in_strides_1; \
    out_offset += (block_idx_z % out_shape_1) * out_strides_1; \
    block_idx_z /= out_shape_1; \
    in_offset += (block_idx_z % out_shape_0) * in_strides_0; \
    out_offset += (block_idx_z % out_shape_0) * out_strides_0; \
\
    for (int i = threadIdx.x; i < out_shape_4; i += blockDim.x * gridDim.x) { \
        output[out_offset + i * out_strides_4] = input[in_offset + i * in_strides_4]; \
    } \
} \
\
extern "C" __global__ void copy_nd6_##name(const T* input, T* output, \
                                           int in_strides_0, int in_strides_1, int in_strides_2, int in_strides_3, int in_strides_4, int in_strides_5, \
                                           int out_shape_0, int out_shape_1, int out_shape_2, int out_shape_3, int out_shape_4, int out_shape_5,\
                                           int out_strides_0, int out_strides_1, int out_strides_2, int out_strides_3, int out_strides_4, int out_strides_5) { \
    int in_offset = blockIdx.x * in_strides_4 + blockIdx.y * in_strides_3; \
    int out_offset = blockIdx.x * out_strides_4 + blockIdx.y * out_strides_3; \
    int block_idx_z = blockIdx.z; \
    in_offset += (block_idx_z % out_shape_2) * in_strides_2; \
    out_offset += (block_idx_z % out_shape_2) * out_strides_2; \
    block_idx_z /= out_shape_2; \
    in_offset += (block_idx_z % out_shape_1) * in_strides_1; \
    out_offset += (block_idx_z % out_shape_1) * out_strides_1; \
    block_idx_z /= out_shape_1; \
    in_offset += (block_idx_z % out_shape_0) * in_strides_0; \
    out_offset += (block_idx_z % out_shape_0) * out_strides_0; \
\
    for (int i = threadIdx.x; i < out_shape_5; i += blockDim.x * gridDim.x) { \
        output[out_offset + i * out_strides_5] = input[in_offset + i * in_strides_5]; \
    } \
} \

#define INSTANTIATE_CAST_AND_COPY(tname, type) \
INSTANTIATE_COPY(tname, type) \
INSTANTIATE_CAST_OP(tname ##_bool, type, bool)    \
INSTANTIATE_CAST_OP(tname ##_f32, type, float)    \
INSTANTIATE_CAST_OP(tname ##_f16, type, __half)     \
INSTANTIATE_CAST_OP(tname ##_u8, type, uint8_t)   \
INSTANTIATE_CAST_OP(tname ##_u16, type, uint16_t) \
INSTANTIATE_CAST_OP(tname ##_u32, type, uint32_t) \
INSTANTIATE_CAST_OP(tname ##_u64, type, uint64_t) \
INSTANTIATE_CAST_OP(tname ##_i8, type, int8_t)    \
INSTANTIATE_CAST_OP(tname ##_i16, type, int16_t)  \
INSTANTIATE_CAST_OP(tname ##_i32, type, int32_t)  \
INSTANTIATE_CAST_OP(tname ##_i64, type, int64_t)  \

#define INSTANTIATE_ALL(tname, type)              \
INSTANTIATE_CAST_AND_COPY(tname, type) \
INSTANTIATE_ROTATE_HALF(tname, type) \

INSTANTIATE_CAST_AND_COPY(bool, bool)
INSTANTIATE_ALL(f32, float)
INSTANTIATE_ALL(f16, __half)
INSTANTIATE_ALL(i8, int8_t)
INSTANTIATE_ALL(i16, int16_t)
INSTANTIATE_ALL(i32, int32_t)
INSTANTIATE_ALL(i64, int64_t)
INSTANTIATE_CAST_AND_COPY(u8, uint8_t)
INSTANTIATE_CAST_AND_COPY(u16, uint16_t)
INSTANTIATE_CAST_AND_COPY(u32, uint32_t)
INSTANTIATE_CAST_AND_COPY(u64, uint64_t)