#include <cuda_runtime.h>
#include "common.cuh"

template <typename T>
__device__ __forceinline__ T op_add(const T a, const T b) {
  return a + b;
}

template <typename T>
__device__ __forceinline__ T op_sub(const T a, const T b) {
  return a - b;
}

template <typename T>
__device__ __forceinline__ T op_mul(const T a, const T b) {
  return a * b;
}

template <typename T>
__device__ __forceinline__ T op_div(const T a, const T b) {
  return a / b;
}

template <typename T>
__device__ __forceinline__ T op_pow(const T a, const T b) {
  return powf((float)a, (float)b);
}

template <typename T>
__device__ __forceinline__ bool op_less(const T a, const T b) {
  return a < b;
}

template <typename T>
__device__ __forceinline__ bool op_less_equal(const T a, const T b) {
  return a <= b;
}

template <typename T>
__device__ __forceinline__ bool op_greater(const T a, const T b) {
  return a > b;
}

template <typename T>
__device__ __forceinline__ bool op_greater_equal(const T a, const T b) {
  return a >= b;
}

template <typename T>
__device__ __forceinline__ bool op_equals(const T a, const T b) {
  return a == b;
}

template <typename T>
__device__ __forceinline__ bool op_not_equals(const T a, const T b) {
  return a != b;
}

static __device__ __forceinline__ bool op_and(const bool a, const bool b) {
  return a && b;
}
static __device__ __forceinline__ bool op_or(const bool a, const bool b) {
  return a || b;
}

#define DEFINE_BINARY_KERNEL(name, tname, T_in, T_out, OP)                     \
  extern "C" __global__ void name##_##tname(                                   \
      const T_in *a, const T_in *b, T_out *out, int b_shape_0, int b_shape_1,  \
      int b_shape_2, int b_shape_3, int out_shape_0, int out_shape_1,          \
      int out_shape_2, int out_shape_3, int a_strides_0, int a_strides_1,      \
      int a_strides_2, int a_strides_3, int b_strides_0, int b_strides_1,      \
      int b_strides_2, int b_strides_3, int o_strides_0, int o_strides_1,      \
      int o_strides_2, int out_strides_3) {                                    \
    const int thread_ix_x = blockDim.x * blockIdx.x + threadIdx.x;             \
    const int thread_ix_y = (blockDim.y * blockIdx.y + threadIdx.y);           \
    const int thread_ix_z =                                                    \
        (blockDim.z * blockIdx.z + threadIdx.z) / out_shape_0;                 \
    const int thread_ix_b =                                                    \
        (blockDim.z * blockIdx.z + threadIdx.z) % out_shape_0;                 \
                                                                               \
    if (thread_ix_x >= out_shape_3 || thread_ix_y >= out_shape_2 ||            \
        thread_ix_z >= out_shape_1 || thread_ix_b >= out_shape_0) {            \
      return;                                                                  \
    }                                                                          \
                                                                               \
    const size_t i_a = thread_ix_b * a_strides_0 + thread_ix_z * a_strides_1 + \
                       thread_ix_y * a_strides_2;                              \
    const size_t i_b = thread_ix_b * b_strides_0 + thread_ix_z * b_strides_1 + \
                       thread_ix_y * b_strides_2;                              \
    const size_t i_out = thread_ix_b * o_strides_0 +                           \
                         thread_ix_z * o_strides_1 +                           \
                         thread_ix_y * o_strides_2;                            \
                                                                               \
    const T_in *a_row = a + i_a;                                               \
    const T_in *b_row = b + i_b;                                               \
    T_out *out_row = out + i_out;                                              \
                                                                               \
    for (int i0 = thread_ix_x; i0 < out_shape_3;                               \
         i0 += blockDim.x * gridDim.x) {                                       \
      out_row[i0] = OP(a_row[i0 * a_strides_3], b_row[i0 * b_strides_3]);      \
    }                                                                          \
  }

#define DEFINE_ARITHMETIC_OP(name, OP)                                         \
  DEFINE_BINARY_KERNEL(name, f32, float, float, OP)                            \
  DEFINE_BINARY_KERNEL(name, f16, __half, __half, OP)                          \
  DEFINE_BINARY_KERNEL(name, u8, uint8_t, uint8_t, OP)                         \
  DEFINE_BINARY_KERNEL(name, u16, uint16_t, uint16_t, OP)                      \
  DEFINE_BINARY_KERNEL(name, u32, uint32_t, uint32_t, OP)                      \
  DEFINE_BINARY_KERNEL(name, u64, uint64_t, uint64_t, OP)                      \
  DEFINE_BINARY_KERNEL(name, i8, int8_t, int8_t, OP)                           \
  DEFINE_BINARY_KERNEL(name, i16, int16_t, int16_t, OP)                        \
  DEFINE_BINARY_KERNEL(name, i32, int32_t, int32_t, OP)                        \
  DEFINE_BINARY_KERNEL(name, i64, int64_t, int64_t, OP)

#define DEFINE_COMP_OP(name, OP)                                               \
  DEFINE_BINARY_KERNEL(name, f32, float, bool, OP)                             \
  DEFINE_BINARY_KERNEL(name, f16, __half, bool, OP)                            \
  DEFINE_BINARY_KERNEL(name, u8, uint8_t, bool, OP)                            \
  DEFINE_BINARY_KERNEL(name, u16, uint16_t, bool, OP)                          \
  DEFINE_BINARY_KERNEL(name, u32, uint32_t, bool, OP)                          \
  DEFINE_BINARY_KERNEL(name, u64, uint64_t, bool, OP)                          \
  DEFINE_BINARY_KERNEL(name, i8, int8_t, bool, OP)                             \
  DEFINE_BINARY_KERNEL(name, i16, int16_t, bool, OP)                           \
  DEFINE_BINARY_KERNEL(name, i32, int32_t, bool, OP)                           \
  DEFINE_BINARY_KERNEL(name, i64, int64_t, bool, OP)

#define DEFINE_LOGIC_OP(name, OP)                                              \
  DEFINE_BINARY_KERNEL(name, bool, bool, bool, OP)

DEFINE_ARITHMETIC_OP(binary_add, op_add)
DEFINE_ARITHMETIC_OP(binary_sub, op_sub)
DEFINE_ARITHMETIC_OP(binary_mul, op_mul)
DEFINE_ARITHMETIC_OP(binary_div, op_div)
DEFINE_ARITHMETIC_OP(binary_pow, op_pow)

DEFINE_COMP_OP(binary_less, op_less)
DEFINE_COMP_OP(binary_less_equal, op_less_equal)
DEFINE_COMP_OP(binary_greater, op_greater)
DEFINE_COMP_OP(binary_greater_equal, op_greater_equal)
DEFINE_COMP_OP(binary_equals, op_equals)
DEFINE_COMP_OP(binary_not_equals, op_not_equals)

DEFINE_LOGIC_OP(binary_and, op_and)
DEFINE_LOGIC_OP(binary_or, op_or)
