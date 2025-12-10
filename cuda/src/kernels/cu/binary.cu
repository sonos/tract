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
  extern "C" __global__ \
  void name##_##tname( \
      const T_in* __restrict__ a, \
      const T_in* __restrict__ b, \
      T_out* __restrict__ out, \
      int32_t b_shape_0, int32_t b_shape_1, \
      int32_t b_shape_2, int32_t b_shape_3, int32_t b_shape_4, \
      int32_t out_shape_0, int32_t out_shape_1, \
      int32_t out_shape_2, int32_t out_shape_3, int32_t out_shape_4, \
      int32_t a_strides_0, int32_t a_strides_1, \
      int32_t a_strides_2, int32_t a_strides_3, int32_t a_strides_4, \
      int32_t b_strides_0, int32_t b_strides_1, \
      int32_t b_strides_2, int32_t b_strides_3, int32_t b_strides_4, \
      int32_t o_strides_0, int32_t o_strides_1, \
      int32_t o_strides_2, int32_t o_strides_3, int32_t o_strides_4) { \
 \
    const int32_t n0 = out_shape_0; \
    const int32_t n1 = out_shape_1; \
    const int32_t n2 = out_shape_2; \
    const int32_t n3 = out_shape_3; \
    const int32_t n4 = out_shape_4; \
 \
    const int32_t total = n0 * n1 * n2 * n3 * n4; \
 \
    for (int32_t linear_idx = blockIdx.x * blockDim.x + threadIdx.x; \
        linear_idx < total; \
        linear_idx += (int32_t)blockDim.x * gridDim.x) { \
 \
      int32_t tmp = linear_idx; \
 \
      const int32_t i4 = tmp % n4; tmp /= n4; \
      const int32_t i3 = tmp % n3; tmp /= n3; \
      const int32_t i2 = tmp % n2; tmp /= n2; \
      const int32_t i1 = tmp % n1; tmp /= n1; \
      const int32_t i0 = tmp; \
 \
      const int32_t ia = \
          i0 * a_strides_0 + \
          i1 * a_strides_1 + \
          i2 * a_strides_2 + \
          i3 * a_strides_3 + \
          i4 * a_strides_4; \
 \
      const int32_t ib = \
          i0 * b_strides_0 + \
          i1 * b_strides_1 + \
          i2 * b_strides_2 + \
          i3 * b_strides_3 + \
          i4 * b_strides_4; \
 \
      const int32_t io = \
          i0 * o_strides_0 + \
          i1 * o_strides_1 + \
          i2 * o_strides_2 + \
          i3 * o_strides_3 + \
          i4 * o_strides_4; \
 \
      out[io] = OP(a[ia], b[ib]); \
    } \
  } \

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
