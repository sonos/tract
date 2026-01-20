#include <cuda_runtime.h>
#include "common.cuh"

template <typename T>
struct OpAdd {
  __device__ __forceinline__ T operator()(T a, T b) const { return a + b; }
};

template <typename T>
struct OpSub {
  __device__ __forceinline__ T operator()(T a, T b) const { return a - b; }
};

template <typename T>
struct OpMul {
  __device__ __forceinline__ T operator()(T a, T b) const { return a * b; }
};

template <typename T>
struct OpDiv {
  __device__ __forceinline__ T operator()(T a, T b) const { return a / b; }
};

template <typename T>
struct OpPow {
  __device__ __forceinline__ T operator()(T a, T b) const {
    return (T)powf((float)a, (float)b);
  }
};

template <typename T>
struct OpMin {
  __device__ __forceinline__ T operator()(T a, T b) const {
    return (T)fmin((float)a, (float)b);
  }
};

template <typename T>
struct OpMax {
  __device__ __forceinline__ T operator()(T a, T b) const {
    return (T)fmax((float)a, (float)b);
  }
};


template <typename T>
struct OpLess {
  __device__ __forceinline__ bool operator()(T a, T b) const { return a < b; }
};

template <typename T>
struct OpLessEqual {
  __device__ __forceinline__ bool operator()(T a, T b) const { return a <= b; }
};

template <typename T>
struct OpGreater {
  __device__ __forceinline__ bool operator()(T a, T b) const { return a > b; }
};

template <typename T>
struct OpGreaterEqual {
  __device__ __forceinline__ bool operator()(T a, T b) const { return a >= b; }
};

template <typename T>
struct OpEquals {
  __device__ __forceinline__ bool operator()(T a, T b) const { return a == b; }
};

template <typename T>
struct OpNotEquals {
  __device__ __forceinline__ bool operator()(T a, T b) const { return a != b; }
};

struct OpAnd {
  __device__ __forceinline__ bool operator()(bool a, bool b) const {
    return a && b;
  }
};

struct OpOr {
  __device__ __forceinline__ bool operator()(bool a, bool b) const {
    return a || b;
  }
};

template <typename T_in, typename T_out, typename Op>
__device__ __forceinline__ void bin_op_generic(
      const T_in* __restrict__ a,
      const T_in* __restrict__ b,
      T_out* __restrict__ out,
      int32_t b_shape_0, int32_t b_shape_1,
      int32_t b_shape_2, int32_t b_shape_3, int32_t b_shape_4,
      int32_t out_shape_0, int32_t out_shape_1,
      int32_t out_shape_2, int32_t out_shape_3, int32_t out_shape_4,
      int32_t a_strides_0, int32_t a_strides_1,
      int32_t a_strides_2, int32_t a_strides_3, int32_t a_strides_4,
      int32_t b_strides_0, int32_t b_strides_1,
      int32_t b_strides_2, int32_t b_strides_3, int32_t b_strides_4,
      int32_t o_strides_0, int32_t o_strides_1,
      int32_t o_strides_2, int32_t o_strides_3, int32_t o_strides_4,
      Op op)
{
      const int32_t n0 = out_shape_0;
      const int32_t n1 = out_shape_1;
      const int32_t n2 = out_shape_2;
      const int32_t n3 = out_shape_3;
      const int32_t n4 = out_shape_4;

      const int32_t total = n0 * n1 * n2 * n3 * n4;

      int32_t tmp = blockIdx.x * blockDim.x + threadIdx.x;
      if(tmp >= total) {
           return;
      }

      const int32_t i4 = tmp % n4; tmp /= n4;
      const int32_t i3 = tmp % n3; tmp /= n3;
      const int32_t i2 = tmp % n2; tmp /= n2;
      const int32_t i1 = tmp % n1; tmp /= n1;
      const int32_t i0 = tmp;

      const int32_t ia =
          i0 * a_strides_0 +
          i1 * a_strides_1 +
          i2 * a_strides_2 +
          i3 * a_strides_3 +
          i4 * a_strides_4;

      const int32_t ib =
          i0 * b_strides_0 +
          i1 * b_strides_1 +
          i2 * b_strides_2 +
          i3 * b_strides_3 +
          i4 * b_strides_4;

      const int32_t io =
          i0 * o_strides_0 +
          i1 * o_strides_1 +
          i2 * o_strides_2 +
          i3 * o_strides_3 +
          i4 * o_strides_4;

      out[io] = op(a[ia], b[ib]);
}

template <typename T_in, typename T_out, typename Op>
__device__ __forceinline__ void bin_op_large(
      const T_in* __restrict__ a,
      const T_in* __restrict__ b,
      T_out* __restrict__ out,
      int32_t b_shape_0, int32_t b_shape_1,
      int32_t b_shape_2, int32_t b_shape_3, int32_t b_shape_4,
      int32_t out_shape_0, int32_t out_shape_1,
      int32_t out_shape_2, int32_t out_shape_3, int32_t out_shape_4,
      int32_t a_strides_0, int32_t a_strides_1,
      int32_t a_strides_2, int32_t a_strides_3, int32_t a_strides_4,
      int32_t b_strides_0, int32_t b_strides_1,
      int32_t b_strides_2, int32_t b_strides_3, int32_t b_strides_4,
      int32_t o_strides_0, int32_t o_strides_1,
      int32_t o_strides_2, int32_t o_strides_3, int32_t o_strides_4,
      Op op)
{
      const int32_t n0 = out_shape_0;
      const int32_t n1 = out_shape_1;
      const int32_t n2 = out_shape_2;
      const int32_t n3 = out_shape_3;
      const int32_t n4 = out_shape_4;

      const int32_t i3 = blockIdx.x;
      const int32_t i2 = blockIdx.y;
      const int32_t bz = blockIdx.z;

      const int32_t i1 = bz % n1;
      const int32_t i0 = bz / n1;

      if (i0 >= n0 || i1 >= n1 || i2 >= n2 || i3 >= n3) return;

      // Base offsets for (i0,i1,i2,i3)
      const int32_t ia_base =
          i0 * a_strides_0 +
          i1 * a_strides_1 +
          i2 * a_strides_2 +
          i3 * a_strides_3;
      const int32_t ib_base =
          i0 * b_strides_0 +
          i1 * b_strides_1 +
          i2 * b_strides_2 +
          i3 * b_strides_3;
      const int32_t io_base =
          i0 * o_strides_0 +
          i1 * o_strides_1 +
          i2 * o_strides_2 +
          i3 * o_strides_3;

      // Each thread handles a strided subset of i4
      for (int32_t i4 = threadIdx.x; i4 < n4; i4 += blockDim.x) {
          const int32_t ia = ia_base + i4 * a_strides_4;
          const int32_t ib = ib_base + i4 * b_strides_4;
          const int32_t io = io_base + i4 * o_strides_4;

          out[io] = op(a[ia], b[ib]);
      }
}

#define DEFINE_BINARY_KERNEL(name, tname, T_in, T_out, OP_TYPE)                 \
  extern "C"                                                                    \
  {                                                                             \
    __global__ void name##_generic_##tname(                                     \
        const T_in* __restrict__ a,                                             \
        const T_in* __restrict__ b,                                             \
        T_out* __restrict__ out,                                                \
        int32_t b_shape_0, int32_t b_shape_1,                                   \
        int32_t b_shape_2, int32_t b_shape_3, int32_t b_shape_4,                \
        int32_t out_shape_0, int32_t out_shape_1,                               \
        int32_t out_shape_2, int32_t out_shape_3, int32_t out_shape_4,          \
        int32_t a_strides_0, int32_t a_strides_1,                               \
        int32_t a_strides_2, int32_t a_strides_3, int32_t a_strides_4,          \
        int32_t b_strides_0, int32_t b_strides_1,                               \
        int32_t b_strides_2, int32_t b_strides_3, int32_t b_strides_4,          \
        int32_t o_strides_0, int32_t o_strides_1,                               \
        int32_t o_strides_2, int32_t o_strides_3, int32_t o_strides_4) {        \
        bin_op_generic<T_in, T_out, OP_TYPE>(                                   \
              a, b, out,                                                        \
              b_shape_0, b_shape_1, b_shape_2, b_shape_3, b_shape_4,            \
              out_shape_0, out_shape_1, out_shape_2, out_shape_3, out_shape_4,  \
              a_strides_0, a_strides_1, a_strides_2, a_strides_3, a_strides_4,  \
              b_strides_0, b_strides_1, b_strides_2, b_strides_3, b_strides_4,  \
              o_strides_0, o_strides_1, o_strides_2, o_strides_3, o_strides_4,  \
              OP_TYPE{});                                                       \
    }                                                                           \
                                                                                \
    __global__ void name##_large_##tname(                                       \
        const T_in* __restrict__ a,                                             \
        const T_in* __restrict__ b,                                             \
        T_out* __restrict__ out,                                                \
        int32_t b_shape_0, int32_t b_shape_1,                                   \
        int32_t b_shape_2, int32_t b_shape_3, int32_t b_shape_4,                \
        int32_t out_shape_0, int32_t out_shape_1,                               \
        int32_t out_shape_2, int32_t out_shape_3, int32_t out_shape_4,          \
        int32_t a_strides_0, int32_t a_strides_1,                               \
        int32_t a_strides_2, int32_t a_strides_3, int32_t a_strides_4,          \
        int32_t b_strides_0, int32_t b_strides_1,                               \
        int32_t b_strides_2, int32_t b_strides_3, int32_t b_strides_4,          \
        int32_t o_strides_0, int32_t o_strides_1,                               \
        int32_t o_strides_2, int32_t o_strides_3, int32_t o_strides_4) {        \
        bin_op_large<T_in, T_out, OP_TYPE>(                                     \
              a, b, out,                                                        \
              b_shape_0, b_shape_1, b_shape_2, b_shape_3, b_shape_4,            \
              out_shape_0, out_shape_1, out_shape_2, out_shape_3, out_shape_4,  \
              a_strides_0, a_strides_1, a_strides_2, a_strides_3, a_strides_4,  \
              b_strides_0, b_strides_1, b_strides_2, b_strides_3, b_strides_4,  \
              o_strides_0, o_strides_1, o_strides_2, o_strides_3, o_strides_4,  \
              OP_TYPE{});                                                       \
    }                                                                           \
  }

#define DEFINE_ARITHMETIC_OP(name, OP)                                         \
  DEFINE_BINARY_KERNEL(name, f32, float, float, OP<float>)                     \
  DEFINE_BINARY_KERNEL(name, f16, __half, __half, OP<half>)                    \
  DEFINE_BINARY_KERNEL(name, u8, uint8_t, uint8_t, OP<uint8_t>)                \
  DEFINE_BINARY_KERNEL(name, u16, uint16_t, uint16_t, OP<uint16_t>)            \
  DEFINE_BINARY_KERNEL(name, u32, uint32_t, uint32_t, OP<uint32_t>)            \
  DEFINE_BINARY_KERNEL(name, u64, uint64_t, uint64_t, OP<uint64_t>)            \
  DEFINE_BINARY_KERNEL(name, i8, int8_t, int8_t, OP<int8_t>)                   \
  DEFINE_BINARY_KERNEL(name, i16, int16_t, int16_t, OP<int16_t>)               \
  DEFINE_BINARY_KERNEL(name, i32, int32_t, int32_t, OP<int32_t>)               \
  DEFINE_BINARY_KERNEL(name, i64, int64_t, int64_t, OP<int64_t>)

#define DEFINE_COMP_OP(name, OP)                                               \
  DEFINE_BINARY_KERNEL(name, f32, float, bool, OP<float>)                      \
  DEFINE_BINARY_KERNEL(name, f16, __half, bool, OP<half>)                      \
  DEFINE_BINARY_KERNEL(name, u8, uint8_t, bool, OP<uint8_t>)                   \
  DEFINE_BINARY_KERNEL(name, u16, uint16_t, bool, OP<uint16_t>)                \
  DEFINE_BINARY_KERNEL(name, u32, uint32_t, bool, OP<uint32_t>)                \
  DEFINE_BINARY_KERNEL(name, u64, uint64_t, bool, OP<uint64_t>)                \
  DEFINE_BINARY_KERNEL(name, i8, int8_t, bool, OP<int8_t>)                     \
  DEFINE_BINARY_KERNEL(name, i16, int16_t, bool, OP<int16_t>)                  \
  DEFINE_BINARY_KERNEL(name, i32, int32_t, bool, OP<int32_t>)                  \
  DEFINE_BINARY_KERNEL(name, i64, int64_t, bool, OP<int64_t>)

#define DEFINE_LOGIC_OP(name, OP)                                              \
  DEFINE_BINARY_KERNEL(name, bool, bool, bool, OP)

DEFINE_ARITHMETIC_OP(binary_add, OpAdd)
DEFINE_ARITHMETIC_OP(binary_sub, OpSub)
DEFINE_ARITHMETIC_OP(binary_mul, OpMul)
DEFINE_ARITHMETIC_OP(binary_div, OpDiv)
DEFINE_ARITHMETIC_OP(binary_pow, OpPow)
DEFINE_ARITHMETIC_OP(binary_min, OpMin)
DEFINE_ARITHMETIC_OP(binary_max, OpMax)

DEFINE_COMP_OP(binary_less, OpLess)
DEFINE_COMP_OP(binary_less_equal, OpLessEqual)
DEFINE_COMP_OP(binary_greater, OpGreater)
DEFINE_COMP_OP(binary_greater_equal, OpGreaterEqual)
DEFINE_COMP_OP(binary_equals, OpEquals)
DEFINE_COMP_OP(binary_not_equals, OpNotEquals)

DEFINE_LOGIC_OP(binary_and, OpAnd)
DEFINE_LOGIC_OP(binary_or, OpOr)
