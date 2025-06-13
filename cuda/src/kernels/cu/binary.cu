#include <cuda_fp16.h>
#include <math.h>

static __device__ __forceinline__ float op_add(const float a, const float b) { return a + b; }
static __device__ __forceinline__ float op_add(const __half a, const __half b) { return a + b; }

static __device__ __forceinline__ float op_sub(const float a, const float b) { return a - b; }
static __device__ __forceinline__ float op_sub(const __half a, const __half b) { return a - b; }

static __device__ __forceinline__ float op_mul(const float a, const float b) { return a * b; }
static __device__ __forceinline__ float op_mul(const __half a, const __half b) { return a * b; }

static __device__ __forceinline__ float op_div(const float a, const float b) { return a / b; }
static __device__ __forceinline__ float op_div(const __half a, const __half b) { return a / b; }

static __device__ __forceinline__ float op_pow(const float a, const float b) { return powf(a, b); }
static __device__ __forceinline__ float op_pow(const __half a, const __half b) { return powf((float)a, (float)b); }

static __device__ __forceinline__ bool op_less(const float a, const float b) { return a < b; }
static __device__ __forceinline__ bool op_less(const __half a, const __half b) { return a < b; }

static __device__ __forceinline__ bool op_less_equal(const float a, const float b) { return a <= b; }
static __device__ __forceinline__ bool op_less_equal(const __half a, const __half b) { return a <= b; }

static __device__ __forceinline__ bool op_greater(const float a, const float b) { return a > b; }
static __device__ __forceinline__ bool op_greater(const __half a, const __half b) { return a > b; }

static __device__ __forceinline__ bool op_greater_equal(const float a, const float b) { return a >= b; }
static __device__ __forceinline__ bool op_greater_equal(const __half a, const __half b) { return a >= b; }

static __device__ __forceinline__ bool op_equals(const float a, const float b) { return a == b; }
static __device__ __forceinline__ bool op_equals(const __half a, const __half b) { return a == b; }

static __device__ __forceinline__ bool op_not_equals(const float a, const float b) { return a != b; }
static __device__ __forceinline__ bool op_not_equals(const __half a, const __half b) { return a != b; }

static __device__ __forceinline__ bool op_and(const bool a, const bool b) { return a && b; }
static __device__ __forceinline__ bool op_or(const bool a, const bool b) { return a || b; }


#define DEFINE_BINARY_KERNEL(name, tname, T_in, T_out, OP) \
    extern "C" __global__ void name##_##tname(const T_in* a, const T_in* b, T_out * out,                        \
                                            int b_shape_0, int b_shape_1, int b_shape_2, int b_shape_3,           \
                                            int out_shape_0, int out_shape_1, int out_shape_2, int out_shape_3,   \
                                            int a_strides_0, int a_strides_1, int a_strides_2, int a_strides_3,                    \
                                            int b_strides_0, int b_strides_1, int b_strides_2, int b_strides_3,                    \
                                            int o_strides_0, int o_strides_1, int o_strides_2, int out_strides_3                     \
                                            ) {                                                                   \
        const int i0s = blockDim.x*blockIdx.x + threadIdx.x;                                                      \
        const int i1 = (blockDim.y*blockIdx.y + threadIdx.y);                                                     \
        const int i2 = (blockDim.z*blockIdx.z + threadIdx.z) / out_shape_0;                                       \
        const int i3 = (blockDim.z*blockIdx.z + threadIdx.z) % out_shape_0;                                       \
                                                                                                                  \
        if (i0s >= out_shape_3 || i1 >= out_shape_2 || i2 >= out_shape_1 || i3 >= out_shape_0) {                  \
            return;                                                                                               \
        }                                                                                                         \
                                                                                                                  \
        const size_t i_a = i3*a_strides_0 + i2*a_strides_1 + i1*a_strides_2;                                   \
        const size_t i_b = i3*b_strides_0 + i2*b_strides_1 + i1*b_strides_2;                                   \
        const size_t i_out = i3*o_strides_0 + i2*o_strides_1 + i1*o_strides_2;                              \
                                                                                                                  \
        const T_in * a_row = a + i_a;                                                                            \
        const T_in * b_row = b + i_b;                                                                            \
        T_out * out_row = out + i_out;                                                                            \
                                                                                                                  \
        for (int i0 = i0s; i0 < out_shape_3; i0 += blockDim.x*gridDim.x) {                                      \
            out_row[i0] = OP(a_row[i0 * a_strides_3], b_row[i0 * b_strides_3]);                                 \
        }                                                                                                         \
    }

#define DEFINE_ARITHMETIC_OP(name, OP) \
    DEFINE_BINARY_KERNEL(name, f32, float, float, OP) \
    DEFINE_BINARY_KERNEL(name, f16, __half, __half, OP) \

#define DEFINE_COMP_OP(name, OP) \
    DEFINE_BINARY_KERNEL(name, f32, float, bool, OP) \
    DEFINE_BINARY_KERNEL(name, f16, __half, bool, OP) \

#define DEFINE_LOGIC_OP(name, OP) \
    DEFINE_BINARY_KERNEL(name, bool, bool, bool, OP) \

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

