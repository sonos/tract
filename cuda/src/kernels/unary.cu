#include <cuda_fp16.h>
#include <math.h>



//template <typename T>
//__device__ T op_sgn(T x) {
//    return (T)(x > 0.f ? 1.f : ((x < 0.f ? -1.f : 0.f)));
//}
//
//template <typename T>
//__device__ T op_neg(T x) {
//    return -x;
//}
//
//template <typename T>
//__device__ T op_step(T x) {
//    return x > 0.0f;
//}
//
//template <typename T>
//__device__ T op_gelu(T x) {
//    const float GELU_COEF_A    = 0.044715f;
//    const float SQRT_2_OVER_PI = 0.79788456080286535587989211986876f;
//
//    return (T)0.5f*x*(1.0f + tanhf(SQRT_2_OVER_PI*x*(1.0f + GELU_COEF_A*x*x)));
//}
//
//template <typename T>
//__device__ T op_gelu_quick(T x) {
//    const float GELU_QUICK_COEF = -1.702f;
//
//    return (T)((float)x * (1.0f / (1.0f + expf(GELU_QUICK_COEF * (float)x))));
//}
//
//template <typename T>
//__device__ T op_silu(T x) {
//    return (T)((float)x / (1.0f + expf((float) -x)));
//}
//
//template <typename T>
//__device__ T op_tanh(T x) {
//    return (T)tanhf((float)x);
//}
//
//template <typename T>
//__device__ T op_relu(T x) {
//    return (T)fmaxf((float)x, 0);
//}
//
//template <typename T>
//__device__ T op_sigmoid(T x) {
//    return (T)(1.0f / (1.0f + expf((float)-x)));
//}
//
//template <typename T>
//__device__ T op_exp(T x) {
//    return (T)expf((float)x);
//}
//
//template <typename T>
//__device__ T op_sqr(T x) {
//    return x * x;
//}
//
//template <typename T>
//__device__ T op_sqrt(T x) {
//    return (T)sqrtf((float)x);
//}
//
//template <typename T>
//__device__ T op_sin(T x) {
//    return (T)sinf((float)x);
//}
//
//template <typename T>
//__device__ T op_cos(T x) {
//    return (T)cosf((float)x);
//}
//
//template <typename T>
//__device__ T op_log(float x) {
//    return (T)logf((float)x);
//}
//

static __device__ __forceinline__ float op_sqr(float x) { return x * x; }
static __device__ __forceinline__ __half op_sqr(__half x) { return __hmul(x, x); }

static __device__ __forceinline__ float op_abs(float x) { return fabsf(x); }
static __device__ __forceinline__ __half op_abs(__half x) { return __habs(x); }

static __device__ __forceinline__ float op_silu(float x) { return (x / (1.0f + expf(-x))); }
static __device__ __forceinline__ __half op_silu(__half x) { return (x / ((__half)1.0f + hexp(-x))); }

#define DEFINE_UNARY_KERNEL(name, tname, T, OP) \
    extern "C" __global__ void name##_##tname(const T* x, T* dst, int k) { \
        int i = blockIdx.x * blockDim.x + threadIdx.x; \
        if (i < k) { \
            dst[i] = OP(x[i]); \
        } \
    }

#define DEFINE_OP_FOR_ALL_TYPES(name, OP) \
    DEFINE_UNARY_KERNEL(name, f32, float, OP) \
    DEFINE_UNARY_KERNEL(name, f16, __half, OP) \


DEFINE_OP_FOR_ALL_TYPES(unary_abs, op_abs)
DEFINE_OP_FOR_ALL_TYPES(unary_sqr, op_sqr)
DEFINE_OP_FOR_ALL_TYPES(unary_silu, op_silu)