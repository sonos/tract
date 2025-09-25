#include <cuda_fp16.h>
#include <cuda_runtime.h>

static __device__ __forceinline__ float op_neg(float x) { return -x; }
static __device__ __forceinline__ __half op_neg(__half x) { return -x; }

static __device__ __forceinline__ float op_abs(float x) { return fabsf(x); }
static __device__ __forceinline__ __half op_abs(__half x) { return __habs(x); }

static __device__ __forceinline__ float op_sqr(float x) { return x * x; }
static __device__ __forceinline__ __half op_sqr(__half x) {
  return __hmul(x, x);
}

static __device__ __forceinline__ float op_sqrt(float x) { return sqrtf(x); }
static __device__ __forceinline__ __half op_sqrt(__half x) { return hsqrt(x); }

static __device__ __forceinline__ float op_rsqrt(float x) {
  return 1.0f / sqrtf(x);
}
static __device__ __forceinline__ __half op_rsqrt(__half x) {
  return hrsqrt(x);
}

static __device__ __forceinline__ float op_recip(float x) { return 1.0f / x; }
static __device__ __forceinline__ __half op_recip(__half x) { return hrcp(x); }

static __device__ __forceinline__ float op_ceil(float x) { return ceilf(x); }
static __device__ __forceinline__ __half op_ceil(__half x) { return hceil(x); }

static __device__ __forceinline__ float op_floor(float x) { return floorf(x); }
static __device__ __forceinline__ __half op_floor(__half x) {
  return hfloor(x);
}

static __device__ __forceinline__ float op_round(float x) { return round(x); }
static __device__ __forceinline__ __half op_round(__half x) {
  return (x < (__half)0.0f) ? hceil(x - (__half)0.5f)
                            : hfloor(x + (__half)0.5f);
}

static __device__ __forceinline__ float op_rint(float x) { return rint(x); }
static __device__ __forceinline__ __half op_rint(__half x) { return hrint(x); }

static __device__ __forceinline__ float op_exp(float x) { return expf(x); }
static __device__ __forceinline__ __half op_exp(__half x) { return hexp(x); }

static __device__ __forceinline__ float op_sin(float x) { return sinf(x); }
static __device__ __forceinline__ __half op_sin(__half x) { return hsin(x); }

static __device__ __forceinline__ float op_sinh(float x) { return sinhf(x); }
static __device__ __forceinline__ __half op_sinh(__half x) {
  return (hexp(x) - hexp(-x)) / (__half)2.0f;
}

static __device__ __forceinline__ float op_asin(float x) { return asinf(x); }
static __device__ __forceinline__ __half op_asin(__half x) {
  return (__half)asinf((float)x);
}

static __device__ __forceinline__ float op_asinh(float x) { return asinhf(x); }
static __device__ __forceinline__ __half op_asinh(__half x) {
  return (__half)asinhf((float)x);
}

static __device__ __forceinline__ float op_cos(float x) { return cosf(x); }
static __device__ __forceinline__ __half op_cos(__half x) { return hcos(x); }

static __device__ __forceinline__ float op_cosh(float x) { return coshf(x); }
static __device__ __forceinline__ __half op_cosh(__half x) {
  return (hexp(x) + hexp(-x)) / (__half)2.0f;
}

static __device__ __forceinline__ float op_acos(float x) { return acosf(x); }
static __device__ __forceinline__ __half op_acos(__half x) {
  return (__half)acosf((float)x);
}

static __device__ __forceinline__ float op_acosh(float x) { return acoshf(x); }
static __device__ __forceinline__ __half op_acosh(__half x) {
  return (__half)acoshf((float)x);
}

static __device__ __forceinline__ float op_tan(float x) { return tanf(x); }
static __device__ __forceinline__ __half op_tan(__half x) {
  return hsin(x) / hcos(x);
}

static __device__ __forceinline__ float op_tanh(float x) { return tanhf(x); }
static __device__ __forceinline__ __half op_tanh(__half x) {
  return (__half)tanhf((float)x);
}

static __device__ __forceinline__ float op_atan(float x) { return atanf(x); }
static __device__ __forceinline__ __half op_atan(__half x) {
  return (__half)atanf((float)x);
}

static __device__ __forceinline__ float op_atanh(float x) { return atanhf(x); }
static __device__ __forceinline__ __half op_atanh(__half x) {
  return (__half)atanhf((float)x);
}

static __device__ __forceinline__ float op_sigmoid(float x) {
  float y = 1.0f / (1.0f + expf(-fabsf(x)));
  return (x < 0.0f) ? 1.0f - y : y;
}

static __device__ __forceinline__ __half op_sigmoid(__half x) {
  __half y = (__half)1.0f / ((__half)1.0f + hexp(-__habs(x)));
  return (x < (__half)0.0f) ? (__half)1.0f - y : y;
  ;
}

static __device__ __forceinline__ float op_ln(float x) { return logf(x); }
static __device__ __forceinline__ __half op_ln(__half x) { return hlog(x); }

static __device__ __forceinline__ float op_erf(float x) { return erff(x); }
static __device__ __forceinline__ __half op_erf(__half x) {
  return (__half)erff((float)x);
}

static __device__ __forceinline__ float op_silu(float x) {
  return (x / (1.0f + expf(-x)));
}
static __device__ __forceinline__ __half op_silu(__half x) {
  return (x / ((__half)1.0f + hexp(-x)));
}

#define DEFINE_UNARY_KERNEL(name, tname, T, OP)                                \
  extern "C" __global__ void name##_##tname(const T *x, T *dst, int k) {       \
    int i = blockIdx.x * blockDim.x + threadIdx.x;                             \
    if (i < k) {                                                               \
      dst[i] = OP(x[i]);                                                       \
    }                                                                          \
  }

#define DEFINE_OP_FOR_ALL_TYPES(name, OP)                                      \
  DEFINE_UNARY_KERNEL(name, f32, float, OP)                                    \
  DEFINE_UNARY_KERNEL(name, f16, __half, OP)

DEFINE_OP_FOR_ALL_TYPES(unary_neg, op_neg)
DEFINE_OP_FOR_ALL_TYPES(unary_abs, op_abs)
DEFINE_OP_FOR_ALL_TYPES(unary_sqr, op_sqr)
DEFINE_OP_FOR_ALL_TYPES(unary_sqrt, op_sqrt)
DEFINE_OP_FOR_ALL_TYPES(unary_rsqrt, op_rsqrt)
DEFINE_OP_FOR_ALL_TYPES(unary_recip, op_recip)
DEFINE_OP_FOR_ALL_TYPES(unary_ceil, op_ceil)
DEFINE_OP_FOR_ALL_TYPES(unary_floor, op_floor)
DEFINE_OP_FOR_ALL_TYPES(unary_round, op_round)
DEFINE_OP_FOR_ALL_TYPES(unary_rint, op_rint)
DEFINE_OP_FOR_ALL_TYPES(unary_sin, op_sin)
DEFINE_OP_FOR_ALL_TYPES(unary_sinh, op_sinh)
DEFINE_OP_FOR_ALL_TYPES(unary_asin, op_asin)
DEFINE_OP_FOR_ALL_TYPES(unary_asinh, op_asinh)
DEFINE_OP_FOR_ALL_TYPES(unary_cos, op_cos)
DEFINE_OP_FOR_ALL_TYPES(unary_cosh, op_cosh)
DEFINE_OP_FOR_ALL_TYPES(unary_acos, op_acos)
DEFINE_OP_FOR_ALL_TYPES(unary_acosh, op_acosh)
DEFINE_OP_FOR_ALL_TYPES(unary_tan, op_tan)
DEFINE_OP_FOR_ALL_TYPES(unary_tanh, op_tanh)
DEFINE_OP_FOR_ALL_TYPES(unary_atan, op_atan)
DEFINE_OP_FOR_ALL_TYPES(unary_atanh, op_atanh)
DEFINE_OP_FOR_ALL_TYPES(unary_exp, op_exp)
DEFINE_OP_FOR_ALL_TYPES(unary_sigmoid, op_sigmoid)
DEFINE_OP_FOR_ALL_TYPES(unary_ln, op_ln)
DEFINE_OP_FOR_ALL_TYPES(unary_erf, op_erf)
DEFINE_OP_FOR_ALL_TYPES(unary_silu, op_silu)
