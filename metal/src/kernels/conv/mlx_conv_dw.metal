// Depthwise and Winograd convolution kernels, ported from Apple MLX
// (https://github.com/ml-explore/mlx) @ fb5133e1049cc1482a330cd3a7135fda62a74b0d,
// MIT License, Copyright (c) 2023-2025 Apple Inc.
//
// Mechanical flattening of the MLX include closure for
// backend/metal/kernels/conv.metal, minus the bf16 instantiations, since tract
// has no bf16 datum type. Resync by re-flattening; do not hand-edit.
//
// clang-format off

// ===== mlx/backend/metal/kernels/conv.metal =====
// Copyright © 2023-2024 Apple Inc.

#include <metal_simdgroup>
#include <metal_simdgroup_matrix>
#include <metal_stdlib>


// ===== mlx/backend/metal/kernels/steel/conv/params.h =====
// Copyright © 2024 Apple Inc.


template <int NDIM>
struct MLXConvParams {
  int N; // Batch size
  int C; // In channels
  int O; // Out channels
  int iS[NDIM]; // Input spatial dim
  int wS[NDIM]; // Weight spatial dim
  int oS[NDIM]; // Output spatial dim
  int str[NDIM]; // Kernel strides
  int pad[NDIM]; // Input padding
  int kdil[NDIM]; // Kernel dilation
  int idil[NDIM]; // Input dilation
  int64_t in_strides[NDIM + 2]; // In strides
  int64_t wt_strides[NDIM + 2]; // Wt strides
  int64_t out_strides[NDIM + 2]; // Out strides
  int groups; // Input channel groups
  bool flip;

  static MLXConvParams<NDIM>
  with_padded_channels(MLXConvParams<NDIM> other, int pad_out, int pad_in) {
    MLXConvParams<NDIM> params = other;

    // Update strides
    for (int i = 0; i < NDIM + 1; i++) {
      params.in_strides[i] =
          (params.in_strides[i] / params.C) * (params.C + pad_in);
      params.wt_strides[i] =
          (params.wt_strides[i] / params.C) * (params.C + pad_in);
      params.out_strides[i] =
          (params.out_strides[i] / params.O) * (params.O + pad_out);
    }
    params.in_strides[NDIM + 1] = 1;
    params.wt_strides[NDIM + 1] = 1;
    params.out_strides[NDIM + 1] = 1;

    // Update channels
    params.C += pad_in;
    params.O += pad_out;

    return params;
  };
};

namespace mlx {
namespace steel {

struct ImplicitGemmConv2DParams {
  const int M;
  const int N;
  const int K;

  const int gemm_k_iterations;

  const int inp_jump_w;
  const int inp_jump_h;
  const int inp_jump_c;

  const int tiles_n;
  const int tiles_m;
  const int swizzle_log;
};

struct ImplicitGemmConv3DParams {
  const int M;
  const int N;
  const int K;

  const int gemm_k_iterations;

  const int inp_jump_w;
  const int inp_jump_h;
  const int inp_jump_d;
  const int inp_jump_c;

  const int tiles_n;
  const int tiles_m;
  const int swizzle_log;
};

struct Conv2DGeneralJumpParams {
  const int f_wgt_jump_h;
  const int f_wgt_jump_w;

  const int f_out_jump_h;
  const int f_out_jump_w;

  const int adj_out_h;
  const int adj_out_w;
  const int adj_out_hw;
  const int adj_implicit_m;
};

struct Conv2DGeneralBaseInfo {
  int weight_base;
  int weight_size;
};

} // namespace steel
} // namespace mlx


// ===== mlx/backend/metal/kernels/utils.h =====
// Copyright © 2023-2024 Apple Inc.


#include <metal_math>


// ===== mlx/backend/metal/kernels/bf16.h =====
// Copyright © 2023 Apple Inc.


#include <metal_stdlib>

using namespace metal;

typedef bfloat bfloat16_t;
inline uint16_t bfloat16_to_uint16(const bfloat16_t x) {
  return as_type<uint16_t>(x);
}

inline bfloat16_t uint16_to_bfloat16(const uint16_t x) {
  return as_type<bfloat16_t>(x);
}


// ===== mlx/backend/metal/kernels/bf16_math.h =====
// Copyright © 2023 Apple Inc.


///////////////////////////////////////////////////////////////////////////////
// Metal math for bfloat16
///////////////////////////////////////////////////////////////////////////////

/*

Following the Metal Shading Language Specification (Metal 3.1)

"bfloat is an extended itypeing point type that only allows implicit conversion
 to a type of greater itypeing point rank. While bfloat can be implicitly
 converted to itype, it cannot be implicitly converted to half, and neither
 itype nor half can be implicitly converted to bfloat."

Further, as far as I can tell, the stdlib math/simd functions are not defined
for bfloat and calling with an argument of type bfloat will result in that
argument getting implicitly converted to itype which then returns an output
that is (likely) a itype which cannot be implicitly converted into a bfloat

This leads to situations where
bfloat a = 5.0bf;
bfloat b = metal::abs(a); // this will throw an error since abs return itype
bfloat c = static_cast<bfloat>(metal::abs(a)); // this is fine

For the moment, I will be adding overloaded instantiations of the math
functions to accordingly automatically handle the casting

*/

#define instantiate_metal_math_funcs(itype, otype, ctype, mfast)               \
                                                                               \
  METAL_FUNC otype abs(itype x) {                                              \
    return static_cast<otype>(__metal_fabs(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype acos(itype x) {                                             \
    return static_cast<otype>(__metal_acos(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype acosh(itype x) {                                            \
    return static_cast<otype>(__metal_acosh(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype asin(itype x) {                                             \
    return static_cast<otype>(__metal_asin(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype asinh(itype x) {                                            \
    return static_cast<otype>(__metal_asinh(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype atan(itype y_over_x) {                                      \
    return static_cast<otype>(                                                 \
        __metal_atan(static_cast<ctype>(y_over_x), mfast));                    \
  }                                                                            \
  METAL_FUNC otype atan2(itype y, itype x) {                                   \
    return static_cast<otype>(                                                 \
        __metal_atan2(static_cast<ctype>(y), static_cast<ctype>(x), mfast));   \
  }                                                                            \
  METAL_FUNC otype atanh(itype x) {                                            \
    return static_cast<otype>(__metal_atanh(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype ceil(itype x) {                                             \
    return static_cast<otype>(__metal_ceil(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype cos(itype x) {                                              \
    return static_cast<otype>(__metal_cos(static_cast<ctype>(x), mfast));      \
  }                                                                            \
  METAL_FUNC otype cosh(itype x) {                                             \
    return static_cast<otype>(__metal_cosh(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype cospi(itype x) {                                            \
    return static_cast<otype>(__metal_cospi(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype divide(itype x, itype y) {                                  \
    return static_cast<otype>(                                                 \
        __metal_divide(static_cast<ctype>(x), static_cast<ctype>(y), mfast));  \
  }                                                                            \
  METAL_FUNC otype exp(itype x) {                                              \
    return static_cast<otype>(__metal_exp(static_cast<ctype>(x), mfast));      \
  }                                                                            \
  METAL_FUNC otype exp10(itype x) {                                            \
    return static_cast<otype>(__metal_exp10(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype exp2(itype x) {                                             \
    return static_cast<otype>(__metal_exp2(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype fabs(itype x) {                                             \
    return static_cast<otype>(__metal_fabs(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype fdim(itype x, itype y) {                                    \
    ctype t = static_cast<ctype>(x - y);                                       \
    return static_cast<otype>(select(t, ctype(0), t < ctype(0) || x == y));    \
  }                                                                            \
  METAL_FUNC otype floor(itype x) {                                            \
    return static_cast<otype>(__metal_floor(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype fma(itype x, itype y, itype z) {                            \
    return static_cast<otype>(__metal_fma(                                     \
        static_cast<ctype>(x), static_cast<ctype>(y), static_cast<ctype>(z))); \
  }                                                                            \
  METAL_FUNC otype fmax(itype x, itype y) {                                    \
    return static_cast<otype>(                                                 \
        __metal_fmax(static_cast<ctype>(x), static_cast<ctype>(y), mfast));    \
  }                                                                            \
  METAL_FUNC otype fmax3(itype x, itype y, itype z) {                          \
    return static_cast<otype>(__metal_fmax3(                                   \
        static_cast<ctype>(x),                                                 \
        static_cast<ctype>(y),                                                 \
        static_cast<ctype>(z),                                                 \
        mfast));                                                               \
  }                                                                            \
  METAL_FUNC otype fmedian3(itype x, itype y, itype z) {                       \
    return static_cast<otype>(__metal_fmedian3(                                \
        static_cast<ctype>(x),                                                 \
        static_cast<ctype>(y),                                                 \
        static_cast<ctype>(z),                                                 \
        mfast));                                                               \
  }                                                                            \
  METAL_FUNC otype fmin(itype x, itype y) {                                    \
    return static_cast<otype>(                                                 \
        __metal_fmin(static_cast<ctype>(x), static_cast<ctype>(y), mfast));    \
  }                                                                            \
  METAL_FUNC otype fmin3(itype x, itype y, itype z) {                          \
    return static_cast<otype>(__metal_fmin3(                                   \
        static_cast<ctype>(x),                                                 \
        static_cast<ctype>(y),                                                 \
        static_cast<ctype>(z),                                                 \
        mfast));                                                               \
  }                                                                            \
  METAL_FUNC otype fmod(itype x, itype y) {                                    \
    return static_cast<otype>(                                                 \
        __metal_fmod(static_cast<ctype>(x), static_cast<ctype>(y), mfast));    \
  }                                                                            \
  METAL_FUNC otype fract(itype x) {                                            \
    return static_cast<otype>(__metal_fract(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype frexp(itype x, thread int& exp) {                           \
    return static_cast<otype>(__metal_frexp(static_cast<ctype>(x), &exp));     \
  }                                                                            \
  METAL_FUNC otype ldexp(itype x, int k) {                                     \
    return static_cast<otype>(__metal_ldexp(static_cast<ctype>(x), k, mfast)); \
  }                                                                            \
  METAL_FUNC otype log(itype x) {                                              \
    return static_cast<otype>(__metal_log(static_cast<ctype>(x), mfast));      \
  }                                                                            \
  METAL_FUNC otype log10(itype x) {                                            \
    return static_cast<otype>(__metal_log10(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype log2(itype x) {                                             \
    return static_cast<otype>(__metal_log2(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype max(itype x, itype y) {                                     \
    return static_cast<otype>(                                                 \
        __metal_fmax(static_cast<ctype>(x), static_cast<ctype>(y), mfast));    \
  }                                                                            \
  METAL_FUNC otype max3(itype x, itype y, itype z) {                           \
    return static_cast<otype>(__metal_fmax3(                                   \
        static_cast<ctype>(x),                                                 \
        static_cast<ctype>(y),                                                 \
        static_cast<ctype>(z),                                                 \
        mfast));                                                               \
  }                                                                            \
  METAL_FUNC otype median3(itype x, itype y, itype z) {                        \
    return static_cast<otype>(__metal_fmedian3(                                \
        static_cast<ctype>(x),                                                 \
        static_cast<ctype>(y),                                                 \
        static_cast<ctype>(z),                                                 \
        mfast));                                                               \
  }                                                                            \
  METAL_FUNC otype min(itype x, itype y) {                                     \
    return static_cast<otype>(                                                 \
        __metal_fmin(static_cast<ctype>(x), static_cast<ctype>(y), mfast));    \
  }                                                                            \
  METAL_FUNC otype min3(itype x, itype y, itype z) {                           \
    return static_cast<otype>(__metal_fmin3(                                   \
        static_cast<ctype>(x),                                                 \
        static_cast<ctype>(y),                                                 \
        static_cast<ctype>(z),                                                 \
        mfast));                                                               \
  }                                                                            \
  METAL_FUNC otype nextafter(itype x, itype y) {                               \
    return static_cast<otype>(                                                 \
        __metal_nextafter(static_cast<ctype>(x), static_cast<ctype>(y)));      \
  }                                                                            \
  METAL_FUNC otype pow(itype x, itype y) {                                     \
    return static_cast<otype>(                                                 \
        __metal_pow(static_cast<ctype>(x), static_cast<ctype>(y), mfast));     \
  }                                                                            \
  METAL_FUNC otype powr(itype x, itype y) {                                    \
    return static_cast<otype>(                                                 \
        __metal_powr(static_cast<ctype>(x), static_cast<ctype>(y), mfast));    \
  }                                                                            \
  METAL_FUNC otype rint(itype x) {                                             \
    return static_cast<otype>(__metal_rint(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype round(itype x) {                                            \
    return static_cast<otype>(__metal_round(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype rsqrt(itype x) {                                            \
    return static_cast<otype>(__metal_rsqrt(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype sin(itype x) {                                              \
    return static_cast<otype>(__metal_sin(static_cast<ctype>(x), mfast));      \
  }                                                                            \
  METAL_FUNC otype sinh(itype x) {                                             \
    return static_cast<otype>(__metal_sinh(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype sinpi(itype x) {                                            \
    return static_cast<otype>(__metal_sinpi(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype sqrt(itype x) {                                             \
    return static_cast<otype>(__metal_sqrt(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype tan(itype x) {                                              \
    return static_cast<otype>(__metal_tan(static_cast<ctype>(x), mfast));      \
  }                                                                            \
  METAL_FUNC otype tanh(itype x) {                                             \
    return static_cast<otype>(__metal_tanh(static_cast<ctype>(x), mfast));     \
  }                                                                            \
  METAL_FUNC otype tanpi(itype x) {                                            \
    return static_cast<otype>(__metal_tanpi(static_cast<ctype>(x), mfast));    \
  }                                                                            \
  METAL_FUNC otype trunc(itype x) {                                            \
    return static_cast<otype>(__metal_trunc(static_cast<ctype>(x), mfast));    \
  }

namespace metal {

instantiate_metal_math_funcs(
    bfloat16_t,
    bfloat16_t,
    float,
    __METAL_MAYBE_FAST_MATH__);

namespace fast {

instantiate_metal_math_funcs(
    bfloat16_t,
    bfloat16_t,
    float,
    __METAL_FAST_MATH__);

} // namespace fast

namespace precise {

instantiate_metal_math_funcs(
    bfloat16_t,
    bfloat16_t,
    float,
    __METAL_PRECISE_MATH__);

} // namespace precise

} // namespace metal

///////////////////////////////////////////////////////////////////////////////
// Metal simd for bfloat16
///////////////////////////////////////////////////////////////////////////////

#define instantiate_metal_simd_comm_funcs(                                   \
    itype, otype, ctype, itype_to_ctype, ctype_to_otype)                     \
                                                                             \
  METAL_FUNC otype simd_broadcast(itype data, ushort broadcast_lane_id) {    \
    return ctype_to_otype(                                                   \
        __metal_simd_broadcast(itype_to_ctype(data), broadcast_lane_id));    \
  }                                                                          \
                                                                             \
  METAL_FUNC otype simd_shuffle(itype data, ushort simd_lane_id) {           \
    return ctype_to_otype(                                                   \
        __metal_simd_shuffle(itype_to_ctype(data), simd_lane_id));           \
  }                                                                          \
                                                                             \
  METAL_FUNC otype simd_shuffle_and_fill_down(                               \
      itype data, itype filling_data, ushort delta, ushort modulo) {         \
    return ctype_to_otype(__metal_simd_shuffle_and_fill_down(                \
        itype_to_ctype(data), itype_to_ctype(filling_data), delta, modulo)); \
  }                                                                          \
                                                                             \
  METAL_FUNC otype simd_shuffle_and_fill_down(                               \
      itype data, itype filling_data, ushort delta) {                        \
    return ctype_to_otype(__metal_simd_shuffle_and_fill_down(                \
        itype_to_ctype(data),                                                \
        itype_to_ctype(filling_data),                                        \
        delta,                                                               \
        __metal_get_simdgroup_size(ushort())));                              \
  }                                                                          \
                                                                             \
  METAL_FUNC otype simd_shuffle_and_fill_up(                                 \
      itype data, itype filling_data, ushort delta, ushort modulo) {         \
    return ctype_to_otype(__metal_simd_shuffle_and_fill_up(                  \
        itype_to_ctype(data), itype_to_ctype(filling_data), delta, modulo)); \
  }                                                                          \
                                                                             \
  METAL_FUNC otype simd_shuffle_and_fill_up(                                 \
      itype data, itype filling_data, ushort delta) {                        \
    return ctype_to_otype(__metal_simd_shuffle_and_fill_up(                  \
        itype_to_ctype(data),                                                \
        itype_to_ctype(filling_data),                                        \
        delta,                                                               \
        __metal_get_simdgroup_size(ushort())));                              \
  }                                                                          \
                                                                             \
  METAL_FUNC otype simd_shuffle_down(itype data, ushort delta) {             \
    return ctype_to_otype(                                                   \
        __metal_simd_shuffle_down(itype_to_ctype(data), delta));             \
  }                                                                          \
                                                                             \
  METAL_FUNC otype simd_shuffle_rotate_down(itype data, ushort delta) {      \
    return ctype_to_otype(                                                   \
        __metal_simd_shuffle_rotate_down(itype_to_ctype(data), delta));      \
  }                                                                          \
                                                                             \
  METAL_FUNC otype simd_shuffle_rotate_up(itype data, ushort delta) {        \
    return ctype_to_otype(                                                   \
        __metal_simd_shuffle_rotate_up(itype_to_ctype(data), delta));        \
  }                                                                          \
                                                                             \
  METAL_FUNC otype simd_shuffle_up(itype data, ushort delta) {               \
    return ctype_to_otype(                                                   \
        __metal_simd_shuffle_up(itype_to_ctype(data), delta));               \
  }                                                                          \
                                                                             \
  METAL_FUNC otype simd_shuffle_xor(itype data, ushort mask) {               \
    return ctype_to_otype(                                                   \
        __metal_simd_shuffle_xor(itype_to_ctype(data), mask));               \
  }

#define instantiate_metal_simd_reduction_funcs(itype, otype, ctype)            \
                                                                               \
  METAL_FUNC otype simd_max(itype data) {                                      \
    return static_cast<otype>(__metal_simd_max(static_cast<ctype>(data)));     \
  }                                                                            \
                                                                               \
  METAL_FUNC otype simd_min(itype data) {                                      \
    return static_cast<otype>(__metal_simd_min(static_cast<ctype>(data)));     \
  }                                                                            \
                                                                               \
  METAL_FUNC otype simd_prefix_exclusive_product(itype data) {                 \
    return static_cast<otype>(                                                 \
        __metal_simd_prefix_exclusive_product(static_cast<ctype>(data)));      \
  }                                                                            \
                                                                               \
  METAL_FUNC otype simd_prefix_exclusive_sum(itype data) {                     \
    return static_cast<otype>(                                                 \
        __metal_simd_prefix_exclusive_sum(static_cast<ctype>(data)));          \
  }                                                                            \
                                                                               \
  METAL_FUNC otype simd_prefix_inclusive_product(itype data) {                 \
    return static_cast<otype>(                                                 \
        __metal_simd_prefix_inclusive_product(static_cast<ctype>(data)));      \
  }                                                                            \
                                                                               \
  METAL_FUNC otype simd_prefix_inclusive_sum(itype data) {                     \
    return static_cast<otype>(                                                 \
        __metal_simd_prefix_inclusive_sum(static_cast<ctype>(data)));          \
  }                                                                            \
                                                                               \
  METAL_FUNC otype simd_product(itype data) {                                  \
    return static_cast<otype>(__metal_simd_product(static_cast<ctype>(data))); \
  }                                                                            \
                                                                               \
  METAL_FUNC otype simd_sum(itype data) {                                      \
    return static_cast<otype>(__metal_simd_sum(static_cast<ctype>(data)));     \
  }                                                                            \
                                                                               \
  METAL_FUNC otype simd_xor(itype data) {                                      \
    return static_cast<otype>(__metal_simd_xor(static_cast<ctype>(data)));     \
  }

namespace metal {

instantiate_metal_simd_comm_funcs(
    bfloat16_t,
    bfloat16_t,
    uint16_t,
    bfloat16_to_uint16,
    uint16_to_bfloat16);
instantiate_metal_simd_reduction_funcs(bfloat16_t, bfloat16_t, float);

} // namespace metal


// ===== mlx/backend/metal/kernels/complex.h =====
// Copyright © 2023 Apple Inc.


#include <metal_stdlib>

using namespace metal;

struct complex64_t;

template <typename T>
static constexpr constant bool can_convert_to_complex64 =
    !is_same_v<T, complex64_t> && is_convertible_v<T, float>;

template <typename T>
static constexpr constant bool can_convert_from_complex64 =
    !is_same_v<T, complex64_t> &&
    (is_convertible_v<float, T> || is_convertible_v<bfloat16_t, T>);

struct complex64_t {
  float real;
  float imag;

  // Constructors
  constexpr complex64_t(float real, float imag) : real(real), imag(imag) {};
  constexpr complex64_t() : real(0), imag(0) {};
  constexpr complex64_t() threadgroup : real(0), imag(0) {};

  // Conversions to complex64_t
  template <
      typename T,
      typename = typename enable_if<can_convert_to_complex64<T>>::type>
  constexpr complex64_t(T x) thread : real(x), imag(0) {}

  template <
      typename T,
      typename = typename enable_if<can_convert_to_complex64<T>>::type>
  constexpr complex64_t(T x) threadgroup : real(x), imag(0) {}

  template <
      typename T,
      typename = typename enable_if<can_convert_to_complex64<T>>::type>
  constexpr complex64_t(T x) device : real(x), imag(0) {}

  template <
      typename T,
      typename = typename enable_if<can_convert_to_complex64<T>>::type>
  constexpr complex64_t(T x) constant : real(x), imag(0) {}

  // Conversions from complex64_t
  template <
      typename T,
      typename = typename enable_if<can_convert_from_complex64<T>>::type>
  constexpr operator T() const thread {
    return static_cast<T>(real);
  }

  template <
      typename T,
      typename = typename enable_if<can_convert_from_complex64<T>>::type>
  constexpr operator T() const threadgroup {
    return static_cast<T>(real);
  }

  template <
      typename T,
      typename = typename enable_if<can_convert_from_complex64<T>>::type>
  constexpr operator T() const device {
    return static_cast<T>(real);
  }

  template <
      typename T,
      typename = typename enable_if<can_convert_from_complex64<T>>::type>
  constexpr operator T() const constant {
    return static_cast<T>(real);
  }
};

constexpr complex64_t operator-(complex64_t x) {
  return {-x.real, -x.imag};
}

constexpr bool operator>=(complex64_t a, complex64_t b) {
  return (a.real > b.real) || (a.real == b.real && a.imag >= b.imag);
}

constexpr bool operator>(complex64_t a, complex64_t b) {
  return (a.real > b.real) || (a.real == b.real && a.imag > b.imag);
}

constexpr bool operator<=(complex64_t a, complex64_t b) {
  return operator>=(b, a);
}

constexpr bool operator<(complex64_t a, complex64_t b) {
  return operator>(b, a);
}

constexpr bool operator==(complex64_t a, complex64_t b) {
  return a.real == b.real && a.imag == b.imag;
}

constexpr complex64_t operator+(complex64_t a, complex64_t b) {
  return {a.real + b.real, a.imag + b.imag};
}

constexpr thread complex64_t& operator+=(thread complex64_t& a, complex64_t b) {
  a.real += b.real;
  a.imag += b.imag;
  return a;
}

constexpr threadgroup complex64_t& operator+=(
    threadgroup complex64_t& a,
    complex64_t b) {
  a.real += b.real;
  a.imag += b.imag;
  return a;
}

constexpr device complex64_t& operator+=(device complex64_t& a, complex64_t b) {
  a.real += b.real;
  a.imag += b.imag;
  return a;
}

constexpr complex64_t operator+(float a, complex64_t b) {
  return {a + b.real, b.imag};
}
constexpr complex64_t operator+(complex64_t a, float b) {
  return {a.real + b, a.imag};
}

constexpr complex64_t operator-(complex64_t a, complex64_t b) {
  return {a.real - b.real, a.imag - b.imag};
}
constexpr complex64_t operator-(float a, complex64_t b) {
  return {a - b.real, -b.imag};
}
constexpr complex64_t operator-(complex64_t a, float b) {
  return {a.real - b, a.imag};
}

constexpr complex64_t operator*(complex64_t a, complex64_t b) {
  return {a.real * b.real - a.imag * b.imag, a.real * b.imag + a.imag * b.real};
}

constexpr complex64_t operator/(complex64_t a, complex64_t b) {
  auto denom = b.real * b.real + b.imag * b.imag;
  auto x = a.real * b.real + a.imag * b.imag;
  auto y = a.imag * b.real - a.real * b.imag;
  return {x / denom, y / denom};
}

constexpr complex64_t operator/(float a, complex64_t b) {
  auto denom = b.real * b.real + b.imag * b.imag;
  auto x = a * b.real;
  auto y = -a * b.imag;
  return {x / denom, y / denom};
}

constexpr complex64_t operator%(complex64_t a, complex64_t b) {
  auto real = a.real - (b.real * static_cast<int64_t>(a.real / b.real));
  auto imag = a.imag - (b.imag * static_cast<int64_t>(a.imag / b.imag));
  if (real != 0 && (real < 0 != b.real < 0)) {
    real += b.real;
  }
  if (imag != 0 && (imag < 0 != b.imag < 0)) {
    imag += b.imag;
  }
  return {real, imag};
}


// ===== mlx/backend/metal/kernels/defines.h =====
// Copyright © 2023 Apple Inc.


#if defined __METAL__ || defined MLX_METAL_JIT
#define MTL_CONST constant
#else
#define MTL_CONST
#endif

static MTL_CONST constexpr int MAX_REDUCE_SPECIALIZED_DIMS = 4;
static MTL_CONST constexpr int REDUCE_N_READS = 4;
static MTL_CONST constexpr int REDUCE_N_WRITES = 4;
static MTL_CONST constexpr int SOFTMAX_N_READS = 4;
static MTL_CONST constexpr int RMS_N_READS = 4;
static MTL_CONST constexpr int RMS_LOOPED_LIMIT = 4096;

// Instantiate a templated kernel.
// Extra args are used as template parameters:
// e.g. instantiate_kernel(binary_int, binary, a, b) ->
// [[host_name(binary_int)]] [kernel] binary<a, b>
#define instantiate_kernel(name, func, ...) \
  template [[host_name(                     \
      name)]] [[kernel]] decltype(func<__VA_ARGS__>) func<__VA_ARGS__>;


// ===== mlx/backend/metal/kernels/logging.h =====
// Copyright © 2025 Apple Inc.


#if defined(__METAL_VERSION__) && (__METAL_VERSION__ >= 320)
#include <metal_logging>

namespace mlx {
using os_log = metal::os_log;
} // namespace mlx

#else

namespace mlx {
struct os_log {
  constexpr os_log(constant char*, constant char*) constant {}

  template <typename... Args>
  void log_debug(constant char*, Args...) const {}

  template <typename... Args>
  void log_debug(constant char*, Args...) const constant {}
};
} // namespace mlx

#endif

typedef half float16_t;

// Work per thread values for different types. The values here are expected to
// match get_work_per_thread in mlx/backend/metal/utils.h
template <typename U>
struct WorkPerThread {
  static_assert(sizeof(U) <= 8, "Type too large");
  static constexpr int constant n = 8 / sizeof(U);
};

///////////////////////////////////////////////////////////////////////////////
// Type limits utils
///////////////////////////////////////////////////////////////////////////////

template <typename U>
struct Limits {
  static const constant U max = metal::numeric_limits<U>::max();
  static const constant U min = metal::numeric_limits<U>::min();
  static const constant U finite_max = metal::numeric_limits<U>::max();
  static const constant U finite_min = metal::numeric_limits<U>::min();
};

#define instantiate_default_limit(type)                                      \
  template <>                                                                \
  struct Limits<type> {                                                      \
    static constexpr constant type max = metal::numeric_limits<type>::max(); \
    static constexpr constant type min = metal::numeric_limits<type>::min(); \
    static constexpr constant type finite_max =                              \
        metal::numeric_limits<type>::max();                                  \
    static constexpr constant type finite_min =                              \
        metal::numeric_limits<type>::min();                                  \
  };

instantiate_default_limit(uint8_t);
instantiate_default_limit(uint16_t);
instantiate_default_limit(uint32_t);
instantiate_default_limit(uint64_t);
instantiate_default_limit(int8_t);
instantiate_default_limit(int16_t);
instantiate_default_limit(int32_t);
instantiate_default_limit(int64_t);

#define instantiate_float_limit(type)             \
  template <>                                     \
  struct Limits<type> {                           \
    static constexpr constant type max =          \
        metal::numeric_limits<type>::infinity();  \
    static constexpr constant type min =          \
        -metal::numeric_limits<type>::infinity(); \
    static constexpr constant type finite_max =   \
        metal::numeric_limits<type>::max();       \
    static constexpr constant type finite_min =   \
        -metal::numeric_limits<type>::max();      \
  };

instantiate_float_limit(half);
instantiate_float_limit(float);
instantiate_float_limit(bfloat16_t);

template <>
struct Limits<bool> {
  static constexpr constant bool max = true;
  static constexpr constant bool min = false;
};

template <>
struct Limits<complex64_t> {
  static constexpr constant complex64_t max = complex64_t(
      metal::numeric_limits<float>::infinity(),
      metal::numeric_limits<float>::infinity());
  static constexpr constant complex64_t min = complex64_t(
      -metal::numeric_limits<float>::infinity(),
      -metal::numeric_limits<float>::infinity());
};

///////////////////////////////////////////////////////////////////////////////
// Indexing utils
///////////////////////////////////////////////////////////////////////////////

#define MLX_MTL_PRAGMA_UNROLL _Pragma("clang loop unroll(full)")

///////////////////////////////////////////////////////////////////////////////
// Single Array with generic dims

template <typename IdxT = int64_t>
METAL_FUNC IdxT elem_to_loc(
    IdxT elem,
    constant const int* shape,
    constant const int64_t* strides,
    int ndim) {
  IdxT loc = 0;
  for (int i = ndim - 1; i >= 0 && elem > 0; --i) {
    loc += (elem % shape[i]) * IdxT(strides[i]);
    elem /= shape[i];
  }
  return loc;
}

// Non templated version to handle arbitrary dims
template <typename IdxT = int64_t>
METAL_FUNC IdxT elem_to_loc(
    uint3 elem,
    constant const int* shape,
    constant const int64_t* strides,
    int ndim) {
  IdxT loc =
      elem.x * IdxT(strides[ndim - 1]) + elem.y * IdxT(strides[ndim - 2]);
  for (int d = ndim - 3; d >= 0; --d) {
    loc += (elem.z % shape[d]) * IdxT(strides[d]);
    elem.z /= shape[d];
  }
  return loc;
}

///////////////////////////////////////////////////////////////////////////////
// Single Array with fixed N dims

template <typename IdxT = int64_t>
METAL_FUNC IdxT elem_to_loc_1(uint elem, constant const int64_t& stride) {
  return elem * IdxT(stride);
}

template <typename IdxT = int64_t>
METAL_FUNC IdxT elem_to_loc_2(uint2 elem, constant const int64_t strides[2]) {
  return elem.x * IdxT(strides[1]) + elem.y * IdxT(strides[0]);
}

template <typename IdxT = int64_t>
METAL_FUNC IdxT elem_to_loc_3(uint3 elem, constant const int64_t strides[3]) {
  return elem.x * IdxT(strides[2]) + elem.y * IdxT(strides[1]) +
      elem.z * IdxT(strides[0]);
}

///////////////////////////////////////////////////////////////////////////////
// Multiple Arrays with generic dims

template <typename IdxT = int64_t>
METAL_FUNC vec<IdxT, 2> elem_to_loc_2_nd(
    uint3 elem,
    constant const int* shape,
    constant const int64_t* a_strides,
    constant const int64_t* b_strides,
    int ndim) {
  vec<IdxT, 2> loc = {
      IdxT(
          elem.x * IdxT(a_strides[ndim - 1]) +
          IdxT(elem.y) * IdxT(a_strides[ndim - 2])),
      IdxT(
          elem.x * IdxT(b_strides[ndim - 1]) +
          elem.y * IdxT(b_strides[ndim - 2]))};
  for (int d = ndim - 3; d >= 0; --d) {
    uint l = elem.z % shape[d];
    loc.x += l * IdxT(a_strides[d]);
    loc.y += l * IdxT(b_strides[d]);
    elem.z /= shape[d];
  }
  return loc;
}

template <typename IdxT = int64_t>
METAL_FUNC vec<IdxT, 3> elem_to_loc_3_nd(
    uint3 elem,
    constant const int* shape,
    constant const int64_t* a_strides,
    constant const int64_t* b_strides,
    constant const int64_t* c_strides,
    int ndim) {
  vec<IdxT, 3> loc = {
      IdxT(elem.x * IdxT(a_strides[ndim - 1])) +
          IdxT(elem.y * IdxT(a_strides[ndim - 2])),
      IdxT(elem.x * IdxT(b_strides[ndim - 1])) +
          IdxT(elem.y * IdxT(b_strides[ndim - 2])),
      IdxT(elem.x * IdxT(c_strides[ndim - 1])) +
          IdxT(elem.y * IdxT(c_strides[ndim - 2]))};
  for (int d = ndim - 3; d >= 0; --d) {
    uint l = elem.z % shape[d];
    loc.x += l * IdxT(a_strides[d]);
    loc.y += l * IdxT(b_strides[d]);
    loc.z += l * IdxT(c_strides[d]);
    elem.z /= shape[d];
  }
  return loc;
}

///////////////////////////////////////////////////////////////////////////////
// Elem to loc in a loop utils
///////////////////////////////////////////////////////////////////////////////

template <int DIM, typename OffsetT = size_t, bool General = true>
struct LoopedElemToLoc {
  int dim;
  LoopedElemToLoc<DIM - 1, OffsetT, General> inner_looper;
  OffsetT offset{0};
  int index{0};

  LoopedElemToLoc(int dim) : dim(dim), inner_looper(dim - 1) {}

  void next(const constant int* shape, const constant int64_t* strides) {
    if (dim == 0) {
      return;
    }
    index++;
    offset += OffsetT(strides[dim - 1]);
    if (index >= shape[dim - 1]) {
      index = 0;
      inner_looper.next(shape, strides);
      offset = inner_looper.offset;
    }
  }

  void next(int n, const constant int* shape, const constant int64_t* strides) {
    if (dim == 0) {
      return;
    }
    index += n;
    offset += n * OffsetT(strides[dim - 1]);

    if (index >= shape[dim - 1]) {
      int extra = index - shape[dim - 1];
      if (extra >= shape[dim - 1]) {
        inner_looper.next(1 + extra / shape[dim - 1], shape, strides);
        extra = extra % shape[dim - 1];
      } else {
        inner_looper.next(shape, strides);
      }
      index = 0;
      offset = inner_looper.offset;
      if (extra > 0) {
        next(extra, shape, strides);
      }
    }
  }

  OffsetT location() {
    return offset;
  }
};

template <typename OffsetT>
struct LoopedElemToLoc<1, OffsetT, true> {
  int dim;
  OffsetT offset{0};
  uint index{0};

  LoopedElemToLoc(int dim) : dim(dim) {}

  void next(const constant int* shape, const constant int64_t* strides) {
    index++;
    if (dim > 1) {
      offset = elem_to_loc<OffsetT>(index, shape, strides, dim);
    } else {
      offset += OffsetT(strides[0]);
    }
  }

  void next(int n, const constant int* shape, const constant int64_t* strides) {
    index += n;
    if (dim > 1) {
      offset = elem_to_loc<OffsetT>(index, shape, strides, dim);
    } else {
      offset = index * OffsetT(strides[0]);
    }
  }

  OffsetT location() {
    return offset;
  }
};

template <typename OffsetT>
struct LoopedElemToLoc<1, OffsetT, false> {
  OffsetT offset{0};

  LoopedElemToLoc(int) {}

  void next(const constant int*, const constant int64_t* strides) {
    offset += OffsetT(strides[0]);
  }

  void next(int n, const constant int*, const constant int64_t* strides) {
    offset += n * OffsetT(strides[0]);
  }

  OffsetT location() {
    return offset;
  }
};

///////////////////////////////////////////////////////////////////////////////
// Calculation utils
///////////////////////////////////////////////////////////////////////////////

/** Compute ceil((float)N/(float)M) */
template <typename T, typename U>
inline T ceildiv(T N, U M) {
  return (N + M - 1) / M;
}

// https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html#1202
inline float log1p(float x) {
  float xp1 = 1.0f + x;
  if (xp1 == Limits<float>::max) {
    return Limits<float>::max;
  }
  if (xp1 == 1.0f) {
    return x;
  }

  return x * (metal::log(xp1) / (xp1 - 1.0f));
}

inline bfloat16_t log1p(bfloat16_t x) {
  float xp1 = 1.0f + static_cast<float>(x);
  if (xp1 == Limits<float>::max) {
    return Limits<bfloat16_t>::max;
  }
  if (xp1 == 1.0f) {
    return x;
  }

  return bfloat16_t(x * (metal::log(xp1) / (xp1 - 1.0f)));
}

inline complex64_t log1p(complex64_t in) {
  float x = in.real;
  float y = in.imag;
  float zabs = metal::precise::sqrt(x * x + y * y);
  float theta = metal::atan2(y, x + 1);
  if (zabs < 0.5f) {
    float r = x * (2 + x) + y * y;
    if (r == 0) { // handle underflow
      return {x, theta};
    }
    return {0.5f * log1p(r), theta};
  } else {
    auto z0 = metal::sqrt((x + 1) * (x + 1) + y * y);
    return {metal::log(z0), theta};
  }
}

///////////////////////////////////////////////////////////////////////////////
// SIMD shuffle ops
///////////////////////////////////////////////////////////////////////////////

inline uint64_t simd_shuffle_down(uint64_t data, uint16_t delta) {
  return as_type<uint64_t>(
      metal::simd_shuffle_down(as_type<uint2>(data), delta));
}

inline int64_t simd_shuffle_down(int64_t data, uint16_t delta) {
  return as_type<int64_t>(
      metal::simd_shuffle_down(as_type<uint2>(data), delta));
}

inline bool simd_shuffle_down(bool data, uint16_t delta) {
  return simd_shuffle_down(static_cast<uint32_t>(data), delta);
}

inline complex64_t simd_shuffle_down(complex64_t data, uint16_t delta) {
  return complex64_t(
      simd_shuffle_down(data.real, delta), simd_shuffle_down(data.imag, delta));
}

inline uint64_t simd_shuffle_up(uint64_t data, uint16_t delta) {
  return as_type<uint64_t>(metal::simd_shuffle_up(as_type<uint2>(data), delta));
}

inline int64_t simd_shuffle_up(int64_t data, uint16_t delta) {
  return as_type<int64_t>(metal::simd_shuffle_up(as_type<uint2>(data), delta));
}

inline bool simd_shuffle_up(bool data, uint16_t delta) {
  return simd_shuffle_up(static_cast<uint32_t>(data), delta);
}

inline complex64_t simd_shuffle_up(complex64_t data, uint16_t delta) {
  return complex64_t(
      simd_shuffle_up(data.real, delta), simd_shuffle_up(data.imag, delta));
}

inline uint64_t
simd_shuffle_and_fill_up(uint64_t data, uint64_t filling, uint16_t delta) {
  return as_type<uint64_t>(metal::simd_shuffle_and_fill_up(
      as_type<uint2>(data), as_type<uint2>(filling), delta));
}

inline int64_t
simd_shuffle_and_fill_up(int64_t data, int64_t filling, uint16_t delta) {
  return as_type<int64_t>(metal::simd_shuffle_and_fill_up(
      as_type<uint2>(data), as_type<uint2>(filling), delta));
}

inline bool simd_shuffle_and_fill_up(bool data, bool filling, uint16_t delta) {
  return simd_shuffle_and_fill_up(
      static_cast<uint32_t>(data), static_cast<uint32_t>(filling), delta);
}

inline complex64_t simd_shuffle_and_fill_up(
    complex64_t data,
    complex64_t filling,
    uint16_t delta) {
  return complex64_t(
      simd_shuffle_and_fill_up(data.real, filling.real, delta),
      simd_shuffle_and_fill_up(data.imag, filling.imag, delta));
}

inline uint64_t simd_shuffle(uint64_t data, uint16_t lane) {
  return as_type<uint64_t>(metal::simd_shuffle(as_type<uint2>(data), lane));
}

inline int64_t simd_shuffle(int64_t data, uint16_t lane) {
  return as_type<int64_t>(metal::simd_shuffle(as_type<uint2>(data), lane));
}

inline bool simd_shuffle(bool data, uint16_t lane) {
  return simd_shuffle(static_cast<uint32_t>(data), lane);
}

inline complex64_t simd_shuffle(complex64_t data, uint16_t lane) {
  return complex64_t(
      simd_shuffle(data.real, lane), simd_shuffle(data.imag, lane));
}

// std::conditional is not included with Metal
template <bool condition, typename T, typename U>
struct ConditionalType {
  using type = U;
};

template <typename T, typename U>
struct ConditionalType<true, T, U> {
  using type = T;
};


#define MLX_MTL_CONST static constant constexpr const

using namespace metal;

///////////////////////////////////////////////////////////////////////////////
/// Naive unfold with dilation
///////////////////////////////////////////////////////////////////////////////

template <typename T, int N>
[[kernel]] void naive_unfold_Nd(
    const device T* in [[buffer(0)]],
    device T* out [[buffer(1)]],
    const constant MLXConvParams<N>* params [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]) {
  int filter_size = params->C;
  for (short i = 0; i < N; i++)
    filter_size *= params->wS[i];

  int out_pixels = 1;
  for (short i = 0; i < N; i++)
    out_pixels *= params->oS[i];

  // Set out
  out += (size_t)gid.z * filter_size + (size_t)gid.y * (params->C);

  // Coordinates in input
  int is[N] = {0};

  // gid.z: N oS (Batch and row in unfolded output)
  // gid.y: wS (Filter location to unfold input)
  // gid.x: C (channel)

  int n = (gid.z) / out_pixels;
  int oS = (gid.z) % out_pixels;
  int wS = gid.y;

  bool valid = n < params->N;

  // Unroll dimensions
  for (int i = N - 1; i >= 0; --i) {
    int os_ = (oS % params->oS[i]);
    int ws_ = (wS % params->wS[i]);

    ws_ = params->flip ? params->wS[i] - ws_ - 1 : ws_;

    int is_ = os_ * params->str[i] - params->pad[i] + ws_ * params->kdil[i];
    int is_max = 1 + params->idil[i] * (params->iS[i] - 1);

    valid &= is_ >= 0 && is_ < is_max && (is_ % params->idil[i] == 0);

    is[i] = is_ / params->idil[i];

    oS /= params->oS[i];
    wS /= params->wS[i];
  }

  if (valid) {
    size_t in_offset = n * params->in_strides[0];

    for (int i = 0; i < N; ++i) {
      in_offset += is[i] * params->in_strides[i + 1];
    }

    out[gid.x] = in[in_offset + gid.x];
  } else {
    out[gid.x] = T(0);
  }
}

// This kernel unfolds the input array of size (N, *spatial_dims, C)
// into an array of size (N x *spatial_dims, C x *kernel_dims).
template <typename T, int N>
[[kernel]] void naive_unfold_transpose_Nd(
    const device T* in [[buffer(0)]],
    device T* out [[buffer(1)]],
    const constant MLXConvParams<N>* params [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]) {
  int filter_size = params->C;
  for (short i = 0; i < N; i++)
    filter_size *= params->wS[i];

  int out_pixels = 1;
  for (short i = 0; i < N; i++)
    out_pixels *= params->oS[i];

  // Set out
  out +=
      (size_t)gid.z * filter_size + (size_t)gid.x * (filter_size / params->C);

  // Coordinates in input
  int is[N] = {0};

  // gid.z: N oS (Batch and row in unfolded output)
  // gid.y: wS (Filter location to unfold input)
  // gid.x: C (channel)

  int n = (gid.z) / out_pixels;
  int oS = (gid.z) % out_pixels;
  int wS = gid.y;

  bool valid = n < params->N;

  // Unroll dimensions
  int kernel_stride = 1;
  for (int i = N - 1; i >= 0; --i) {
    int os_ = (oS % params->oS[i]);
    int ws_ = (wS % params->wS[i]);
    out += ws_ * kernel_stride;

    ws_ = params->flip ? params->wS[i] - ws_ - 1 : ws_;

    int is_ = os_ * params->str[i] - params->pad[i] + ws_ * params->kdil[i];
    int is_max = 1 + params->idil[i] * (params->iS[i] - 1);

    valid &= is_ >= 0 && is_ < is_max && (is_ % params->idil[i] == 0);

    is[i] = is_ / params->idil[i];

    oS /= params->oS[i];
    wS /= params->wS[i];

    kernel_stride *= params->wS[i];
  }

  if (valid) {
    size_t in_offset = n * params->in_strides[0];

    for (int i = 0; i < N; ++i) {
      in_offset += is[i] * params->in_strides[i + 1];
    }

    out[0] = in[in_offset + gid.x];
  } else {
    out[0] = T(0);
  }
}

#define instantiate_naive_unfold_nd(name, itype, n)                            \
  template [[host_name("naive_unfold_nd_" #name "_" #n)]] [[kernel]] void      \
  naive_unfold_Nd(                                                             \
      const device itype* in [[buffer(0)]],                                    \
      device itype* out [[buffer(1)]],                                         \
      const constant MLXConvParams<n>* params [[buffer(2)]],                   \
      uint3 gid [[thread_position_in_grid]]);                                  \
  template                                                                     \
      [[host_name("naive_unfold_transpose_nd_" #name "_" #n)]] [[kernel]] void \
      naive_unfold_transpose_Nd(                                               \
          const device itype* in [[buffer(0)]],                                \
          device itype* out [[buffer(1)]],                                     \
          const constant MLXConvParams<n>* params [[buffer(2)]],               \
          uint3 gid [[thread_position_in_grid]]);

#define instantiate_naive_unfold_nd_dims(name, itype)                      \
  instantiate_naive_unfold_nd(name, itype, 1) instantiate_naive_unfold_nd( \
      name, itype, 2) instantiate_naive_unfold_nd(name, itype, 3)

instantiate_naive_unfold_nd_dims(float32, float);
instantiate_naive_unfold_nd_dims(float16, half);


///////////////////////////////////////////////////////////////////////////////
/// Depthwise convolution kernels
///////////////////////////////////////////////////////////////////////////////

constant int ker_h [[function_constant(00)]];
constant int ker_w [[function_constant(01)]];
constant int str_h [[function_constant(10)]];
constant int str_w [[function_constant(11)]];
constant int tgp_h [[function_constant(100)]];
constant int tgp_w [[function_constant(101)]];
constant bool do_flip [[function_constant(200)]];

constant int span_h = tgp_h * str_h + ker_h - 1;
constant int span_w = tgp_w * str_w + ker_w - 1;
constant int span_hw = span_h * span_w;

template <typename T>
[[kernel]] void depthwise_conv_2d(
    const device T* in [[buffer(0)]],
    const device T* wt [[buffer(1)]],
    device T* out [[buffer(2)]],
    const constant MLXConvParams<2>& params [[buffer(3)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 gid [[thread_position_in_grid]],
    uint simd_gid [[simdgroup_index_in_threadgroup]],
    uint simd_lid [[thread_index_in_simdgroup]]) {
  constexpr int tc = 8;
  constexpr int tw = 8;
  constexpr int th = 4;

  constexpr int c_per_thr = 8;

  constexpr int TGH = th * 2 + 6;
  constexpr int TGW = tw * 2 + 6;
  constexpr int TGC = tc;

  threadgroup T ins[TGH * TGW * TGC];

  const int n_tgblocks_h = (params.oS[0] + th - 1) / th;
  const int n = tid.z / n_tgblocks_h;
  const int tghid = tid.z % n_tgblocks_h;
  const int oh = tghid * th + lid.z;
  const int ow = gid.y;
  const int c = gid.x;

  in += n * params.in_strides[0];

  // Load in
  {
    constexpr int n_threads = th * tw * tc;
    const int tg_oh = (tghid * th) * str_h - params.pad[0];
    const int tg_ow = (tid.y * tw) * str_w - params.pad[1];
    const int tg_c = tid.x * tc;

    const int thread_idx = simd_gid * 32 + simd_lid;
    constexpr int thr_per_hw = tc / c_per_thr;
    constexpr int hw_per_group = n_threads / thr_per_hw;

    const int thr_c = thread_idx % thr_per_hw;
    const int thr_hw = thread_idx / thr_per_hw;

    for (int hw = thr_hw; hw < span_hw; hw += hw_per_group) {
      const int h = hw / span_w;
      const int w = hw % span_w;

      const int ih = tg_oh + h;
      const int iw = tg_ow + w;

      const int in_s_offset = h * span_w * TGC + w * TGC;

      if (ih >= 0 && ih < params.iS[0] && iw >= 0 && iw < params.iS[1]) {
        const auto in_load =
            in + ih * params.in_strides[1] + iw * params.in_strides[2] + tg_c;

        MLX_MTL_PRAGMA_UNROLL
        for (int cc = 0; cc < c_per_thr; ++cc) {
          ins[in_s_offset + c_per_thr * thr_c + cc] =
              in_load[c_per_thr * thr_c + cc];
        }
      } else {
        MLX_MTL_PRAGMA_UNROLL
        for (int cc = 0; cc < c_per_thr; ++cc) {
          ins[in_s_offset + c_per_thr * thr_c + cc] = T(0);
        }
      }
    }
  }

  threadgroup_barrier(mem_flags::mem_threadgroup);
  wt += c * params.wt_strides[0];

  const auto ins_ptr =
      &ins[lid.z * str_h * span_w * TGC + lid.y * str_w * TGC + lid.x];
  float o = 0.;
  for (int h = 0; h < ker_h; ++h) {
    for (int w = 0; w < ker_w; ++w) {
      int wt_h = h;
      int wt_w = w;
      if (do_flip) {
        wt_h = ker_h - h - 1;
        wt_w = ker_w - w - 1;
      }
      auto inv = ins_ptr[h * span_w * TGC + w * TGC];
      auto wtv = wt[wt_h * ker_w + wt_w];
      o += inv * wtv;
    }
  }
  threadgroup_barrier(mem_flags::mem_none);

  if (oh >= params.oS[0] || ow >= params.oS[1]) {
    return;
  }

  out += n * params.out_strides[0] + oh * params.out_strides[1] +
      ow * params.out_strides[2];
  out[c] = static_cast<T>(o);
}

#define instantiate_depthconv2d(iname, itype) \
  instantiate_kernel("depthwise_conv_2d_" #iname, depthwise_conv_2d, itype)

instantiate_depthconv2d(float32, float);
instantiate_depthconv2d(float16, half);


template <typename T, typename IdxT>
[[kernel]] void depthwise_conv_1d(
    const device T* in [[buffer(0)]],
    const device T* w [[buffer(1)]],
    device T* out [[buffer(2)]],
    constant const IdxT strides[3],
    constant const int& kernel_size,
    uint3 tid [[thread_position_in_grid]],
    uint3 grid_dim [[threads_per_grid]]) {
  out += (tid.z * static_cast<IdxT>(grid_dim.y) + tid.y) * grid_dim.x + tid.x;
  in += tid.z * strides[0] + tid.y * strides[1] + tid.x * strides[2];
  w += tid.x * kernel_size;

  float acc = 0.0;
  for (int i = 0; i < kernel_size; ++i) {
    acc += static_cast<float>(in[0]) * w[i];
    in += strides[1];
  }
  *out = static_cast<T>(acc);
}

#define instantiate_depthconv1d(iname, itype)                         \
  instantiate_kernel(                                                 \
      "depthwise_conv_1d_" #iname, depthwise_conv_1d, itype, int32_t) \
      instantiate_kernel(                                             \
          "depthwise_conv_1d_" #iname "_large",                       \
          depthwise_conv_1d,                                          \
          itype,                                                      \
          int64_t)

instantiate_depthconv1d(float32, float);
instantiate_depthconv1d(float16, half);


///////////////////////////////////////////////////////////////////////////////
/// Winograd kernels
///////////////////////////////////////////////////////////////////////////////

template <int M, int R, int S>
struct WinogradTransforms {};

template <>
struct WinogradTransforms<6, 3, 8> {
  MLX_MTL_CONST int OUT_TILE_SIZE = 6;
  MLX_MTL_CONST int FILTER_SIZE = 3;
  MLX_MTL_CONST int IN_TILE_SIZE = OUT_TILE_SIZE + FILTER_SIZE - 1;
  MLX_MTL_CONST int SIMD_MATRIX_SIZE = 8;
  MLX_MTL_CONST float in_transform[SIMD_MATRIX_SIZE][SIMD_MATRIX_SIZE] = {
      {1.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f},
      {0.00f, 1.00f, -1.00f, 0.50f, -0.50f, 2.00f, -2.00f, -1.00f},
      {-5.25f, 1.00f, 1.00f, 0.25f, 0.25f, 4.00f, 4.00f, 0.00f},
      {0.00f, -4.25f, 4.25f, -2.50f, 2.50f, -2.50f, 2.50f, 5.25f},
      {5.25f, -4.25f, -4.25f, -1.25f, -1.25f, -5.00f, -5.00f, 0.00f},
      {0.00f, 1.00f, -1.00f, 2.00f, -2.00f, 0.50f, -0.50f, -5.25f},
      {-1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 0.00f},
      {0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 1.00f},
  };

  MLX_MTL_CONST float out_transform[SIMD_MATRIX_SIZE][SIMD_MATRIX_SIZE] = {
      {1.00f, 0.00f, 0.00f, 0.00f, 0.00f, 0.00f},
      {1.00f, 1.00f, 1.00f, 1.00f, 1.00f, 1.00f},
      {1.00f, -1.00f, 1.00f, -1.00f, 1.00f, -1.00f},
      {1.00f, 2.00f, 4.00f, 8.00f, 16.00f, 32.00f},
      {1.00f, -2.00f, 4.00f, -8.00f, 16.00f, -32.00f},
      {1.00f, 0.50f, 0.25f, 0.125f, 0.0625f, 0.03125f},
      {1.00f, -0.50f, 0.25f, -0.125f, 0.0625f, -0.03125f},
      {0.00f, 0.00f, 0.00f, 0.00f, 0.00f, 1.00f},
  };

  MLX_MTL_CONST float wt_transform[SIMD_MATRIX_SIZE][SIMD_MATRIX_SIZE] = {
      {1.00, 0.00, 0.00},
      {-2.0 / 9.00, -2.0 / 9.00, -2.0 / 9.00},
      {-2.0 / 9.00, 2.0 / 9.00, -2.0 / 9.00},
      {1.0 / 90.0, 1.0 / 45.0, 2.0 / 45.0},
      {1.0 / 90.0, -1.0 / 45.0, 2.0 / 45.0},
      {32.0 / 45.0, 16.0 / 45.0, 8.0 / 45.0},
      {32.0 / 45.0, -16.0 / 45.0, 8.0 / 45.0},
      {0.00, 0.00, 1.00},
  };
};

constant constexpr const float WinogradTransforms<6, 3, 8>::wt_transform[8][8];
constant constexpr const float WinogradTransforms<6, 3, 8>::in_transform[8][8];
constant constexpr const float WinogradTransforms<6, 3, 8>::out_transform[8][8];

template <typename T, int BC = 32, int BO = 4, int M = 6, int R = 3>
[[kernel, max_total_threads_per_threadgroup(BO * 32)]] void
winograd_conv_2d_weight_transform(
    const device T* wt_in [[buffer(0)]],
    device T* wt_out [[buffer(1)]],
    const constant int& C [[buffer(2)]],
    const constant int& O [[buffer(3)]],
    uint tid [[threadgroup_position_in_grid]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]]) {
  using WGT = WinogradTransforms<M, R, 8>;

  // Get lane position in simdgroup
  const short qid = simd_lane_id / 4;
  const short sm = (qid & 4) + (simd_lane_id / 2) % 4;
  const short sn = (qid & 2) * 2 + (simd_lane_id % 2) * 2;

  // Initialize G matrix
  simdgroup_matrix<float, 8, 8> G;
  G.thread_elements()[0] = WGT::wt_transform[sm][sn];
  G.thread_elements()[1] = WGT::wt_transform[sm][sn + 1];

  // Initialize Gt matrix
  simdgroup_matrix<float, 8, 8> Gt;
  Gt.thread_elements()[0] = WGT::wt_transform[sn][sm];
  Gt.thread_elements()[1] = WGT::wt_transform[sn + 1][sm];

  // Move to the correct output filter
  size_t ko = BO * tid + simd_group_id;
  wt_in += ko * R * R * C;

  // wt_out is stored transposed (A x A x C x O)
  short ohw_0 = sm * 8 + sn;
  short ohw_1 = sm * 8 + sn + 1;
  device T* wt_out_0 = wt_out + ohw_0 * C * O + ko;
  device T* wt_out_1 = wt_out + ohw_1 * C * O + ko;

  // Prepare shared memory
  threadgroup T Ws[BO][R][R][BC];

  // Loop over C
  for (int bc = 0; bc < C; bc += BC) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Read into shared memory
    for (int kh = 0; kh < R; ++kh) {
      for (int kw = 0; kw < R; ++kw) {
        for (int kc = simd_lane_id; kc < BC; kc += 32) {
          Ws[simd_group_id][kh][kw][kc] = wt_in[kh * R * C + kw * C + kc];
        }
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Do transform and store the result
    for (int c = 0; c < BC; ++c) {
      simdgroup_matrix<float, 8, 8> g;
      g.thread_elements()[0] =
          sm < R && sn < R ? Ws[simd_group_id][sm][sn][c] : T(0);
      g.thread_elements()[1] =
          sm < R && sn + 1 < R ? Ws[simd_group_id][sm][sn + 1][c] : T(0);

      simdgroup_matrix<float, 8, 8> g_out = (G * g) * Gt;
      wt_out_0[c * O] = static_cast<T>(g_out.thread_elements()[0]);
      wt_out_1[c * O] = static_cast<T>(g_out.thread_elements()[1]);
    }

    wt_in += BC;
    wt_out_0 += BC * O;
    wt_out_1 += BC * O;
  }
}

#define instantiate_winograd_conv_2d_weight_transform_base(name, itype, bc)   \
  template [[host_name(                                                       \
      "winograd_conv_2d_weight_transform_" #name "_bc" #bc)]] [[kernel]] void \
  winograd_conv_2d_weight_transform<itype, bc>(                               \
      const device itype* wt_in [[buffer(0)]],                                \
      device itype* wt_out [[buffer(1)]],                                     \
      const constant int& C [[buffer(2)]],                                    \
      const constant int& O [[buffer(3)]],                                    \
      uint tid [[threadgroup_position_in_grid]],                              \
      uint simd_group_id [[simdgroup_index_in_threadgroup]],                  \
      uint simd_lane_id [[thread_index_in_simdgroup]]);

template <typename T, int BC, int WM, int WN, int M = 6, int R = 3>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void
winograd_conv_2d_input_transform(
    const device T* inp_in [[buffer(0)]],
    device T* inp_out [[buffer(1)]],
    const constant MLXConvParams<2>& params [[buffer(2)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 tgp_per_grid [[threadgroups_per_grid]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]]) {
  (void)lid;

  using WGT = WinogradTransforms<M, R, 8>;
  constexpr int A = WGT::IN_TILE_SIZE;
  constexpr int N_SIMD_GROUPS = WM * WN;

  // Get lane position in simdgroup
  const short qid = simd_lane_id / 4;
  const short sm = (qid & 4) + (simd_lane_id / 2) % 4;
  const short sn = (qid & 2) * 2 + (simd_lane_id % 2) * 2;

  // Initialize B matrix
  simdgroup_matrix<float, 8, 8> B;
  B.thread_elements()[0] = WGT::in_transform[sm][sn];
  B.thread_elements()[1] = WGT::in_transform[sm][sn + 1];

  // Initialize Bt matrix
  simdgroup_matrix<float, 8, 8> Bt;
  Bt.thread_elements()[0] = WGT::in_transform[sn][sm];
  Bt.thread_elements()[1] = WGT::in_transform[sn + 1][sm];

  // Resolve input tile
  constexpr int TH = (A / WM);
  constexpr int TW = (A / WN);
  int kh = TH * (simd_group_id / WN);
  int kw = TW * (simd_group_id % WN);
  int bh = M * tid.y + kh;
  int bw = M * tid.x + kw;

  // Move to the correct input tile
  inp_in += tid.z * params.in_strides[0] + bh * params.in_strides[1] +
      bw * params.in_strides[2];

  // Pre compute strides
  int jump_in[TH][TW];

  for (int h = 0; h < TH; h++) {
    for (int w = 0; w < TW; w++) {
      jump_in[h][w] = h * params.in_strides[1] + w * params.in_strides[2];
    }
  }

  // inp_out is stored interleaved (A x A x tiles x C)
  size_t N_TILES = tgp_per_grid.x * tgp_per_grid.y * tgp_per_grid.z;
  size_t tile_id =
      tid.z * tgp_per_grid.x * tgp_per_grid.y + tid.y * tgp_per_grid.x + tid.x;
  size_t ohw_0 = sm * 8 + sn;
  size_t ohw_1 = sm * 8 + sn + 1;
  device T* inp_out_0 =
      inp_out + ohw_0 * N_TILES * params.C + tile_id * params.C;
  device T* inp_out_1 =
      inp_out + ohw_1 * N_TILES * params.C + tile_id * params.C;

  // Prepare shared memory
  threadgroup T Is[A][A][BC];

  // Loop over C
  for (int bc = 0; bc < params.C; bc += BC) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Read into shared memory
    for (int h = 0; h < TH; h++) {
      for (int w = 0; w < TW; w++) {
        const device T* in_ptr = inp_in + jump_in[h][w];
        for (int c = simd_lane_id; c < BC; c += 32) {
          Is[kh + h][kw + w][c] = in_ptr[c];
        }
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Do transform and store the result
    for (int c = simd_group_id; c < BC; c += N_SIMD_GROUPS) {
      simdgroup_matrix<float, 8, 8> I;
      I.thread_elements()[0] = Is[sm][sn][c];
      I.thread_elements()[1] = Is[sm][sn + 1][c];

      simdgroup_matrix<float, 8, 8> I_out = (Bt * I) * B;
      inp_out_0[c] = static_cast<T>(I_out.thread_elements()[0]);
      inp_out_1[c] = static_cast<T>(I_out.thread_elements()[1]);
    }

    inp_in += BC;
    inp_out_0 += BC;
    inp_out_1 += BC;
  }
}

#define instantiate_winograd_conv_2d_input_transform(name, itype, bc)        \
  template [[host_name(                                                      \
      "winograd_conv_2d_input_transform_" #name "_bc" #bc)]] [[kernel]] void \
  winograd_conv_2d_input_transform<itype, bc, 2, 2>(                         \
      const device itype* inp_in [[buffer(0)]],                              \
      device itype* inp_out [[buffer(1)]],                                   \
      const constant MLXConvParams<2>& params [[buffer(2)]],                 \
      uint3 tid [[threadgroup_position_in_grid]],                            \
      uint3 lid [[thread_position_in_threadgroup]],                          \
      uint3 tgp_per_grid [[threadgroups_per_grid]],                          \
      uint simd_group_id [[simdgroup_index_in_threadgroup]],                 \
      uint simd_lane_id [[thread_index_in_simdgroup]]);

template <typename T, int BO, int WM, int WN, int M = 6, int R = 3>
[[kernel, max_total_threads_per_threadgroup(WM * WN * 32)]] void
winograd_conv_2d_output_transform(
    const device T* out_in [[buffer(0)]],
    device T* out_out [[buffer(1)]],
    const constant MLXConvParams<2>& params [[buffer(2)]],
    uint3 tid [[threadgroup_position_in_grid]],
    uint3 lid [[thread_position_in_threadgroup]],
    uint3 tgp_per_grid [[threadgroups_per_grid]],
    uint simd_group_id [[simdgroup_index_in_threadgroup]],
    uint simd_lane_id [[thread_index_in_simdgroup]]) {
  (void)lid;

  using WGT = WinogradTransforms<M, R, 8>;
  constexpr int N_SIMD_GROUPS = WM * WN;

  // Get lane position in simdgroup
  const short qid = simd_lane_id / 4;
  const short sm = (qid & 4) + (simd_lane_id / 2) % 4;
  const short sn = (qid & 2) * 2 + (simd_lane_id % 2) * 2;

  // Initialize A matrix
  simdgroup_matrix<float, 8, 8> B;
  B.thread_elements()[0] = WGT::out_transform[sm][sn];
  B.thread_elements()[1] = WGT::out_transform[sm][sn + 1];

  // Initialize At matrix
  simdgroup_matrix<float, 8, 8> Bt;
  Bt.thread_elements()[0] = WGT::out_transform[sn][sm];
  Bt.thread_elements()[1] = WGT::out_transform[sn + 1][sm];

  // Out_in comes in shape (A x A x tiles x O)
  // We do transform and then write out to out_out in shape (N, H, W, O)

  // Resolve output tile
  constexpr int TH = (M / WM);
  constexpr int TW = (M / WN);
  int kh = TH * (simd_group_id / WN);
  int kw = TW * (simd_group_id % WN);
  int bh = M * tid.y + kh;
  int bw = M * tid.x + kw;

  // Move to the correct input tile
  out_out += tid.z * params.out_strides[0] + bh * params.out_strides[1] +
      bw * params.out_strides[2];

  // Pre compute strides
  int jump_in[TH][TW];

  for (int h = 0; h < TH; h++) {
    for (int w = 0; w < TW; w++) {
      bool valid = ((bh + h) < params.oS[0]) && ((bw + w) < params.oS[1]);
      jump_in[h][w] =
          valid ? h * params.out_strides[1] + w * params.out_strides[2] : -1;
    }
  }

  // out_in is stored interleaved (A x A x tiles x O)
  size_t N_TILES = tgp_per_grid.x * tgp_per_grid.y * tgp_per_grid.z;
  size_t tile_id =
      tid.z * tgp_per_grid.x * tgp_per_grid.y + tid.y * tgp_per_grid.x + tid.x;
  size_t ohw_0 = sm * 8 + sn;
  size_t ohw_1 = sm * 8 + sn + 1;
  const device T* out_in_0 =
      out_in + ohw_0 * N_TILES * params.O + tile_id * params.O;
  const device T* out_in_1 =
      out_in + ohw_1 * N_TILES * params.O + tile_id * params.O;

  // Prepare shared memory
  threadgroup T Os[M][M][BO];

  // Loop over O
  for (int bo = 0; bo < params.O; bo += BO) {
    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Do transform and store the result
    for (int c = simd_group_id; c < BO; c += N_SIMD_GROUPS) {
      simdgroup_matrix<float, 8, 8> O_mat;
      O_mat.thread_elements()[0] = out_in_0[c];
      O_mat.thread_elements()[1] = out_in_1[c];

      simdgroup_matrix<float, 8, 8> O_out = (Bt * (O_mat * B));
      if ((sm < M) && (sn < M)) {
        Os[sm][sn][c] = static_cast<T>(O_out.thread_elements()[0]);
      }
      if ((sm < M) && ((sn + 1) < M)) {
        Os[sm][sn + 1][c] = static_cast<T>(O_out.thread_elements()[1]);
      }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);
    // Read out from shared memory
    for (int h = 0; h < TH; h++) {
      for (int w = 0; w < TW; w++) {
        if (jump_in[h][w] >= 0) {
          device T* out_ptr = out_out + jump_in[h][w];
          for (int c = simd_lane_id; c < BO; c += 32) {
            out_ptr[c] = Os[kh + h][kw + w][c];
          }
        }
      }
    }

    out_out += BO;
    out_in_0 += BO;
    out_in_1 += BO;
  }
}

#define instantiate_winograd_conv_2d_output_transform(name, itype, bo)        \
  template [[host_name(                                                       \
      "winograd_conv_2d_output_transform_" #name "_bo" #bo)]] [[kernel]] void \
  winograd_conv_2d_output_transform<itype, bo, 2, 2>(                         \
      const device itype* out_in [[buffer(0)]],                               \
      device itype* out_out [[buffer(1)]],                                    \
      const constant MLXConvParams<2>& params [[buffer(2)]],                  \
      uint3 tid [[threadgroup_position_in_grid]],                             \
      uint3 lid [[thread_position_in_threadgroup]],                           \
      uint3 tgp_per_grid [[threadgroups_per_grid]],                           \
      uint simd_group_id [[simdgroup_index_in_threadgroup]],                  \
      uint simd_lane_id [[thread_index_in_simdgroup]]);

// clang-format off
#define instantiate_winograd_conv_2d(name, itype)                     \
  instantiate_winograd_conv_2d_weight_transform_base(name, itype, 32) \
  instantiate_winograd_conv_2d_input_transform(name, itype, 32)       \
  instantiate_winograd_conv_2d_output_transform(name, itype, 32) // clang-format on

// clang-format off
instantiate_winograd_conv_2d(float32, float);

instantiate_winograd_conv_2d(float16, half); // clang-format on

