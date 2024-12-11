#include <metal_stdlib>
#include <metal_math>

using namespace metal;

#define NUM_SIMDGROUP 32

METAL_FUNC uint indices_to_idx_2(uint2 indices, constant const size_t strides[2]) {
  return indices.x * strides[1] + indices.y * strides[0];
}

METAL_FUNC uint indices_to_idx_3(uint3 indices, constant const size_t strides[3]) {
  return indices.x * strides[2] + indices.y * strides[1] + indices.z * strides[0];
}

METAL_FUNC uint indices_to_idx_4(uint3 indices,
                                 constant const size_t shape[4], 
                                 constant const size_t strides[4]) {
  auto idx = indices.x * strides[3] + indices.y * strides[2];
  idx += (indices.z % shape[1]) * strides[1];
  indices.z /= shape[1];
  idx += indices.z * strides[0];
  return idx;
}

template <typename U>
struct MeanOfSquares {
  float simd_reduce(float val, size_t reduce_dim) {
    return simd_sum(val) / static_cast<float>(reduce_dim);
  }

  static constexpr constant float init = 0.0;

  // Operator
  float operator()(float acc, U a) {
    float a_f = static_cast<float>(a);
    return acc + a_f * a_f;
  }
};

template <typename U>
struct Sum {
  U simd_reduce(U val, size_t reduce_dim) {
    return simd_sum(val);
  }

  static constexpr constant U init = U(0);

  // Operator
  U operator()(U acc, U a) {
    return acc + a;
  }
};

template <typename U>
struct Min {
  template <typename T>
  T simd_reduce(T val, size_t reduce_dim) {
    return simd_min(val);
  }

  static constexpr constant U init = metal::numeric_limits<U>::infinity();

  // Operator
  U operator()(U a, U b) {
    return a < b ? a : b;
  }
};

template <typename U>
struct Max {
  template <typename T>
  T simd_reduce(T val, size_t reduce_dim) {
    return simd_max(val);
  }

  static constexpr constant U init = -metal::numeric_limits<U>::infinity();

  // Operator
  U operator()(U a, U b) {
    return a > b ? a : b;
  }
};



template <typename U>
struct Prod {
  U simd_reduce(U val, size_t reduce_dim) {
    return simd_product(val);
  }

  static constexpr constant U init = U(1);

  // Operator
  U operator()(U acc, U a) {
    return acc * a;
  }
};


template<typename F, typename Op>  
[[kernel]] void reduce_nd3(
                device const void *input_b,
                device void *output_b,
                constant const size_t input_shape[3], 
                constant const size_t input_strides[3],
                constant const size_t output_strides[3],
                uint3  tgpig[[threadgroup_position_in_grid]],
        		    uint  tiisg[[thread_index_in_simdgroup]],
        		    uint  tpsg[[threads_per_simdgroup]]
                ) {

    device const F *input = (device const F *)input_b;
    device F *output = (device F *)output_b;

    Op op = Op();

    size_t reduce_dim = input_shape[1];

    size_t out_idx = tgpig.x * output_strides[2] 
            + tgpig.y * output_strides[1] 
            + tgpig.z * output_strides[0];

    size_t base_in_idx = tgpig.x * input_strides[2] 
            + tgpig.z * input_strides[0];

    auto partial_acc = Op::init;
    for (size_t i = tiisg; i < reduce_dim; i += tpsg) {
        F el = input[base_in_idx + i * input_strides[1]];
        partial_acc = op(partial_acc, el);
    }
    auto acc = op.simd_reduce(partial_acc, reduce_dim);

    if (tiisg == 0) {
       output[out_idx] = acc;
    }
}

typedef decltype(reduce_nd3<float, Prod<float>>) reduce_nd3_t;

#define INSTANTIATE_REDUCE(name, op, tname, type)                    \
template [[host_name("nn_ops::reduce_" #name "_nd3_" #tname)]]       \
[[kernel]] reduce_nd3_t reduce_nd3<type, op<type>>;


INSTANTIATE_REDUCE(mean_of_squares, MeanOfSquares, f32, float)
INSTANTIATE_REDUCE(mean_of_squares, MeanOfSquares, f16, half)
INSTANTIATE_REDUCE(sum, Sum, f32, float)
INSTANTIATE_REDUCE(sum, Sum, f16, half)
INSTANTIATE_REDUCE(min, Min, f32, float)
INSTANTIATE_REDUCE(min, Min, f16, half)
INSTANTIATE_REDUCE(max, Max, f32, float)
INSTANTIATE_REDUCE(max, Max, f16, half)
INSTANTIATE_REDUCE(prod, Prod, f32, float)
INSTANTIATE_REDUCE(prod, Prod, f16, half)


template<typename F>  
[[kernel]] void rms_norm_nd3(
                device const void *input_b,
                constant void * eps_b,
                device void *output_b,
                constant const size_t shape[3], 
                constant const size_t strides[3],
                uint3  tgpig[[threadgroup_position_in_grid]],
                uint  tiisg[[thread_index_in_simdgroup]],
                uint  tpsg[[threads_per_simdgroup]]
                ) {

    device const F* input = (device const F*) input_b;
    F eps = ((constant F *)eps_b)[0];
    device F * output = (device F*) output_b;

    size_t dim = shape[1];

    size_t base_idx = tgpig.x * strides[2] 
            + tgpig.z * strides[0];

    float partial_acc = 0.0;
    for (size_t i = tiisg; i < dim; i += tpsg) {
        float el = static_cast<float>(input[base_idx + i * strides[1]]);
        partial_acc += el * el;
    }
    float mean_of_squares = simd_sum(partial_acc) / static_cast<float>(dim);

    F norm = static_cast<F>(metal::rsqrt(mean_of_squares + static_cast<float>(eps)));

    for (size_t i = tiisg; i < dim; i += tpsg) {
        auto idx = base_idx + i * strides[1];
        output[idx] = input[idx] * norm;
    }
}

typedef decltype(rms_norm_nd3<float>) rms_norm_nd3_t;

template [[host_name("nn_ops::rms_norm_nd3_f32")]] [[kernel]] rms_norm_nd3_t rms_norm_nd3<float>;
template [[host_name("nn_ops::rms_norm_nd3_f16")]] [[kernel]] rms_norm_nd3_t rms_norm_nd3<half>;


struct Sigmoid {
  template <typename T>
  T operator()(T x) {
    auto y = 1 / (1 + metal::exp(-metal::abs(x)));
    return (x < 0) ? 1 - y : y;
  }
};

template<typename T>
[[kernel]] void silu(device const void *input_b [[buffer(0)]],
                             device void *output_b [[buffer(1)]],
                             uint tpig[[thread_position_in_grid]]) {
   device const T *input = (device const T *)input_b;
   device T *output = (device T *)output_b;

   output[tpig] = Sigmoid()(input[tpig]) * input[tpig];
}

typedef decltype(silu<float>) silu_t;

template [[host_name("nn_ops::silu_f32")]] [[kernel]] silu_t silu<float>;
template [[host_name("nn_ops::silu_f16")]] [[kernel]] silu_t silu<half>;


template<typename F>  
[[kernel]] void softmax_nd3(
                device const void *input_b,
                device void *output_b,
                constant const size_t shape[3], 
                constant const size_t strides[3],
                uint3  tgpig[[threadgroup_position_in_grid]],
                uint  tiisg[[thread_index_in_simdgroup]],
                uint  tpsg[[threads_per_simdgroup]]
                ) {

    device const F *input = (device const F *)input_b;
    device F *output = (device F *)output_b;

    size_t dim = shape[1];

    size_t base_idx = tgpig.x * strides[2] 
            + tgpig.z * strides[0];

    // Get max value on softmax dim
    float partial_max = -INFINITY;
    for (size_t i = tiisg; i < dim; i += tpsg) {
        auto idx = base_idx + i * strides[1];
        float el = static_cast<float>(input[idx]);
        partial_max = max(partial_max, el);
    }

    float axis_max = simd_max(partial_max);

    // Compute Sum(exp(x - max))
    float partial_norm = 0;
    for (size_t i = tiisg; i < dim; i += tpsg) {
        auto idx = base_idx + i * strides[1];
        float el = static_cast<float>(input[idx]);
        float exp_el = fast::exp(el - axis_max);
        partial_norm += exp_el;
        output[idx] = static_cast<F>(exp_el);
    }

    float axis_norm = simd_sum(partial_norm);
    float inv_axis_norm = 1.0 / axis_norm;

    for (size_t i = tiisg; i < dim; i += tpsg) {
        auto idx = base_idx + i * strides[1];
        float exp_el = static_cast<float>(output[idx]);
        output[idx] = static_cast<F>(exp_el * inv_axis_norm);
    }
}

typedef decltype(softmax_nd3<float>) softmax_nd3_t;

template [[host_name("nn_ops::softmax_nd3_f32")]] [[kernel]] softmax_nd3_t softmax_nd3<float>;
template [[host_name("nn_ops::softmax_nd3_f16")]] [[kernel]] softmax_nd3_t softmax_nd3<half>;

template<typename F>  
[[kernel]] void scaled_masked_softmax_nd3(
                device const void *input_b,
                device const void *mask_b,
                constant void *scale_b,
                device void *output_b,
                constant const size_t shape[3], 
                constant const size_t strides[3],
                constant const size_t mask_strides[3],
                uint3  tgpig[[threadgroup_position_in_grid]],
                uint  tiisg[[thread_index_in_simdgroup]],
                uint  tpsg[[threads_per_simdgroup]]
                ) {

    device const F *input = (device const F *)input_b;
    device const F *mask = (device const F *)mask_b;
    F scale = ((constant F *)scale_b)[0];
    device F *output = (device F *)output_b;

    size_t reduce_dim = shape[2];

    size_t base_idx = tgpig.y * strides[1] 
            + tgpig.z * strides[0];

    size_t mask_base_idx = tgpig.y * mask_strides[1] 
            + tgpig.z * mask_strides[0];

    // Get max value on softmax reduce_dim after applying scale and mask
    float partial_max = -INFINITY;
    for (size_t i = tiisg; i < reduce_dim; i += tpsg) {
        auto idx = base_idx + i * strides[2];
        auto mask_idx = mask_base_idx + i * mask_strides[2];
        output[idx] = input[idx] * scale + mask[mask_idx];
        float el = static_cast<float>(output[idx]);
        partial_max = max(partial_max, el);
    }

   float axis_max = simd_max(partial_max);

   // Compute Sum(exp(x - max))
   float partial_norm = 0;
   for (size_t i = tiisg; i < reduce_dim; i += tpsg) {
       auto idx = base_idx + i * strides[2];
       float el = static_cast<float>(output[idx]);
       float exp_el = fast::exp(el - axis_max);
       partial_norm += exp_el;
   }

   float axis_norm = simd_sum(partial_norm);
   float inv_axis_norm = 1.0 / axis_norm;

   for (size_t i = tiisg; i < reduce_dim; i += tpsg) {
       auto idx = base_idx + i * strides[2];
       float el = static_cast<float>(output[idx]);
       float exp_el = fast::exp(el - axis_max);
       output[idx] = static_cast<F>(exp_el * inv_axis_norm);
   }
}

typedef decltype(scaled_masked_softmax_nd3<float>) scaled_masked_softmax_nd3_t;

template [[host_name("nn_ops::scaled_masked_softmax_nd3_f32")]] [[kernel]] scaled_masked_softmax_nd3_t scaled_masked_softmax_nd3<float>;
template [[host_name("nn_ops::scaled_masked_softmax_nd3_f16")]] [[kernel]] scaled_masked_softmax_nd3_t scaled_masked_softmax_nd3<half>;

constant float GELU_COEF_A     = 0.044715f;
constant float SQRT_2_OVER_PI  = 0.79788456080286535587989211986876f;

template<typename F>  
[[kernel]] void new_gelu(
                device const void *input_b,
                device void *output_b,
                uint tpig[[thread_position_in_grid]]
                ) {

    device const F *input = (device const F *)input_b;
    device F *output = (device F *)output_b;

    float x = static_cast<float>(input[tpig]);
    float output_f32 = 0.5 * x * (
      1.0 + precise::tanh(SQRT_2_OVER_PI
          *(x + GELU_COEF_A * metal::powr(x, 3))));
    output[tpig] = static_cast<F>(output_f32);
}

typedef decltype(new_gelu<float>) new_gelu_t;

template [[host_name("nn_ops::new_gelu_f32")]] [[kernel]] new_gelu_t new_gelu<float>;
template [[host_name("nn_ops::new_gelu_f16")]] [[kernel]] new_gelu_t new_gelu<half>;

template<typename F>  
[[kernel]] void new_gelu_fast(
                device const void *input_b,
                device void *output_b,
                uint tpig[[thread_position_in_grid]]
                ) {

    device const F *input = (device const F *)input_b;
    device F *output = (device F *)output_b;

    float x = static_cast<float>(input[tpig]);
    float output_f32 = 0.5 * x * (
      1.0 + precise::tanh(SQRT_2_OVER_PI
          *(x + GELU_COEF_A * metal::powr(x, 2))));
    output[tpig] = static_cast<F>(output_f32);
}

typedef decltype(new_gelu_fast<float>) new_gelu_fast_t;

template [[host_name("nn_ops::new_gelu_fast_f32")]] [[kernel]] new_gelu_fast_t new_gelu_fast<float>;
template [[host_name("nn_ops::new_gelu_fast_f16")]] [[kernel]] new_gelu_fast_t new_gelu_fast<half>;



template<typename T>  
[[kernel]] void apply_rope_nd2(             
      device const void *input_b [[buffer(0)]],
      device const void *cos_b [[buffer(1)]],
      device const void *sin_b [[buffer(2)]],                 
      device void *output_b [[buffer(3)]],                        
      constant const size_t * shape [[buffer(4)]],
      constant const size_t * strides [[buffer(5)]],
      constant const size_t * cos_sin_strides [[buffer(6)]],              
      uint2 tpig[[thread_position_in_grid]]                   
) {
  device const T *input = (device const T *)input_b;
  device const T *cos = (device const T *)cos_b;
  device const T *sin = (device const T *)sin_b;

  device T* output = (device T *) output_b;

  uint2 rotated_tpig = tpig;
  rotated_tpig.x += shape[1] / 2;

  auto idx = indices_to_idx_2(tpig, strides);
  auto rot_idx = indices_to_idx_2(rotated_tpig, strides);

  auto cos_sin_idx = indices_to_idx_2(tpig, cos_sin_strides);
  auto rot_cos_sin_idx = indices_to_idx_2(rotated_tpig, cos_sin_strides);

  output[idx] = input[idx] * cos[cos_sin_idx] - input[rot_idx] * sin[cos_sin_idx];
  output[rot_idx] = input[rot_idx] * cos[rot_cos_sin_idx] 
          + input[idx] * sin[rot_cos_sin_idx];
}

template<typename T>  
[[kernel]] void apply_rope_nd3(             
      device const void *input_b [[buffer(0)]],
      device const void *cos_b [[buffer(1)]],
      device const void *sin_b [[buffer(2)]],                 
      device void *output_b [[buffer(3)]],                        
      constant const size_t * shape [[buffer(4)]],
      constant const size_t * strides [[buffer(5)]],
      constant const size_t * cos_sin_strides [[buffer(6)]],              
      uint3 tpig[[thread_position_in_grid]]                   
) {
  device const T *input = (device const T *)input_b;
  device const T *cos = (device const T *)cos_b;
  device const T *sin = (device const T *)sin_b;

  device T* output = (device T *) output_b;

  uint3 rotated_tpig = tpig;
  rotated_tpig.x += shape[2] / 2;

  auto idx = indices_to_idx_3(tpig, strides);
  auto rot_idx = indices_to_idx_3(rotated_tpig, strides);

  auto cos_sin_idx = indices_to_idx_3(tpig, cos_sin_strides);
  auto rot_cos_sin_idx = indices_to_idx_3(rotated_tpig, cos_sin_strides);

  output[idx] = input[idx] * cos[cos_sin_idx] - input[rot_idx] * sin[cos_sin_idx];
  output[rot_idx] = input[rot_idx] * cos[rot_cos_sin_idx] 
          + input[idx] * sin[rot_cos_sin_idx];
}

template<typename T>  
[[kernel]] void apply_rope_nd4(             
      device const void *input_b [[buffer(0)]],
      device const void *cos_b [[buffer(1)]],
      device const void *sin_b [[buffer(2)]],                 
      device void *output_b [[buffer(3)]],                        
      constant const size_t * shape [[buffer(4)]],
      constant const size_t * strides [[buffer(5)]],
      constant const size_t * cos_sin_strides [[buffer(6)]],              
      uint3 tpig[[thread_position_in_grid]]                   
) {
  device const T *input = (device const T *)input_b;
  device const T *cos = (device const T *)cos_b;
  device const T *sin = (device const T *)sin_b;

  device T* output = (device T *) output_b;

  uint3 rotated_tpig = tpig;
  rotated_tpig.x += shape[3] / 2;

  auto idx = indices_to_idx_4(tpig, shape, strides);
  auto rot_idx = indices_to_idx_4(rotated_tpig, shape, strides);

  auto cos_sin_idx = indices_to_idx_4(tpig, shape, cos_sin_strides);
  auto rot_cos_sin_idx = indices_to_idx_4(rotated_tpig, shape, cos_sin_strides);

  output[idx] = input[idx] * cos[cos_sin_idx] - input[rot_idx] * sin[cos_sin_idx];
  output[rot_idx] = input[rot_idx] * cos[rot_cos_sin_idx] 
          + input[idx] * sin[rot_cos_sin_idx];
}


typedef decltype(apply_rope_nd2<float>) apply_rope_nd2_t;
typedef decltype(apply_rope_nd3<float>) apply_rope_nd3_t;
typedef decltype(apply_rope_nd4<float>) apply_rope_nd4_t;

template [[host_name("nn_ops::apply_rope_nd2_f32")]] [[kernel]] apply_rope_nd2_t apply_rope_nd2<float>;
template [[host_name("nn_ops::apply_rope_nd3_f32")]] [[kernel]] apply_rope_nd3_t apply_rope_nd3<float>;
template [[host_name("nn_ops::apply_rope_nd4_f32")]] [[kernel]] apply_rope_nd4_t apply_rope_nd4<float>;

template [[host_name("nn_ops::apply_rope_nd2_f16")]] [[kernel]] apply_rope_nd2_t apply_rope_nd2<half>;
template [[host_name("nn_ops::apply_rope_nd3_f16")]] [[kernel]] apply_rope_nd3_t apply_rope_nd3<half>;
template [[host_name("nn_ops::apply_rope_nd4_f16")]] [[kernel]] apply_rope_nd4_t apply_rope_nd4<half>;


