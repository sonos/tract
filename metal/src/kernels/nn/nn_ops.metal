#include <metal_stdlib>
#include <metal_math>

using namespace metal;

#define NUM_SIMDGROUP 32

METAL_FUNC uint indices_to_idx_3(uint3 indices, constant const uint strides[3]) {
  return indices.x * strides[2] + indices.y * strides[1] + indices.z * strides[0];
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

    F norm = metal::rsqrt(static_cast<F>(mean_of_squares) + eps);

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


