#include <metal_stdlib>
#include <metal_simdgroup_matrix>  // Available from Metal version 2.3 released with OS X 11.0+

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
                device const F *input [[buffer(0)]],
                device F *output [[buffer(1)]],
                constant const size_t input_shape[3], 
                constant const size_t input_strides[3],
                constant const size_t output_strides[3],
                uint3  tgpig[[threadgroup_position_in_grid]],
        		uint  tiisg[[thread_index_in_simdgroup]],
        		uint  tpsg[[threads_per_simdgroup]]
                ) {

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

#define INSTANTIATE_REDUCE(name, op, tname, type)                    \
template [[host_name("nn_ops::reduce_" #name "_nd3_" #tname)]]       \
[[kernel]] void reduce_nd3<type, op<type>>(                          \
        device const type *input [[buffer(0)]],                      \
        device type *output [[buffer(1)]],                           \
        constant const size_t input_shape[3],                        \
        constant const size_t input_strides[3],                      \
        constant const size_t output_strides[3],                     \
        uint3  tgpig[[threadgroup_position_in_grid]],                \
        uint  tiisg[[thread_index_in_simdgroup]],                    \
        uint  tpsg[[threads_per_simdgroup]]                          \
    );

INSTANTIATE_REDUCE(mean_of_squares, MeanOfSquares, f32, float)
INSTANTIATE_REDUCE(mean_of_squares, MeanOfSquares, f16, half)
INSTANTIATE_REDUCE(sum, Sum, f32, float)
INSTANTIATE_REDUCE(sum, Sum, f16, half)
INSTANTIATE_REDUCE(prod, Prod, f32, float)
INSTANTIATE_REDUCE(prod, Prod, f16, half)

