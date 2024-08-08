#include <metal_stdlib>
#include <metal_integer>
#include <metal_math>
#include <metal_simdgroup_matrix>  // Available from Metal version 2.3 released with OS X 11.0+

using namespace metal;

namespace utils {
    
    METAL_FUNC uint indices_to_idx_1(uint index, constant const size_t strides[1]) {
        return index * strides[0];
    }
    
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
    
    METAL_FUNC uint indices_to_idx_5(uint3 indices,
                                     constant const size_t shape[5], 
                                     constant const size_t strides[5]) {
        auto idx = indices.x * strides[4] + indices.y * strides[3];
        idx += (indices.z % shape[2]) * strides[2];
        indices.z /= shape[2];
        idx += (indices.z % shape[1]) * strides[1];
        indices.z /= shape[1];
        idx += indices.z * strides[0];
        return idx;
    }
}

/*
 * Based on code from:
 * https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/binary_ops.h
 */

struct Add {
    template <typename T>
    T operator()(T x, T y) {
        return x + y;
    }
};

struct Div {
    template <typename T>
    T operator()(T x, T y) {
        return x / y;
    }
};

struct Sub {
    template <typename T>
    T operator()(T x, T y) {
        return x - y;
    }
};

struct Mul {
    template <typename T>
    T operator()(T x, T y) {
        return x * y;
    }
};

struct Equals {
    template <typename T>
    bool operator()(T x, T y) {
        return x == y;
    }
};

struct NotEquals {
    template <typename T>
    bool operator()(T x, T y) {
        return x != y;
    }
};

struct Greater {
    template <typename T>
    bool operator()(T x, T y) {
        return x > y;
    }
};

struct GreaterEqual {
    template <typename T>
    bool operator()(T x, T y) {
        return x >= y;
    }
};

struct Less {
    template <typename T>
    bool operator()(T x, T y) {
        return x < y;
    }
};

struct LessEqual {
    template <typename T>
    bool operator()(T x, T y) {
        return x <= y;
    }
};

struct And {
    template <typename T>
    T operator()(T x, T y) {
        return x && y;
    };
};

struct Or {
    template <typename T>
    T operator()(T x, T y) {
        return x || y;
    };
};

struct Pow {
    template <typename T>
    metal::enable_if_t<!metal::is_integral_v<T>, T>
    operator()(T base, T exp) {
        return metal::pow(base, exp);
    }
    
    template <typename T>
    metal::enable_if_t<metal::is_integral_v<T>, T>
    operator()(T base, T exp) {
        T res = 1;
        while (exp) {
            if (exp & 1) {
                res *= base;
            }
            exp >>= 1;
            base *= base;
        }
        return res;
    }
};

#define INSTANTIATE_BIN_OP(name, op, itname, itype, otype)                    \
template [[host_name("bin_ops::" #name "_unicast_" #itname)]] [[kernel]]      \
bin_op_unicast_t bin_op_unicast<itype, otype, op>;                            \
template [[host_name("bin_ops::" #name "_by_scalar_rhs_" #itname)]]           \
[[kernel]] bin_op_by_scalar_rhs_t bin_op_by_scalar_rhs<itype, otype, op>;     \
template [[host_name("bin_ops::" #name "_by_scalar_lhs_" #itname)]]           \
[[kernel]] bin_op_by_scalar_lhs_t bin_op_by_scalar_lhs<itype, otype, op>;     \
template [[host_name("bin_ops::" #name "_nd2_" #itname)]]                     \
[[kernel]] bin_op_nd2_t bin_op_nd2<itype, otype, op>;                         \
template [[host_name("bin_ops::" #name "_nd3_" #itname)]]                     \
[[kernel]] bin_op_nd3_t bin_op_nd3<itype, otype, op>;                         \
template [[host_name("bin_ops::" #name "_nd4_" #itname)]]                     \
[[kernel]] bin_op_nd4_t bin_op_nd4<itype, otype, op>;                         \
template [[host_name("bin_ops::" #name "_nd5_" #itname)]]                     \
[[kernel]] bin_op_nd5_t bin_op_nd5<itype, otype, op>;


#define INSTANTIATE_FLOAT(name, op)                     \
INSTANTIATE_BIN_OP(name, op, f32, float, float)         \
INSTANTIATE_BIN_OP(name, op, f16, half, half)          

#define INSTANTIATE_FLOAT_BOOL(name, op)                \
INSTANTIATE_BIN_OP(name, op, f32, float, bool)          \
INSTANTIATE_BIN_OP(name, op, f16, half, bool)          

#define INSTANTIATE_INTEGER(name, op)                    \
INSTANTIATE_BIN_OP(name, op, u8,  uint8_t, uint8_t)      \
INSTANTIATE_BIN_OP(name, op, u16, uint16_t, uint16_t)    \
INSTANTIATE_BIN_OP(name, op, u32, uint32_t, uint32_t)    \
INSTANTIATE_BIN_OP(name, op, u64, uint64_t, uint64_t)    \
INSTANTIATE_BIN_OP(name, op, i8,  int8_t, int8_t)        \
INSTANTIATE_BIN_OP(name, op, i16, int16_t, int16_t)      \
INSTANTIATE_BIN_OP(name, op, i32, int32_t, int32_t)      \
INSTANTIATE_BIN_OP(name, op, i64, int64_t, int64_t)       

#define INSTANTIATE_INTEGER_BOOL(name, op)               \
INSTANTIATE_BIN_OP(name, op, u8,  uint8_t, bool)         \
INSTANTIATE_BIN_OP(name, op, u16, uint16_t, bool)        \
INSTANTIATE_BIN_OP(name, op, u32, uint32_t, bool)        \
INSTANTIATE_BIN_OP(name, op, u64, uint64_t, bool)        \
INSTANTIATE_BIN_OP(name, op, i8,  int8_t, bool)          \
INSTANTIATE_BIN_OP(name, op, i16, int16_t, bool)         \
INSTANTIATE_BIN_OP(name, op, i32, int32_t, bool)         \
INSTANTIATE_BIN_OP(name, op, i64, int64_t, bool)        

#define INSTANTIATE_ALL_TYPES(name, op)                  \
INSTANTIATE_FLOAT(name, op)                              \
INSTANTIATE_INTEGER(name, op)  

#define INSTANTIATE_ALL_TYPES_BOOL(name, op)             \
INSTANTIATE_FLOAT_BOOL(name, op)                         \
INSTANTIATE_INTEGER_BOOL(name, op)                

template<typename In, typename Out, typename Op>
[[kernel]] void bin_op_unicast(device const void *lhs_b [[buffer(0)]],
                               device const void *rhs_b [[buffer(1)]],
                               device void *output_b [[buffer(2)]],
                               uint tpig[[thread_position_in_grid]]) {
    device const In * lhs = (device const In *)lhs_b;
    device const In * rhs = (device const In *)rhs_b;
    device  Out * output = (device  Out *)output_b;
    output[tpig] = Op()(lhs[tpig], rhs[tpig]);
}

typedef decltype(bin_op_unicast<float, float, Mul>) bin_op_unicast_t;

template<typename In, typename Out, typename Op>
[[kernel]] void bin_op_by_scalar_rhs(device const void *lhs_b [[buffer(0)]],
                                     device const void *rhs_b [[buffer(1)]],
                                     device void *output_b [[buffer(2)]],
                                     uint tpig[[thread_position_in_grid]]) {
    device const In * lhs = (device const In *)lhs_b;
    device const In * rhs = (device const In *)rhs_b;
    device  Out * output = (device  Out *)output_b;
    output[tpig] = Op()(lhs[tpig], rhs[0]);
}

typedef decltype(bin_op_by_scalar_rhs<float, float, Mul>) bin_op_by_scalar_rhs_t;

template<typename In, typename Out, typename Op>
[[kernel]] void bin_op_by_scalar_lhs(device const void *lhs_b [[buffer(0)]],
                                     device const void *rhs_b [[buffer(1)]],
                                     device void *output_b [[buffer(2)]],
                                     uint tpig[[thread_position_in_grid]]) {
    device const In * lhs = (device const In *)lhs_b;
    device const In * rhs = (device const In *)rhs_b;
    device  Out * output = (device  Out *)output_b;
    output[tpig] = Op()(lhs[0], rhs[tpig]);
}

typedef decltype(bin_op_by_scalar_lhs<float, float, Mul>) bin_op_by_scalar_lhs_t;

template<typename In, typename Out, typename Op>
[[kernel]] void bin_op_nd2(device const void *lhs_b [[buffer(0)]],
                           constant const size_t * lhs_strides [[buffer(1)]],
                           device const void *rhs_b [[buffer(2)]],
                           constant const size_t * rhs_strides [[buffer(3)]],
                           device void *output_b [[buffer(4)]],
                           constant const size_t * out_shape [[buffer(5)]],
                           uint2 tpig[[thread_position_in_grid]],
                           uint2 grid_dim [[threads_per_grid]]) {
    device const In * lhs = (device const In *)lhs_b;
    device const In * rhs = (device const In *)rhs_b;
    device  Out * output = (device  Out *)output_b;
    auto lhs_idx = utils::indices_to_idx_2(tpig, lhs_strides);
    auto rhs_idx = utils::indices_to_idx_2(tpig, rhs_strides);
    auto out_idx = tpig.x + grid_dim.x * tpig.y;
    output[out_idx] = Op()(lhs[lhs_idx], rhs[rhs_idx]);
}

typedef decltype(bin_op_nd2<float, float, Mul>) bin_op_nd2_t;

template<typename In, typename Out, typename Op>
[[kernel]] void bin_op_nd3(device const void *lhs_b [[buffer(0)]],
                           constant const size_t * lhs_strides [[buffer(1)]],
                           device const void *rhs_b [[buffer(2)]],
                           constant const size_t * rhs_strides [[buffer(3)]],
                           device void *output_b [[buffer(4)]],
                           constant const size_t * out_shape [[buffer(5)]],
                           uint3 tpig[[thread_position_in_grid]],
                           uint3 grid_dim [[threads_per_grid]]) {
    
    device const In * lhs = (device const In *)lhs_b;
    device const In * rhs = (device const In *)rhs_b;
    device  Out * output = (device  Out *)output_b;
    
    auto lhs_idx = utils::indices_to_idx_3(tpig, lhs_strides);
    auto rhs_idx = utils::indices_to_idx_3(tpig, rhs_strides);
    auto out_idx = tpig.x + grid_dim.x * (tpig.y + grid_dim.y * tpig.z);
    output[out_idx] = Op()(lhs[lhs_idx], rhs[rhs_idx]);
}

typedef decltype(bin_op_nd3<float, float, Mul>) bin_op_nd3_t;

template<typename In, typename Out, typename Op>
[[kernel]] void bin_op_nd4(device const void *lhs_b [[buffer(0)]],
                           constant const size_t * lhs_strides [[buffer(1)]],
                           device const void *rhs_b [[buffer(2)]],
                           constant const size_t * rhs_strides [[buffer(3)]],
                           device void *output_b [[buffer(4)]],
                           constant const size_t * out_shape [[buffer(5)]],
                           uint3 tpig[[thread_position_in_grid]],
                           uint3 grid_dim [[threads_per_grid]]) {
    
    device const In * lhs = (device const In *)lhs_b;
    device const In * rhs = (device const In *)rhs_b;
    device  Out * output = (device  Out *)output_b;
    
    auto lhs_idx = utils::indices_to_idx_4(tpig, out_shape, lhs_strides);
    auto rhs_idx = utils::indices_to_idx_4(tpig, out_shape, rhs_strides);
    auto out_idx = tpig.x + grid_dim.x * (tpig.y + grid_dim.y * tpig.z);
    output[out_idx] =  Op()(lhs[lhs_idx], rhs[rhs_idx]);
}

typedef decltype(bin_op_nd4<float, float, Mul>) bin_op_nd4_t;

template<typename In, typename Out, typename Op>
[[kernel]] void bin_op_nd5(device const void *lhs_b [[buffer(0)]],
                           constant const size_t * lhs_strides [[buffer(1)]],
                           device const void *rhs_b [[buffer(2)]],
                           constant const size_t * rhs_strides [[buffer(3)]],
                           device void *output_b [[buffer(4)]],
                           constant const size_t * out_shape [[buffer(5)]],
                           uint3 tpig[[thread_position_in_grid]],
                           uint3 grid_dim [[threads_per_grid]]) {
    
    device const In * lhs = (device const In *)lhs_b;
    device const In * rhs = (device const In *)rhs_b;
    device  Out * output = (device  Out *)output_b;
    
    auto lhs_idx = utils::indices_to_idx_5(tpig, out_shape, lhs_strides);
    auto rhs_idx = utils::indices_to_idx_5(tpig, out_shape, rhs_strides);
    auto out_idx = tpig.x + grid_dim.x * (tpig.y + grid_dim.y * tpig.z);
    output[out_idx] =  Op()(lhs[lhs_idx], rhs[rhs_idx]);
}

typedef decltype(bin_op_nd5<float, float, Mul>) bin_op_nd5_t;

INSTANTIATE_ALL_TYPES(mul, Mul)
INSTANTIATE_ALL_TYPES(div, Div)
INSTANTIATE_ALL_TYPES(add, Add)
INSTANTIATE_ALL_TYPES(sub, Sub)
INSTANTIATE_ALL_TYPES(pow, Pow)
INSTANTIATE_ALL_TYPES_BOOL(less, Less)
INSTANTIATE_ALL_TYPES_BOOL(greater, Greater)
INSTANTIATE_ALL_TYPES_BOOL(less_equal, LessEqual)
INSTANTIATE_ALL_TYPES_BOOL(greater_equal, GreaterEqual)
INSTANTIATE_ALL_TYPES_BOOL(equals, Equals)
INSTANTIATE_ALL_TYPES_BOOL(not_equals, NotEquals)
INSTANTIATE_BIN_OP(and, And, bool, bool, bool)
INSTANTIATE_BIN_OP(or, Or, bool, bool, bool)
