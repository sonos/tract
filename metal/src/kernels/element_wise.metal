#include <metal_stdlib>
#include <metal_integer>
#include <metal_math>
#include <metal_simdgroup_matrix>  // Available from Metal version 2.3 released with OS X 11.0+

using namespace metal;

METAL_FUNC float erf_f32(float x ) {
    const float a1 = 0.0705230784;
    const float a2 = 0.0422820123;
    const float a3 = 0.0092705272;
    const float a4 = 0.0001520143;
    const float a5 = 0.0002765672;
    const float a6 = 0.0000430638;
    
    float abs = metal::abs(x);
    float y = a6 * abs;
    y = (a5 + y) * abs;
    y = (a4 + y) * abs;
    y = (a3 + y) * abs;
    y = (a2 + y) * abs;
    y = (a1 + y) * abs;
    y = 1.0 - (1.0 / metal::powr(y + 1.0, 16));
    y = metal::copysign(y, x);
    return y;
}

/*
 * Based on code from:
 * https://github.com/ml-explore/mlx/blob/main/mlx/backend/metal/kernels/unary_ops.h
 */

struct Abs {
    template <typename T>
    metal::enable_if_t<metal::is_integral_v<T> & !metal::is_signed_v<T>, T>
    operator()(T x) {
        return x;
    }
    
    template <typename T>
    metal::enable_if_t<metal::is_integral_v<T> & metal::is_signed_v<T>, T>
    operator()(T x) {
        return metal::abs(x);
    };
    
    template <typename T>
    metal::enable_if_t<!metal::is_integral_v<T>, T>
    operator()(T x) {
        return metal::abs(x);
    };
};

struct Ceil {
    template <typename T>
    metal::enable_if_t<!metal::is_integral_v<T>, T>
    operator()(T x) {
        return metal::ceil(x);
    }
    
    template <typename T>
    metal::enable_if_t<metal::is_integral_v<T>, T>
    operator()(T x) {
        return x;
    }
};

struct Floor {
    template <typename T>
    metal::enable_if_t<!metal::is_integral_v<T>, T>
    operator()(T x) {
        return metal::floor(x);
    }
    
    template <typename T>
    metal::enable_if_t<metal::is_integral_v<T>, T>
    operator()(T x) {
        return x;
    }
};

struct Round {
    template <typename T>
    metal::enable_if_t<!metal::is_integral_v<T>, T>
    operator()(T x) {
        return metal::round(x);
    }
    
    template <typename T>
    metal::enable_if_t<metal::is_integral_v<T>, T>
    operator()(T x) {
        return x;
    }
};

struct RoundHalfToEven {
    template <typename T>
    metal::enable_if_t<!metal::is_integral_v<T>, T>
    operator()(T x) {
        return metal::rint(x);
    }
    
    template <typename T>
    metal::enable_if_t<metal::is_integral_v<T>, T>
    operator()(T x) {
        return x;
    }
};

struct Recip {
    template <typename T>
    T operator()(T x) {
        return 1 / x;
    }
};

struct Erf {
    template <typename T>
    T operator()(T x) {
        return static_cast<T>(erf_f32(static_cast<float>(x)));
    };
};

struct Exp {
    template <typename T>
    T operator()(T x) {
        return metal::precise::exp(x);
    };
};

struct Ln {
    template <typename T>
    T operator()(T x) {
        return metal::precise::log(x);
    };
};

struct Sigmoid {
    template <typename T>
    T operator()(T x) {
        auto y = 1 / (1 + metal::exp(-metal::abs(x)));
        return (x < 0) ? 1 - y : y;
    }
};

// Cosine of x
struct Cos {
    template <typename T>
    T operator()(T x) {
        return metal::cos(x);
    }
};

// Hyperbolic cosine of x
struct Cosh {
    template <typename T>
    T operator()(T x) {
        return metal::cosh(x);
    }
};

// Arc cosine of x
struct Acos {
    template <typename T>
    T operator()(T x) {
        return metal::acos(x);
    }
};

// Inverse hyperbolic cosine of x
struct Acosh {
    template <typename T>
    T operator()(T x) {
        return metal::acosh(x);
    }
};

// Sine of x
struct Sin {
    template <typename T>
    T operator()(T x) {
        return metal::sin(x);
    }
};

// Hyperbolic sine of x
struct Sinh {
    template <typename T>
    T operator()(T x) {
        return metal::sinh(x);
    }
};

// Arc sine of x
struct Asin {
    template <typename T>
    T operator()(T x) {
        return metal::asin(x);
    }
};

// Inverse hyperbolic sine of x
struct Asinh {
    template <typename T>
    T operator()(T x) {
        return metal::asinh(x);
    }
};

// Tangent of x
struct Tan {
    template <typename T>
    T operator()(T x) {
        return metal::tan(x);
    }
};

// Arc tangent of x
struct Atan {
    template <typename T>
    T operator()(T x) {
        return metal::precise::atan(x);
    }
};

// Inverse hyperbolic tangent of x
struct Atanh {
    template <typename T>
    T operator()(T x) {
        return metal::precise::atanh(x);
    }
};

// Hyperbolic tangent of x
struct Tanh {
    template <typename T>
    T operator()(T x) {
        // Use precise to avoid NaN for large value with fast implementation 
        return metal::precise::tanh(x);
    }
};

struct Square {
    template <typename T>
    T operator()(T x) {
        return metal::pow(x, static_cast<T>(2.0));
    }
};

struct Sqrt {
    template <typename T>
    T operator()(T x) {
        return metal::precise::sqrt(x);
    };
};

struct Rsqrt {
    template <typename T>
    T operator()(T x) {
        return metal::precise::rsqrt(x);
    };
};

struct Neg {
    template <typename T>
    T operator()(T x) {
        return -x;
    };
};

struct Sign {
    template <typename T>
    metal::enable_if_t<!metal::is_integral_v<T>, T>
    operator()(T x) {
        return (x > T(0)) ? T(1) : ((x < T(0)) ? T(-1) : T(0));
    }

    template <typename T>
    metal::enable_if_t<metal::is_integral_v<T>, T>
    operator()(T x) {
        return (x > T(0)) - (x < T(0));
    }
};

struct HardSwish {
    template <typename T>
    T operator()(T x) {
        return x * metal::max(T(0), metal::min(T(1), x / T(6) + T(0.5)));
    }
};

struct Silu {
    template <typename T>
    T operator()(T x) {
        return x / (T(1) + metal::exp(-x));
    }
};

struct BitNot {
    template <typename T>
    metal::enable_if_t<metal::is_integral_v<T>, T>
    operator()(T x) {
        return ~x;
    }

    bool operator()(bool x) {
        return !x;
    }
};

template<typename T, typename Op>
[[kernel]] void eval_out_of_place(device const T *input[  [buffer(0)]],
                                  device T *output [[buffer(1)]],
                                  uint tpig[[thread_position_in_grid]]) {
    output[tpig] = Op()(input[tpig]);
}

template<typename T, typename Op>
[[kernel]] void eval_in_place(device T *inout[  [buffer(0)]],
                              uint tpig[[thread_position_in_grid]]) {
    inout[tpig] = Op()(inout[tpig]);
}

#define INSTANTIATE_ELEMENT_WISE_OP(name, op, tname, type)            \
template [[host_name("element_wise_ops::" #name "_out_of_place_" #tname)]] [[kernel]] void eval_out_of_place<type, op>(                                   \
device const type *input [[buffer(0)]],                    \
device type *output [[buffer(1)]],                        \
uint tpig[[thread_position_in_grid]]                       \
);                                                             \
template [[host_name("element_wise_ops::" #name "_in_place_" #tname)]] [[kernel]] void eval_in_place<type, op>(                                  \
device type *inout [[buffer(0)]],                          \
uint tpig[[thread_position_in_grid]]                       \
);


#define INSTANTIATE_FLOAT(name, op)                      \
INSTANTIATE_ELEMENT_WISE_OP(name, op, f32,  float)       \
INSTANTIATE_ELEMENT_WISE_OP(name, op, f16, half)         \

#define INSTANTIATE_INTEGER_SIGNED(name, op)             \
INSTANTIATE_ELEMENT_WISE_OP(name, op, i8,  int8_t)       \
INSTANTIATE_ELEMENT_WISE_OP(name, op, i16, int16_t)      \
INSTANTIATE_ELEMENT_WISE_OP(name, op, i32, int32_t)      \
INSTANTIATE_ELEMENT_WISE_OP(name, op, i64, int64_t)

#define INSTANTIATE_INTEGER_UNSIGNED(name, op)                    \
INSTANTIATE_ELEMENT_WISE_OP(name, op, u8,  uint8_t)      \
INSTANTIATE_ELEMENT_WISE_OP(name, op, u16, uint16_t)     \
INSTANTIATE_ELEMENT_WISE_OP(name, op, u32, uint32_t)     \
INSTANTIATE_ELEMENT_WISE_OP(name, op, u64, uint64_t)     \

#define INSTANTIATE_INTEGER(name, op)                    \
INSTANTIATE_INTEGER_SIGNED(name, op)                     \
INSTANTIATE_INTEGER_UNSIGNED(name, op)                   \

#define INSTANTIATE_ALL_TYPES(name, op)                  \
INSTANTIATE_FLOAT(name, op)                              \
INSTANTIATE_INTEGER(name, op)

INSTANTIATE_ALL_TYPES(abs, Abs)
INSTANTIATE_FLOAT(exp, Exp)
INSTANTIATE_FLOAT(ln, Ln)
INSTANTIATE_FLOAT(sqrt, Sqrt)
INSTANTIATE_FLOAT(rsqrt, Rsqrt)
INSTANTIATE_FLOAT(sigmoid, Sigmoid)
INSTANTIATE_FLOAT(square, Square)
INSTANTIATE_FLOAT(recip, Recip)
INSTANTIATE_ALL_TYPES(ceil, Ceil)
INSTANTIATE_ALL_TYPES(floor, Floor)
INSTANTIATE_ALL_TYPES(round, Round)
INSTANTIATE_ALL_TYPES(roundhalftoeven, RoundHalfToEven)
INSTANTIATE_FLOAT(cos, Cos)
INSTANTIATE_FLOAT(acos, Acos)
INSTANTIATE_FLOAT(acosh, Acosh)
INSTANTIATE_FLOAT(cosh, Cosh)
INSTANTIATE_FLOAT(sin, Sin)
INSTANTIATE_FLOAT(asin, Asin)
INSTANTIATE_FLOAT(asinh, Asinh)
INSTANTIATE_FLOAT(sinh, Sinh)
INSTANTIATE_FLOAT(tan, Tan)
INSTANTIATE_FLOAT(atan, Atan)
INSTANTIATE_FLOAT(atanh, Atanh)
INSTANTIATE_FLOAT(tanh, Tanh)
INSTANTIATE_FLOAT(erf, Erf)
INSTANTIATE_FLOAT(neg, Neg)
INSTANTIATE_INTEGER_SIGNED(neg, Neg)
INSTANTIATE_FLOAT(sign, Sign)
INSTANTIATE_INTEGER_SIGNED(sign, Sign)
INSTANTIATE_FLOAT(hardswish, HardSwish)
INSTANTIATE_FLOAT(silu, Silu)
INSTANTIATE_INTEGER(bitnot, BitNot)
INSTANTIATE_ELEMENT_WISE_OP(bitnot, BitNot, bool, bool)

// ---------------------------------------------------------------------------
// Fused elementwise chain.
//
// Executes a small RPN program over up to FUSED_EW_MAX_INPUTS input tensors,
// one thread per output element (or 4 in the _v4 variant), entirely in f32
// registers. The program is baked into the pipeline through function
// constants, so the compiler unrolls it into straight-line code (the
// pipeline cache keys on the constant values: one compile per distinct
// program). Each step can ask for its result to be rounded through f16,
// which reproduces the numerics of the original per-op kernels when the
// source chain computed in half. Rust-side encoding lives in
// kernels/element_wise.rs.
// ---------------------------------------------------------------------------

constant constexpr int FUSED_EW_MAX_INPUTS = 6;
constant constexpr int FUSED_EW_MAX_STEPS = 24;
constant constexpr int FUSED_EW_MAX_STACK = 8;

// Opcodes (keep in sync with kernels/element_wise.rs).
constant constexpr uint FEW_OP_MASK = 0xffu;
constant constexpr uint FEW_FLAG_ROUND_F16 = 0x100u;
constant constexpr uint FEW_SRC_SHIFT = 16;
constant constexpr uint FEW_PUSH_INPUT = 1;
constant constexpr uint FEW_PUSH_SCALAR = 2;
constant constexpr uint FEW_NEG = 16;
constant constexpr uint FEW_EXP = 17;
constant constexpr uint FEW_LN = 18;
constant constexpr uint FEW_SIGMOID = 19;
constant constexpr uint FEW_SILU = 20;
constant constexpr uint FEW_TANH = 21;
constant constexpr uint FEW_SQRT = 22;
constant constexpr uint FEW_RSQRT = 23;
constant constexpr uint FEW_RECIP = 24;
constant constexpr uint FEW_ABS = 25;
constant constexpr uint FEW_SQUARE = 26;
constant constexpr uint FEW_ID = 27; // identity: carries a round-to-f16 flag
constant constexpr uint FEW_ADD = 48;
constant constexpr uint FEW_SUB = 49;
constant constexpr uint FEW_MUL = 50;
constant constexpr uint FEW_DIV = 51;
constant constexpr uint FEW_MIN = 52;
constant constexpr uint FEW_MAX = 53;
constant constexpr uint FEW_POW = 54;

// Program function constants: step count, output dtype, per-input f16 bit
// mask, then one code (op | flags | src << 16) and one f32 immediate per step.
constant uint few_n_steps [[function_constant(0)]];
constant bool few_out_f16 [[function_constant(1)]];
constant uint few_in_f16_mask [[function_constant(2)]];
constant uint few_code_0 [[function_constant(10)]];
constant uint few_code_1 [[function_constant(11)]];
constant uint few_code_2 [[function_constant(12)]];
constant uint few_code_3 [[function_constant(13)]];
constant uint few_code_4 [[function_constant(14)]];
constant uint few_code_5 [[function_constant(15)]];
constant uint few_code_6 [[function_constant(16)]];
constant uint few_code_7 [[function_constant(17)]];
constant uint few_code_8 [[function_constant(18)]];
constant uint few_code_9 [[function_constant(19)]];
constant uint few_code_10 [[function_constant(20)]];
constant uint few_code_11 [[function_constant(21)]];
constant uint few_code_12 [[function_constant(22)]];
constant uint few_code_13 [[function_constant(23)]];
constant uint few_code_14 [[function_constant(24)]];
constant uint few_code_15 [[function_constant(25)]];
constant uint few_code_16 [[function_constant(26)]];
constant uint few_code_17 [[function_constant(27)]];
constant uint few_code_18 [[function_constant(28)]];
constant uint few_code_19 [[function_constant(29)]];
constant uint few_code_20 [[function_constant(30)]];
constant uint few_code_21 [[function_constant(31)]];
constant uint few_code_22 [[function_constant(32)]];
constant uint few_code_23 [[function_constant(33)]];
constant float few_imm_0 [[function_constant(40)]];
constant float few_imm_1 [[function_constant(41)]];
constant float few_imm_2 [[function_constant(42)]];
constant float few_imm_3 [[function_constant(43)]];
constant float few_imm_4 [[function_constant(44)]];
constant float few_imm_5 [[function_constant(45)]];
constant float few_imm_6 [[function_constant(46)]];
constant float few_imm_7 [[function_constant(47)]];
constant float few_imm_8 [[function_constant(48)]];
constant float few_imm_9 [[function_constant(49)]];
constant float few_imm_10 [[function_constant(50)]];
constant float few_imm_11 [[function_constant(51)]];
constant float few_imm_12 [[function_constant(52)]];
constant float few_imm_13 [[function_constant(53)]];
constant float few_imm_14 [[function_constant(54)]];
constant float few_imm_15 [[function_constant(55)]];
constant float few_imm_16 [[function_constant(56)]];
constant float few_imm_17 [[function_constant(57)]];
constant float few_imm_18 [[function_constant(58)]];
constant float few_imm_19 [[function_constant(59)]];
constant float few_imm_20 [[function_constant(60)]];
constant float few_imm_21 [[function_constant(61)]];
constant float few_imm_22 [[function_constant(62)]];
constant float few_imm_23 [[function_constant(63)]];

METAL_FUNC uint few_code(uint s) {
    switch (s) {
        case 0: return few_code_0;
        case 1: return few_code_1;
        case 2: return few_code_2;
        case 3: return few_code_3;
        case 4: return few_code_4;
        case 5: return few_code_5;
        case 6: return few_code_6;
        case 7: return few_code_7;
        case 8: return few_code_8;
        case 9: return few_code_9;
        case 10: return few_code_10;
        case 11: return few_code_11;
        case 12: return few_code_12;
        case 13: return few_code_13;
        case 14: return few_code_14;
        case 15: return few_code_15;
        case 16: return few_code_16;
        case 17: return few_code_17;
        case 18: return few_code_18;
        case 19: return few_code_19;
        case 20: return few_code_20;
        case 21: return few_code_21;
        case 22: return few_code_22;
        case 23: return few_code_23;
        default: return 0;
    }
}

METAL_FUNC float few_imm(uint s) {
    switch (s) {
        case 0: return few_imm_0;
        case 1: return few_imm_1;
        case 2: return few_imm_2;
        case 3: return few_imm_3;
        case 4: return few_imm_4;
        case 5: return few_imm_5;
        case 6: return few_imm_6;
        case 7: return few_imm_7;
        case 8: return few_imm_8;
        case 9: return few_imm_9;
        case 10: return few_imm_10;
        case 11: return few_imm_11;
        case 12: return few_imm_12;
        case 13: return few_imm_13;
        case 14: return few_imm_14;
        case 15: return few_imm_15;
        case 16: return few_imm_16;
        case 17: return few_imm_17;
        case 18: return few_imm_18;
        case 19: return few_imm_19;
        case 20: return few_imm_20;
        case 21: return few_imm_21;
        case 22: return few_imm_22;
        case 23: return few_imm_23;
        default: return 0.0f;
    }
}

// Runtime-only parameters (shapes carry the symbolic dims of the graph).
struct FusedEwRtParams {
    uint total;                              // number of output elements
    uint out_shape[4];                       // output padded to rank 4
    uint in_strides[FUSED_EW_MAX_INPUTS][4]; // element strides, 0 on broadcast axes
};

METAL_FUNC float few_load(device const void *in, bool f16, uint off) {
    return f16 ? (float)((device const half *)in)[off] : ((device const float *)in)[off];
}

METAL_FUNC float few_apply_unary(uint op, float x) {
    switch (op) {
        case FEW_NEG: return -x;
        case FEW_EXP: return metal::precise::exp(x);
        case FEW_LN: return metal::precise::log(x);
        case FEW_SIGMOID: {
            float y = 1.0f / (1.0f + metal::exp(-metal::abs(x)));
            return (x < 0.0f) ? 1.0f - y : y;
        }
        case FEW_SILU: return x / (1.0f + metal::exp(-x));
        case FEW_TANH: return metal::precise::tanh(x);
        case FEW_SQRT: return metal::precise::sqrt(x);
        case FEW_RSQRT: return metal::precise::rsqrt(x);
        case FEW_RECIP: return 1.0f / x;
        case FEW_ABS: return metal::abs(x);
        case FEW_SQUARE: return x * x;
        default: return x; // FEW_ID
    }
}

METAL_FUNC float few_apply_binary(uint op, float a, float b) {
    switch (op) {
        case FEW_ADD: return a + b;
        case FEW_SUB: return a - b;
        case FEW_MUL: return a * b;
        case FEW_DIV: return a / b;
        case FEW_MIN: return a < b ? a : b;
        case FEW_MAX: return a > b ? a : b;
        default: return metal::pow(a, b); // FEW_POW
    }
}

[[kernel]] void fused_elementwise_chain(
    device const void *in0 [[buffer(0)]],
    device const void *in1 [[buffer(1)]],
    device const void *in2 [[buffer(2)]],
    device const void *in3 [[buffer(3)]],
    device const void *in4 [[buffer(4)]],
    device const void *in5 [[buffer(5)]],
    device void *out [[buffer(6)]],
    constant FusedEwRtParams &p [[buffer(7)]],
    uint tpig [[thread_position_in_grid]]) {
    if (tpig >= p.total) {
        return;
    }
    device const void *ins[FUSED_EW_MAX_INPUTS] = {in0, in1, in2, in3, in4, in5};

    uint coords[4];
    uint idx = tpig;
    for (int a = 3; a >= 0; a--) {
        coords[a] = idx % p.out_shape[a];
        idx /= p.out_shape[a];
    }

    float stack[FUSED_EW_MAX_STACK];
    int sp = 0;
    for (uint s = 0; s < few_n_steps; s++) {
        const uint code = few_code(s);
        const uint op = code & FEW_OP_MASK;
        if (op == FEW_PUSH_INPUT) {
            const uint i = code >> FEW_SRC_SHIFT;
            const uint off = coords[0] * p.in_strides[i][0] + coords[1] * p.in_strides[i][1]
                + coords[2] * p.in_strides[i][2] + coords[3] * p.in_strides[i][3];
            stack[sp++] = few_load(ins[i], (few_in_f16_mask >> i) & 1, off);
        } else if (op == FEW_PUSH_SCALAR) {
            stack[sp++] = few_imm(s);
        } else if (op < FEW_ADD) {
            stack[sp - 1] = few_apply_unary(op, stack[sp - 1]);
        } else {
            float b = stack[--sp];
            stack[sp - 1] = few_apply_binary(op, stack[sp - 1], b);
        }
        if (code & FEW_FLAG_ROUND_F16) {
            stack[sp - 1] = (float)(half)stack[sp - 1];
        }
    }
    if (few_out_f16) {
        ((device half *)out)[tpig] = (half)stack[0];
    } else {
        ((device float *)out)[tpig] = stack[0];
    }
}

METAL_FUNC float4 few_load4(device const void *in, bool f16, uint off, uint stride) {
    if (stride == 0) {
        return float4(few_load(in, f16, off));
    }
    if (f16) {
        return float4(half4(*(device const packed_half4 *)((device const half *)in + off)));
    }
    return float4(*(device const packed_float4 *)((device const float *)in + off));
}

METAL_FUNC float4 few_apply_unary4(uint op, float4 x) {
    switch (op) {
        case FEW_NEG: return -x;
        case FEW_EXP: return metal::precise::exp(x);
        case FEW_LN: return metal::precise::log(x);
        case FEW_SIGMOID: {
            float4 y = 1.0f / (1.0f + metal::exp(-metal::abs(x)));
            return metal::select(y, 1.0f - y, x < 0.0f);
        }
        case FEW_SILU: return x / (1.0f + metal::exp(-x));
        case FEW_TANH: return metal::precise::tanh(x);
        case FEW_SQRT: return metal::precise::sqrt(x);
        case FEW_RSQRT: return metal::precise::rsqrt(x);
        case FEW_RECIP: return 1.0f / x;
        case FEW_ABS: return metal::abs(x);
        case FEW_SQUARE: return x * x;
        default: return x; // FEW_ID
    }
}

METAL_FUNC float4 few_apply_binary4(uint op, float4 a, float4 b) {
    switch (op) {
        case FEW_ADD: return a + b;
        case FEW_SUB: return a - b;
        case FEW_MUL: return a * b;
        case FEW_DIV: return a / b;
        case FEW_MIN: return metal::min(a, b);
        case FEW_MAX: return metal::max(a, b);
        default: return metal::pow(a, b); // FEW_POW
    }
}

// 4-wide variant: each thread computes 4 consecutive output elements along
// the innermost axis. Dispatched when every input's innermost stride is 0 or
// 1 and the innermost output dim is a multiple of 4, which amortizes the
// program per 4 elements.
[[kernel]] void fused_elementwise_chain_v4(
    device const void *in0 [[buffer(0)]],
    device const void *in1 [[buffer(1)]],
    device const void *in2 [[buffer(2)]],
    device const void *in3 [[buffer(3)]],
    device const void *in4 [[buffer(4)]],
    device const void *in5 [[buffer(5)]],
    device void *out [[buffer(6)]],
    constant FusedEwRtParams &p [[buffer(7)]],
    uint tpig [[thread_position_in_grid]]) {
    if (tpig * 4 >= p.total) {
        return;
    }
    device const void *ins[FUSED_EW_MAX_INPUTS] = {in0, in1, in2, in3, in4, in5};

    uint coords[4];
    uint idx = tpig * 4;
    for (int a = 3; a >= 0; a--) {
        coords[a] = idx % p.out_shape[a];
        idx /= p.out_shape[a];
    }

    float4 stack[FUSED_EW_MAX_STACK];
    int sp = 0;
    for (uint s = 0; s < few_n_steps; s++) {
        const uint code = few_code(s);
        const uint op = code & FEW_OP_MASK;
        if (op == FEW_PUSH_INPUT) {
            const uint i = code >> FEW_SRC_SHIFT;
            const uint off = coords[0] * p.in_strides[i][0] + coords[1] * p.in_strides[i][1]
                + coords[2] * p.in_strides[i][2] + coords[3] * p.in_strides[i][3];
            stack[sp++] = few_load4(ins[i], (few_in_f16_mask >> i) & 1, off, p.in_strides[i][3]);
        } else if (op == FEW_PUSH_SCALAR) {
            stack[sp++] = float4(few_imm(s));
        } else if (op < FEW_ADD) {
            stack[sp - 1] = few_apply_unary4(op, stack[sp - 1]);
        } else {
            float4 b = stack[--sp];
            stack[sp - 1] = few_apply_binary4(op, stack[sp - 1], b);
        }
        if (code & FEW_FLAG_ROUND_F16) {
            stack[sp - 1] = float4(half4(stack[sp - 1]));
        }
    }
    if (few_out_f16) {
        *(device packed_half4 *)((device half *)out + tpig * 4) = packed_half4(half4(stack[0]));
    } else {
        *(device packed_float4 *)((device float *)out + tpig * 4) = packed_float4(stack[0]);
    }
}
