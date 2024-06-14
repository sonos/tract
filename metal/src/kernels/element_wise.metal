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

namespace element_wise {
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
         return metal::atan(x);
      }
    };

    // Inverse hyperbolic tangent of x
    struct Atanh {
      template <typename T>
      T operator()(T x) {
         return metal::atanh(x);
      }
    };

    // Hyperbolic tangent of x
    struct Tanh {
      template <typename T>
      T operator()(T x) {
         return metal::tanh(x);
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
    INSTANTIATE_ALL_TYPES(round_half_to_even, RoundHalfToEven)
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
}