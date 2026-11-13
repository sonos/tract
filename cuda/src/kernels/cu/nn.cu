#include "common.cuh"
#include <cuda_runtime.h>
#include <math_constants.h>

#define GELU_COEF_A 0.044715f
#define SQRT_2_OVER_PI 0.79788456080286535587989211986876f

template <typename Acc> struct MaxOp {
    using acc_t = Acc;
    __device__ __forceinline__ static acc_t identity() { return -CUDART_INF_F; }
    __device__ __forceinline__ static acc_t pre(acc_t a) { return a; }
    __device__ __forceinline__ static acc_t combine(acc_t a, acc_t b) {
        return a > b ? a : b;
    }
    __device__ __forceinline__ static acc_t norm(acc_t a, int32_t size) {
        return a;
    }
};

template <typename Acc> struct MinOp {
    using acc_t = Acc;
    __device__ __forceinline__ static acc_t identity() { return CUDART_INF_F; }
    __device__ __forceinline__ static acc_t pre(acc_t a) { return a; }
    __device__ __forceinline__ static acc_t combine(acc_t a, acc_t b) {
        return a < b ? a : b;
    }
    __device__ __forceinline__ static acc_t norm(acc_t a, int32_t size) {
        return a;
    }
};

template <typename Acc> struct AddOp {
    using acc_t = Acc;
    __device__ __forceinline__ static acc_t identity() { return acc_t(0); }
    __device__ __forceinline__ static acc_t pre(acc_t a) { return a; }
    __device__ __forceinline__ static acc_t combine(acc_t a, acc_t b) {
        return a + b;
    }
    __device__ __forceinline__ static acc_t norm(acc_t a, int32_t size) {
        return a;
    }
};

template <typename Acc> struct MulOp {
    using acc_t = Acc;
    __device__ __forceinline__ static acc_t identity() { return acc_t(1); }
    __device__ __forceinline__ static acc_t pre(acc_t a) { return a; }
    __device__ __forceinline__ static acc_t combine(acc_t a, acc_t b) {
        return a * b;
    }
    __device__ __forceinline__ static acc_t norm(acc_t a, int32_t size) {
        return a;
    }
};

// Accumulates in float whatever the data type, so an f16 mean does not
// overflow where the f16 sum would.
template <typename Acc> struct MeanOp {
    using acc_t = float;
    __device__ __forceinline__ static acc_t identity() { return acc_t(0); }
    __device__ __forceinline__ static acc_t pre(acc_t a) { return a; }
    __device__ __forceinline__ static acc_t combine(acc_t a, acc_t b) {
        return a + b;
    }
    __device__ __forceinline__ static acc_t norm(acc_t a, int32_t size) {
        return a / (acc_t)size;
    }
};

template <typename Acc> struct MeanOfSquaresOp {
    using acc_t = Acc;
    __device__ __forceinline__ static acc_t identity() { return acc_t(0); }
    __device__ __forceinline__ static acc_t pre(acc_t a) { return a * a; }
    __device__ __forceinline__ static acc_t combine(acc_t a, acc_t b) {
        return a + b;
    }
    __device__ __forceinline__ static acc_t norm(acc_t a, int32_t size) {
        return a / (acc_t)size;
    }
};

template <typename Acc> struct BoolOrOp {
    using acc_t = Acc;
    __device__ __forceinline__ static acc_t identity() { return acc_t(0); }
    __device__ __forceinline__ static acc_t pre(acc_t a) { return a; }
    __device__ __forceinline__ static acc_t combine(acc_t a, acc_t b) {
        return a || b;
    }
    __device__ __forceinline__ static acc_t norm(acc_t a, int32_t size) {
        return a;
    }
};

template <typename Acc> struct BoolAndOp {
    using acc_t = Acc;
    __device__ __forceinline__ static acc_t identity() { return acc_t(1); }
    __device__ __forceinline__ static acc_t pre(acc_t a) { return a; }
    __device__ __forceinline__ static acc_t combine(acc_t a, acc_t b) {
        return a && b;
    }
    __device__ __forceinline__ static acc_t norm(acc_t a, int32_t size) {
        return a;
    }
};

template <class Op>
__device__ __forceinline__ typename Op::acc_t
warp_reduce(typename Op::acc_t v) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        auto other = __shfl_xor_sync(0xffffffff, v, offset, WARP_SIZE);
        v = Op::combine(v, other);
    }
    return v;
}

template <typename T, int BLOCK_THREADS, class Op>
__device__ void reduce(const T *input, T *output, const int32_t shape_0,
                       const int32_t shape_1, const int32_t shape_2,
                       const int32_t in_stride_0, const int32_t in_stride_1,
                       const int32_t in_stride_2, const int32_t out_stride_0,
                       const int32_t out_stride_1, const int32_t out_stride_2) {

    using Acc = typename Op::acc_t;

    int64_t linear =
        (int64_t)blockIdx.x + (int64_t)blockIdx.y * (int64_t)gridDim.x;
    int64_t total = (int64_t)shape_0 * (int64_t)shape_2;
    if (linear >= total)
        return;

    int32_t y = (int32_t)(linear / shape_2);
    int32_t x = (int32_t)(linear - (int64_t)y * shape_2);

    input += (int64_t)y * in_stride_0 + (int64_t)x * in_stride_2;
    output += (int64_t)y * out_stride_0 + (int64_t)x * out_stride_2;

    Acc accu = Op::identity();
    // each thread computes reduction over BLOCK_SIZE values
    _Pragma("unroll") for (int i = threadIdx.x; i < shape_1;
                           i += BLOCK_THREADS) {
        accu = Op::combine(accu, Op::pre(input[i * in_stride_1]));
    }

    // each warp reduce the values of its 32 threads
    // (every wrap member converge to the value)
    accu = warp_reduce<Op>(accu);

    constexpr int NUM_WARPS = (BLOCK_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    if constexpr (NUM_WARPS == 1) {
        if (threadIdx.x == 0) {
            *output = Op::norm(accu, shape_1);
        }
    } else {
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;

        __shared__ Acc shared[NUM_WARPS];

        if (lane_id == 0) {
            shared[warp_id] = accu;
        }
        __syncthreads();

        if (warp_id == 0) {
            accu = (lane_id < NUM_WARPS) ? shared[lane_id] : Op::identity();
            accu = warp_reduce<Op>(accu);
            if (lane_id == 0) {
                *output = Op::norm(accu, shape_1);
            }
        }
    }
}

#define INSTANTIATE_REDUCE_1(op_name, name, T, Op, bname, block_size)          \
    extern "C" __global__ void CAT5(reduce_, op_name, _, bname, name)(         \
        const T *input, T *output, const int32_t shape_0,                      \
        const int32_t shape_1, const int32_t shape_2,                          \
        const int32_t in_stride_0, const int32_t in_stride_1,                  \
        const int32_t in_stride_2, const int32_t out_stride_0,                 \
        const int32_t out_stride_1, const int32_t out_stride_2) {              \
        reduce<T, block_size, Op<T>>(input, output, shape_0, shape_1, shape_2, \
                                     in_stride_0, in_stride_1, in_stride_2,    \
                                     out_stride_0, out_stride_1,               \
                                     out_stride_2);                            \
    }

#define INSTANTIATE_REDUCE(name, T, bname, block_size)                         \
    INSTANTIATE_REDUCE_1(max, name, T, MaxOp, bname, block_size)               \
    INSTANTIATE_REDUCE_1(min, name, T, MinOp, bname, block_size)               \
    INSTANTIATE_REDUCE_1(sum, name, T, AddOp, bname, block_size)               \
    INSTANTIATE_REDUCE_1(prod, name, T, MulOp, bname, block_size)              \
    INSTANTIATE_REDUCE_1(mean, name, T, MeanOp, bname, block_size)             \
    INSTANTIATE_REDUCE_1(mean_of_squares, name, T, MeanOfSquaresOp, bname,     \
                         block_size)

// Largest finite value of T, as float: the starting candidate of an arg
// reduction is (index 0, +-max) like core's fold from T::max_value().
template <typename T> struct ArgLimit;
template <> struct ArgLimit<float> {
    __device__ __forceinline__ static float max() { return FLT_MAX; }
};
template <> struct ArgLimit<__half> {
    __device__ __forceinline__ static float max() { return 65504.0f; }
};

// Strict total order on (value, index) pairs: the reduction result does not
// depend on how threads split the axis.
template <bool IS_MAX, bool LAST>
__device__ __forceinline__ bool arg_better(float bv, int32_t bi, float av,
                                           int32_t ai) {
    if (IS_MAX ? bv > av : bv < av)
        return true;
    return bv == av && (LAST ? bi > ai : bi < ai);
}

template <bool IS_MAX, bool LAST>
__device__ __forceinline__ void arg_warp_reduce(float &v, int32_t &i) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        float ov = __shfl_xor_sync(0xffffffff, v, offset, WARP_SIZE);
        int32_t oi = __shfl_xor_sync(0xffffffff, i, offset, WARP_SIZE);
        if (arg_better<IS_MAX, LAST>(ov, oi, v, i)) {
            v = ov;
            i = oi;
        }
    }
}

// Index of the extreme value along the middle axis of a rank-3 view, written
// as int64. NaN elements are skipped, as in core's ArgMax/ArgMin.
template <typename T, int BLOCK_THREADS, bool IS_MAX, bool LAST>
__device__ void
arg_reduce(const T *input, int64_t *output, const int32_t shape_0,
           const int32_t shape_1, const int32_t shape_2,
           const int32_t in_stride_0, const int32_t in_stride_1,
           const int32_t in_stride_2, const int32_t out_stride_0,
           const int32_t out_stride_1, const int32_t out_stride_2) {
    int64_t linear =
        (int64_t)blockIdx.x + (int64_t)blockIdx.y * (int64_t)gridDim.x;
    int64_t total = (int64_t)shape_0 * (int64_t)shape_2;
    if (linear >= total)
        return;

    int32_t y = (int32_t)(linear / shape_2);
    int32_t x = (int32_t)(linear - (int64_t)y * shape_2);

    input += (int64_t)y * in_stride_0 + (int64_t)x * in_stride_2;
    output += (int64_t)y * out_stride_0 + (int64_t)x * out_stride_2;

    const float start = IS_MAX ? -ArgLimit<T>::max() : ArgLimit<T>::max();
    float best = start;
    int32_t best_i = 0;
    for (int32_t i = threadIdx.x; i < shape_1; i += BLOCK_THREADS) {
        float v = (float)input[(int64_t)i * in_stride_1];
        if (!isnan(v) && arg_better<IS_MAX, LAST>(v, i, best, best_i)) {
            best = v;
            best_i = i;
        }
    }

    arg_warp_reduce<IS_MAX, LAST>(best, best_i);

    constexpr int NUM_WARPS = (BLOCK_THREADS + WARP_SIZE - 1) / WARP_SIZE;
    if constexpr (NUM_WARPS == 1) {
        if (threadIdx.x == 0) {
            *output = (int64_t)best_i;
        }
    } else {
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;

        __shared__ float shared_v[NUM_WARPS];
        __shared__ int32_t shared_i[NUM_WARPS];

        if (lane_id == 0) {
            shared_v[warp_id] = best;
            shared_i[warp_id] = best_i;
        }
        __syncthreads();

        if (warp_id == 0) {
            best = lane_id < NUM_WARPS ? shared_v[lane_id] : start;
            best_i = lane_id < NUM_WARPS ? shared_i[lane_id] : 0;
            arg_warp_reduce<IS_MAX, LAST>(best, best_i);
            if (lane_id == 0) {
                *output = (int64_t)best_i;
            }
        }
    }
}

#define INSTANTIATE_ARG_REDUCE_1(op_name, name, T, IS_MAX, LAST, bname,       \
                                 block_size)                                   \
    extern "C" __global__ void CAT5(reduce_, op_name, _, bname, name)(         \
        const T *input, int64_t *output, const int32_t shape_0,                \
        const int32_t shape_1, const int32_t shape_2,                          \
        const int32_t in_stride_0, const int32_t in_stride_1,                  \
        const int32_t in_stride_2, const int32_t out_stride_0,                 \
        const int32_t out_stride_1, const int32_t out_stride_2) {              \
        arg_reduce<T, block_size, IS_MAX, LAST>(                               \
            input, output, shape_0, shape_1, shape_2, in_stride_0,             \
            in_stride_1, in_stride_2, out_stride_0, out_stride_1,              \
            out_stride_2);                                                     \
    }

#define INSTANTIATE_ARG_REDUCE(name, T, bname, block_size)                     \
    INSTANTIATE_ARG_REDUCE_1(argmax, name, T, true, false, bname, block_size)  \
    INSTANTIATE_ARG_REDUCE_1(argmax_last, name, T, true, true, bname,          \
                             block_size)                                       \
    INSTANTIATE_ARG_REDUCE_1(argmin, name, T, false, false, bname, block_size) \
    INSTANTIATE_ARG_REDUCE_1(argmin_last, name, T, false, true, bname,         \
                             block_size)

extern "C" __global__ void gelu_approx_f32(const float *input, float *output,
                                           int32_t len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) {
        float x = input[i];
        float output_f32 = 0.5 * x *
                           (1.0 + tanhf(SQRT_2_OVER_PI *
                                        (x + GELU_COEF_A * powf(x, (float)3))));
        output[i] = output_f32;
    }
}

extern "C" __global__ void gelu_approx_f16(const __half *input, __half *output,
                                           int32_t len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) {
        float x = (float)input[i];
        float output_f32 = 0.5 * x *
                           (1.0 + tanhf(SQRT_2_OVER_PI *
                                        (x + GELU_COEF_A * powf(x, (float)3))));
        output[i] = (__half)output_f32;
    }
}

extern "C" __global__ void gelu_approx_fast_f32(const float *input,
                                                float *output, int32_t len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) {
        float x = input[i];
        float output_f32 = 0.5 * x *
                           (1.0 + tanhf(SQRT_2_OVER_PI *
                                        (x + GELU_COEF_A * powf(x, (float)2))));
        output[i] = output_f32;
    }
}

extern "C" __global__ void gelu_approx_fast_f16(const __half *input,
                                                __half *output, int32_t len) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) {
        float x = (float)input[i];
        float output_f32 = 0.5 * x *
                           (1.0 + tanhf(SQRT_2_OVER_PI *
                                        (x + GELU_COEF_A * powf(x, (float)2))));
        output[i] = (__half)output_f32;
    }
}

extern "C" __global__ void leaky_relu_f32(const float *input, float *output,
                                          int32_t len, float alpha) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) {
        float x = input[i];
        output[i] = x * (x < 0 ? alpha : 1.0);
    }
}

extern "C" __global__ void leaky_relu_f16(const __half *input, __half *output,
                                          int32_t len, float alpha) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    __half alpha_f16 = (__half)alpha;
    if (i < len) {
        __half x = input[i];
        output[i] = x * (x < (__half)0.0 ? alpha_f16 : (__half)1.0);
    }
}

static __device__ __forceinline__ int
indices_to_idx_2(int x, int y, int x_strides, int y_strides) {
    return x * x_strides + y * y_strides;
}

static __device__ __forceinline__ int indices_to_idx_3(int x, int y, int z,
                                                       int x_strides,
                                                       int y_strides,
                                                       int z_strides) {
    return x * x_strides + y * y_strides + z * z_strides;
}

static __device__ __forceinline__ int
indices_to_idx_4(int x, int y, int z, int x_shape, int y_shape, int z_shape,
                 int b_shape, int x_strides, int y_strides, int z_strides,
                 int b_strides) {
    int idx = x * x_strides + y * y_strides;
    idx += (z % z_shape) * z_strides;
    z /= z_shape;
    idx += z * b_strides;
    return idx;
}

#define INSTANTIATE_APPLY_ROPE(name, T)                                        \
    extern "C" __global__ void apply_rope_nd2_##name(                          \
        const T *input, const T *cos, const T *sin, T *output,                 \
        int32_t in_shape_0, int32_t in_shape_1, int32_t in_strides_0,          \
        int32_t in_strides_1, int32_t cos_sin_strides_0,                       \
        int32_t cos_sin_strides_1, int32_t out_strides_0,                      \
        int32_t out_strides_1) {                                               \
        int thread_idx_x = blockIdx.x * blockDim.x + threadIdx.x;              \
        int thread_idx_y = blockIdx.y * blockDim.y + threadIdx.y;              \
        if (thread_idx_x >= in_shape_1 / 2 || thread_idx_y >= in_shape_0) {    \
            return;                                                            \
        }                                                                      \
        int rotated_idx_x = thread_idx_x + in_shape_1 / 2;                     \
                                                                               \
        int idx = indices_to_idx_2(thread_idx_x, thread_idx_y, in_strides_1,   \
                                   in_strides_0);                              \
        int rot_idx = indices_to_idx_2(rotated_idx_x, thread_idx_y,            \
                                       in_strides_1, in_strides_0);            \
        int out_idx = indices_to_idx_2(thread_idx_x, thread_idx_y,             \
                                       out_strides_1, out_strides_0);          \
        int out_rot_idx = indices_to_idx_2(rotated_idx_x, thread_idx_y,        \
                                           out_strides_1, out_strides_0);      \
                                                                               \
        int cos_sin_idx = indices_to_idx_2(                                    \
            thread_idx_x, thread_idx_y, cos_sin_strides_1, cos_sin_strides_0); \
        int rot_cos_sin_idx =                                                  \
            indices_to_idx_2(rotated_idx_x, thread_idx_y, cos_sin_strides_1,   \
                             cos_sin_strides_0);                               \
                                                                               \
        output[out_idx] =                                                      \
            input[idx] * cos[cos_sin_idx] - input[rot_idx] * sin[cos_sin_idx]; \
        output[out_rot_idx] = input[rot_idx] * cos[rot_cos_sin_idx] +          \
                              input[idx] * sin[rot_cos_sin_idx];               \
    }                                                                          \
                                                                               \
    extern "C" __global__ void apply_rope_nd3_##name(                          \
        const T *input, const T *cos, const T *sin, T *output,                 \
        int32_t in_shape_0, int32_t in_shape_1, int32_t in_shape_2,            \
        int32_t in_strides_0, int32_t in_strides_1, int32_t in_strides_2,      \
        int32_t cos_sin_strides_0, int32_t cos_sin_strides_1,                  \
        int32_t cos_sin_strides_2, int32_t out_strides_0,                      \
        int32_t out_strides_1, int32_t out_strides_2) {                        \
        int thread_idx_x = blockIdx.x * blockDim.x + threadIdx.x;              \
        int thread_idx_y = blockIdx.y * blockDim.y + threadIdx.y;              \
        int thread_idx_z = blockIdx.z * blockDim.z + threadIdx.z;              \
        if (thread_idx_x >= in_shape_2 / 2 || thread_idx_y >= in_shape_1 ||    \
            thread_idx_z >= in_shape_0) {                                      \
            return;                                                            \
        }                                                                      \
        int rotated_idx_x = thread_idx_x + in_shape_2 / 2;                     \
                                                                               \
        int idx = indices_to_idx_3(thread_idx_x, thread_idx_y, thread_idx_z,   \
                                   in_strides_2, in_strides_1, in_strides_0);  \
        int rot_idx =                                                          \
            indices_to_idx_3(rotated_idx_x, thread_idx_y, thread_idx_z,        \
                             in_strides_2, in_strides_1, in_strides_0);        \
        int out_idx =                                                          \
            indices_to_idx_3(thread_idx_x, thread_idx_y, thread_idx_z,         \
                             out_strides_2, out_strides_1, out_strides_0);     \
        int out_rot_idx =                                                      \
            indices_to_idx_3(rotated_idx_x, thread_idx_y, thread_idx_z,        \
                             out_strides_2, out_strides_1, out_strides_0);     \
                                                                               \
        int cos_sin_idx = indices_to_idx_3(                                    \
            thread_idx_x, thread_idx_y, thread_idx_z, cos_sin_strides_2,       \
            cos_sin_strides_1, cos_sin_strides_0);                             \
        int rot_cos_sin_idx = indices_to_idx_3(                                \
            rotated_idx_x, thread_idx_y, thread_idx_z, cos_sin_strides_2,      \
            cos_sin_strides_1, cos_sin_strides_0);                             \
                                                                               \
        output[out_idx] =                                                      \
            input[idx] * cos[cos_sin_idx] - input[rot_idx] * sin[cos_sin_idx]; \
        output[out_rot_idx] = input[rot_idx] * cos[rot_cos_sin_idx] +          \
                              input[idx] * sin[rot_cos_sin_idx];               \
    }                                                                          \
                                                                               \
    extern "C" __global__ void apply_rope_nd4_##name(                          \
        const T *input, const T *cos, const T *sin, T *output,                 \
        int32_t in_shape_0, int32_t in_shape_1, int32_t in_shape_2,            \
        int32_t in_shape_3, int32_t in_strides_0, int32_t in_strides_1,        \
        int32_t in_strides_2, int32_t in_strides_3, int32_t cos_sin_strides_0, \
        int32_t cos_sin_strides_1, int32_t cos_sin_strides_2,                  \
        int32_t cos_sin_strides_3, int32_t out_strides_0,                      \
        int32_t out_strides_1, int32_t out_strides_2, int32_t out_strides_3) { \
        int thread_idx_x = blockIdx.x * blockDim.x + threadIdx.x;              \
        int thread_idx_y = blockIdx.y * blockDim.y + threadIdx.y;              \
        int thread_idx_z = blockIdx.z * blockDim.z + threadIdx.z;              \
        if (thread_idx_x >= in_shape_3 / 2 || thread_idx_y >= in_shape_2 ||    \
            thread_idx_z >= (in_shape_1 * in_shape_0)) {                       \
            return;                                                            \
        }                                                                      \
        int rotated_idx_x = thread_idx_x + in_shape_3 / 2;                     \
                                                                               \
        int idx = indices_to_idx_4(thread_idx_x, thread_idx_y, thread_idx_z,   \
                                   in_shape_3, in_shape_2, in_shape_1,         \
                                   in_shape_0, in_strides_3, in_strides_2,     \
                                   in_strides_1, in_strides_0);                \
        int rot_idx = indices_to_idx_4(                                        \
            rotated_idx_x, thread_idx_y, thread_idx_z, in_shape_3, in_shape_2, \
            in_shape_1, in_shape_0, in_strides_3, in_strides_2, in_strides_1,  \
            in_strides_0);                                                     \
        int out_idx = indices_to_idx_4(                                        \
            thread_idx_x, thread_idx_y, thread_idx_z, in_shape_3, in_shape_2,  \
            in_shape_1, in_shape_0, out_strides_3, out_strides_2,              \
            out_strides_1, out_strides_0);                                     \
        int out_rot_idx = indices_to_idx_4(                                    \
            rotated_idx_x, thread_idx_y, thread_idx_z, in_shape_3, in_shape_2, \
            in_shape_1, in_shape_0, out_strides_3, out_strides_2,              \
            out_strides_1, out_strides_0);                                     \
                                                                               \
        int cos_sin_idx = indices_to_idx_4(                                    \
            thread_idx_x, thread_idx_y, thread_idx_z, in_shape_3, in_shape_2,  \
            in_shape_1, in_shape_0, cos_sin_strides_3, cos_sin_strides_2,      \
            cos_sin_strides_1, cos_sin_strides_0);                             \
        int rot_cos_sin_idx = indices_to_idx_4(                                \
            rotated_idx_x, thread_idx_y, thread_idx_z, in_shape_3, in_shape_2, \
            in_shape_1, in_shape_0, cos_sin_strides_3, cos_sin_strides_2,      \
            cos_sin_strides_1, cos_sin_strides_0);                             \
                                                                               \
        output[out_idx] =                                                      \
            input[idx] * cos[cos_sin_idx] - input[rot_idx] * sin[cos_sin_idx]; \
        output[out_rot_idx] = input[rot_idx] * cos[rot_cos_sin_idx] +          \
                              input[idx] * sin[rot_cos_sin_idx];               \
    }

// A running exp-sum taken from its max `from` to `to`. A sum that has only
// seen -inf holds nothing: exp(-inf - -inf) would be NaN.
static __device__ __forceinline__ float softmax_rescaled(float sum, float from,
                                                         float to) {
    return from == -CUDART_INF_F ? 0.0f : sum * __expf(from - to);
}

static __device__ __forceinline__ void softmax_merge(float &max_val, float &sum,
                                                     float other_max,
                                                     float other_sum) {
    const float m = fmaxf(max_val, other_max);
    sum = softmax_rescaled(sum, max_val, m) +
          softmax_rescaled(other_sum, other_max, m);
    max_val = m;
}

static __device__ __forceinline__ void warp_reduce_max_sum(float &max_val,
                                                           float &sum) {
#pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
        softmax_merge(max_val, sum,
                      __shfl_xor_sync(0xffffffff, max_val, offset),
                      __shfl_xor_sync(0xffffffff, sum, offset));
    }
}

#define INSTANTIATE_SOFTMAX(name, T, op, bname, block_size, log)               \
    extern "C" __global__ void op##_##bname##name(                             \
        const T *x, T *dst, const int32_t shape_0, const int32_t shape_1,      \
        const int32_t shape_2, const int32_t stride_0, const int32_t stride_1, \
        const int32_t stride_2) {                                              \
        int offset = (blockIdx.x % shape_2) * stride_2 +                       \
                     (blockIdx.x / shape_2) * stride_0;                        \
        x += offset;                                                           \
        dst += offset;                                                         \
                                                                               \
        const int warp_id = threadIdx.x / WARP_SIZE;                           \
        const int lane_id = threadIdx.x % WARP_SIZE;                           \
                                                                               \
        float max_val = -CUDART_INF_F;                                         \
        float sum = 0.0f;                                                      \
        for (int i = threadIdx.x; i < shape_1; i += blockDim.x) {              \
            const float el = x[i * stride_1];                                  \
            if (el > max_val) {                                                \
                sum = softmax_rescaled(sum, max_val, el);                      \
                max_val = el;                                                  \
            }                                                                  \
            if (max_val != -CUDART_INF_F) {                                    \
                sum += __expf(el - max_val);                                   \
            }                                                                  \
        }                                                                      \
                                                                               \
        warp_reduce_max_sum(max_val, sum);                                     \
        if (block_size > WARP_SIZE) {                                          \
            __shared__ float s_max[32];                                        \
            __shared__ float s_sum[32];                                        \
            if (warp_id == 0) {                                                \
                s_max[lane_id] = -CUDART_INF_F;                                \
                s_sum[lane_id] = 0.0f;                                         \
            }                                                                  \
            __syncthreads();                                                   \
            if (lane_id == 0) {                                                \
                s_max[warp_id] = max_val;                                      \
                s_sum[warp_id] = sum;                                          \
            }                                                                  \
            __syncthreads();                                                   \
            max_val = s_max[lane_id];                                          \
            sum = s_sum[lane_id];                                              \
            warp_reduce_max_sum(max_val, sum);                                 \
        }                                                                      \
                                                                               \
        if (log) {                                                             \
            const float shift = max_val + logf(sum);                           \
            for (int i = threadIdx.x; i < shape_1; i += blockDim.x) {          \
                dst[i * stride_1] = (float)x[i * stride_1] - shift;            \
            }                                                                  \
        } else {                                                               \
            const float inv_sum = 1.0f / sum;                                  \
            for (int i = threadIdx.x; i < shape_1; i += blockDim.x) {          \
                dst[i * stride_1] =                                            \
                    __expf((float)x[i * stride_1] - max_val) * inv_sum;        \
            }                                                                  \
        }                                                                      \
    }

/// basic 4D f32-masked softmax, doing softmax on last axis (*_3)
// if input is [ b, h, rows, cols ], softmax is performed alongside the col
// dimension
// supports mask broadcast by mask_stride tuning
// grid dim is (rows, h, b)
// block_size is WARP_SIZE * 2^n
// shared_mem: (32 + next_poswer_of_2(cols) * sizeof(T)
template <typename T, int BLOCK_SIZE>
__device__ void scaled_masked_softmax(
    const T *x, const T *mask, const float scale, T *dst, const int32_t shape_0,
    const int32_t shape_1, const int32_t shape_2, const int32_t shape_3,
    const int32_t shape_4, const int32_t stride_0, const int32_t stride_1,
    const int32_t stride_2, const int32_t stride_3, const int32_t stride_4,
    const int32_t mask_stride_0, const int32_t mask_stride_1,
    const int32_t mask_stride_2, const int32_t mask_stride_3,
    const int32_t mask_stride_4, const int32_t out_stride_0,
    const int32_t out_stride_1, const int32_t out_stride_2,
    const int32_t out_stride_3, const int32_t out_stride_4) {
    int32_t z0 = blockIdx.z / shape_1;
    int32_t z1 = blockIdx.z % shape_1;
    x += blockIdx.x * stride_3 + blockIdx.y * stride_2 + z1 * stride_1 +
         z0 * stride_0;
    mask += mask ? blockIdx.x * mask_stride_3 + blockIdx.y * mask_stride_2 +
                       z1 * mask_stride_1 + z0 * mask_stride_0
                 : 0;
    dst += blockIdx.x * out_stride_3 + blockIdx.y * out_stride_2 +
           z1 * out_stride_1 + z0 * out_stride_0;

    const int block_size = BLOCK_SIZE == 0 ? blockDim.x : BLOCK_SIZE;

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    extern __shared__ float data_soft_max_f32[];
    float *buf_iw = data_soft_max_f32;
    float *vals = buf_iw + WARP_SIZE;

    float max_val = -CUDART_INF_F;
    _Pragma("unroll") for (int col0 = 0; col0 < shape_4; col0 += block_size) {
        const int col = col0 + threadIdx.x;
        if (col >= shape_4) {
            break;
        }

        const float m = mask ? (float)mask[col * mask_stride_4] : 0.0f;
        const float val = ((float)x[col * stride_4]) * scale + m;
        vals[col] = val;
        max_val = max(max_val, val);
    }

    max_val = warp_reduce_max(max_val);
    if (block_size > WARP_SIZE) {
        if (warp_id == 0) {
            buf_iw[lane_id] = -CUDART_INF_F;
        }
        __syncthreads();

        if (lane_id == 0) {
            buf_iw[warp_id] = max_val;
        }
        __syncthreads();

        max_val = buf_iw[lane_id];
        max_val = warp_reduce_max(max_val);
    }

    float tmp = 0.0f;
    _Pragma("unroll") for (int col0 = 0; col0 < shape_4; col0 += block_size) {
        const int col = col0 + threadIdx.x;
        if (col >= shape_4) {
            break;
        }

        const float val = __expf(vals[col] - max_val);
        tmp += val;
        vals[col] = val;
    }

    tmp = warp_reduce_sum(tmp);
    if (block_size > WARP_SIZE) {
        __syncthreads();
        if (warp_id == 0) {
            buf_iw[lane_id] = 0.0f;
        }
        __syncthreads();

        if (lane_id == 0) {
            buf_iw[warp_id] = tmp;
        }
        __syncthreads();

        tmp = buf_iw[lane_id];
        tmp = warp_reduce_sum(tmp);
    }

    const float inv_sum = 1.0f / tmp;

    _Pragma("unroll") for (int col0 = 0; col0 < shape_4; col0 += block_size) {
        const int col = col0 + threadIdx.x;
        if (col >= shape_4) {
            return;
        }
        dst[col * out_stride_4] = vals[col] * inv_sum;
    }
}

#define INSTANTIATE_SCALED_MASKED_SOFTMAX(name, T, bname, block_size_template) \
    extern "C" __global__ void scaled_masked_softmax_##bname##name(            \
        const T *x, const T *mask, const float scale, T *dst,                  \
        const int32_t shape_0, const int32_t shape_1, const int32_t shape_2,   \
        const int32_t shape_3, const int32_t shape_4, const int32_t stride_0,  \
        const int32_t stride_1, const int32_t stride_2,                        \
        const int32_t stride_3, const int32_t stride_4,                        \
        const int32_t mask_stride_0, const int32_t mask_stride_1,              \
        const int32_t mask_stride_2, const int32_t mask_stride_3,              \
        const int32_t mask_stride_4, const int32_t out_stride_0,               \
        const int32_t out_stride_1, const int32_t out_stride_2,                \
        const int32_t out_stride_3, const int32_t out_stride_4) {              \
        scaled_masked_softmax<T, block_size_template>(                         \
            x, mask, scale, dst, shape_0, shape_1, shape_2, shape_3, shape_4,  \
            stride_0, stride_1, stride_2, stride_3, stride_4, mask_stride_0,   \
            mask_stride_1, mask_stride_2, mask_stride_3, mask_stride_4,        \
            out_stride_0, out_stride_1, out_stride_2, out_stride_3,            \
            out_stride_4);                                                     \
    }

// Bool-mask variant: mask is char (0/1), substitutes -inf at masked positions
// before the softmax.  When post_mask is non-zero, fully-masked rows (all
// inputs -inf) are written as 0 instead of the NaN the naive softmax would
// emit.  Partially-masked rows are unaffected: exp(-inf) = 0 already zeros
// masked positions in the output.
template <typename T, int BLOCK_SIZE>
__device__ void scaled_bool_masked_softmax(
    const T *x, const char *mask, const float scale, T *dst,
    const int32_t post_mask, const int32_t shape_0, const int32_t shape_1,
    const int32_t shape_2, const int32_t shape_3, const int32_t shape_4,
    const int32_t stride_0, const int32_t stride_1, const int32_t stride_2,
    const int32_t stride_3, const int32_t stride_4,
    const int32_t mask_stride_0, const int32_t mask_stride_1,
    const int32_t mask_stride_2, const int32_t mask_stride_3,
    const int32_t mask_stride_4, const int32_t out_stride_0,
    const int32_t out_stride_1, const int32_t out_stride_2,
    const int32_t out_stride_3, const int32_t out_stride_4) {
    int32_t z0 = blockIdx.z / shape_1;
    int32_t z1 = blockIdx.z % shape_1;
    x += blockIdx.x * stride_3 + blockIdx.y * stride_2 + z1 * stride_1 +
         z0 * stride_0;
    mask += blockIdx.x * mask_stride_3 + blockIdx.y * mask_stride_2 +
            z1 * mask_stride_1 + z0 * mask_stride_0;
    dst += blockIdx.x * out_stride_3 + blockIdx.y * out_stride_2 +
           z1 * out_stride_1 + z0 * out_stride_0;

    const int block_size = BLOCK_SIZE == 0 ? blockDim.x : BLOCK_SIZE;

    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;

    extern __shared__ float data_soft_max_f32[];
    float *buf_iw = data_soft_max_f32;
    float *vals = buf_iw + WARP_SIZE;

    float max_val = -CUDART_INF_F;
    _Pragma("unroll") for (int col0 = 0; col0 < shape_4; col0 += block_size) {
        const int col = col0 + threadIdx.x;
        if (col >= shape_4) {
            break;
        }

        const bool m = mask[col * mask_stride_4] != 0;
        const float val = m ? ((float)x[col * stride_4]) * scale : -CUDART_INF_F;
        vals[col] = val;
        max_val = max(max_val, val);
    }

    max_val = warp_reduce_max(max_val);
    if (block_size > WARP_SIZE) {
        if (warp_id == 0) {
            buf_iw[lane_id] = -CUDART_INF_F;
        }
        __syncthreads();

        if (lane_id == 0) {
            buf_iw[warp_id] = max_val;
        }
        __syncthreads();

        max_val = buf_iw[lane_id];
        max_val = warp_reduce_max(max_val);
    }

    float tmp = 0.0f;
    _Pragma("unroll") for (int col0 = 0; col0 < shape_4; col0 += block_size) {
        const int col = col0 + threadIdx.x;
        if (col >= shape_4) {
            break;
        }

        const float val = __expf(vals[col] - max_val);
        tmp += val;
        vals[col] = val;
    }

    tmp = warp_reduce_sum(tmp);
    if (block_size > WARP_SIZE) {
        __syncthreads();
        if (warp_id == 0) {
            buf_iw[lane_id] = 0.0f;
        }
        __syncthreads();

        if (lane_id == 0) {
            buf_iw[warp_id] = tmp;
        }
        __syncthreads();

        tmp = buf_iw[lane_id];
        tmp = warp_reduce_sum(tmp);
    }

    // Row-uniform: tmp <= 0 (or NaN) iff every position was masked.  When
    // post_mask is set we write 0 in that case to scrub the NaN; otherwise
    // we fall through to the normal 1/sum path and let it propagate.
    const bool zero_row = post_mask && !(tmp > 0.0f);
    const float inv_sum = 1.0f / tmp;

    _Pragma("unroll") for (int col0 = 0; col0 < shape_4; col0 += block_size) {
        const int col = col0 + threadIdx.x;
        if (col >= shape_4) {
            return;
        }
        dst[col * out_stride_4] = zero_row ? (T)0.0f : (T)(vals[col] * inv_sum);
    }
}

#define INSTANTIATE_SCALED_BOOL_MASKED_SOFTMAX(name, T, bname,                 \
                                               block_size_template)            \
    extern "C" __global__ void scaled_bool_masked_softmax_##bname##name(       \
        const T *x, const char *mask, const float scale, T *dst,               \
        const int32_t post_mask, const int32_t shape_0,                        \
        const int32_t shape_1, const int32_t shape_2, const int32_t shape_3,   \
        const int32_t shape_4, const int32_t stride_0,                         \
        const int32_t stride_1, const int32_t stride_2,                        \
        const int32_t stride_3, const int32_t stride_4,                        \
        const int32_t mask_stride_0, const int32_t mask_stride_1,              \
        const int32_t mask_stride_2, const int32_t mask_stride_3,              \
        const int32_t mask_stride_4, const int32_t out_stride_0,               \
        const int32_t out_stride_1, const int32_t out_stride_2,                \
        const int32_t out_stride_3, const int32_t out_stride_4) {              \
        scaled_bool_masked_softmax<T, block_size_template>(                    \
            x, mask, scale, dst, post_mask, shape_0, shape_1, shape_2,         \
            shape_3, shape_4, stride_0, stride_1, stride_2, stride_3,          \
            stride_4, mask_stride_0, mask_stride_1, mask_stride_2,             \
            mask_stride_3, mask_stride_4, out_stride_0, out_stride_1,          \
            out_stride_2, out_stride_3, out_stride_4);                         \
    }

// `HAS_RESIDUAL`/`HAS_SCALE` are compile-time: the dead branch (e.g. reading
// `residual`/`scale`, writing `sum_out`) is eliminated by `if constexpr`, so
// the plain variant never touches those pointers and callers may pass any
// valid device pointer (e.g. `dst` itself) as a placeholder for the ones it
// doesn't use. `InT`/`OutT` may differ (e.g. an f16 activation normalized
// into an f32 buffer for a higher-precision downstream op): `residual` and
// `sum_out` are always `InT` (the residual stream stays in the input
// dtype, same contract as Metal's kernel), only `dst` is `OutT`.
// Reads the value the norm operates on: the plain input, or `input +
// residual` summed in InT. The two f16 operands add exactly in f32, so the
// single cast back to InT is the correctly-rounded InT sum -- i.e. exactly
// what a standalone Add dispatch (and Metal's kernel) produces.
template <typename InT, bool HAS_RESIDUAL>
__device__ __forceinline__ InT load_normed_input(const InT *x, const InT *residual,
                                                 const int idx) {
    if constexpr (HAS_RESIDUAL) {
        return (InT)((float)x[idx] + (float)residual[idx]);
    } else {
        return x[idx];
    }
}

template <typename InT, typename OutT, int BLOCK_SIZE, bool HAS_RESIDUAL, bool HAS_SCALE,
          bool ROUND_SCALE_F16>
__device__ void rms_norm_body(const InT *x, const InT *residual, const float *scale,
                               OutT *dst, InT *sum_out, const int32_t shape_0,
                               const int32_t shape_1, const int32_t shape_2,
                               const int32_t strides_0, const int32_t strides_1,
                               const int32_t strides_2, const float eps) {
    int base_idx =
        (blockIdx.x % shape_2) * strides_2 + (blockIdx.x / shape_2) * strides_0;

    float tmp = 0.0f;
    for (int i = threadIdx.x; i < shape_1; i += blockDim.x) {
        const int idx = base_idx + i * strides_1;
        // The residual sum is ROUNDED BACK TO InT before it is used, so the
        // fusion stays bit-identical to the standalone Add dispatch it
        // replaces (and matches Metal's `input[idx] + res[idx]`). Summing in
        // f32 and normalizing that unrounded value would drop a rounding
        // step the source graph performs.
        float xi = (float)load_normed_input<InT, HAS_RESIDUAL>(x, residual, idx);
        tmp += xi * xi;
    }

    tmp = warp_reduce_sum(tmp);
    if constexpr (BLOCK_SIZE > WARP_SIZE) {
        __shared__ float s_sum[32];
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            s_sum[warp_id] = tmp;
        }
        __syncthreads();
        tmp = s_sum[lane_id];
        tmp = warp_reduce_sum(tmp);
    }

    const float mean = tmp / shape_1;
    const float rscale = rsqrtf(mean + eps);

    for (int i = threadIdx.x; i < shape_1; i += blockDim.x) {
        const int idx = base_idx + i * strides_1;
        const InT v = load_normed_input<InT, HAS_RESIDUAL>(x, residual, idx);
        if constexpr (HAS_RESIDUAL) {
            sum_out[idx] = v;
        }
        float normed = rscale * (float)v;
        if constexpr (HAS_SCALE) {
            // The graph this fusion replaces materializes the norm output at
            // its own precision before the gamma multiply; keeping the f32
            // accumulator here would make the op more precise than that graph.
            if constexpr (ROUND_SCALE_F16) {
                normed = (float)(__half)normed;
            }
            normed *= scale[i];
        }
        dst[idx] = (OutT)normed;
    }
}

#define INSTANTIATE_RMS_NORM_VARIANT(suffix, has_residual, has_scale, round16,   \
                                     iname, InT, oname, OutT, bname, block_size)  \
    extern "C" __global__ void rms_norm##suffix##_##bname##iname##_##oname(     \
        const InT *x, const InT *residual, const float *scale, OutT *dst,       \
        InT *sum_out, const int32_t shape_0, const int32_t shape_1,             \
        const int32_t shape_2, const int32_t strides_0, const int32_t strides_1, \
        const int32_t strides_2, const float eps) {                             \
        rms_norm_body<InT, OutT, block_size, has_residual, has_scale, round16>( \
            x, residual, scale, dst, sum_out, shape_0, shape_1, shape_2,        \
            strides_0, strides_1, strides_2, eps);                              \
    }

#define INSTANTIATE_RMS_NORM_INOUT(iname, InT, oname, OutT, bname, block_size)  \
    INSTANTIATE_RMS_NORM_VARIANT(, false, false, false, iname, InT, oname,      \
                                  OutT, bname, block_size)                      \
    INSTANTIATE_RMS_NORM_VARIANT(_scaled, false, true, false, iname, InT,       \
                                  oname, OutT, bname, block_size)               \
    INSTANTIATE_RMS_NORM_VARIANT(_scaled_r16, false, true, true, iname, InT,    \
                                  oname, OutT, bname, block_size)               \
    INSTANTIATE_RMS_NORM_VARIANT(_add, true, false, false, iname, InT, oname,   \
                                  OutT, bname, block_size)                      \
    INSTANTIATE_RMS_NORM_VARIANT(_scaled_add, true, true, false, iname, InT,    \
                                  oname, OutT, bname, block_size)               \
    INSTANTIATE_RMS_NORM_VARIANT(_scaled_r16_add, true, true, true, iname, InT, \
                                  oname, OutT, bname, block_size)

// Layer norm, one warp per contiguous row: the row is held in registers (`NV` 16-byte
// vectors per lane), so it is read once, its variance taken from the centered values, and
// written once. Requires rows of at most NV * 32 vectors and 16-byte aligned rows.
template <typename T, int NV, bool AFFINE>
__device__ void layer_norm_rows(const T *x, const float *scale, const float *bias, T *dst,
                                const int32_t rows, const int32_t cols, const float eps) {
    constexpr int VEC = 16 / sizeof(T);
    const int row = blockIdx.x * (blockDim.x / WARP_SIZE) + threadIdx.x / WARP_SIZE;
    const int lane = threadIdx.x % WARP_SIZE;
    if (row >= rows) {
        return;
    }
    const T *xr = x + (size_t)row * cols;
    T *dr = dst + (size_t)row * cols;

    float v[NV][VEC];
    float sum = 0.0f;
    _Pragma("unroll") for (int k = 0; k < NV; k++) {
        const int c = (k * WARP_SIZE + lane) * VEC;
        if (c < cols) {
            alignas(16) T e[VEC];
            *reinterpret_cast<uint4 *>(e) = *reinterpret_cast<const uint4 *>(xr + c);
            _Pragma("unroll") for (int j = 0; j < VEC; j++) {
                v[k][j] = (float)e[j];
                sum += v[k][j];
            }
        }
    }
    const float mean = warp_reduce_sum(sum) / cols;
    float sq = 0.0f;
    _Pragma("unroll") for (int k = 0; k < NV; k++) {
        if ((k * WARP_SIZE + lane) * VEC < cols) {
            _Pragma("unroll") for (int j = 0; j < VEC; j++) {
                const float d = v[k][j] - mean;
                sq += d * d;
            }
        }
    }
    const float inv = rsqrtf(warp_reduce_sum(sq) / cols + eps);
    _Pragma("unroll") for (int k = 0; k < NV; k++) {
        const int c = (k * WARP_SIZE + lane) * VEC;
        if (c < cols) {
            alignas(16) T e[VEC];
            _Pragma("unroll") for (int j = 0; j < VEC; j++) {
                float y = (v[k][j] - mean) * inv;
                if constexpr (AFFINE) {
                    y = y * scale[c + j] + bias[c + j];
                }
                e[j] = (T)y;
            }
            *reinterpret_cast<uint4 *>(dr + c) = *reinterpret_cast<const uint4 *>(e);
        }
    }
}

template <int BLOCK_SIZE>
__device__ __forceinline__ float layer_norm_block_sum(float v, float *shared) {
    v = warp_reduce_sum(v);
    if constexpr (BLOCK_SIZE > WARP_SIZE) {
        const int warp_id = threadIdx.x / WARP_SIZE;
        const int lane_id = threadIdx.x % WARP_SIZE;
        if (lane_id == 0) {
            shared[warp_id] = v;
        }
        __syncthreads();
        v = warp_reduce_sum(shared[lane_id]);
    }
    return v;
}

// Layer norm over the middle axis of a (shape_0, shape_1, shape_2) view, one block per
// normalized row, any strides.
template <typename T, int BLOCK_SIZE, bool AFFINE>
__device__ void layer_norm_strided(const T *x, const float *scale, const float *bias, T *dst,
                                   const int32_t shape_0, const int32_t shape_1,
                                   const int32_t shape_2, const int32_t strides_0,
                                   const int32_t strides_1, const int32_t strides_2,
                                   const float eps) {
    __shared__ float s_sum[32];
    __shared__ float s_sq[32];
    const int base_idx =
        (blockIdx.x % shape_2) * strides_2 + (blockIdx.x / shape_2) * strides_0;
    float sum = 0.0f;
    for (int i = threadIdx.x; i < shape_1; i += blockDim.x) {
        sum += (float)x[base_idx + i * strides_1];
    }
    const float mean = layer_norm_block_sum<BLOCK_SIZE>(sum, s_sum) / shape_1;
    float sq = 0.0f;
    for (int i = threadIdx.x; i < shape_1; i += blockDim.x) {
        const float d = (float)x[base_idx + i * strides_1] - mean;
        sq += d * d;
    }
    const float inv = rsqrtf(layer_norm_block_sum<BLOCK_SIZE>(sq, s_sq) / shape_1 + eps);
    for (int i = threadIdx.x; i < shape_1; i += blockDim.x) {
        const int idx = base_idx + i * strides_1;
        float y = ((float)x[idx] - mean) * inv;
        if constexpr (AFFINE) {
            y = y * scale[i] + bias[i];
        }
        dst[idx] = (T)y;
    }
}

#define INSTANTIATE_LAYER_NORM_ROWS(tname, T, nv, affine, aname)                \
    extern "C" __global__ void layer_norm_rows##nv##aname##_##tname(            \
        const T *x, const float *scale, const float *bias, T *dst,              \
        const int32_t rows, const int32_t cols, const float eps) {              \
        layer_norm_rows<T, nv, affine>(x, scale, bias, dst, rows, cols, eps);   \
    }

#define INSTANTIATE_LAYER_NORM_STRIDED(tname, T, bname, block_size, affine,     \
                                       aname)                                   \
    extern "C" __global__ void layer_norm##aname##_##bname##tname(              \
        const T *x, const float *scale, const float *bias, T *dst,              \
        const int32_t shape_0, const int32_t shape_1, const int32_t shape_2,    \
        const int32_t strides_0, const int32_t strides_1,                       \
        const int32_t strides_2, const float eps) {                             \
        layer_norm_strided<T, block_size, affine>(                              \
            x, scale, bias, dst, shape_0, shape_1, shape_2, strides_0,          \
            strides_1, strides_2, eps);                                         \
    }

#define INSTANTIATE_LAYER_NORM_AFFINE(tname, T, affine, aname)                  \
    INSTANTIATE_LAYER_NORM_ROWS(tname, T, 1, affine, aname)                     \
    INSTANTIATE_LAYER_NORM_ROWS(tname, T, 2, affine, aname)                     \
    INSTANTIATE_LAYER_NORM_ROWS(tname, T, 3, affine, aname)                     \
    INSTANTIATE_LAYER_NORM_ROWS(tname, T, 4, affine, aname)                     \
    INSTANTIATE_LAYER_NORM_ROWS(tname, T, 5, affine, aname)                     \
    INSTANTIATE_LAYER_NORM_ROWS(tname, T, 6, affine, aname)                     \
    INSTANTIATE_LAYER_NORM_ROWS(tname, T, 7, affine, aname)                     \
    INSTANTIATE_LAYER_NORM_ROWS(tname, T, 8, affine, aname)                     \
    INSTANTIATE_LAYER_NORM_STRIDED(tname, T, small_, 32, affine, aname)         \
    INSTANTIATE_LAYER_NORM_STRIDED(tname, T, , 1024, affine, aname)

#define INSTANTIATE_LAYER_NORM(tname, T)                                        \
    INSTANTIATE_LAYER_NORM_AFFINE(tname, T, false, )                            \
    INSTANTIATE_LAYER_NORM_AFFINE(tname, T, true, _affine)

INSTANTIATE_LAYER_NORM(f32, float)
INSTANTIATE_LAYER_NORM(f16, __half)

INSTANTIATE_APPLY_ROPE(f32, float)
INSTANTIATE_APPLY_ROPE(f16, __half)

INSTANTIATE_RMS_NORM_INOUT(f32, float, f32, float, small_, 32)
INSTANTIATE_RMS_NORM_INOUT(f32, float, f32, float, , 1024)
INSTANTIATE_RMS_NORM_INOUT(f16, __half, f16, __half, small_, 32)
INSTANTIATE_RMS_NORM_INOUT(f16, __half, f16, __half, , 1024)
INSTANTIATE_RMS_NORM_INOUT(f16, __half, f32, float, small_, 32)
INSTANTIATE_RMS_NORM_INOUT(f16, __half, f32, float, , 1024)
INSTANTIATE_RMS_NORM_INOUT(f32, float, f16, __half, small_, 32)
INSTANTIATE_RMS_NORM_INOUT(f32, float, f16, __half, , 1024)

INSTANTIATE_SOFTMAX(f32, float, softmax, small_, 32, false)
INSTANTIATE_SOFTMAX(f32, float, softmax, , 1024, false)
INSTANTIATE_SOFTMAX(f16, __half, softmax, small_, 32, false)
INSTANTIATE_SOFTMAX(f16, __half, softmax, , 1024, false)
INSTANTIATE_SOFTMAX(f32, float, log_softmax, small_, 32, true)
INSTANTIATE_SOFTMAX(f32, float, log_softmax, , 1024, true)
INSTANTIATE_SOFTMAX(f16, __half, log_softmax, small_, 32, true)
INSTANTIATE_SOFTMAX(f16, __half, log_softmax, , 1024, true)

// Max and exp-sum of the whole block, in every thread. blockDim.x is a
// multiple of WARP_SIZE.
static __device__ __forceinline__ void block_reduce_max_sum(float &max_val,
                                                            float &sum) {
    __shared__ float s_max[32];
    __shared__ float s_sum[32];
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    warp_reduce_max_sum(max_val, sum);
    if (lane_id == 0) {
        s_max[warp_id] = max_val;
        s_sum[warp_id] = sum;
    }
    __syncthreads();
    const bool live = lane_id < blockDim.x / WARP_SIZE;
    max_val = live ? s_max[lane_id] : -CUDART_INF_F;
    sum = live ? s_sum[lane_id] : 0.0f;
    warp_reduce_max_sum(max_val, sum);
}

// Max and exp-sum of chunk blockIdx.x of row blockIdx.y of a contiguous
// [rows, k] tensor, written to part[row * gridDim.x + chunk].
template <typename T>
static __device__ void softmax_split_partial(const T *x, float2 *part,
                                             const int32_t k,
                                             const int32_t chunk) {
    const T *row = x + (long long)blockIdx.y * k;
    const int end = min(k, (int)(blockIdx.x + 1) * chunk);
    float max_val = -CUDART_INF_F;
    float sum = 0.0f;
    for (int i = blockIdx.x * chunk + threadIdx.x; i < end; i += blockDim.x) {
        const float el = row[i];
        if (el > max_val) {
            sum = softmax_rescaled(sum, max_val, el);
            max_val = el;
        }
        if (max_val != -CUDART_INF_F) {
            sum += __expf(el - max_val);
        }
    }
    block_reduce_max_sum(max_val, sum);
    if (threadIdx.x == 0) {
        part[blockIdx.y * gridDim.x + blockIdx.x] = make_float2(max_val, sum);
    }
}

// Merges the gridDim.x partials of row blockIdx.y, then writes chunk
// blockIdx.x of its softmax or log-softmax.
template <typename T, bool log>
static __device__ void softmax_split_finalize(const T *x, T *dst,
                                              const float2 *part,
                                              const int32_t k,
                                              const int32_t chunk) {
    const long long offset = (long long)blockIdx.y * k;
    const float2 *row_part = part + blockIdx.y * gridDim.x;
    float max_val = -CUDART_INF_F;
    float sum = 0.0f;
    for (int j = 0; j < gridDim.x; j++) {
        const float2 p = row_part[j];
        softmax_merge(max_val, sum, p.x, p.y);
    }
    const int end = min(k, (int)(blockIdx.x + 1) * chunk);
    if (log) {
        const float shift = max_val + logf(sum);
        for (int i = blockIdx.x * chunk + threadIdx.x; i < end;
             i += blockDim.x) {
            dst[offset + i] = (float)x[offset + i] - shift;
        }
    } else {
        const float inv_sum = 1.0f / sum;
        for (int i = blockIdx.x * chunk + threadIdx.x; i < end;
             i += blockDim.x) {
            dst[offset + i] = __expf((float)x[offset + i] - max_val) * inv_sum;
        }
    }
}

#define INSTANTIATE_SOFTMAX_SPLIT(name, T)                                     \
    extern "C" __global__ void softmax_split_partial_##name(                   \
        const T *x, float2 *part, const int32_t k, const int32_t chunk) {      \
        softmax_split_partial<T>(x, part, k, chunk);                           \
    }                                                                          \
    extern "C" __global__ void softmax_split_finalize_##name(                  \
        const T *x, T *dst, const float2 *part, const int32_t k,               \
        const int32_t chunk) {                                                 \
        softmax_split_finalize<T, false>(x, dst, part, k, chunk);              \
    }                                                                          \
    extern "C" __global__ void log_softmax_split_finalize_##name(              \
        const T *x, T *dst, const float2 *part, const int32_t k,               \
        const int32_t chunk) {                                                 \
        softmax_split_finalize<T, true>(x, dst, part, k, chunk);               \
    }

INSTANTIATE_SOFTMAX_SPLIT(f32, float)
INSTANTIATE_SOFTMAX_SPLIT(f16, __half)

#define INSTANTIATE_SCALED_MASKED_SOFTMAX_FOR_T(name, T)                       \
    INSTANTIATE_SCALED_MASKED_SOFTMAX(name, T, 32_, 32)                        \
    INSTANTIATE_SCALED_MASKED_SOFTMAX(name, T, 64_, 64)                        \
    INSTANTIATE_SCALED_MASKED_SOFTMAX(name, T, 128_, 128)                      \
    INSTANTIATE_SCALED_MASKED_SOFTMAX(name, T, 256_, 256)                      \
    INSTANTIATE_SCALED_MASKED_SOFTMAX(name, T, 512_, 512)                      \
    INSTANTIATE_SCALED_MASKED_SOFTMAX(name, T, 1024_, 1024)                    \
    INSTANTIATE_SCALED_MASKED_SOFTMAX(name, T, 2048_, 1024)                    \
    INSTANTIATE_SCALED_MASKED_SOFTMAX(name, T, 4096_, 1024)                    \
    INSTANTIATE_SCALED_MASKED_SOFTMAX(name, T, 8192_, 1024)                    \
    INSTANTIATE_SCALED_MASKED_SOFTMAX(name, T, 16384_, 1024)                   \
    INSTANTIATE_SCALED_MASKED_SOFTMAX(name, T, 32768_, 1024)                   \
    INSTANTIATE_SCALED_MASKED_SOFTMAX(name, T, 0_, 0)

#define INSTANTIATE_SCALED_BOOL_MASKED_SOFTMAX_FOR_T(name, T)                  \
    INSTANTIATE_SCALED_BOOL_MASKED_SOFTMAX(name, T, 32_, 32)                   \
    INSTANTIATE_SCALED_BOOL_MASKED_SOFTMAX(name, T, 64_, 64)                   \
    INSTANTIATE_SCALED_BOOL_MASKED_SOFTMAX(name, T, 128_, 128)                 \
    INSTANTIATE_SCALED_BOOL_MASKED_SOFTMAX(name, T, 256_, 256)                 \
    INSTANTIATE_SCALED_BOOL_MASKED_SOFTMAX(name, T, 512_, 512)                 \
    INSTANTIATE_SCALED_BOOL_MASKED_SOFTMAX(name, T, 1024_, 1024)               \
    INSTANTIATE_SCALED_BOOL_MASKED_SOFTMAX(name, T, 2048_, 1024)               \
    INSTANTIATE_SCALED_BOOL_MASKED_SOFTMAX(name, T, 4096_, 1024)               \
    INSTANTIATE_SCALED_BOOL_MASKED_SOFTMAX(name, T, 8192_, 1024)               \
    INSTANTIATE_SCALED_BOOL_MASKED_SOFTMAX(name, T, 16384_, 1024)              \
    INSTANTIATE_SCALED_BOOL_MASKED_SOFTMAX(name, T, 32768_, 1024)              \
    INSTANTIATE_SCALED_BOOL_MASKED_SOFTMAX(name, T, 0_, 0)

INSTANTIATE_SCALED_MASKED_SOFTMAX_FOR_T(f32, float)
INSTANTIATE_SCALED_MASKED_SOFTMAX_FOR_T(f16, __half)
INSTANTIATE_SCALED_BOOL_MASKED_SOFTMAX_FOR_T(f32, float)
INSTANTIATE_SCALED_BOOL_MASKED_SOFTMAX_FOR_T(f16, __half)

INSTANTIATE_REDUCE(f32, float, small_, 32)
INSTANTIATE_REDUCE(f32, float, , 1024)
INSTANTIATE_REDUCE(f16, __half, small_, 32)
INSTANTIATE_REDUCE(f16, __half, , 1024)
INSTANTIATE_ARG_REDUCE(f32, float, small_, 32)
INSTANTIATE_ARG_REDUCE(f32, float, , 1024)
INSTANTIATE_ARG_REDUCE(f16, __half, small_, 32)
INSTANTIATE_ARG_REDUCE(f16, __half, , 1024)
INSTANTIATE_REDUCE_1(any, bool, char, BoolOrOp, small_, 32)
INSTANTIATE_REDUCE_1(any, bool, char, BoolOrOp, , 1024)
INSTANTIATE_REDUCE_1(all, bool, char, BoolAndOp, small_, 32)
INSTANTIATE_REDUCE_1(all, bool, char, BoolAndOp, , 1024)
INSTANTIATE_REDUCE_1(sum, i64, int64_t, AddOp, small_, 32)
INSTANTIATE_REDUCE_1(sum, i64, int64_t, AddOp, , 1024)
INSTANTIATE_REDUCE_1(prod, i64, int64_t, MulOp, small_, 32)
INSTANTIATE_REDUCE_1(prod, i64, int64_t, MulOp, , 1024)

// Max or sum pooling of a 2D window, one thread per output element. Positions
// are walked in output memory order, channels fastest when `channels_last`.
// A 1D pool comes in with a unit height. `count_include_pad` counts the taps
// inside the padded input, i.e. before `ih + pad_after_h` and
// `iw + pad_after_w`, which ONNX ceil mode can make smaller than the kernel.
template <typename T, bool IS_MAX>
static __device__ __forceinline__ void
pool_2d(const T *input, T *output, int32_t n, int32_t c, int32_t ih, int32_t iw,
        int32_t oh, int32_t ow, int32_t kh, int32_t kw, int32_t stride_h,
        int32_t stride_w, int32_t pad_h, int32_t pad_w, int32_t pad_after_h,
        int32_t pad_after_w, int32_t dil_h, int32_t dil_w, int32_t in_sn,
        int32_t in_sc, int32_t in_sh, int32_t in_sw, int32_t out_sn,
        int32_t out_sc, int32_t out_sh, int32_t out_sw, int32_t channels_last,
        int32_t count_include_pad, int32_t normalize) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (int64_t)n * c * oh * ow)
        return;
    int32_t ci, x, y, b;
    if (channels_last) {
        ci = idx % c;
        idx /= c;
        x = idx % ow;
        idx /= ow;
        y = idx % oh;
        b = idx / oh;
    } else {
        x = idx % ow;
        idx /= ow;
        y = idx % oh;
        idx /= oh;
        ci = idx % c;
        b = idx / c;
    }
    const T *plane = input + (int64_t)b * in_sn + (int64_t)ci * in_sc;
    const int32_t y0 = y * stride_h - pad_h;
    const int32_t x0 = x * stride_w - pad_w;
    float acc = IS_MAX ? -CUDART_INF_F : 0.0f;
    int32_t valid = 0;
    for (int32_t ky = 0; ky < kh; ky++) {
        const int32_t iy = y0 + ky * dil_h;
        if (iy < 0 || iy >= ih)
            continue;
        for (int32_t kx = 0; kx < kw; kx++) {
            const int32_t ix = x0 + kx * dil_w;
            if (ix < 0 || ix >= iw)
                continue;
            const float v =
                (float)plane[(int64_t)iy * in_sh + (int64_t)ix * in_sw];
            if (IS_MAX) {
                acc = v > acc ? v : acc;
            } else {
                acc += v;
            }
            valid++;
        }
    }
    if (!IS_MAX && normalize) {
        int32_t divisor = valid;
        if (count_include_pad) {
            int32_t rows = 0, cols = 0;
            for (int32_t ky = 0; ky < kh; ky++)
                rows += y0 + ky * dil_h < ih + pad_after_h;
            for (int32_t kx = 0; kx < kw; kx++)
                cols += x0 + kx * dil_w < iw + pad_after_w;
            divisor = rows * cols;
        }
        acc /= (float)(divisor > 0 ? divisor : 1);
    }
    output[(int64_t)b * out_sn + (int64_t)ci * out_sc + (int64_t)y * out_sh +
           (int64_t)x * out_sw] = (T)acc;
}

#define INSTANTIATE_POOL(kind, is_max, name, T)                                \
    extern "C" __global__ void kind##_pool_2d_##name(                          \
        const T *input, T *output, int32_t n, int32_t c, int32_t ih,           \
        int32_t iw, int32_t oh, int32_t ow, int32_t kh, int32_t kw,            \
        int32_t stride_h, int32_t stride_w, int32_t pad_h, int32_t pad_w,      \
        int32_t pad_after_h, int32_t pad_after_w, int32_t dil_h,               \
        int32_t dil_w, int32_t in_sn, int32_t in_sc, int32_t in_sh,            \
        int32_t in_sw, int32_t out_sn, int32_t out_sc, int32_t out_sh,         \
        int32_t out_sw, int32_t channels_last, int32_t count_include_pad,      \
        int32_t normalize) {                                                   \
        pool_2d<T, is_max>(input, output, n, c, ih, iw, oh, ow, kh, kw,        \
                           stride_h, stride_w, pad_h, pad_w, pad_after_h,      \
                           pad_after_w, dil_h, dil_w, in_sn, in_sc, in_sh,     \
                           in_sw, out_sn, out_sc, out_sh, out_sw,              \
                           channels_last, count_include_pad, normalize);       \
    }

INSTANTIATE_POOL(max, true, f32, float)
INSTANTIATE_POOL(max, true, f16, __half)
INSTANTIATE_POOL(sum, false, f32, float)
INSTANTIATE_POOL(sum, false, f16, __half)

// Group norm over an (outer, pre, channels, post) view, in three passes.
// Stats writes `entries` (count, mean, m2) Welford triples per group: channel
// first, a block per group chunk of the flattened (p, c, q) range; channel last,
// a thread per channel over a chunk of `pre`, so a warp reads contiguous
// channels. Finalize merges a group's triples with Chan's formula and stores
// (mean, inv_std) over its first triple. Apply is element-wise.
#define GROUP_NORM_BLOCK 256

struct Moments {
    float n, mean, m2;
};

static __device__ __forceinline__ Moments merge_moments(Moments a, Moments b) {
    const float n = a.n + b.n;
    if (n == 0.0f) {
        return a;
    }
    const float d = b.mean - a.mean;
    const float f = b.n / n;
    return {n, a.mean + d * f, a.m2 + b.m2 + d * d * a.n * f};
}

static __device__ __forceinline__ Moments warp_merge_moments(Moments m) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        Moments o = {__shfl_xor_sync(0xffffffff, m.n, offset, 32),
                     __shfl_xor_sync(0xffffffff, m.mean, offset, 32),
                     __shfl_xor_sync(0xffffffff, m.m2, offset, 32)};
        m = merge_moments(m, o);
    }
    return m;
}

// Valid in thread 0 only.
static __device__ __forceinline__ Moments block_merge_moments(Moments m) {
    __shared__ Moments s_moments[GROUP_NORM_BLOCK / WARP_SIZE];
    m = warp_merge_moments(m);
    const int warp_id = threadIdx.x / WARP_SIZE;
    const int lane_id = threadIdx.x % WARP_SIZE;
    if (lane_id == 0) {
        s_moments[warp_id] = m;
    }
    __syncthreads();
    if (warp_id == 0) {
        m = lane_id < blockDim.x / WARP_SIZE ? s_moments[lane_id] : Moments{0.0f, 0.0f, 0.0f};
        m = warp_merge_moments(m);
    }
    return m;
}

static __device__ __forceinline__ Moments shifted_moments(float n, float shift, float s1,
                                                          float s2) {
    return {n, n > 0.0f ? shift + s1 / n : 0.0f, n > 0.0f ? s2 - s1 * s1 / n : 0.0f};
}

static __device__ __forceinline__ void store_moments(float *out, Moments m) {
    out[0] = m.n;
    out[1] = m.mean;
    out[2] = m.m2;
}

template <typename T>
__device__ void group_norm_stats(const T *x, float *partials, int32_t pre, int32_t channels,
                                 int32_t post, int32_t groups, int32_t splits) {
    const int o = blockIdx.x / groups;
    const int g = blockIdx.x % groups;
    const int cg = channels / groups;
    const int len = pre * cg * post;
    const int chunk = (len + splits - 1) / splits;
    const int start = blockIdx.y * chunk;
    const int end = min(len, start + chunk);

    float shift = 0.0f, s1 = 0.0f, s2 = 0.0f, n = 0.0f;
    for (int j = start + threadIdx.x; j < end; j += blockDim.x) {
        const int q = j % post;
        const int t = j / post;
        const float v =
            (float)x[((o * pre + t / cg) * channels + g * cg + t % cg) * post + q];
        if (n == 0.0f) {
            shift = v;
        }
        s1 += v - shift;
        s2 += (v - shift) * (v - shift);
        n += 1.0f;
    }
    const Moments m = block_merge_moments(shifted_moments(n, shift, s1, s2));
    if (threadIdx.x == 0) {
        store_moments(partials + (blockIdx.x * splits + blockIdx.y) * 3, m);
    }
}

template <typename T>
__device__ void group_norm_stats_channel_last(const T *x, float *partials, int32_t pre,
                                              int32_t channels, int32_t splits) {
    const int channel_blocks = (channels + blockDim.x - 1) / blockDim.x;
    const int o = blockIdx.x / channel_blocks;
    const int c = (blockIdx.x % channel_blocks) * blockDim.x + threadIdx.x;
    if (c >= channels) {
        return;
    }
    const int chunk = (pre + splits - 1) / splits;
    const int start = blockIdx.y * chunk;
    const int end = min(pre, start + chunk);

    float shift = 0.0f, s1 = 0.0f, s2 = 0.0f, n = 0.0f;
    for (int p = start; p < end; p++) {
        const float v = (float)x[(o * pre + p) * channels + c];
        if (n == 0.0f) {
            shift = v;
        }
        s1 += v - shift;
        s2 += (v - shift) * (v - shift);
        n += 1.0f;
    }
    store_moments(partials + ((o * channels + c) * splits + blockIdx.y) * 3,
                  shifted_moments(n, shift, s1, s2));
}

extern "C" __global__ void group_norm_finalize(float *partials, int32_t entries, float eps) {
    float *group = partials + blockIdx.x * entries * 3;
    Moments m = {0.0f, 0.0f, 0.0f};
    for (int e = threadIdx.x; e < entries; e += blockDim.x) {
        m = merge_moments(m, Moments{group[3 * e], group[3 * e + 1], group[3 * e + 2]});
    }
    m = block_merge_moments(m);
    if (threadIdx.x == 0) {
        group[0] = m.mean;
        group[1] = rsqrtf(m.m2 / m.n + eps);
    }
}

template <typename T>
__device__ void group_norm_apply(const T *x, const float *partials, T *y, int32_t len,
                                 int32_t pre, int32_t channels, int32_t post, int32_t groups,
                                 int32_t entries) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= len) {
        return;
    }
    const int c = (idx / post) % channels;
    const int o = idx / (pre * channels * post);
    const float *stats = partials + (o * groups + c / (channels / groups)) * entries * 3;
    y[idx] = (T)(((float)x[idx] - stats[0]) * stats[1]);
}

#define INSTANTIATE_GROUP_NORM(name, T)                                                   \
    extern "C" __global__ void group_norm_stats_##name(                                   \
        const T *x, float *partials, int32_t pre, int32_t channels, int32_t post,         \
        int32_t groups, int32_t splits) {                                                 \
        group_norm_stats<T>(x, partials, pre, channels, post, groups, splits);            \
    }                                                                                     \
    extern "C" __global__ void group_norm_stats_channel_last_##name(                      \
        const T *x, float *partials, int32_t pre, int32_t channels, int32_t splits) {     \
        group_norm_stats_channel_last<T>(x, partials, pre, channels, splits);             \
    }                                                                                     \
    extern "C" __global__ void group_norm_apply_##name(                                   \
        const T *x, const float *partials, T *y, int32_t len, int32_t pre,                \
        int32_t channels, int32_t post, int32_t groups, int32_t entries) {                \
        group_norm_apply<T>(x, partials, y, len, pre, channels, post, groups, entries);   \
    }

INSTANTIATE_GROUP_NORM(f32, float)
INSTANTIATE_GROUP_NORM(f16, __half)

// Mamba-2 recurrence: one warp per state row (batch, head, p), the row's N
// values spread over the lanes and kept in registers across the S steps.
#define MAMBA2_MAX_PER_LANE 8

template <typename T, typename D, typename St>
static __device__ void mamba2_recurrent(
    const T *x, const T *b, const T *c, const D *log_decay, const St *initial_state,
    T *output, St *final_state, const int32_t batch, const int32_t steps,
    const int32_t heads, const int32_t head_dim, const int32_t groups,
    const int32_t d_state) {
    const int32_t row = (blockIdx.x * blockDim.x + threadIdx.x) / WARP_SIZE;
    const int32_t lane = threadIdx.x % WARP_SIZE;
    if (row >= batch * heads * head_dim)
        return;
    const int32_t p = row % head_dim;
    const int32_t h = (row / head_dim) % heads;
    const int32_t bi = row / (head_dim * heads);
    const int32_t group = h / (heads / groups);
    const int64_t state_base = (int64_t)row * d_state;

    float s[MAMBA2_MAX_PER_LANE];
#pragma unroll
    for (int k = 0; k < MAMBA2_MAX_PER_LANE; k++) {
        const int32_t n = lane + k * WARP_SIZE;
        s[k] = n < d_state ? (float)initial_state[state_base + n] : 0.0f;
    }
    for (int32_t t = 0; t < steps; t++) {
        const int64_t step = (int64_t)bi * steps + t;
        const int64_t xi = (step * heads + h) * head_dim + p;
        const int64_t bc = (step * groups + group) * d_state;
        const float decay = expf((float)log_decay[step * heads + h]);
        const float xv = (float)x[xi];
        float acc = 0.0f;
#pragma unroll
        for (int k = 0; k < MAMBA2_MAX_PER_LANE; k++) {
            const int32_t n = lane + k * WARP_SIZE;
            if (n < d_state) {
                s[k] = s[k] * decay + xv * (float)b[bc + n];
                acc += s[k] * (float)c[bc + n];
            }
        }
        acc = warp_reduce_sum(acc);
        if (lane == 0)
            output[xi] = (T)acc;
    }
#pragma unroll
    for (int k = 0; k < MAMBA2_MAX_PER_LANE; k++) {
        const int32_t n = lane + k * WARP_SIZE;
        if (n < d_state)
            final_state[state_base + n] = (St)s[k];
    }
}

#define INSTANTIATE_MAMBA2(tn, T, dn, D, sn, St)                               \
    extern "C" __global__ void mamba2_recurrent_##tn##_##dn##_##sn(            \
        const T *x, const T *b, const T *c, const D *log_decay,                \
        const St *initial_state, T *output, St *final_state,                   \
        const int32_t batch, const int32_t steps, const int32_t heads,         \
        const int32_t head_dim, const int32_t groups, const int32_t d_state) { \
        mamba2_recurrent<T, D, St>(x, b, c, log_decay, initial_state, output,  \
                                   final_state, batch, steps, heads, head_dim, \
                                   groups, d_state);                           \
    }

#define INSTANTIATE_MAMBA2_STATES(tn, T, dn, D)                                \
    INSTANTIATE_MAMBA2(tn, T, dn, D, f32, float)                               \
    INSTANTIATE_MAMBA2(tn, T, dn, D, f16, __half)

INSTANTIATE_MAMBA2_STATES(f32, float, f32, float)
INSTANTIATE_MAMBA2_STATES(f32, float, f16, __half)
INSTANTIATE_MAMBA2_STATES(f16, __half, f32, float)
INSTANTIATE_MAMBA2_STATES(f16, __half, f16, __half)
