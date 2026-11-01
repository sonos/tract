#include <metal_integer>
#include <metal_math>
#include <metal_simdgroup_matrix> // Available from Metal version 2.3 released with OS X 11.0+
#include <metal_stdlib>

using namespace metal;

namespace utils {
METAL_FUNC uint indices_to_idx_2(uint2 indices,
                                 constant const size_t strides[2]) {
    return indices.x * strides[1] + indices.y * strides[0];
}
} // namespace utils

#define INSTANTIATE_COPY(tname, type)                                          \
    template [[host_name("array_ops::copy_nd1_" #tname)]] [[kernel]]           \
    copy_nd_t<type> copy_nd<type, 1>;                                          \
    template [[host_name("array_ops::copy_nd2_" #tname)]] [[kernel]]           \
    copy_nd_t<type> copy_nd<type, 2>;                                          \
    template [[host_name("array_ops::copy_nd3_" #tname)]] [[kernel]]           \
    copy_nd_t<type> copy_nd<type, 3>;                                          \
    template [[host_name("array_ops::copy_nd4_" #tname)]] [[kernel]]           \
    copy_nd_t<type> copy_nd<type, 4>;                                          \
    template [[host_name("array_ops::copy_nd5_" #tname)]] [[kernel]]           \
    copy_nd_t<type> copy_nd<type, 5>;                                          \
    template [[host_name("array_ops::copy_nd6_" #tname)]] [[kernel]]           \
    copy_nd_t<type> copy_nd<type, 6>;                                          \
    template [[host_name(                                                      \
        "array_ops::copy_transpose_" #tname)]] [[kernel]] copy_transpose_t     \
        copy_transpose<type>;                                                  \
    template [[host_name(                                                      \
        "array_ops::copy_unicast_" #tname)]] [[kernel]] copy_unicast_t         \
        copy_unicast<type>;

#define INSTANTIATE_CAST_OP(tname, itype, otype)                               \
    template [[host_name(                                                      \
        "array_ops::cast_" #tname)]] [[kernel]] cast_t cast<itype, otype>;

template <typename In, typename Out>
[[kernel]] void cast(device const void *input_b [[buffer(0)]],
                     device void *output_b [[buffer(1)]],
                     uint tpig [[thread_position_in_grid]]) {
    device const In *input = (device const In *)input_b;
    device Out *output = (device Out *)output_b;
    output[tpig] = static_cast<Out>(input[tpig]);
}

typedef decltype(cast<float, float>) cast_t;

template <typename T>
[[kernel]] void copy_unicast(device const void *input_b [[buffer(0)]],
                             device void *output_b [[buffer(1)]],
                             uint tpig [[thread_position_in_grid]]) {
    device const T *input = (device const T *)input_b;
    device T *output = (device T *)output_b;
    output[tpig] = input[tpig];
}

typedef decltype(copy_unicast<float>) copy_unicast_t;

// A group covers whole rows of the innermost axis: `threads / inner` of them,
// or one it strides through when a row is wider than the group. The groups'
// `x` takes the rows, `y + z * groups.y` the axes beyond the innermost two;
// the last of `z`'s groups can run past them. See tract_gpu::utils::copy_launch.
template <typename T>
METAL_FUNC void copy_last2(device const T *input, int in_offset,
                           int in_stride_prev, int in_stride_inner,
                           device T *output, int out_offset,
                           int out_stride_prev, int out_stride_inner, int prev,
                           int inner, int group, int lane, int threads) {
    const int rows = threads / inner;
    if (rows == 0) {
        in_offset += group * in_stride_prev;
        out_offset += group * out_stride_prev;
        for (int i = lane; i < inner; i += threads) {
            output[out_offset + i * out_stride_inner] =
                input[in_offset + i * in_stride_inner];
        }
        return;
    }
    const int p = group * rows + lane / inner;
    if (p >= prev) {
        return;
    }
    const int i = lane % inner;
    output[out_offset + p * out_stride_prev + i * out_stride_inner] =
        input[in_offset + p * in_stride_prev + i * in_stride_inner];
}

template <typename T, int R>
[[kernel]] void copy_nd(device const void *input_b [[buffer(0)]],
                        constant const size_t *in_strides [[buffer(1)]],
                        device void *output_b [[buffer(2)]],
                        constant const size_t *out_shape [[buffer(3)]],
                        constant const size_t *out_strides [[buffer(4)]],
                        uint3 tgpig [[threadgroup_position_in_grid]],
                        uint3 tgpg [[threadgroups_per_grid]],
                        ushort3 tpitg [[thread_position_in_threadgroup]],
                        ushort3 ntg [[threads_per_threadgroup]]) {
    device const T *input = (device const T *)input_b;
    device T *output = (device T *)output_b;
    if (R == 1) {
        const int n = (int)out_shape[0];
        for (int i = tgpig.x * ntg.x + tpitg.x; i < n; i += ntg.x * tgpg.x) {
            output[i * (int)out_strides[0]] = input[i * (int)in_strides[0]];
        }
        return;
    }
    int in_offset = 0;
    int out_offset = 0;
    if (R > 2) {
        int b = tgpig.y + tgpig.z * tgpg.y;
        for (int k = R - 3; k > 0; k--) {
            const int d = (int)out_shape[k];
            const int c = b % d;
            b /= d;
            in_offset += c * (int)in_strides[k];
            out_offset += c * (int)out_strides[k];
        }
        if (b >= (int)out_shape[0]) {
            return;
        }
        in_offset += b * (int)in_strides[0];
        out_offset += b * (int)out_strides[0];
    }
    copy_last2<T>(input, in_offset, (int)in_strides[R - 2],
                  (int)in_strides[R - 1], output, out_offset,
                  (int)out_strides[R - 2], (int)out_strides[R - 1],
                  (int)out_shape[R - 2], (int)out_shape[R - 1], tgpig.x,
                  tpitg.x, ntg.x);
}

template <typename T> using copy_nd_t = decltype(copy_nd<T, 1>);

// A batch of 2D transposes, see tract_gpu::utils::TransposeLaunch. A group
// stages a TILE square through threadgroup memory so that both the read, along
// the input's unit-stride rows, and the write, along the output's unit-stride
// columns, are contiguous. The extra column keeps the transposed read off a
// single bank.
#define TRANSPOSE_TILE 32
template <typename T>
[[kernel]] void copy_transpose(device const void *input_b [[buffer(0)]],
                               device void *output_b [[buffer(1)]],
                               constant const int *geo [[buffer(2)]],
                               uint3 tgpig [[threadgroup_position_in_grid]],
                               ushort3 tpitg [[thread_position_in_threadgroup]],
                               ushort3 ntg [[threads_per_threadgroup]]) {
    threadgroup T tile[TRANSPOSE_TILE][TRANSPOSE_TILE + 1];
    const int in_batch_stride = geo[0];
    const int in_col_stride = geo[1];
    const int out_batch_stride = geo[2];
    const int out_row_stride = geo[3];
    const int rows = geo[4];
    const int cols = geo[5];
    device const T *input =
        (device const T *)input_b + (int)tgpig.z * in_batch_stride;
    device T *output = (device T *)output_b + (int)tgpig.z * out_batch_stride;
    const int r0 = tgpig.y * TRANSPOSE_TILE;
    const int c0 = tgpig.x * TRANSPOSE_TILE;
    for (int j = tpitg.y; j < TRANSPOSE_TILE; j += ntg.y) {
        const int r = r0 + tpitg.x;
        const int c = c0 + j;
        if (r < rows && c < cols) {
            tile[j][tpitg.x] = input[r + c * in_col_stride];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int j = tpitg.y; j < TRANSPOSE_TILE; j += ntg.y) {
        const int r = r0 + j;
        const int c = c0 + tpitg.x;
        if (r < rows && c < cols) {
            output[r * out_row_stride + c] = tile[tpitg.x][j];
        }
    }
}

typedef decltype(copy_transpose<float>) copy_transpose_t;

// Rotate half of the input buffer
//
// Y = Concat(Neg(Slice(X, X.shape[-1]/2.., -1)), Slice(X, ..X.shape[-1]/2, -1))
//
template <typename T>
[[kernel]] void rotate_half_nd2(device const void *input_b [[buffer(0)]],
                                device void *output_b [[buffer(1)]],
                                constant const size_t *shape [[buffer(2)]],
                                constant const size_t *strides [[buffer(3)]],
                                uint2 tpig [[thread_position_in_grid]]) {
    device const T *input = (device const T *)input_b;
    device T *output = (device T *)output_b;

    uint2 rotated_tpig = tpig;
    rotated_tpig.x += shape[1] / 2;

    // output[tpig] = -1 * input[rotated_tpig]
    // output[rotated_tpig] = input[tpig]

    auto rotated_idx = utils::indices_to_idx_2(rotated_tpig, strides);
    auto out_idx = utils::indices_to_idx_2(tpig, strides);

    output[out_idx] = -input[rotated_idx];

    auto idx = utils::indices_to_idx_2(tpig, strides);
    auto rotated_out_idx = utils::indices_to_idx_2(rotated_tpig, strides);

    output[rotated_out_idx] = input[idx];
}

typedef decltype(rotate_half_nd2<float>) rotate_half_nd2_t;

#define INSTANTIATE_ROTATE_HALF_OP(tname, type)                                \
    template [[host_name(                                                      \
        "array_ops::rotate_half_nd2_" #tname)]] [[kernel]] rotate_half_nd2_t   \
        rotate_half_nd2<type>;

// Diagonal gather (Transformer-XL rel-pos skew, folded):
//   out[..., i, k] = in[..., i, offset + k - i], 0 on out-of-bounds.
// Leading axes are flattened by the host into one batch axis.  Each thread
// owns one (b, i, k) output element.
//
// params layout: [offset, t_q, r_in, out_len,
//                 in_stride_b, in_stride_i, in_stride_r,
//                 out_stride_b, out_stride_i, out_stride_k]
template <typename T>
[[kernel]] void diag_gather(device const void *input_b [[buffer(0)]],
                            device void *output_b [[buffer(1)]],
                            constant const int32_t *params [[buffer(2)]],
                            uint3 tpig [[thread_position_in_grid]]) {
    const int32_t k = (int32_t)tpig.x;
    const int32_t i = (int32_t)tpig.y;
    const int32_t b = (int32_t)tpig.z;

    const int32_t offset = params[0];
    const int32_t t_q = params[1];
    const int32_t r_in = params[2];
    const int32_t out_len = params[3];
    const int32_t in_stride_b = params[4];
    const int32_t in_stride_i = params[5];
    const int32_t in_stride_r = params[6];
    const int32_t out_stride_b = params[7];
    const int32_t out_stride_i = params[8];
    const int32_t out_stride_k = params[9];

    if (k >= out_len || i >= t_q)
        return;

    device const T *input = (device const T *)input_b;
    device T *output = (device T *)output_b;

    const int32_t out_idx = b * out_stride_b + i * out_stride_i + k * out_stride_k;
    const int32_t r = offset + k - i;
    if (r >= 0 && r < r_in) {
        const int32_t in_idx = b * in_stride_b + i * in_stride_i + r * in_stride_r;
        output[out_idx] = input[in_idx];
    } else {
        output[out_idx] = (T)0;
    }
}

typedef decltype(diag_gather<float>) diag_gather_t;

#define INSTANTIATE_DIAG_GATHER(tname, type)                                   \
    template [[host_name(                                                      \
        "array_ops::diag_gather_" #tname)]] [[kernel]] diag_gather_t           \
        diag_gather<type>;

// Gather along one axis:
//   out[i_pre, i_n, i_post] = data[i_pre, indices[i_n], i_post]
// where the host flattens to (pre × a_size × post) for data and
// (pre × n_indices × post) for output.  Negative indices wrap with `a_size`,
// matching the CPU contract.
//
// params layout: [pre, a_size, post, n_indices]
template <typename T>
[[kernel]] void gather(device const void *data_b [[buffer(0)]],
                       device const void *indices_b [[buffer(1)]],
                       device void *output_b [[buffer(2)]],
                       constant const int32_t *params [[buffer(3)]],
                       uint3 tpig [[thread_position_in_grid]]) {
    const int32_t i_post = (int32_t)tpig.x;
    const int32_t i_n = (int32_t)tpig.y;
    const int32_t i_pre = (int32_t)tpig.z;

    const int32_t pre = params[0];
    const int32_t a_size = params[1];
    const int32_t post = params[2];
    const int32_t n_indices = params[3];

    if (i_post >= post || i_n >= n_indices || i_pre >= pre)
        return;

    device const T *data = (device const T *)data_b;
    device const long *indices = (device const long *)indices_b;
    device T *output = (device T *)output_b;

    long k = indices[i_n];
    if (k < 0)
        k += a_size;

    const long in_off = ((long)i_pre * a_size + k) * post + i_post;
    const long out_off = ((long)i_pre * n_indices + i_n) * post + i_post;
    output[out_off] = data[in_off];
}

typedef decltype(gather<float>) gather_t;

#define INSTANTIATE_GATHER(tname, type)                                        \
    template [[host_name(                                                      \
        "array_ops::gather_" #tname)]] [[kernel]] gather_t gather<type>;

// GatherNd with f32 coordinate rows: output element i copies the element of
// slice `row = i / inner` at `i % inner`, the slice being picked in batch
// `row / rows` by the n coordinates of that row. Negative coordinates wrap.
//
// params layout: [len, inner, rows, n, batch_stride, dims[4], strides[4]]
template <typename T>
[[kernel]] void gather_nd(device const void *data_b [[buffer(0)]],
                          device const float *indices [[buffer(1)]],
                          device void *output_b [[buffer(2)]],
                          constant const long *params [[buffer(3)]],
                          uint tpig [[thread_position_in_grid]]) {
    const long i = (long)tpig;
    if (i >= params[0])
        return;
    const long inner = params[1];
    const long rows = params[2];
    const long n = params[3];
    device const T *data = (device const T *)data_b;
    device T *output = (device T *)output_b;
    const long row = i / inner;
    long src = (row / rows) * params[4] + i % inner;
    for (long k = 0; k < n; k++) {
        long c = (long)indices[row * n + k];
        if (c < 0)
            c += params[5 + k];
        src += c * params[9 + k];
    }
    output[i] = data[src];
}

typedef decltype(gather_nd<float>) gather_nd_t;

#define INSTANTIATE_GATHER_ND(tname, type)                                     \
    template [[host_name(                                                      \
        "array_ops::gather_nd_" #tname)]] [[kernel]] gather_nd_t               \
        gather_nd<type>;

// Resample one axis against a host-built plan: every output index owns a window
// of `window` input indices (already clamped) and their weights, so nearest,
// linear, cubic and their antialiased variants all run through this kernel.
// The axes around the resampled one are flattened into `outer` and `inner`.
//
// params layout: [outer, len_in, len_out, inner, window]
template <typename T>
[[kernel]] void resize_axis(device const void *input_b [[buffer(0)]],
                            device void *output_b [[buffer(1)]],
                            device const int32_t *indices [[buffer(2)]],
                            device const float *weights [[buffer(3)]],
                            constant const int32_t *params [[buffer(4)]],
                            uint3 tpig [[thread_position_in_grid]]) {
    const int32_t len_in = params[1];
    const int32_t len_out = params[2];
    const int32_t inner = params[3];
    const int32_t window = params[4];

    const int32_t i = (int32_t)tpig.x;
    const int32_t x = (int32_t)tpig.y;
    const int32_t o = (int32_t)tpig.z;

    if (i >= inner || x >= len_out)
        return;

    device const T *input = (device const T *)input_b;
    device T *output = (device T *)output_b;

    const int32_t in_base = o * len_in * inner + i;
    float acc = 0.0f;
    for (int32_t k = 0; k < window; ++k) {
        const float w = weights[x * window + k];
        if (w != 0.0f)
            acc += w * (float)input[in_base + indices[x * window + k] * inner];
    }
    output[o * len_out * inner + x * inner + i] = (T)acc;
}

typedef decltype(resize_axis<float>) resize_axis_t;

#define INSTANTIATE_RESIZE_AXIS(tname, type)                                   \
    template [[host_name(                                                      \
        "array_ops::resize_axis_" #tname)]] [[kernel]] resize_axis_t           \
        resize_axis<type>;

// Copy kernels: only u8/u16/u32/u64 (copy is type-size based)
INSTANTIATE_COPY(u8, uint8_t)
INSTANTIATE_COPY(u16, uint16_t)
INSTANTIATE_COPY(u32, uint32_t)
INSTANTIATE_COPY(u64, uint64_t)

// Cast kernels: all types
#define INSTANTIATE_CAST_FROM(tname, type)                                     \
    INSTANTIATE_CAST_OP(tname##_bool, type, bool)                              \
    INSTANTIATE_CAST_OP(tname##_f32, type, float)                              \
    INSTANTIATE_CAST_OP(tname##_f16, type, half)                               \
    INSTANTIATE_CAST_OP(tname##_u8, type, uint8_t)                             \
    INSTANTIATE_CAST_OP(tname##_u16, type, uint16_t)                           \
    INSTANTIATE_CAST_OP(tname##_u32, type, uint32_t)                           \
    INSTANTIATE_CAST_OP(tname##_u64, type, uint64_t)                           \
    INSTANTIATE_CAST_OP(tname##_i8, type, int8_t)                              \
    INSTANTIATE_CAST_OP(tname##_i16, type, int16_t)                            \
    INSTANTIATE_CAST_OP(tname##_i32, type, int32_t)                            \
    INSTANTIATE_CAST_OP(tname##_i64, type, int64_t)

INSTANTIATE_CAST_FROM(bool, bool)
INSTANTIATE_CAST_FROM(f32, float)
INSTANTIATE_CAST_FROM(f16, half)
INSTANTIATE_CAST_FROM(i8, int8_t)
INSTANTIATE_CAST_FROM(i16, int16_t)
INSTANTIATE_CAST_FROM(i32, int32_t)
INSTANTIATE_CAST_FROM(i64, int64_t)
INSTANTIATE_CAST_FROM(u8, uint8_t)
INSTANTIATE_CAST_FROM(u16, uint16_t)
INSTANTIATE_CAST_FROM(u32, uint32_t)
INSTANTIATE_CAST_FROM(u64, uint64_t)

// Rotate half: only float types
INSTANTIATE_ROTATE_HALF_OP(f32, float)
INSTANTIATE_ROTATE_HALF_OP(f16, half)

// Diagonal gather: f32 and f16 only.
INSTANTIATE_DIAG_GATHER(f32, float)
INSTANTIATE_DIAG_GATHER(f16, half)

// Axis Gather: f32, f16 and i64 (indices are int64).
INSTANTIATE_GATHER(f32, float)
INSTANTIATE_GATHER(f16, half)
INSTANTIATE_GATHER(i64, int64_t)

// GatherNd: f32, f16 and i64 (indices are f32).
INSTANTIATE_GATHER_ND(f32, float)
INSTANTIATE_GATHER_ND(f16, half)
INSTANTIATE_GATHER_ND(i64, int64_t)

// Axis resample: f32 and f16 only (plan indices are int32, weights f32).
INSTANTIATE_RESIZE_AXIS(f32, float)
INSTANTIATE_RESIZE_AXIS(f16, half)

// Cumulative sum along one axis of a contiguous tensor viewed as (outer, len, inner):
// one thread per (outer, inner) lane walks the axis, accumulating in Acc. `exclusive`
// writes the sum of the previous elements, `reverse` walks from the end.
//
// params layout: [outer, len, inner, exclusive, reverse]
template <typename T, typename Acc>
[[kernel]] void cumsum(device const void *input_b [[buffer(0)]],
                       device void *output_b [[buffer(1)]],
                       constant const int32_t *params [[buffer(2)]],
                       uint tpig [[thread_position_in_grid]]) {
    const int32_t outer = params[0];
    const int32_t len = params[1];
    const int32_t inner = params[2];
    const bool exclusive = params[3] != 0;
    const bool reverse = params[4] != 0;
    const int32_t lane = (int32_t)tpig;
    if (lane >= outer * inner)
        return;

    device const T *input = (device const T *)input_b;
    device T *output = (device T *)output_b;
    const int64_t base = (int64_t)(lane / inner) * len * inner + lane % inner;
    Acc acc = (Acc)0;
    for (int32_t s = 0; s < len; ++s) {
        const int64_t ix = base + (int64_t)(reverse ? len - 1 - s : s) * inner;
        const Acc v = (Acc)input[ix];
        if (exclusive) {
            output[ix] = (T)acc;
            acc += v;
        } else {
            acc += v;
            output[ix] = (T)acc;
        }
    }
}

typedef decltype(cumsum<float, float>) cumsum_t;

#define INSTANTIATE_CUMSUM(tname, type, acc)                                   \
    template [[host_name("array_ops::cumsum_" #tname)]] [[kernel]] cumsum_t   \
        cumsum<type, acc>;

INSTANTIATE_CUMSUM(f32, float, float)
INSTANTIATE_CUMSUM(f16, half, float)
INSTANTIATE_CUMSUM(i32, int32_t, int32_t)
INSTANTIATE_CUMSUM(i64, int64_t, int64_t)
