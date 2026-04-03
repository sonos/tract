#include <metal_stdlib>
using namespace metal;

// Direct convolution kernel — one thread per output spatial position.
// Grid: (ceil(spatial_out / threads_per_group), output_channels, batch_size)
//
// Buffer layout:
//   0: input         [T]
//   1: in_shape      [N, C, spatial...]  (2 + georank ints)
//   2: in_strides    [N, C, spatial...]  (2 + georank ints)
//   3: weights       [T]
//   4: ker_params    [groups, co_per_group, ci_per_group, ker_spatial...]  (3 + georank ints)
//   5: ker_strides   [g_stride, o_stride, i_stride, spatial...]  (3 + georank ints)
//   6: bias          [T] (may be empty)
//   7: bias_stride   scalar int32 (-1 = no bias)
//   8: pad           [spatial...]  (georank ints)
//   9: strides       [spatial...]  (georank ints)
//  10: dilations     [spatial...]  (georank ints)
//  11: output        [T]
//  12: out_shape     [N, C, spatial...]  (2 + georank ints)
//  13: out_strides   [N, C, spatial...]  (2 + georank ints)

template <typename T, int GEORANK>
void conv_generic_impl(
    device const T *input,
    constant int32_t *in_shape,
    constant int32_t *in_strides,
    device const T *weights,
    constant int32_t *ker_params,
    constant int32_t *ker_strides,
    device const T *bias,
    int32_t bias_stride,
    constant int32_t *p,
    constant int32_t *str,
    constant int32_t *dil,
    device T *output,
    constant int32_t *out_shape,
    constant int32_t *out_strides,
    uint3 gid)
{
    int n  = gid.z;
    int co = gid.y;
    int xyz = gid.x;

    int co_per_group = ker_params[1];
    int ci_per_group = ker_params[2];
    int group        = co / co_per_group;

    // Decompose linear index into per-axis output coords (last axis fastest)
    int ox[GEORANK];
    {
        int rem = xyz;
        for (int d = GEORANK - 1; d >= 0; d--) {
            int dim = out_shape[2 + d];
            ox[d] = rem % dim;
            rem /= dim;
        }
    }

    // Bounds check
    for (int d = 0; d < GEORANK; d++) {
        if (ox[d] >= out_shape[2 + d]) return;
    }
    if (n >= out_shape[0] || co >= out_shape[1]) return;

    device const T *pfi = input + n * in_strides[0]
                          + ci_per_group * group * in_strides[1];
    device const T *pfk = weights + co * ker_strides[1];

    float sum = (bias_stride >= 0) ? float(bias[co * bias_stride]) : 0.0f;

    for (int ci = 0; ci < ci_per_group; ci++) {
        // Recursive-style nested loop over spatial kernel dims.
        // Unrolled at compile time thanks to constexpr GEORANK.
        if (GEORANK == 1) {
            for (int k0 = 0; k0 < ker_params[3]; k0++) {
                int x0 = ox[0] * str[0] + k0 * dil[0] - p[0];
                if (x0 < 0 || x0 >= in_shape[2]) continue;
                sum += float(pfi[ci * in_strides[1] + x0 * in_strides[2]])
                     * float(pfk[ci * ker_strides[2] + k0 * ker_strides[3]]);
            }
        } else if (GEORANK == 2) {
            for (int k0 = 0; k0 < ker_params[3]; k0++) {
                int x0 = ox[0] * str[0] + k0 * dil[0] - p[0];
                if (x0 < 0 || x0 >= in_shape[2]) continue;
                for (int k1 = 0; k1 < ker_params[4]; k1++) {
                    int x1 = ox[1] * str[1] + k1 * dil[1] - p[1];
                    if (x1 < 0 || x1 >= in_shape[3]) continue;
                    sum += float(pfi[ci * in_strides[1] + x0 * in_strides[2] + x1 * in_strides[3]])
                         * float(pfk[ci * ker_strides[2] + k0 * ker_strides[3] + k1 * ker_strides[4]]);
                }
            }
        } else if (GEORANK == 3) {
            for (int k0 = 0; k0 < ker_params[3]; k0++) {
                int x0 = ox[0] * str[0] + k0 * dil[0] - p[0];
                if (x0 < 0 || x0 >= in_shape[2]) continue;
                for (int k1 = 0; k1 < ker_params[4]; k1++) {
                    int x1 = ox[1] * str[1] + k1 * dil[1] - p[1];
                    if (x1 < 0 || x1 >= in_shape[3]) continue;
                    for (int k2 = 0; k2 < ker_params[5]; k2++) {
                        int x2 = ox[2] * str[2] + k2 * dil[2] - p[2];
                        if (x2 < 0 || x2 >= in_shape[4]) continue;
                        sum += float(pfi[ci * in_strides[1] + x0 * in_strides[2]
                                        + x1 * in_strides[3] + x2 * in_strides[4]])
                             * float(pfk[ci * ker_strides[2] + k0 * ker_strides[3]
                                        + k1 * ker_strides[4] + k2 * ker_strides[5]]);
                    }
                }
            }
        } else if (GEORANK == 4) {
            for (int k0 = 0; k0 < ker_params[3]; k0++) {
                int x0 = ox[0] * str[0] + k0 * dil[0] - p[0];
                if (x0 < 0 || x0 >= in_shape[2]) continue;
                for (int k1 = 0; k1 < ker_params[4]; k1++) {
                    int x1 = ox[1] * str[1] + k1 * dil[1] - p[1];
                    if (x1 < 0 || x1 >= in_shape[3]) continue;
                    for (int k2 = 0; k2 < ker_params[5]; k2++) {
                        int x2 = ox[2] * str[2] + k2 * dil[2] - p[2];
                        if (x2 < 0 || x2 >= in_shape[4]) continue;
                        for (int k3 = 0; k3 < ker_params[6]; k3++) {
                            int x3 = ox[3] * str[3] + k3 * dil[3] - p[3];
                            if (x3 < 0 || x3 >= in_shape[5]) continue;
                            sum += float(pfi[ci * in_strides[1] + x0 * in_strides[2]
                                            + x1 * in_strides[3] + x2 * in_strides[4]
                                            + x3 * in_strides[5]])
                                 * float(pfk[ci * ker_strides[2] + k0 * ker_strides[3]
                                            + k1 * ker_strides[4] + k2 * ker_strides[5]
                                            + k3 * ker_strides[6]]);
                        }
                    }
                }
            }
        }
    }

    int out_offset = n * out_strides[0] + co * out_strides[1];
    for (int d = 0; d < GEORANK; d++) {
        out_offset += ox[d] * out_strides[2 + d];
    }
    output[out_offset] = T(sum);
}

// --- Kernel entry points: 8 variants (f32/f16 × georank 1-4) ---

#define CONV_ENTRY(GEORANK, SUFFIX, T)                                              \
kernel void conv##GEORANK##d_##SUFFIX##_generic(                                    \
    device const T *input          [[buffer(0)]],                                   \
    constant int32_t *in_shape     [[buffer(1)]],                                   \
    constant int32_t *in_strides   [[buffer(2)]],                                   \
    device const T *weights        [[buffer(3)]],                                   \
    constant int32_t *ker_params   [[buffer(4)]],                                   \
    constant int32_t *ker_strides  [[buffer(5)]],                                   \
    device const T *bias           [[buffer(6)]],                                   \
    constant int32_t &bias_stride  [[buffer(7)]],                                   \
    constant int32_t *p            [[buffer(8)]],                                   \
    constant int32_t *str          [[buffer(9)]],                                   \
    constant int32_t *dil          [[buffer(10)]],                                  \
    device T *output               [[buffer(11)]],                                  \
    constant int32_t *out_shape    [[buffer(12)]],                                  \
    constant int32_t *out_strides  [[buffer(13)]],                                  \
    uint3 gid                      [[thread_position_in_grid]])                     \
{                                                                                   \
    conv_generic_impl<T, GEORANK>(input, in_shape, in_strides, weights, ker_params, \
        ker_strides, bias, bias_stride, p, str, dil, output, out_shape,             \
        out_strides, gid);                                                          \
}

CONV_ENTRY(1, f32, float)
CONV_ENTRY(2, f32, float)
CONV_ENTRY(3, f32, float)
CONV_ENTRY(4, f32, float)

CONV_ENTRY(1, f16, half)
CONV_ENTRY(2, f16, half)
CONV_ENTRY(3, f16, half)
CONV_ENTRY(4, f16, half)
