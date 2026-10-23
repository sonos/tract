#include <metal_stdlib>

using namespace metal;

// F(6x6, 3x3) Winograd convolution, run as three transforms around a batched
// GEMM: U = G g G^T per (c, o), V = B^T d B per (input tile, c), M = U V summed
// over c by the GEMM, then Y = A^T M A per (output tile, o). The transforms and
// the GEMM stay in f32: the constants of F(6, 3) reach 32 and amplify any f16
// rounding of U, V or M an order of magnitude past a direct convolution's.
//
// Layouts: the kernel is OHWI, data is NHWC, U is [64, C, O], V is [64, T, C]
// and M is [64, T, O], T being the output tiles N * tiles_h * tiles_w in that
// order, and 64 the (a, b) positions of the 8x8 transformed tile.

constant float WINO_G[8][3] = {
    {1.0f, 0.0f, 0.0f},
    {-2.0f / 9.0f, -2.0f / 9.0f, -2.0f / 9.0f},
    {-2.0f / 9.0f, 2.0f / 9.0f, -2.0f / 9.0f},
    {1.0f / 90.0f, 1.0f / 45.0f, 2.0f / 45.0f},
    {1.0f / 90.0f, -1.0f / 45.0f, 2.0f / 45.0f},
    {32.0f / 45.0f, 16.0f / 45.0f, 8.0f / 45.0f},
    {32.0f / 45.0f, -16.0f / 45.0f, 8.0f / 45.0f},
    {0.0f, 0.0f, 1.0f},
};

// B, indexed [i][a]: V[a][b] = sum_ij B[i][a] d[i][j] B[j][b].
constant float WINO_B[8][8] = {
    {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
    {0.0f, 1.0f, -1.0f, 0.5f, -0.5f, 2.0f, -2.0f, -1.0f},
    {-5.25f, 1.0f, 1.0f, 0.25f, 0.25f, 4.0f, 4.0f, 0.0f},
    {0.0f, -4.25f, 4.25f, -2.5f, 2.5f, -2.5f, 2.5f, 5.25f},
    {5.25f, -4.25f, -4.25f, -1.25f, -1.25f, -5.0f, -5.0f, 0.0f},
    {0.0f, 1.0f, -1.0f, 2.0f, -2.0f, 0.5f, -0.5f, -5.25f},
    {-1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 0.0f},
    {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f},
};

// A, indexed [a][i]: Y[i][j] = sum_ab A[a][i] M[a][b] A[b][j].
constant float WINO_A[8][6] = {
    {1.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f},
    {1.0f, 1.0f, 1.0f, 1.0f, 1.0f, 1.0f},
    {1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f},
    {1.0f, 2.0f, 4.0f, 8.0f, 16.0f, 32.0f},
    {1.0f, -2.0f, 4.0f, -8.0f, 16.0f, -32.0f},
    {1.0f, 0.5f, 0.25f, 0.125f, 0.0625f, 0.03125f},
    {1.0f, -0.5f, 0.25f, -0.125f, 0.0625f, -0.03125f},
    {0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 1.0f},
};

// Mirrors tract_metal::kernels::conv::winograd::WinogradGeometry.
struct WinogradGeometry {
    int n;
    int h;
    int w;
    int c;
    int o;
    int oh;
    int ow;
    int pad_h;
    int pad_w;
    int tiles_h;
    int tiles_w;
};

// One thread per (o, c): the writes, along o, are the bulk of the traffic.
template <typename T>
[[kernel]] void winograd_weights(device const T *kernel_ohwi [[buffer(0)]],
                                 device float *u [[buffer(1)]],
                                 constant WinogradGeometry &geo [[buffer(2)]],
                                 uint2 gid [[thread_position_in_grid]]) {
    const int o = gid.x;
    const int c = gid.y;
    if (o >= geo.o || c >= geo.c) {
        return;
    }
    float g[3][3];
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            g[i][j] = kernel_ohwi[(o * 9 + i * 3 + j) * geo.c + c];
        }
    }
    float t[8][3];
    for (int a = 0; a < 8; a++) {
        for (int j = 0; j < 3; j++) {
            t[a][j] = WINO_G[a][0] * g[0][j] + WINO_G[a][1] * g[1][j] +
                      WINO_G[a][2] * g[2][j];
        }
    }
    const int plane = geo.c * geo.o;
    for (int a = 0; a < 8; a++) {
        for (int b = 0; b < 8; b++) {
            u[(a * 8 + b) * plane + c * geo.o + o] =
                t[a][0] * WINO_G[b][0] + t[a][1] * WINO_G[b][1] +
                t[a][2] * WINO_G[b][2];
        }
    }
}

// One thread per (c, tile). The padding is read as zeros.
template <typename T>
[[kernel]] void winograd_input(device const T *input [[buffer(0)]],
                               device float *v [[buffer(1)]],
                               constant WinogradGeometry &geo [[buffer(2)]],
                               uint2 gid [[thread_position_in_grid]]) {
    const int c = gid.x;
    const int t = gid.y;
    const int tiles = geo.n * geo.tiles_h * geo.tiles_w;
    if (c >= geo.c || t >= tiles) {
        return;
    }
    const int tw = t % geo.tiles_w;
    const int th = (t / geo.tiles_w) % geo.tiles_h;
    const int n = t / (geo.tiles_w * geo.tiles_h);
    const int y0 = th * 6 - geo.pad_h;
    const int x0 = tw * 6 - geo.pad_w;
    device const T *image = input + n * geo.h * geo.w * geo.c + c;
    float d[8][8];
    for (int i = 0; i < 8; i++) {
        const int y = y0 + i;
        for (int j = 0; j < 8; j++) {
            const int x = x0 + j;
            d[i][j] = (y >= 0 && y < geo.h && x >= 0 && x < geo.w)
                          ? float(image[(y * geo.w + x) * geo.c])
                          : 0.0f;
        }
    }
    float s[8][8];
    for (int a = 0; a < 8; a++) {
        for (int j = 0; j < 8; j++) {
            float acc = 0.0f;
            for (int i = 0; i < 8; i++) {
                acc += WINO_B[i][a] * d[i][j];
            }
            s[a][j] = acc;
        }
    }
    const int plane = tiles * geo.c;
    for (int a = 0; a < 8; a++) {
        for (int b = 0; b < 8; b++) {
            float acc = 0.0f;
            for (int j = 0; j < 8; j++) {
                acc += s[a][j] * WINO_B[j][b];
            }
            v[(a * 8 + b) * plane + t * geo.c + c] = acc;
        }
    }
}

// One thread per (o, tile). Rows and columns past the output are cropped.
template <typename T>
[[kernel]] void winograd_output(device const float *m [[buffer(0)]],
                                device T *output [[buffer(1)]],
                                constant WinogradGeometry &geo [[buffer(2)]],
                                uint2 gid [[thread_position_in_grid]]) {
    const int o = gid.x;
    const int t = gid.y;
    const int tiles = geo.n * geo.tiles_h * geo.tiles_w;
    if (o >= geo.o || t >= tiles) {
        return;
    }
    const int plane = tiles * geo.o;
    float s[6][8];
    for (int i = 0; i < 6; i++) {
        for (int b = 0; b < 8; b++) {
            s[i][b] = 0.0f;
        }
    }
    for (int a = 0; a < 8; a++) {
        float row[8];
        for (int b = 0; b < 8; b++) {
            row[b] = m[(a * 8 + b) * plane + t * geo.o + o];
        }
        for (int i = 0; i < 6; i++) {
            for (int b = 0; b < 8; b++) {
                s[i][b] += WINO_A[a][i] * row[b];
            }
        }
    }
    const int tw = t % geo.tiles_w;
    const int th = (t / geo.tiles_w) % geo.tiles_h;
    const int n = t / (geo.tiles_w * geo.tiles_h);
    device T *image = output + n * geo.oh * geo.ow * geo.o + o;
    for (int i = 0; i < 6; i++) {
        const int y = th * 6 + i;
        if (y >= geo.oh) {
            break;
        }
        for (int j = 0; j < 6; j++) {
            const int x = tw * 6 + j;
            if (x >= geo.ow) {
                break;
            }
            float acc = 0.0f;
            for (int b = 0; b < 8; b++) {
                acc += s[i][b] * WINO_A[b][j];
            }
            image[(y * geo.ow + x) * geo.o] = T(acc);
        }
    }
}

#define INSTANTIATE_WINOGRAD(tname, type)                                      \
    template [[host_name("winograd_weights_" #tname)]] [[kernel]] void         \
    winograd_weights<type>(device const type *, device float *,               \
                           constant WinogradGeometry &, uint2);                \
    template [[host_name("winograd_input_" #tname)]] [[kernel]] void           \
    winograd_input<type>(device const type *, device float *,                 \
                         constant WinogradGeometry &, uint2);                  \
    template [[host_name("winograd_output_" #tname)]] [[kernel]] void          \
    winograd_output<type>(device const float *, device type *,                \
                          constant WinogradGeometry &, uint2);

INSTANTIATE_WINOGRAD(f32, float)
INSTANTIATE_WINOGRAD(f16, half)
