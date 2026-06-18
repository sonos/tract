// Power-of-two radix-2 Stockham autosort complex FFT, forward, one FFT per threadgroup.
// Interleaved-complex f32 (float2). Out-of-place, ping-pong in threadgroup memory.
// Twiddles W_N^k = exp(-2*pi*i*k/N) computed in-kernel. Port of cu/fft.cu; entry points
// are generated per size (256/512/1024/2048) via DEFINE_FFT. NOTE: 2048 uses 2*2048*8 =
// 32 KB of threadgroup memory, at the Metal limit — adjust if a device rejects it.

#include <metal_stdlib>
using namespace metal;

static inline float2 fft_cmul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// N-point forward FFT (N a power of two), ping-ponging between two threadgroup buffers;
// N/2 threads run one butterfly each per pass. Returns the buffer with the result.
static threadgroup float2 *fft_stockham(
    threadgroup float2 *src, threadgroup float2 *dst, uint tid, int N) {
    const int H = N / 2;
    for (int pass = 0; (1 << pass) < N; ++pass) {
        const int L = 1 << pass;
        const int g = tid >> pass;   // tid / L
        const int j = tid & (L - 1); // tid % L
        const float ang = -6.283185307179586f * (float)(j * (H >> pass)) / (float)N;
        const float2 w = float2(cos(ang), sin(ang)); // W_N^{j*(N/2)/L}
        const float2 a = src[tid];
        const float2 b = src[tid + H];
        const float2 t = fft_cmul(w, b);
        const int oa = (g << (pass + 1)) + j; // g*2L + j
        dst[oa] = float2(a.x + t.x, a.y + t.y);
        dst[oa + L] = float2(a.x - t.x, a.y - t.y);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        threadgroup float2 *tmp = src;
        src = dst;
        dst = tmp;
    }
    return src;
}

#define DEFINE_FFT(N)                                                              \
    kernel void fft##N##_forward(                                                  \
        device const float2 *in [[buffer(0)]],                                     \
        device float2 *out [[buffer(1)]],                                          \
        uint3 tgpig [[threadgroup_position_in_grid]],                              \
        uint tid [[thread_index_in_threadgroup]]) {                                \
        threadgroup float2 a[N];                                                   \
        threadgroup float2 b[N];                                                   \
        const uint base = tgpig.x * N;                                             \
        a[tid] = in[base + tid];                                                   \
        a[tid + N / 2] = in[base + tid + N / 2];                                   \
        threadgroup_barrier(mem_flags::mem_threadgroup);                           \
        threadgroup float2 *res = fft_stockham(a, b, tid, N);                      \
        out[base + tid] = res[tid];                                                \
        out[base + tid + N / 2] = res[tid + N / 2];                                \
    }                                                                              \
    kernel void stft##N##_forward(                                                 \
        device const float2 *in [[buffer(0)]],                                     \
        device const float *win [[buffer(1)]],                                     \
        device float2 *out [[buffer(2)]],                                          \
        constant int &T [[buffer(3)]],                                             \
        constant int &frames [[buffer(4)]],                                        \
        constant int &stride [[buffer(5)]],                                        \
        uint3 tgpig [[threadgroup_position_in_grid]],                              \
        uint tid [[thread_index_in_threadgroup]]) {                                \
        threadgroup float2 a[N];                                                   \
        threadgroup float2 b[N];                                                   \
        const uint f = tgpig.x;                                                    \
        const uint bb = tgpig.y;                                                   \
        const uint in_base = bb * (uint)T;                                         \
        const uint out_base = (bb * (uint)frames + f) * N;                         \
        for (int r = 0; r < 2; ++r) {                                              \
            const uint i = tid + (uint)(r * (N / 2));                              \
            const float2 v = in[in_base + f * (uint)stride + i];                   \
            const float w = win[i];                                                \
            a[i] = float2(v.x * w, v.y * w);                                       \
        }                                                                          \
        threadgroup_barrier(mem_flags::mem_threadgroup);                           \
        threadgroup float2 *res = fft_stockham(a, b, tid, N);                      \
        out[out_base + tid] = res[tid];                                            \
        out[out_base + tid + N / 2] = res[tid + N / 2];                            \
    }

DEFINE_FFT(256)
DEFINE_FFT(512)
DEFINE_FFT(1024)
DEFINE_FFT(2048)
