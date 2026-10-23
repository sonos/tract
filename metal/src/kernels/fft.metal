// Power-of-two radix-2 Stockham autosort complex FFT, forward + inverse, one FFT per
// threadgroup. Interleaved-complex f32 (float2). Out-of-place, ping-pong in threadgroup
// memory. Twiddles computed in-kernel; inverse is UNNORMALIZED (matches rustfft/core Fft).
// Port of cu/fft.cu; entry points generated per size (256/512/1024/2048) via DEFINE_FFT.
// Other lengths go through the direct DFT kernels at the end of this file.
// NOTE: 2048 uses 2*2048*8 = 32 KB threadgroup memory, at the Metal limit.

#include <metal_stdlib>
using namespace metal;

static inline float2 fft_cmul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// N-point FFT (N a power of two), ping-ponging between two threadgroup buffers; N/2
// threads run one butterfly each per pass. `sign` is -1 forward / +1 inverse. Returns
// the buffer holding the natural-order result.
static threadgroup float2 *fft_stockham(
    threadgroup float2 *src, threadgroup float2 *dst, uint tid, int N, float sign) {
    const int H = N / 2;
    for (int pass = 0; (1 << pass) < N; ++pass) {
        const int L = 1 << pass;
        const int g = tid >> pass;   // tid / L
        const int j = tid & (L - 1); // tid % L
        const float ang = sign * 6.283185307179586f * (float)(j * (H >> pass)) / (float)N;
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

#define DEFINE_PLAIN_FFT(NAME, N, SIGN)                                            \
    kernel void NAME(                                                              \
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
        threadgroup float2 *res = fft_stockham(a, b, tid, N, SIGN);                \
        out[base + tid] = res[tid];                                                \
        out[base + tid + N / 2] = res[tid + N / 2];                                \
    }

#define DEFINE_FFT(N)                                                              \
    DEFINE_PLAIN_FFT(fft##N##_forward, N, -1.0f)                                    \
    DEFINE_PLAIN_FFT(fft##N##_inverse, N, 1.0f)                                     \
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
        threadgroup float2 *res = fft_stockham(a, b, tid, N, -1.0f);               \
        out[out_base + tid] = res[tid];                                            \
        out[out_base + tid + N / 2] = res[tid + N / 2];                            \
    }

DEFINE_FFT(256)
DEFINE_FFT(512)
DEFINE_FFT(1024)
DEFINE_FFT(2048)

// Direct DFT for the lengths without a Stockham kernel, one transform per threadgroup. The
// n-point input and its twiddle table W_n^j = exp(sign*2*pi*i*j/n) sit in the threadgroup
// buffer bound at index 0 (2 * n float2); thread t accumulates output bins t, t + nth, ...
// O(n^2) per transform, which stays negligible at featurizer frame sizes.
static void dft_block(threadgroup const float2 *x, threadgroup float2 *tw, device float2 *out,
                      int n, float sign, uint tid, uint nth) {
    for (int j = tid; j < n; j += nth) {
        float c;
        const float s = sincos(sign * 6.283185307179586f * (float)j / (float)n, c);
        tw[j] = float2(c, s);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (int k = tid; k < n; k += nth) {
        float2 acc = float2(0.0f, 0.0f);
        int idx = 0; // (j * k) mod n
        for (int j = 0; j < n; ++j) {
            acc += fft_cmul(x[j], tw[idx]);
            idx += k;
            if (idx >= n) {
                idx -= n;
            }
        }
        out[k] = acc;
    }
}

#define DEFINE_PLAIN_DFT(NAME, SIGN)                                               \
    kernel void NAME(                                                              \
        device const float2 *in [[buffer(0)]],                                     \
        device float2 *out [[buffer(1)]],                                          \
        constant int &n [[buffer(2)]],                                             \
        threadgroup float2 *smem [[threadgroup(0)]],                               \
        uint3 tgpig [[threadgroup_position_in_grid]],                              \
        uint tid [[thread_index_in_threadgroup]],                                  \
        uint nth [[threads_per_threadgroup]]) {                                    \
        const uint base = tgpig.x * (uint)n;                                       \
        for (int j = tid; j < n; j += nth) {                                       \
            smem[j] = in[base + j];                                                \
        }                                                                          \
        dft_block(smem, smem + n, out + base, n, SIGN, tid, nth);                  \
    }

DEFINE_PLAIN_DFT(dft_forward, -1.0f)
DEFINE_PLAIN_DFT(dft_inverse, 1.0f)

// Fused STFT over the direct DFT: same layout as stft<N>_forward, with n at buffer 3.
kernel void stft_dft_forward(
    device const float2 *in [[buffer(0)]],
    device const float *win [[buffer(1)]],
    device float2 *out [[buffer(2)]],
    constant int &n [[buffer(3)]],
    constant int &T [[buffer(4)]],
    constant int &frames [[buffer(5)]],
    constant int &stride [[buffer(6)]],
    threadgroup float2 *smem [[threadgroup(0)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]],
    uint nth [[threads_per_threadgroup]]) {
    const uint f = tgpig.x;
    const uint bb = tgpig.y;
    const uint in_base = bb * (uint)T + f * (uint)stride;
    for (int j = tid; j < n; j += nth) {
        const float2 v = in[in_base + j];
        smem[j] = float2(v.x * win[j], v.y * win[j]);
    }
    dft_block(smem, smem + n, out + (bb * (uint)frames + f) * (uint)n, n, -1.0f, tid, nth);
}
