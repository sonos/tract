// 512-point radix-2 Stockham autosort complex FFT, forward, one FFT per threadgroup.
// Interleaved-complex f32 (float2). Out-of-place, ping-pong in threadgroup memory.
// Twiddles W_512^k = exp(-2*pi*i*k/512) are computed in-kernel. Port of cu/fft.cu.

#include <metal_stdlib>
using namespace metal;

#define FFT_N 512
#define FFT_HALF 256
#define FFT_LOGN 9

static inline float2 fft_cmul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// 512-point forward FFT, ping-ponging between two threadgroup buffers. `src` holds
// the input; 256 threads run one butterfly each per pass. Returns the buffer holding
// the natural-order result.
static threadgroup float2 *fft512_stockham(
    threadgroup float2 *src, threadgroup float2 *dst, uint tid) {
    for (int pass = 0; pass < FFT_LOGN; ++pass) {
        const int L = 1 << pass;
        const int g = tid >> pass;   // tid / L
        const int j = tid & (L - 1); // tid % L
        const float ang =
            -6.283185307179586f * (float)(j * (FFT_HALF >> pass)) / (float)FFT_N;
        const float2 w = float2(cos(ang), sin(ang)); // W_512^{j*256/L}
        const float2 a = src[tid];
        const float2 b = src[tid + FFT_HALF];
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

// Plain batched FFT: in/out are [batch, 512] complex, contiguous. grid = (batch).
kernel void fft512_forward(
    device const float2 *in [[buffer(0)]],
    device float2 *out [[buffer(1)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float2 a[FFT_N];
    threadgroup float2 b[FFT_N];
    const uint base = tgpig.x * FFT_N;

    a[tid] = in[base + tid];
    a[tid + FFT_HALF] = in[base + tid + FFT_HALF];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float2 *res = fft512_stockham(a, b, tid);

    out[base + tid] = res[tid];
    out[base + tid + FFT_HALF] = res[tid + FFT_HALF];
}

// Fused STFT: gather a strided 512-sample frame, apply the (pre-padded) window, then
// FFT. in is [batch, T] complex; win is [512] real; out is [batch, frames, 512]
// complex. grid = (frames, batch).
kernel void stft512_forward(
    device const float2 *in [[buffer(0)]],
    device const float *win [[buffer(1)]],
    device float2 *out [[buffer(2)]],
    constant int &T [[buffer(3)]],
    constant int &frames [[buffer(4)]],
    constant int &stride [[buffer(5)]],
    uint3 tgpig [[threadgroup_position_in_grid]],
    uint tid [[thread_index_in_threadgroup]]) {
    threadgroup float2 a[FFT_N];
    threadgroup float2 b[FFT_N];
    const uint f = tgpig.x;
    const uint bb = tgpig.y;
    const uint in_base = bb * (uint)T;
    const uint out_base = (bb * (uint)frames + f) * FFT_N;

    for (int r = 0; r < 2; ++r) {
        const uint i = tid + (uint)(r * FFT_HALF);
        const float2 v = in[in_base + f * (uint)stride + i];
        const float w = win[i];
        a[i] = float2(v.x * w, v.y * w);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    threadgroup float2 *res = fft512_stockham(a, b, tid);

    out[out_base + tid] = res[tid];
    out[out_base + tid + FFT_HALF] = res[tid + FFT_HALF];
}
