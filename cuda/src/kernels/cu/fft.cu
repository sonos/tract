// 512-point radix-2 Stockham autosort complex FFT, forward, one FFT per block.
// Interleaved-complex f32 (float2). Out-of-place, ping-pong in shared memory.
// Twiddles W_512^k = exp(-2*pi*i*k/512) are computed in-kernel.

#define FFT_N 512
#define FFT_HALF 256
#define FFT_LOGN 9

static __device__ __forceinline__ float2 fft_cmul(float2 a, float2 b) {
    return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// In-place (ping-pong) 512-point forward FFT over `buf`, already loaded into
// buf[0]. One butterfly per thread (256 threads). Returns the buffer index
// holding the natural-order result.
static __device__ int fft512_stockham(float2 buf[2][FFT_N], int tid) {
    int cur = 0;
#pragma unroll
    for (int pass = 0; pass < FFT_LOGN; ++pass) {
        const int L = 1 << pass;
        const int g = tid >> pass;   // tid / L
        const int j = tid & (L - 1); // tid % L

        float s, c;
        sincosf(-6.283185307179586f * (float)(j * (FFT_HALF >> pass)) / (float)FFT_N, &s, &c);
        const float2 w = make_float2(c, s); // W_512^{j*256/L}

        const float2 a = buf[cur][tid];
        const float2 b = buf[cur][tid + FFT_HALF];
        const float2 t = fft_cmul(w, b);
        const int oa = (g << (pass + 1)) + j; // g*2L + j
        buf[cur ^ 1][oa] = make_float2(a.x + t.x, a.y + t.y);
        buf[cur ^ 1][oa + L] = make_float2(a.x - t.x, a.y - t.y);
        cur ^= 1;
        __syncthreads();
    }
    return cur;
}

// Plain batched FFT: in/out are [batch, 512] complex, contiguous.
extern "C" __global__ void fft512_forward(const float2 *__restrict__ in,
                                          float2 *__restrict__ out) {
    __shared__ float2 buf[2][FFT_N];
    const int tid = threadIdx.x;
    const int base = blockIdx.x * FFT_N;

    buf[0][tid] = in[base + tid];
    buf[0][tid + FFT_HALF] = in[base + tid + FFT_HALF];
    __syncthreads();

    const int cur = fft512_stockham(buf, tid);

    out[base + tid] = buf[cur][tid];
    out[base + tid + FFT_HALF] = buf[cur][tid + FFT_HALF];
}

// Fused STFT: gather a strided 512-sample frame, apply the (pre-padded) window,
// then FFT. in is [batch, T] complex; win is [512] real; out is [batch, frames,
// 512] complex. grid = (frames, batch).
extern "C" __global__ void stft512_forward(const float2 *__restrict__ in,
                                           const float *__restrict__ win,
                                           float2 *__restrict__ out, int T, int frames,
                                           int stride) {
    __shared__ float2 buf[2][FFT_N];
    const int tid = threadIdx.x;
    const int f = blockIdx.x;
    const int b = blockIdx.y;
    const int in_base = b * T;
    const int out_base = (b * frames + f) * FFT_N;

#pragma unroll
    for (int r = 0; r < 2; ++r) {
        const int i = tid + r * FFT_HALF;
        const float2 v = in[in_base + f * stride + i];
        const float w = win[i];
        buf[0][i] = make_float2(v.x * w, v.y * w);
    }
    __syncthreads();

    const int cur = fft512_stockham(buf, tid);

    out[out_base + tid] = buf[cur][tid];
    out[out_base + tid + FFT_HALF] = buf[cur][tid + FFT_HALF];
}
