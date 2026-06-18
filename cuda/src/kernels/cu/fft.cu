// Power-of-two radix-2 Stockham autosort complex FFT, forward, one FFT per block.
// Interleaved-complex f32 (float2). Out-of-place, ping-pong in shared memory. Twiddles
// W_N^k = exp(-2*pi*i*k/N) are computed in-kernel. Entry points are generated per size
// (256/512/1024/2048) via DEFINE_FFT so the shared buffer and thread count are constant.

static __device__ __forceinline__ float2 fft_cmul(float2 a, float2 b) {
    return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

// In-place (ping-pong) N-point forward FFT over buf[0], N a power of two, N/2 threads,
// one butterfly per thread. Returns the buffer index holding the natural-order result.
template <int N>
static __device__ int fft_stockham(float2 buf[2][N], int tid) {
    const int H = N / 2;
    int cur = 0;
    for (int pass = 0; (1 << pass) < N; ++pass) {
        const int L = 1 << pass;
        const int g = tid >> pass;   // tid / L
        const int j = tid & (L - 1); // tid % L

        float s, c;
        sincosf(-6.283185307179586f * (float)(j * (H >> pass)) / (float)N, &s, &c);
        const float2 w = make_float2(c, s); // W_N^{j*(N/2)/L}

        const float2 a = buf[cur][tid];
        const float2 b = buf[cur][tid + H];
        const float2 t = fft_cmul(w, b);
        const int oa = (g << (pass + 1)) + j; // g*2L + j
        buf[cur ^ 1][oa] = make_float2(a.x + t.x, a.y + t.y);
        buf[cur ^ 1][oa + L] = make_float2(a.x - t.x, a.y - t.y);
        cur ^= 1;
        __syncthreads();
    }
    return cur;
}

template <int N>
static __device__ void fft_forward_impl(const float2 *in, float2 *out) {
    __shared__ float2 buf[2][N];
    const int H = N / 2;
    const int tid = threadIdx.x;
    const int base = blockIdx.x * N;

    buf[0][tid] = in[base + tid];
    buf[0][tid + H] = in[base + tid + H];
    __syncthreads();

    const int cur = fft_stockham<N>(buf, tid);

    out[base + tid] = buf[cur][tid];
    out[base + tid + H] = buf[cur][tid + H];
}

// Fused STFT: gather a strided N-sample frame, apply the (pre-padded) window, then FFT.
// in is [batch, T] complex; win is [N] real; out is [batch, frames, N] complex.
template <int N>
static __device__ void stft_forward_impl(const float2 *in, const float *win, float2 *out,
                                         int T, int frames, int stride) {
    __shared__ float2 buf[2][N];
    const int H = N / 2;
    const int tid = threadIdx.x;
    const int f = blockIdx.x;
    const int b = blockIdx.y;
    const int in_base = b * T;
    const int out_base = (b * frames + f) * N;

    for (int r = 0; r < 2; ++r) {
        const int i = tid + r * H;
        const float2 v = in[in_base + f * stride + i];
        const float w = win[i];
        buf[0][i] = make_float2(v.x * w, v.y * w);
    }
    __syncthreads();

    const int cur = fft_stockham<N>(buf, tid);

    out[out_base + tid] = buf[cur][tid];
    out[out_base + tid + H] = buf[cur][tid + H];
}

#define DEFINE_FFT(N)                                                                       \
    extern "C" __global__ void fft##N##_forward(const float2 *in, float2 *out) {            \
        fft_forward_impl<N>(in, out);                                                       \
    }                                                                                       \
    extern "C" __global__ void stft##N##_forward(const float2 *in, const float *win,        \
                                                 float2 *out, int T, int frames,            \
                                                 int stride) {                              \
        stft_forward_impl<N>(in, win, out, T, frames, stride);                              \
    }

DEFINE_FFT(256)
DEFINE_FFT(512)
DEFINE_FFT(1024)
DEFINE_FFT(2048)
