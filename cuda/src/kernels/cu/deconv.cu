// Gather-based 2D transposed convolution: one thread per output element sums the
// kernel taps that land on it, accumulating in float. A 1D deconv runs as 2D with a
// unit height. Weights are OIHW: [group * co_per_group, ci_per_group, kh, kw].
template <typename T>
__device__ void deconv2d(const T *input, const T *weights, T *output, int32_t n,
                         int32_t co, int32_t ih, int32_t iw, int32_t oh,
                         int32_t ow, int32_t kh, int32_t kw, int32_t ci_pg,
                         int32_t co_pg, int32_t pad_h, int32_t pad_w,
                         int32_t stride_h, int32_t stride_w, int32_t dil_h,
                         int32_t dil_w, int32_t in_sn, int32_t in_sc,
                         int32_t in_sh, int32_t in_sw, int32_t w_so,
                         int32_t w_si, int32_t w_sh, int32_t w_sw,
                         int32_t out_sn, int32_t out_sc, int32_t out_sh,
                         int32_t out_sw) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t nout = (int64_t)n * co * oh * ow;
    if (i >= nout) {
        return;
    }
    int32_t x = (int32_t)(i % ow);
    i /= ow;
    int32_t y = (int32_t)(i % oh);
    i /= oh;
    int32_t c = (int32_t)(i % co);
    int32_t b = (int32_t)(i / co);
    int32_t ci0 = (c / co_pg) * ci_pg;

    float acc = 0.0f;
    // Which taps land on an input pixel does not depend on the channel, and finding
    // out costs two integer divisions, so the channels run innermost.
    for (int32_t ky = 0; ky < kh; ky++) {
        int32_t y_num = y + pad_h - ky * dil_h;
        if (y_num % stride_h != 0) {
            continue;
        }
        int32_t iy = y_num / stride_h;
        if (iy < 0 || iy >= ih) {
            continue;
        }
        for (int32_t kx = 0; kx < kw; kx++) {
            int32_t x_num = x + pad_w - kx * dil_w;
            if (x_num % stride_w != 0) {
                continue;
            }
            int32_t ix = x_num / stride_w;
            if (ix < 0 || ix >= iw) {
                continue;
            }
            const T *in = input + (int64_t)b * in_sn + (int64_t)ci0 * in_sc +
                          (int64_t)iy * in_sh + (int64_t)ix * in_sw;
            const T *w = weights + (int64_t)c * w_so + (int64_t)ky * w_sh +
                         (int64_t)kx * w_sw;
            for (int32_t ci = 0; ci < ci_pg; ci++) {
                acc += (float)in[(int64_t)ci * in_sc] *
                       (float)w[(int64_t)ci * w_si];
            }
        }
    }
    output[(int64_t)b * out_sn + (int64_t)c * out_sc + (int64_t)y * out_sh +
           (int64_t)x * out_sw] = (T)acc;
}

#define INSTANTIATE_DECONV(name, T)                                             \
    extern "C" __global__ void deconv2d_##name(                                 \
        const T *input, const T *weights, T *output, int32_t n, int32_t co,    \
        int32_t ih, int32_t iw, int32_t oh, int32_t ow, int32_t kh,            \
        int32_t kw, int32_t ci_pg, int32_t co_pg, int32_t pad_h,               \
        int32_t pad_w, int32_t stride_h, int32_t stride_w, int32_t dil_h,      \
        int32_t dil_w, int32_t in_sn, int32_t in_sc, int32_t in_sh,            \
        int32_t in_sw, int32_t w_so, int32_t w_si, int32_t w_sh, int32_t w_sw, \
        int32_t out_sn, int32_t out_sc, int32_t out_sh, int32_t out_sw) {      \
        deconv2d<T>(input, weights, output, n, co, ih, iw, oh, ow, kh, kw,     \
                    ci_pg, co_pg, pad_h, pad_w, stride_h, stride_w, dil_h,     \
                    dil_w, in_sn, in_sc, in_sh, in_sw, w_so, w_si, w_sh, w_sw, \
                    out_sn, out_sc, out_sh, out_sw);                           \
    }

INSTANTIATE_DECONV(f32, float)
INSTANTIATE_DECONV(f16, __half)
