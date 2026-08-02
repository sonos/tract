#include <metal_stdlib>
using namespace metal;

// 2D pooling over a channels-last tensor. One thread owns one (n, oh, ow, c),
// so consecutive threads walk the contiguous channel axis and every window read
// is coalesced.
//
// Buffer layout:
//   0: input   [N, iH, iW, C]
//   1: output  [N, oH, oW, C]
//   2: params  (see PoolParams)
struct PoolParams {
    int n;
    int ih;
    int iw;
    int c;
    int oh;
    int ow;
    int kh;
    int kw;
    int stride_h;
    int stride_w;
    int pad_h;
    int pad_w;
    int dil_h;
    int dil_w;
    // Divide the sum by the window area including padding, rather than by the
    // number of positions that actually landed inside the input.
    int count_include_pad;
    int normalize;
};

template <typename T>
[[kernel]] void max_pool_2d(
    const device T* input [[buffer(0)]],
    device T* output [[buffer(1)]],
    const constant PoolParams& p [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]) {
    const int c = int(gid.x);
    const int ow = int(gid.y);
    const int rest = int(gid.z);
    const int oh = rest % p.oh;
    const int n = rest / p.oh;
    if (c >= p.c || ow >= p.ow || n >= p.n) {
        return;
    }

    const int h_start = oh * p.stride_h - p.pad_h;
    const int w_start = ow * p.stride_w - p.pad_w;

    // An all-padding window has no value to take, and tract's CPU op leaves
    // -inf there too.
    T best = T(-INFINITY);
    for (int kh = 0; kh < p.kh; ++kh) {
        const int ih = h_start + kh * p.dil_h;
        if (ih < 0 || ih >= p.ih) {
            continue;
        }
        for (int kw = 0; kw < p.kw; ++kw) {
            const int iw = w_start + kw * p.dil_w;
            if (iw < 0 || iw >= p.iw) {
                continue;
            }
            const int64_t idx =
                ((int64_t(n) * p.ih + ih) * p.iw + iw) * p.c + c;
            best = max(best, input[idx]);
        }
    }
    const int64_t out_idx = ((int64_t(n) * p.oh + oh) * p.ow + ow) * p.c + c;
    output[out_idx] = best;
}

template <typename T>
[[kernel]] void sum_pool_2d(
    const device T* input [[buffer(0)]],
    device T* output [[buffer(1)]],
    const constant PoolParams& p [[buffer(2)]],
    uint3 gid [[thread_position_in_grid]]) {
    const int c = int(gid.x);
    const int ow = int(gid.y);
    const int rest = int(gid.z);
    const int oh = rest % p.oh;
    const int n = rest / p.oh;
    if (c >= p.c || ow >= p.ow || n >= p.n) {
        return;
    }

    const int h_start = oh * p.stride_h - p.pad_h;
    const int w_start = ow * p.stride_w - p.pad_w;

    float acc = 0.0f;
    int counted = 0;
    for (int kh = 0; kh < p.kh; ++kh) {
        const int ih = h_start + kh * p.dil_h;
        if (ih < 0 || ih >= p.ih) {
            continue;
        }
        for (int kw = 0; kw < p.kw; ++kw) {
            const int iw = w_start + kw * p.dil_w;
            if (iw < 0 || iw >= p.iw) {
                continue;
            }
            const int64_t idx =
                ((int64_t(n) * p.ih + ih) * p.iw + iw) * p.c + c;
            acc += float(input[idx]);
            counted += 1;
        }
    }
    if (p.normalize) {
        const int divisor = p.count_include_pad ? (p.kh * p.kw) : max(counted, 1);
        acc /= float(divisor);
    }
    const int64_t out_idx = ((int64_t(n) * p.oh + oh) * p.ow + ow) * p.c + c;
    output[out_idx] = T(acc);
}

#define instantiate_pool(name, tname, itype)                                  \
    template [[host_name(#name "_" #tname)]] [[kernel]] void name<itype>(     \
        const device itype* input [[buffer(0)]],                              \
        device itype* output [[buffer(1)]],                                   \
        const constant PoolParams& p [[buffer(2)]],                           \
        uint3 gid [[thread_position_in_grid]]);

instantiate_pool(max_pool_2d, f32, float)
instantiate_pool(max_pool_2d, f16, half)
instantiate_pool(sum_pool_2d, f32, float)
instantiate_pool(sum_pool_2d, f16, half)
