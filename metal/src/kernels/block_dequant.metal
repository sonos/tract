#include <metal_stdlib>
using namespace metal;

// ggml dequant_q8_0: little-endian f16 scale, then 32 i8 codes. out = code * scale.
template <typename T>
[[kernel]] void dequant_q8_0(device const uchar *quant [[buffer(0)]],
                             device T *out [[buffer(1)]],
                             constant uint &n_blocks [[buffer(2)]],
                             uint tid [[thread_position_in_grid]]) {
    if (tid >= n_blocks) {
        return;
    }
    device const uchar *block = quant + tid * 34u;
    ushort bits = ushort(block[0]) | (ushort(block[1]) << 8);
    float d = float(as_type<half>(bits));
    device T *dst = out + tid * 32u;
    for (uint j = 0u; j < 32u; ++j) {
        float q = float(as_type<char>(block[2u + j]));
        dst[j] = static_cast<T>(q * d);
    }
}

template [[host_name("block_dequant::q8_0_f32")]] [[kernel]] void
dequant_q8_0<float>(device const uchar *quant [[buffer(0)]],
                    device float *out [[buffer(1)]],
                    constant uint &n_blocks [[buffer(2)]],
                    uint tid [[thread_position_in_grid]]);

template [[host_name("block_dequant::q8_0_f16")]] [[kernel]] void
dequant_q8_0<half>(device const uchar *quant [[buffer(0)]],
                   device half *out [[buffer(1)]],
                   constant uint &n_blocks [[buffer(2)]],
                   uint tid [[thread_position_in_grid]]);
