#include <metal_stdlib>
using namespace metal;

METAL_FUNC float load_f16_le(device const uchar *p) {
    ushort bits = ushort(p[0]) | (ushort(p[1]) << 8);
    return float(as_type<half>(bits));
}

// ggml scale_min_k4. j is the group index inside one super-block.
METAL_FUNC void scale_min_k4(device const uchar *q, uint j, thread uint &sc, thread uint &m) {
    if (j < 4u) {
        sc = q[j] & 63u;
        m = q[j + 4u] & 63u;
    } else {
        sc = (q[j + 4u] & 0xFu) | ((q[j - 4u] >> 6) << 4);
        m = (q[j + 4u] >> 4) | ((q[j] >> 6) << 4);
    }
}

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

// ggml dequant_q6_k. Scale sits at bytes 208..210. Codes are 6-bit, biased by 32.
template <typename T>
[[kernel]] void dequant_q6_k(device const uchar *quant [[buffer(0)]],
                             device T *out [[buffer(1)]],
                             constant uint &n_blocks [[buffer(2)]],
                             uint tid [[thread_position_in_grid]]) {
    if (tid >= n_blocks) {
        return;
    }
    device const uchar *q = quant + tid * 210u;
    ushort bits = ushort(q[208]) | (ushort(q[209]) << 8);
    float d = float(as_type<half>(bits));
    device T *dst = out + tid * 256u;
    for (uint half_block = 0u; half_block < 2u; ++half_block) {
        uint ql = half_block * 64u;
        uint qh = 128u + half_block * 32u;
        uint sc = 192u + half_block * 8u;
        uint y = half_block * 128u;
        for (uint l = 0u; l < 32u; ++l) {
            uint is = l / 16u;
            uchar qhl = q[qh + l];
            int q1 = int((q[ql + l] & 0xFu) | ((qhl & 3u) << 4)) - 32;
            int q2 = int((q[ql + 32u + l] & 0xFu) | (((qhl >> 2) & 3u) << 4)) - 32;
            int q3 = int((q[ql + l] >> 4) | (((qhl >> 4) & 3u) << 4)) - 32;
            int q4 = int((q[ql + 32u + l] >> 4) | (((qhl >> 6) & 3u) << 4)) - 32;
            float s0 = float(as_type<char>(q[sc + is]));
            float s2 = float(as_type<char>(q[sc + is + 2u]));
            float s4 = float(as_type<char>(q[sc + is + 4u]));
            float s6 = float(as_type<char>(q[sc + is + 6u]));
            dst[y + l] = static_cast<T>(d * s0 * float(q1));
            dst[y + 32u + l] = static_cast<T>(d * s2 * float(q2));
            dst[y + 64u + l] = static_cast<T>(d * s4 * float(q3));
            dst[y + 96u + l] = static_cast<T>(d * s6 * float(q4));
        }
    }
}

template [[host_name("block_dequant::q6_k_f32")]] [[kernel]] void
dequant_q6_k<float>(device const uchar *quant [[buffer(0)]],
                    device float *out [[buffer(1)]],
                    constant uint &n_blocks [[buffer(2)]],
                    uint tid [[thread_position_in_grid]]);

template [[host_name("block_dequant::q6_k_f16")]] [[kernel]] void
dequant_q6_k<half>(device const uchar *quant [[buffer(0)]],
                   device half *out [[buffer(1)]],
                   constant uint &n_blocks [[buffer(2)]],
                   uint tid [[thread_position_in_grid]]);

// ggml dequant_q5_k. High bit in qh adds 16 to the nibble. 176 bytes, 256 values.
template <typename T>
[[kernel]] void dequant_q5_k(device const uchar *quant [[buffer(0)]],
                             device T *out [[buffer(1)]],
                             constant uint &n_blocks [[buffer(2)]],
                             uint tid [[thread_position_in_grid]]) {
    if (tid >= n_blocks) {
        return;
    }
    device const uchar *q = quant + tid * 176u;
    float d = load_f16_le(q);
    float mn = load_f16_le(q + 2);
    device const uchar *scales = q + 4;
    device const uchar *qh = q + 16;
    uint ql = 48u;
    uint y = 0u;
    uint u1 = 1u;
    uint u2 = 2u;
    device T *dst = out + tid * 256u;
    for (uint is = 0u; is < 8u; is += 2u) {
        uint sc, m;
        scale_min_k4(scales, is, sc, m);
        float d1 = d * float(sc);
        float m1 = mn * float(m);
        scale_min_k4(scales, is + 1u, sc, m);
        float d2 = d * float(sc);
        float m2 = mn * float(m);
        for (uint l = 0u; l < 32u; ++l) {
            float hi = (qh[l] & u1) != 0u ? 16.0f : 0.0f;
            dst[y] = static_cast<T>(d1 * (float(q[ql + l] & 0xFu) + hi) - m1);
            y += 1u;
        }
        for (uint l = 0u; l < 32u; ++l) {
            float hi = (qh[l] & u2) != 0u ? 16.0f : 0.0f;
            dst[y] = static_cast<T>(d2 * (float(q[ql + l] >> 4) + hi) - m2);
            y += 1u;
        }
        ql += 32u;
        u1 <<= 2;
        u2 <<= 2;
    }
}

template [[host_name("block_dequant::q5_k_f32")]] [[kernel]] void
dequant_q5_k<float>(device const uchar *quant [[buffer(0)]],
                    device float *out [[buffer(1)]],
                    constant uint &n_blocks [[buffer(2)]],
                    uint tid [[thread_position_in_grid]]);

template [[host_name("block_dequant::q5_k_f16")]] [[kernel]] void
dequant_q5_k<half>(device const uchar *quant [[buffer(0)]],
                   device half *out [[buffer(1)]],
                   constant uint &n_blocks [[buffer(2)]],
                   uint tid [[thread_position_in_grid]]);

// ggml dequant_q4_k. Nibbles only, no high bit. 144 bytes, 256 values.
template <typename T>
[[kernel]] void dequant_q4_k(device const uchar *quant [[buffer(0)]],
                             device T *out [[buffer(1)]],
                             constant uint &n_blocks [[buffer(2)]],
                             uint tid [[thread_position_in_grid]]) {
    if (tid >= n_blocks) {
        return;
    }
    device const uchar *q = quant + tid * 144u;
    float d = load_f16_le(q);
    float mn = load_f16_le(q + 2);
    device const uchar *scales = q + 4;
    uint qs = 16u;
    uint y = 0u;
    device T *dst = out + tid * 256u;
    for (uint is = 0u; is < 8u; is += 2u) {
        uint sc, m;
        scale_min_k4(scales, is, sc, m);
        float d1 = d * float(sc);
        float m1 = mn * float(m);
        scale_min_k4(scales, is + 1u, sc, m);
        float d2 = d * float(sc);
        float m2 = mn * float(m);
        for (uint l = 0u; l < 32u; ++l) {
            dst[y] = static_cast<T>(d1 * float(q[qs + l] & 0xFu) - m1);
            y += 1u;
        }
        for (uint l = 0u; l < 32u; ++l) {
            dst[y] = static_cast<T>(d2 * float(q[qs + l] >> 4) - m2);
            y += 1u;
        }
        qs += 32u;
    }
}

template [[host_name("block_dequant::q4_k_f32")]] [[kernel]] void
dequant_q4_k<float>(device const uchar *quant [[buffer(0)]],
                    device float *out [[buffer(1)]],
                    constant uint &n_blocks [[buffer(2)]],
                    uint tid [[thread_position_in_grid]]);

template [[host_name("block_dequant::q4_k_f16")]] [[kernel]] void
dequant_q4_k<half>(device const uchar *quant [[buffer(0)]],
                   device half *out [[buffer(1)]],
                   constant uint &n_blocks [[buffer(2)]],
                   uint tid [[thread_position_in_grid]]);
