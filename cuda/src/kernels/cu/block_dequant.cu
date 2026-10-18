#include "common.cuh"

// Little-endian f16 at an even offset. ggml stores every scale this way.
__device__ __forceinline__ float load_f16_le(const uint8_t *p) {
    __half_raw raw;
    raw.x = (unsigned short)p[0] | ((unsigned short)p[1] << 8);
    return __half2float(__half(raw));
}

// ggml dequant_q8_0. 34 bytes, 32 values. out = i8 code * f16 scale.
template <typename T>
__device__ void dequant_q8_0_block(const uint8_t *__restrict__ quant, T *__restrict__ out,
                                   uint32_t n_blocks) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_blocks) {
        return;
    }
    const uint8_t *block = quant + tid * 34u;
    float d = load_f16_le(block);
    T *dst = out + tid * 32u;
    for (int j = 0; j < 32; ++j) {
        float q = (float)(int8_t)block[2 + j];
        dst[j] = (T)(q * d);
    }
}

extern "C" __global__ void dequant_q8_0_f32(const uint8_t *__restrict__ quant,
                                            float *__restrict__ out, uint32_t n_blocks) {
    dequant_q8_0_block<float>(quant, out, n_blocks);
}

extern "C" __global__ void dequant_q8_0_f16(const uint8_t *__restrict__ quant,
                                            __half *__restrict__ out, uint32_t n_blocks) {
    dequant_q8_0_block<__half>(quant, out, n_blocks);
}

// ggml dequant_q6_k. Scale sits at bytes 208..210. Codes are 6-bit, biased by 32.
// 210 bytes, 256 values.
template <typename T>
__device__ void dequant_q6_k_block(const uint8_t *__restrict__ quant, T *__restrict__ out,
                                   uint32_t n_blocks) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_blocks) {
        return;
    }
    const uint8_t *q = quant + tid * 210u;
    float d = load_f16_le(q + 208);
    T *dst = out + tid * 256u;
    for (int half = 0; half < 2; ++half) {
        int ql = half * 64;
        int qh = 128 + half * 32;
        int sc = 192 + half * 8;
        int y = half * 128;
        for (int l = 0; l < 32; ++l) {
            int is = l / 16;
            uint8_t qhl = q[qh + l];
            int q1 = (int)((q[ql + l] & 0xF) | ((qhl & 3) << 4)) - 32;
            int q2 = (int)((q[ql + 32 + l] & 0xF) | (((qhl >> 2) & 3) << 4)) - 32;
            int q3 = (int)((q[ql + l] >> 4) | (((qhl >> 4) & 3) << 4)) - 32;
            int q4 = (int)((q[ql + 32 + l] >> 4) | (((qhl >> 6) & 3) << 4)) - 32;
            float s0 = (float)(int8_t)q[sc + is];
            float s2 = (float)(int8_t)q[sc + is + 2];
            float s4 = (float)(int8_t)q[sc + is + 4];
            float s6 = (float)(int8_t)q[sc + is + 6];
            dst[y + l] = (T)(d * s0 * (float)q1);
            dst[y + 32 + l] = (T)(d * s2 * (float)q2);
            dst[y + 64 + l] = (T)(d * s4 * (float)q3);
            dst[y + 96 + l] = (T)(d * s6 * (float)q4);
        }
    }
}

extern "C" __global__ void dequant_q6_k_f32(const uint8_t *__restrict__ quant,
                                            float *__restrict__ out, uint32_t n_blocks) {
    dequant_q6_k_block<float>(quant, out, n_blocks);
}

extern "C" __global__ void dequant_q6_k_f16(const uint8_t *__restrict__ quant,
                                            __half *__restrict__ out, uint32_t n_blocks) {
    dequant_q6_k_block<__half>(quant, out, n_blocks);
}

// ggml scale_min_k4. j is the group index inside one super-block. Scales are 6-bit.
__device__ __forceinline__ void scale_min_k4(const uint8_t *q, uint32_t j, uint32_t &sc,
                                             uint32_t &m) {
    if (j < 4u) {
        sc = q[j] & 63u;
        m = q[j + 4u] & 63u;
    } else {
        sc = (q[j + 4u] & 0xFu) | ((q[j - 4u] >> 6) << 4);
        m = (q[j + 4u] >> 4) | ((q[j] >> 6) << 4);
    }
}

// ggml dequant_q5_k. High bit in qh adds 16 to the nibble. 176 bytes, 256 values.
template <typename T>
__device__ void dequant_q5_k_block(const uint8_t *__restrict__ quant, T *__restrict__ out,
                                   uint32_t n_blocks) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_blocks) {
        return;
    }
    const uint8_t *q = quant + tid * 176u;
    float d = load_f16_le(q);
    float mn = load_f16_le(q + 2);
    const uint8_t *scales = q + 4;
    const uint8_t *qh = q + 16;
    uint32_t ql = 48u;
    uint32_t y = 0u;
    uint32_t u1 = 1u;
    uint32_t u2 = 2u;
    T *dst = out + tid * 256u;
    for (uint32_t is = 0u; is < 8u; is += 2u) {
        uint32_t sc, m;
        scale_min_k4(scales, is, sc, m);
        float d1 = d * (float)sc;
        float m1 = mn * (float)m;
        scale_min_k4(scales, is + 1u, sc, m);
        float d2 = d * (float)sc;
        float m2 = mn * (float)m;
        for (uint32_t l = 0u; l < 32u; ++l) {
            float hi = (qh[l] & u1) != 0u ? 16.0f : 0.0f;
            dst[y] = (T)(d1 * ((float)(q[ql + l] & 0xFu) + hi) - m1);
            y += 1u;
        }
        for (uint32_t l = 0u; l < 32u; ++l) {
            float hi = (qh[l] & u2) != 0u ? 16.0f : 0.0f;
            dst[y] = (T)(d2 * ((float)(q[ql + l] >> 4) + hi) - m2);
            y += 1u;
        }
        ql += 32u;
        u1 <<= 2;
        u2 <<= 2;
    }
}

extern "C" __global__ void dequant_q5_k_f32(const uint8_t *__restrict__ quant,
                                            float *__restrict__ out, uint32_t n_blocks) {
    dequant_q5_k_block<float>(quant, out, n_blocks);
}

extern "C" __global__ void dequant_q5_k_f16(const uint8_t *__restrict__ quant,
                                            __half *__restrict__ out, uint32_t n_blocks) {
    dequant_q5_k_block<__half>(quant, out, n_blocks);
}

// ggml dequant_q4_k. Nibbles only, no high bit. 144 bytes, 256 values.
template <typename T>
__device__ void dequant_q4_k_block(const uint8_t *__restrict__ quant, T *__restrict__ out,
                                   uint32_t n_blocks) {
    uint32_t tid = blockIdx.x * blockDim.x + threadIdx.x;
    if (tid >= n_blocks) {
        return;
    }
    const uint8_t *q = quant + tid * 144u;
    float d = load_f16_le(q);
    float mn = load_f16_le(q + 2);
    const uint8_t *scales = q + 4;
    uint32_t qs = 16u;
    uint32_t y = 0u;
    T *dst = out + tid * 256u;
    for (uint32_t is = 0u; is < 8u; is += 2u) {
        uint32_t sc, m;
        scale_min_k4(scales, is, sc, m);
        float d1 = d * (float)sc;
        float m1 = mn * (float)m;
        scale_min_k4(scales, is + 1u, sc, m);
        float d2 = d * (float)sc;
        float m2 = mn * (float)m;
        for (uint32_t l = 0u; l < 32u; ++l) {
            dst[y] = (T)(d1 * (float)(q[qs + l] & 0xFu) - m1);
            y += 1u;
        }
        for (uint32_t l = 0u; l < 32u; ++l) {
            dst[y] = (T)(d2 * (float)(q[qs + l] >> 4) - m2);
            y += 1u;
        }
        qs += 32u;
    }
}

extern "C" __global__ void dequant_q4_k_f32(const uint8_t *__restrict__ quant,
                                            float *__restrict__ out, uint32_t n_blocks) {
    dequant_q4_k_block<float>(quant, out, n_blocks);
}

extern "C" __global__ void dequant_q4_k_f16(const uint8_t *__restrict__ quant,
                                            __half *__restrict__ out, uint32_t n_blocks) {
    dequant_q4_k_block<__half>(quant, out, n_blocks);
}
