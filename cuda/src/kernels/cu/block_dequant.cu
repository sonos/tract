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
