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
