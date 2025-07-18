#include <cuda_fp16.h>
#include <math.h>

#define WARP_SIZE   32

template<int width = WARP_SIZE>
static __device__ __forceinline__ float warp_reduce_sum(float x) {
#pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, width);
    }
    return x;
}

template<int width = WARP_SIZE>
static __device__ __forceinline__ __half warp_reduce_sum(__half x) {
#pragma unroll
    for (int offset = width/2; offset > 0; offset >>= 1) {
        x += __shfl_xor_sync(0xffffffff, x, offset, width);
    }
    return x;
}

static __device__ __forceinline__ float mat_vec_acc(const float* x, const float* y, const int tid, const int64_t ncols2, const int block_size) {
    float sumf = 0.0f;
    const float2 * x2 = (const float2 *) x;
    const float2 * y2 = (const float2 *) y;

    for (int64_t col2 = tid; col2 < ncols2; col2 += block_size) {
        const float2 tmpx = x2[col2];
        const float2 tmpy = y2[col2];
        sumf += tmpx.x*tmpy.x;
        sumf += tmpx.y*tmpy.y;
    }
    return sumf;
}

static __device__ __forceinline__ float mat_vec_acc(const half* x, const half* y, const int tid, const int64_t ncols2, const int block_size) {
    float sumf = 0.0f;
    const half2 * x2 = (const half2 *) x;
    const half2 * y2 = (const half2 *) y;

    for (int64_t col2 = tid; col2 < ncols2; col2 += block_size) {
        const half2 tmpx = x2[col2];
        const half2 tmpy = y2[col2];
        sumf += __half2float(tmpx.x * tmpy.x);
        sumf += __half2float(tmpx.y * tmpy.y);
    }

    return sumf;
}

#define INSTANTIATE_MAT_VEC(type_name, T, block_size_name, block_size) \
extern "C" __global__ void ggml_matvec_##type_name##block_size_name( \
        const T * __restrict__ x, const T * __restrict__ y, T * __restrict__ dst, \
        const int64_t ncols2, const int64_t nchannels_y, const int64_t stride_row, \
        const int64_t channel_ratio, const int64_t stride_channel_x, const int64_t stride_channel_y, const int64_t stride_channel_dst) { \
    const int64_t row         = blockIdx.x; \
    const int64_t channel_dst = blockIdx.y; \
    const int64_t channel_x   = channel_dst / channel_ratio; \
    const int64_t channel_y   = channel_dst; \
    const int     tid         = threadIdx.x; \
 \
    x   += channel_x  *stride_channel_x   + row*stride_row; \
    y   += channel_y  *stride_channel_y; \
    dst += channel_dst*stride_channel_dst; \
 \
    extern __shared__ char data_mmv[]; \
    float * buf_iw = (float *) data_mmv; \
 \
    if (block_size > WARP_SIZE) { \
        if (tid < WARP_SIZE) { \
            buf_iw[tid] = 0.0f; \
        } \
        __syncthreads(); \
    } \
 \
    float sumf = mat_vec_acc(x, y, tid, ncols2, block_size); \
 \
    sumf = warp_reduce_sum(sumf); \
 \
    if (block_size > WARP_SIZE) { \
        buf_iw[tid/WARP_SIZE] = sumf; \
        __syncthreads(); \
        if (tid >= WARP_SIZE) { \
            return; \
        } \
        sumf = buf_iw[tid]; \
        sumf = warp_reduce_sum(sumf); \
    } \
 \
    if (tid != 0) { \
        return; \
    } \
 \
    dst[row] = sumf; \
} \

#define INSTANTIATE_MAT_VEC_FOR_T(name, T) \
    INSTANTIATE_MAT_VEC(name, T, bs_32, 32) \
    INSTANTIATE_MAT_VEC(name, T, bs_64, 64) \
    INSTANTIATE_MAT_VEC(name, T, bs_96, 96) \
    INSTANTIATE_MAT_VEC(name, T, bs_128, 128) \
    INSTANTIATE_MAT_VEC(name, T, bs_160, 160) \
    INSTANTIATE_MAT_VEC(name, T, bs_196, 196) \
    INSTANTIATE_MAT_VEC(name, T, bs_224, 224) \
    INSTANTIATE_MAT_VEC(name, T, bs_256, 256) \

INSTANTIATE_MAT_VEC_FOR_T(f32_, float)
INSTANTIATE_MAT_VEC_FOR_T(f16_, __half)
