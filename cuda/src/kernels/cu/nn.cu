#include <cuda_fp16.h>
#include <math.h>

#define GELU_COEF_A    0.044715f
#define SQRT_2_OVER_PI 0.79788456080286535587989211986876f

#define WARP_SIZE 32

extern "C" __global__ void gelu_approx_f32(
                const float *input,
                float *output,
                int len
                ) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) {
        float x = input[i];
        float output_f32 = 0.5 * x * (
        1.0 + tanhf(SQRT_2_OVER_PI
            *(x + GELU_COEF_A * powf(x, (float) 3))));
        output[i] = output_f32;
    }
}

extern "C" __global__ void gelu_approx_f16(
                const __half *input,
                __half *output,
                int len
                ) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) {
        float x = (float) input[i];
        float output_f32 = 0.5 * x * (
        1.0 + tanhf(SQRT_2_OVER_PI
            *(x + GELU_COEF_A * powf(x, (float) 3))));
        output[i] = (__half) output_f32;
    }
}

extern "C" __global__ void gelu_approx_fast_f32(
                const float *input,
                float *output,
                int len
                ) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) {
        float x = input[i];
        float output_f32 = 0.5 * x * (
        1.0 + tanhf(SQRT_2_OVER_PI
            *(x + GELU_COEF_A * powf(x, (float)2))));
        output[i] = output_f32;
    }
}

extern "C" __global__ void gelu_approx_fast_f16(
                const __half *input,
                __half *output,
                int len
                ) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < len) {
        float x = (float) input[i];
        float output_f32 = 0.5 * x * (
        1.0 + tanhf(SQRT_2_OVER_PI
            *(x + GELU_COEF_A * powf(x, (float)2))));
        output[i] = (__half) output_f32;
    }
}

static __device__ __forceinline__ int indices_to_idx_2(int x, int y, int x_strides, int y_strides) {
  return x * x_strides + y * y_strides;
}

static __device__ __forceinline__ int indices_to_idx_3(int x, int y, int z, int x_strides, int y_strides, int z_strides) {
  return x * x_strides + y * y_strides + z * z_strides;
}

static __device__ __forceinline__ int indices_to_idx_4(int x, int y, int z,
                                                        int x_shape, int y_shape, int z_shape, int b_shape,
                                                        int x_strides, int y_strides, int z_strides, int b_strides) {
  int idx = x * x_strides + y * y_strides;
  idx += (z % z_shape) * z_strides;
  z /= z_shape;
  idx += z * b_strides;
  return idx;
}
//static __device__ __forceinline__ int indices_to_idx_4(uint3 indices,
//                                 constant const size_t shape[4], 
//                                 constant const size_t strides[4]) {
//  auto idx = indices.x * strides[3] + indices.y * strides[2];
//  idx += (indices.z % shape[1]) * strides[1];
//  indices.z /= shape[1];
//  idx += indices.z * strides[0];
//  return idx;
//}


#define INSTANTIATE_APPLY_ROPE(name, T) \
extern "C" __global__ void apply_rope_nd2_##name(const T *input, const T *cos, const T *sin, T *output, \
                                                int in_shape_0, int in_shape_1, int in_strides_0, int in_strides_1, \
                                                int cos_sin_strides_0, int cos_sin_strides_1, int out_strides_0, int out_strides_1) { \
    int thread_idx_x = blockIdx.x * blockDim.x + threadIdx.x; \
    int thread_idx_y = blockIdx.y * blockDim.y + threadIdx.y; \
    if (thread_idx_x >= in_shape_1 / 2 || thread_idx_y >= in_shape_0) { \
        return; \
    } \
    int rotated_idx_x = thread_idx_x + in_shape_1 / 2; \
\
    int idx = indices_to_idx_2(thread_idx_x, thread_idx_y, in_strides_1, in_strides_0); \
    int rot_idx = indices_to_idx_2(rotated_idx_x, thread_idx_y, in_strides_1, in_strides_0); \
    int out_idx = indices_to_idx_2(thread_idx_x, thread_idx_y, out_strides_1, out_strides_0); \
    int out_rot_idx = indices_to_idx_2(rotated_idx_x, thread_idx_y, out_strides_1, out_strides_0); \
\
    int cos_sin_idx = indices_to_idx_2(thread_idx_x, thread_idx_y, cos_sin_strides_1, cos_sin_strides_0); \
    int rot_cos_sin_idx = indices_to_idx_2(rotated_idx_x, thread_idx_y, cos_sin_strides_1, cos_sin_strides_0); \
\
    output[out_idx] = input[idx] * cos[cos_sin_idx] - input[rot_idx] * sin[cos_sin_idx]; \
    output[out_rot_idx] = input[rot_idx] * cos[rot_cos_sin_idx] \
            + input[idx] * sin[rot_cos_sin_idx]; \
} \
\
extern "C" __global__ void apply_rope_nd3_##name(const T *input, const T *cos, const T *sin, T *output, \
                                                int in_shape_0, int in_shape_1, int in_shape_2, \
                                                int in_strides_0, int in_strides_1, int in_strides_2, \
                                                int cos_sin_strides_0, int cos_sin_strides_1, int cos_sin_strides_2, \
                                                int out_strides_0, int out_strides_1, int out_strides_2) { \
    int thread_idx_x = blockIdx.x * blockDim.x + threadIdx.x; \
    int thread_idx_y = blockIdx.y * blockDim.y + threadIdx.y; \
    int thread_idx_z = blockIdx.z * blockDim.z + threadIdx.z; \
    if (thread_idx_x >= in_shape_2 / 2 || thread_idx_y >= in_shape_1 || thread_idx_z >= in_shape_0) { \
        return; \
    } \
    int rotated_idx_x = thread_idx_x + in_shape_2 / 2; \
\
    int idx = indices_to_idx_3(thread_idx_x, thread_idx_y, thread_idx_z, in_strides_2, in_strides_1, in_strides_0); \
    int rot_idx = indices_to_idx_3(rotated_idx_x, thread_idx_y, thread_idx_z, in_strides_2, in_strides_1, in_strides_0); \
    int out_idx = indices_to_idx_3(thread_idx_x, thread_idx_y, thread_idx_z, out_strides_2, out_strides_1, out_strides_0); \
    int out_rot_idx = indices_to_idx_3(rotated_idx_x, thread_idx_y, thread_idx_z, out_strides_2, out_strides_1, out_strides_0); \
\
    int cos_sin_idx = indices_to_idx_3(thread_idx_x, thread_idx_y, thread_idx_z, cos_sin_strides_2, cos_sin_strides_1, cos_sin_strides_0); \
    int rot_cos_sin_idx = indices_to_idx_3(rotated_idx_x, thread_idx_y, thread_idx_z, cos_sin_strides_2, cos_sin_strides_1, cos_sin_strides_0); \
\
    output[out_idx] = input[idx] * cos[cos_sin_idx] - input[rot_idx] * sin[cos_sin_idx]; \
    output[out_rot_idx] = input[rot_idx] * cos[rot_cos_sin_idx] \
            + input[idx] * sin[rot_cos_sin_idx]; \
} \
\
extern "C" __global__ void apply_rope_nd4_##name(const T *input, const T *cos, const T *sin, T *output, \
                                                int in_shape_0, int in_shape_1, int in_shape_2, int in_shape_3, \
                                                int in_strides_0, int in_strides_1, int in_strides_2, int in_strides_3, \
                                                int cos_sin_strides_0, int cos_sin_strides_1, int cos_sin_strides_2, int cos_sin_strides_3, \
                                                int out_strides_0, int out_strides_1, int out_strides_2, int out_strides_3) { \
    int thread_idx_x = blockIdx.x * blockDim.x + threadIdx.x; \
    int thread_idx_y = blockIdx.y * blockDim.y + threadIdx.y; \
    int thread_idx_z = blockIdx.z * blockDim.z + threadIdx.z; \
    if (thread_idx_x >= in_shape_3 / 2 || thread_idx_y >= in_shape_2 || thread_idx_z >= (in_shape_1 * in_shape_0)) { \
        return; \
    } \
    int rotated_idx_x = thread_idx_x + in_shape_3 / 2; \
\
    int idx = indices_to_idx_4(thread_idx_x, thread_idx_y, thread_idx_z, \
        in_shape_3, in_shape_2, in_shape_1, in_shape_0, \
        in_strides_3, in_strides_2, in_strides_1, in_strides_0); \
    int rot_idx = indices_to_idx_4(rotated_idx_x, thread_idx_y, thread_idx_z, \
        in_shape_3, in_shape_2, in_shape_1, in_shape_0, \
        in_strides_3, in_strides_2, in_strides_1, in_strides_0); \
    int out_idx = indices_to_idx_4(thread_idx_x, thread_idx_y, thread_idx_z, \
        in_shape_3, in_shape_2, in_shape_1, in_shape_0, \
        out_strides_3, out_strides_2, out_strides_1, out_strides_0); \
    int out_rot_idx = indices_to_idx_4(rotated_idx_x, thread_idx_y, thread_idx_z, \
        in_shape_3, in_shape_2, in_shape_1, in_shape_0, \
        out_strides_3, out_strides_2, out_strides_1, out_strides_0); \
\
    int cos_sin_idx = indices_to_idx_4(thread_idx_x, thread_idx_y, thread_idx_z, \
        in_shape_3, in_shape_2, in_shape_1, in_shape_0, \
        cos_sin_strides_3, cos_sin_strides_2, cos_sin_strides_1, cos_sin_strides_0); \
    int rot_cos_sin_idx = indices_to_idx_4(rotated_idx_x, thread_idx_y, thread_idx_z, \
        in_shape_3, in_shape_2, in_shape_1, in_shape_0, \
        cos_sin_strides_3, cos_sin_strides_2, cos_sin_strides_1, cos_sin_strides_0); \
\
    output[out_idx] = input[idx] * cos[cos_sin_idx] - input[rot_idx] * sin[cos_sin_idx]; \
    output[out_rot_idx] = input[rot_idx] * cos[rot_cos_sin_idx] \
            + input[idx] * sin[rot_cos_sin_idx]; \
} \
\

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

#define INSTANTIATE_RMS_NORM(name, T, bname, block_size) \
extern "C" __global__ void rms_norm_##bname##name( \
        const T * x, T * dst, const int shape_0, const int shape_1, const int shape_2, \
        const int strides_0, const int strides_1,const int strides_2, const T eps) { \
    int base_idx = (blockIdx.x % shape_2) * strides_2 + (blockIdx.x / shape_2) * strides_0; \
 \
    float tmp = 0.0f; \
 \
    for (int i = threadIdx.x; i < shape_1; i += blockDim.x) { \
        const float xi = x[base_idx + i * strides_1]; \
        tmp += xi * xi; \
    } \
 \
    tmp = warp_reduce_sum(tmp); \
    if constexpr (block_size > WARP_SIZE) { \
        __shared__ float s_sum[32]; \
        const int warp_id = threadIdx.x / WARP_SIZE; \
        const int lane_id = threadIdx.x % WARP_SIZE; \
        if (lane_id == 0) { \
            s_sum[warp_id] = tmp; \
        } \
        __syncthreads(); \
        tmp = s_sum[lane_id]; \
        tmp = warp_reduce_sum(tmp); \
    } \
\
    float eps_f = (float) eps; \
    const float mean = tmp / shape_1; \
    const float scale = rsqrtf(mean + eps_f); \
\
    for (int i = threadIdx.x; i < shape_1; i += blockDim.x) { \
        int idx =  base_idx + i * strides_1; \
        dst[idx] = (T) scale * x[idx]; \
    } \
} \

INSTANTIATE_APPLY_ROPE(f32, float)
INSTANTIATE_APPLY_ROPE(f16, __half)

INSTANTIATE_RMS_NORM(f32, float, small_, 32)
INSTANTIATE_RMS_NORM(f32, float, , 1024)
INSTANTIATE_RMS_NORM(f16, __half, small_, 32)
INSTANTIATE_RMS_NORM(f16, __half, , 1024)