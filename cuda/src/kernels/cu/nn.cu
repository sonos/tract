#include <cuda_fp16.h>
#include <math.h>

#define GELU_COEF_A    0.044715f
#define SQRT_2_OVER_PI 0.79788456080286535587989211986876f

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
//template<typename T>  
//[[kernel]] void apply_rope_nd3(             
//      device const void *input_b [[buffer(0)]],
//      device const void *cos_b [[buffer(1)]],
//      device const void *sin_b [[buffer(2)]],                 
//      device void *output_b [[buffer(3)]],                        
//      constant const size_t * shape [[buffer(4)]],
//      constant const size_t * strides [[buffer(5)]],
//      constant const size_t * cos_sin_strides [[buffer(6)]],
//      constant const size_t * out_strides [[buffer(7)]],
//      uint3 tpig[[thread_position_in_grid]]
//) {
//  device const T *input = (device const T *)input_b;
//  device const T *cos = (device const T *)cos_b;
//  device const T *sin = (device const T *)sin_b;
//
//  device T* output = (device T *) output_b;
//
//  uint3 rotated_tpig = tpig;
//  rotated_tpig.x += shape[2] / 2;
//
//  auto idx = indices_to_idx_3(tpig, strides);
//  auto rot_idx = indices_to_idx_3(rotated_tpig, strides);
//  auto out_idx = indices_to_idx_3(tpig, out_strides);
//  auto out_rot_idx = indices_to_idx_3(rotated_tpig, out_strides);
//
//  auto cos_sin_idx = indices_to_idx_3(tpig, cos_sin_strides);
//  auto rot_cos_sin_idx = indices_to_idx_3(rotated_tpig, cos_sin_strides);
//
//  output[out_idx] = input[idx] * cos[cos_sin_idx] - input[rot_idx] * sin[cos_sin_idx];
//  output[out_rot_idx] = input[rot_idx] * cos[rot_cos_sin_idx]
//          + input[idx] * sin[rot_cos_sin_idx];
//}
//
//template<typename T>  
//[[kernel]] void apply_rope_nd4(             
//      device const void *input_b [[buffer(0)]],
//      device const void *cos_b [[buffer(1)]],
//      device const void *sin_b [[buffer(2)]],                 
//      device void *output_b [[buffer(3)]],                        
//      constant const size_t * shape [[buffer(4)]],
//      constant const size_t * strides [[buffer(5)]],
//      constant const size_t * cos_sin_strides [[buffer(6)]],
//      constant const size_t * out_strides [[buffer(7)]],
//      uint3 tpig[[thread_position_in_grid]]
//) {
//  device const T *input = (device const T *)input_b;
//  device const T *cos = (device const T *)cos_b;
//  device const T *sin = (device const T *)sin_b;
//
//  device T* output = (device T *) output_b;
//
//  uint3 rotated_tpig = tpig;
//  rotated_tpig.x += shape[3] / 2;
//
//  auto idx = indices_to_idx_4(tpig, shape, strides);
//  auto rot_idx = indices_to_idx_4(rotated_tpig, shape, strides);
//  auto out_idx = indices_to_idx_4(tpig, shape, out_strides);
//  auto out_rot_idx = indices_to_idx_4(rotated_tpig, shape, out_strides);
//
//  auto cos_sin_idx = indices_to_idx_4(tpig, shape, cos_sin_strides);
//  auto rot_cos_sin_idx = indices_to_idx_4(rotated_tpig, shape, cos_sin_strides);
//
//  output[out_idx] = input[idx] * cos[cos_sin_idx] - input[rot_idx] * sin[cos_sin_idx];
//  output[out_rot_idx] = input[rot_idx] * cos[rot_cos_sin_idx]
//          + input[idx] * sin[rot_cos_sin_idx];
//}
//

INSTANTIATE_APPLY_ROPE(f32, float)
INSTANTIATE_APPLY_ROPE(f16, __half)