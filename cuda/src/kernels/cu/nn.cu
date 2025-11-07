#include <cuda_runtime.h>
#include <math_constants.h>
#include "common.cuh"

#define GELU_COEF_A 0.044715f
#define SQRT_2_OVER_PI 0.79788456080286535587989211986876f

#define INSTANTIATE_REDUCE(name, T, bname, block_size)                         \
  extern "C" __global__ void reduce_max_##bname##name(                         \
      const T *input, T *output, const int shape_0, const int shape_1,         \
      const int shape_2, const int in_stride_0, const int in_stride_1,         \
      const int in_stride_2, const int out_stride_0, const int out_stride_1,   \
      const int out_stride_2) {                                                \
    input += blockIdx.z * in_stride_0 + blockIdx.x * in_stride_2;              \
    output += blockIdx.z * out_stride_0 + blockIdx.x * out_stride_2;           \
                                                                               \
    const int warp_id = threadIdx.x / WARP_SIZE;                               \
    const int lane_id = threadIdx.x % WARP_SIZE;                               \
                                                                               \
    float max_val = -CUDART_INF_F;                                                 \
    _Pragma("unroll") for (int i = threadIdx.x; i < shape_1;                   \
                           i += blockDim.x) {                                  \
      max_val = max(max_val, input[i * in_stride_1]);                          \
    }                                                                          \
                                                                               \
    max_val = warp_reduce_max(max_val);                                        \
    if (block_size > WARP_SIZE) {                                              \
      __shared__ float s_max[32];                                              \
      if (warp_id == 0) {                                                      \
        s_max[lane_id] = -CUDART_INF_F;                                            \
      }                                                                        \
      __syncthreads();                                                         \
                                                                               \
      if (lane_id == 0) {                                                      \
        s_max[warp_id] = max_val;                                              \
      }                                                                        \
      __syncthreads();                                                         \
                                                                               \
      max_val = s_max[lane_id];                                                \
      max_val = warp_reduce_max(max_val);                                      \
    }                                                                          \
                                                                               \
    if (threadIdx.x == 0) {                                                    \
      *output = max_val;                                                       \
    }                                                                          \
  }                                                                            \
                                                                               \
  extern "C" __global__ void reduce_min_##bname##name(                         \
      const T *input, T *output, const int shape_0, const int shape_1,         \
      const int shape_2, const int in_stride_0, const int in_stride_1,         \
      const int in_stride_2, const int out_stride_0, const int out_stride_1,   \
      const int out_stride_2) {                                                \
    input += blockIdx.z * in_stride_0 + blockIdx.x * in_stride_2;              \
    output += blockIdx.z * out_stride_0 + blockIdx.x * out_stride_2;           \
                                                                               \
    const int warp_id = threadIdx.x / WARP_SIZE;                               \
    const int lane_id = threadIdx.x % WARP_SIZE;                               \
                                                                               \
    float min_val = CUDART_INF_F;                                                  \
    _Pragma("unroll") for (int i = threadIdx.x; i < shape_1;                   \
                           i += blockDim.x) {                                  \
      min_val = min(min_val, input[i * in_stride_1]);                          \
    }                                                                          \
                                                                               \
    min_val = warp_reduce_min(min_val);                                        \
    if (block_size > WARP_SIZE) {                                              \
      __shared__ float s_min[32];                                              \
      if (warp_id == 0) {                                                      \
        s_min[lane_id] = -CUDART_INF_F;                                            \
      }                                                                        \
      __syncthreads();                                                         \
                                                                               \
      if (lane_id == 0) {                                                      \
        s_min[warp_id] = min_val;                                              \
      }                                                                        \
      __syncthreads();                                                         \
                                                                               \
      min_val = s_min[lane_id];                                                \
      min_val = warp_reduce_min(min_val);                                      \
    }                                                                          \
                                                                               \
    if (threadIdx.x == 0) {                                                    \
      *output = min_val;                                                       \
    }                                                                          \
  }                                                                            \
                                                                               \
  extern "C" __global__ void reduce_sum_##bname##name(                         \
      const T *input, T *output, const int shape_0, const int shape_1,         \
      const int shape_2, const int in_stride_0, const int in_stride_1,         \
      const int in_stride_2, const int out_stride_0, const int out_stride_1,   \
      const int out_stride_2) {                                                \
    input += blockIdx.z * in_stride_0 + blockIdx.x * in_stride_2;              \
    output += blockIdx.z * out_stride_0 + blockIdx.x * out_stride_2;           \
                                                                               \
    const int warp_id = threadIdx.x / WARP_SIZE;                               \
    const int lane_id = threadIdx.x % WARP_SIZE;                               \
                                                                               \
    T sum_val = 0.0f;                                                          \
    _Pragma("unroll") for (int i = threadIdx.x; i < shape_1;                   \
                           i += blockDim.x) {                                  \
      sum_val += input[i * in_stride_1];                                       \
    }                                                                          \
                                                                               \
    sum_val = warp_reduce_sum(sum_val);                                        \
    if (block_size > WARP_SIZE) {                                              \
      __shared__ T s_sum[32];                                                  \
      if (warp_id == 0) {                                                      \
        s_sum[lane_id] = (T)0.0f;                                              \
      }                                                                        \
      __syncthreads();                                                         \
                                                                               \
      if (lane_id == 0) {                                                      \
        s_sum[warp_id] = sum_val;                                              \
      }                                                                        \
      __syncthreads();                                                         \
                                                                               \
      sum_val = s_sum[lane_id];                                                \
      sum_val = warp_reduce_sum(sum_val);                                      \
    }                                                                          \
                                                                               \
    if (threadIdx.x == 0) {                                                    \
      *output = sum_val;                                                       \
    }                                                                          \
  }                                                                            \
                                                                               \
  extern "C" __global__ void reduce_prod_##bname##name(                        \
      const T *input, T *output, const int shape_0, const int shape_1,         \
      const int shape_2, const int in_stride_0, const int in_stride_1,         \
      const int in_stride_2, const int out_stride_0, const int out_stride_1,   \
      const int out_stride_2) {                                                \
    input += blockIdx.z * in_stride_0 + blockIdx.x * in_stride_2;              \
    output += blockIdx.z * out_stride_0 + blockIdx.x * out_stride_2;           \
                                                                               \
    const int warp_id = threadIdx.x / WARP_SIZE;                               \
    const int lane_id = threadIdx.x % WARP_SIZE;                               \
                                                                               \
    T prod_val = (T)1.0f;                                                      \
    _Pragma("unroll") for (int i = threadIdx.x; i < shape_1;                   \
                           i += blockDim.x) {                                  \
      prod_val *= input[i * in_stride_1];                                      \
    }                                                                          \
                                                                               \
    prod_val = warp_reduce_prod(prod_val);                                     \
    if (block_size > WARP_SIZE) {                                              \
      __shared__ T s_prod[32];                                                 \
      if (warp_id == 0) {                                                      \
        s_prod[lane_id] = (T)0.0f;                                             \
      }                                                                        \
      __syncthreads();                                                         \
                                                                               \
      if (lane_id == 0) {                                                      \
        s_prod[warp_id] = prod_val;                                            \
      }                                                                        \
      __syncthreads();                                                         \
                                                                               \
      prod_val = s_prod[lane_id];                                              \
      prod_val = warp_reduce_prod(prod_val);                                   \
    }                                                                          \
                                                                               \
    if (threadIdx.x == 0) {                                                    \
      *output = prod_val;                                                      \
    }                                                                          \
  }                                                                            \
                                                                               \
  extern "C" __global__ void reduce_mean_of_squares_##bname##name(             \
      const T *input, T *output, const int shape_0, const int shape_1,         \
      const int shape_2, const int in_stride_0, const int in_stride_1,         \
      const int in_stride_2, const int out_stride_0, const int out_stride_1,   \
      const int out_stride_2) {                                                \
    input += blockIdx.z * in_stride_0 + blockIdx.x * in_stride_2;              \
    output += blockIdx.z * out_stride_0 + blockIdx.x * out_stride_2;           \
                                                                               \
    const int warp_id = threadIdx.x / WARP_SIZE;                               \
    const int lane_id = threadIdx.x % WARP_SIZE;                               \
                                                                               \
    T square_sum_val = (T)0.0f;                                                \
    _Pragma("unroll") for (int i = threadIdx.x; i < shape_1;                   \
                           i += blockDim.x) {                                  \
      square_sum_val += input[i * in_stride_1] * input[i * in_stride_1];       \
    }                                                                          \
                                                                               \
    square_sum_val = warp_reduce_sum(square_sum_val);                          \
    if (block_size > WARP_SIZE) {                                              \
      __shared__ T s_prod[32];                                                 \
      if (warp_id == 0) {                                                      \
        s_prod[lane_id] = (T)0.0f;                                             \
      }                                                                        \
      __syncthreads();                                                         \
                                                                               \
      if (lane_id == 0) {                                                      \
        s_prod[warp_id] = square_sum_val;                                      \
      }                                                                        \
      __syncthreads();                                                         \
                                                                               \
      square_sum_val = s_prod[lane_id];                                        \
      square_sum_val = warp_reduce_sum(square_sum_val);                        \
    }                                                                          \
                                                                               \
    if (threadIdx.x == 0) {                                                    \
      *output = square_sum_val / (T)shape_1;                                   \
    }                                                                          \
  }

extern "C" __global__ void gelu_approx_f32(const float *input, float *output,
                                           int len) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    float x = input[i];
    float output_f32 =
        0.5 * x *
        (1.0 + tanhf(SQRT_2_OVER_PI * (x + GELU_COEF_A * powf(x, (float)3))));
    output[i] = output_f32;
  }
}

extern "C" __global__ void gelu_approx_f16(const __half *input, __half *output,
                                           int len) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    float x = (float)input[i];
    float output_f32 =
        0.5 * x *
        (1.0 + tanhf(SQRT_2_OVER_PI * (x + GELU_COEF_A * powf(x, (float)3))));
    output[i] = (__half)output_f32;
  }
}

extern "C" __global__ void gelu_approx_fast_f32(const float *input,
                                                float *output, int len) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    float x = input[i];
    float output_f32 =
        0.5 * x *
        (1.0 + tanhf(SQRT_2_OVER_PI * (x + GELU_COEF_A * powf(x, (float)2))));
    output[i] = output_f32;
  }
}

extern "C" __global__ void gelu_approx_fast_f16(const __half *input,
                                                __half *output, int len) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    float x = (float)input[i];
    float output_f32 =
        0.5 * x *
        (1.0 + tanhf(SQRT_2_OVER_PI * (x + GELU_COEF_A * powf(x, (float)2))));
    output[i] = (__half)output_f32;
  }
}

extern "C" __global__ void leaky_relu_f32(const float *input,
                                                float *output, int len,
                                                float alpha) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    float x = input[i];
    output[i] = x * (x < 0 ? alpha : 1.0);
  }
}

extern "C" __global__ void leaky_relu_f16(const __half *input,
                                                __half *output, int len,
                                                float alpha) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  __half alpha_f16 = (__half) alpha;
  if (i < len) {
    __half x = input[i];
    output[i] = x * (x < (__half) 0.0 ? alpha_f16 : (__half) 1.0);
  }
}

static __device__ __forceinline__ int
indices_to_idx_2(int x, int y, int x_strides, int y_strides) {
  return x * x_strides + y * y_strides;
}

static __device__ __forceinline__ int indices_to_idx_3(int x, int y, int z,
                                                       int x_strides,
                                                       int y_strides,
                                                       int z_strides) {
  return x * x_strides + y * y_strides + z * z_strides;
}

static __device__ __forceinline__ int
indices_to_idx_4(int x, int y, int z, int x_shape, int y_shape, int z_shape,
                 int b_shape, int x_strides, int y_strides, int z_strides,
                 int b_strides) {
  int idx = x * x_strides + y * y_strides;
  idx += (z % z_shape) * z_strides;
  z /= z_shape;
  idx += z * b_strides;
  return idx;
}

#define INSTANTIATE_APPLY_ROPE(name, T)                                        \
  extern "C" __global__ void apply_rope_nd2_##name(                            \
      const T *input, const T *cos, const T *sin, T *output, int in_shape_0,   \
      int in_shape_1, int in_strides_0, int in_strides_1,                      \
      int cos_sin_strides_0, int cos_sin_strides_1, int out_strides_0,         \
      int out_strides_1) {                                                     \
    int thread_idx_x = blockIdx.x * blockDim.x + threadIdx.x;                  \
    int thread_idx_y = blockIdx.y * blockDim.y + threadIdx.y;                  \
    if (thread_idx_x >= in_shape_1 / 2 || thread_idx_y >= in_shape_0) {        \
      return;                                                                  \
    }                                                                          \
    int rotated_idx_x = thread_idx_x + in_shape_1 / 2;                         \
                                                                               \
    int idx = indices_to_idx_2(thread_idx_x, thread_idx_y, in_strides_1,       \
                               in_strides_0);                                  \
    int rot_idx = indices_to_idx_2(rotated_idx_x, thread_idx_y, in_strides_1,  \
                                   in_strides_0);                              \
    int out_idx = indices_to_idx_2(thread_idx_x, thread_idx_y, out_strides_1,  \
                                   out_strides_0);                             \
    int out_rot_idx = indices_to_idx_2(rotated_idx_x, thread_idx_y,            \
                                       out_strides_1, out_strides_0);          \
                                                                               \
    int cos_sin_idx = indices_to_idx_2(thread_idx_x, thread_idx_y,             \
                                       cos_sin_strides_1, cos_sin_strides_0);  \
    int rot_cos_sin_idx = indices_to_idx_2(                                    \
        rotated_idx_x, thread_idx_y, cos_sin_strides_1, cos_sin_strides_0);    \
                                                                               \
    output[out_idx] =                                                          \
        input[idx] * cos[cos_sin_idx] - input[rot_idx] * sin[cos_sin_idx];     \
    output[out_rot_idx] = input[rot_idx] * cos[rot_cos_sin_idx] +              \
                          input[idx] * sin[rot_cos_sin_idx];                   \
  }                                                                            \
                                                                               \
  extern "C" __global__ void apply_rope_nd3_##name(                            \
      const T *input, const T *cos, const T *sin, T *output, int in_shape_0,   \
      int in_shape_1, int in_shape_2, int in_strides_0, int in_strides_1,      \
      int in_strides_2, int cos_sin_strides_0, int cos_sin_strides_1,          \
      int cos_sin_strides_2, int out_strides_0, int out_strides_1,             \
      int out_strides_2) {                                                     \
    int thread_idx_x = blockIdx.x * blockDim.x + threadIdx.x;                  \
    int thread_idx_y = blockIdx.y * blockDim.y + threadIdx.y;                  \
    int thread_idx_z = blockIdx.z * blockDim.z + threadIdx.z;                  \
    if (thread_idx_x >= in_shape_2 / 2 || thread_idx_y >= in_shape_1 ||        \
        thread_idx_z >= in_shape_0) {                                          \
      return;                                                                  \
    }                                                                          \
    int rotated_idx_x = thread_idx_x + in_shape_2 / 2;                         \
                                                                               \
    int idx = indices_to_idx_3(thread_idx_x, thread_idx_y, thread_idx_z,       \
                               in_strides_2, in_strides_1, in_strides_0);      \
    int rot_idx = indices_to_idx_3(rotated_idx_x, thread_idx_y, thread_idx_z,  \
                                   in_strides_2, in_strides_1, in_strides_0);  \
    int out_idx =                                                              \
        indices_to_idx_3(thread_idx_x, thread_idx_y, thread_idx_z,             \
                         out_strides_2, out_strides_1, out_strides_0);         \
    int out_rot_idx =                                                          \
        indices_to_idx_3(rotated_idx_x, thread_idx_y, thread_idx_z,            \
                         out_strides_2, out_strides_1, out_strides_0);         \
                                                                               \
    int cos_sin_idx = indices_to_idx_3(thread_idx_x, thread_idx_y,             \
                                       thread_idx_z, cos_sin_strides_2,        \
                                       cos_sin_strides_1, cos_sin_strides_0);  \
    int rot_cos_sin_idx = indices_to_idx_3(                                    \
        rotated_idx_x, thread_idx_y, thread_idx_z, cos_sin_strides_2,          \
        cos_sin_strides_1, cos_sin_strides_0);                                 \
                                                                               \
    output[out_idx] =                                                          \
        input[idx] * cos[cos_sin_idx] - input[rot_idx] * sin[cos_sin_idx];     \
    output[out_rot_idx] = input[rot_idx] * cos[rot_cos_sin_idx] +              \
                          input[idx] * sin[rot_cos_sin_idx];                   \
  }                                                                            \
                                                                               \
  extern "C" __global__ void apply_rope_nd4_##name(                            \
      const T *input, const T *cos, const T *sin, T *output, int in_shape_0,   \
      int in_shape_1, int in_shape_2, int in_shape_3, int in_strides_0,        \
      int in_strides_1, int in_strides_2, int in_strides_3,                    \
      int cos_sin_strides_0, int cos_sin_strides_1, int cos_sin_strides_2,     \
      int cos_sin_strides_3, int out_strides_0, int out_strides_1,             \
      int out_strides_2, int out_strides_3) {                                  \
    int thread_idx_x = blockIdx.x * blockDim.x + threadIdx.x;                  \
    int thread_idx_y = blockIdx.y * blockDim.y + threadIdx.y;                  \
    int thread_idx_z = blockIdx.z * blockDim.z + threadIdx.z;                  \
    if (thread_idx_x >= in_shape_3 / 2 || thread_idx_y >= in_shape_2 ||        \
        thread_idx_z >= (in_shape_1 * in_shape_0)) {                           \
      return;                                                                  \
    }                                                                          \
    int rotated_idx_x = thread_idx_x + in_shape_3 / 2;                         \
                                                                               \
    int idx =                                                                  \
        indices_to_idx_4(thread_idx_x, thread_idx_y, thread_idx_z, in_shape_3, \
                         in_shape_2, in_shape_1, in_shape_0, in_strides_3,     \
                         in_strides_2, in_strides_1, in_strides_0);            \
    int rot_idx = indices_to_idx_4(rotated_idx_x, thread_idx_y, thread_idx_z,  \
                                   in_shape_3, in_shape_2, in_shape_1,         \
                                   in_shape_0, in_strides_3, in_strides_2,     \
                                   in_strides_1, in_strides_0);                \
    int out_idx =                                                              \
        indices_to_idx_4(thread_idx_x, thread_idx_y, thread_idx_z, in_shape_3, \
                         in_shape_2, in_shape_1, in_shape_0, out_strides_3,    \
                         out_strides_2, out_strides_1, out_strides_0);         \
    int out_rot_idx = indices_to_idx_4(                                        \
        rotated_idx_x, thread_idx_y, thread_idx_z, in_shape_3, in_shape_2,     \
        in_shape_1, in_shape_0, out_strides_3, out_strides_2, out_strides_1,   \
        out_strides_0);                                                        \
                                                                               \
    int cos_sin_idx = indices_to_idx_4(                                        \
        thread_idx_x, thread_idx_y, thread_idx_z, in_shape_3, in_shape_2,      \
        in_shape_1, in_shape_0, cos_sin_strides_3, cos_sin_strides_2,          \
        cos_sin_strides_1, cos_sin_strides_0);                                 \
    int rot_cos_sin_idx = indices_to_idx_4(                                    \
        rotated_idx_x, thread_idx_y, thread_idx_z, in_shape_3, in_shape_2,     \
        in_shape_1, in_shape_0, cos_sin_strides_3, cos_sin_strides_2,          \
        cos_sin_strides_1, cos_sin_strides_0);                                 \
                                                                               \
    output[out_idx] =                                                          \
        input[idx] * cos[cos_sin_idx] - input[rot_idx] * sin[cos_sin_idx];     \
    output[out_rot_idx] = input[rot_idx] * cos[rot_cos_sin_idx] +              \
                          input[idx] * sin[rot_cos_sin_idx];                   \
  }

#define INSTANTIATE_SOFTMAX(name, T, bname, block_size)                        \
  extern "C" __global__ void softmax_##bname##name(                            \
      const T *x, T *dst, const int shape_0, const int shape_1,                \
      const int shape_2, const int stride_0, const int stride_1,               \
      const int stride_2) {                                                    \
    int offset =                                                               \
        (blockIdx.x % shape_2) * stride_2 + (blockIdx.x / shape_2) * stride_0; \
    x += offset;                                                               \
    dst += offset;                                                             \
                                                                               \
    const int warp_id = threadIdx.x / WARP_SIZE;                               \
    const int lane_id = threadIdx.x % WARP_SIZE;                               \
                                                                               \
    float max_val = -CUDART_INF_F;                                                 \
    _Pragma("unroll") for (int i = threadIdx.x; i < shape_1;                   \
                           i += blockDim.x) {                                  \
      max_val = max(max_val, x[i * stride_1]);                                 \
    }                                                                          \
                                                                               \
    max_val = warp_reduce_max(max_val);                                        \
    if (block_size > WARP_SIZE) {                                              \
      __shared__ float s_max[32];                                              \
      if (warp_id == 0) {                                                      \
        s_max[lane_id] = -CUDART_INF_F;                                            \
      }                                                                        \
      __syncthreads();                                                         \
                                                                               \
      if (lane_id == 0) {                                                      \
        s_max[warp_id] = max_val;                                              \
      }                                                                        \
      __syncthreads();                                                         \
                                                                               \
      max_val = s_max[lane_id];                                                \
      max_val = warp_reduce_max(max_val);                                      \
    }                                                                          \
                                                                               \
    float tmp = 0.0f;                                                          \
    for (int i = threadIdx.x; i < shape_1; i += blockDim.x) {                  \
      float el = x[i * stride_1];                                              \
      const float val = expf(el - max_val);                                    \
      tmp += val;                                                              \
      dst[i * stride_1] = val;                                                 \
    }                                                                          \
                                                                               \
    tmp = warp_reduce_sum(tmp);                                                \
    if (block_size > WARP_SIZE) {                                              \
      __shared__ float s_sum[32];                                              \
      if (warp_id == 0) {                                                      \
        s_sum[lane_id] = 0.0f;                                                 \
      }                                                                        \
      __syncthreads();                                                         \
                                                                               \
      if (lane_id == 0) {                                                      \
        s_sum[warp_id] = tmp;                                                  \
      }                                                                        \
      __syncthreads();                                                         \
                                                                               \
      tmp = s_sum[lane_id];                                                    \
      tmp = warp_reduce_sum(tmp);                                              \
    }                                                                          \
                                                                               \
    const float inv_sum = 1.0f / tmp;                                          \
                                                                               \
    for (int i = threadIdx.x; i < shape_1; i += blockDim.x) {                  \
      dst[i * stride_1] *= inv_sum;                                            \
    }                                                                          \
  }

#define INSTANTIATE_SCALED_MASKED_SOFTMAX(name, T, bname, block_size_template) \
  extern "C" __global__ void scaled_masked_softmax_##bname##name(              \
      const T *x, const T *mask, const T scale, T *dst, const int shape_0,     \
      const int shape_1, const int shape_2, const int stride_0,                \
      const int stride_1, const int stride_2, const int mask_stride_0,         \
      const int mask_stride_1, const int mask_stride_2,                        \
      const int out_stride_0, const int out_stride_1,                          \
      const int out_stride_2) {                                                \
    x += blockIdx.y * stride_1 + blockIdx.z * stride_0;                        \
    mask +=                                                                    \
        mask ? blockIdx.y * mask_stride_1 + blockIdx.z * mask_stride_0 : 0;    \
    dst += blockIdx.y * out_stride_1 + blockIdx.z * out_stride_0;              \
                                                                               \
    const int block_size =                                                     \
        block_size_template == 0 ? blockDim.x : block_size_template;           \
                                                                               \
    const int warp_id = threadIdx.x / WARP_SIZE;                               \
    const int lane_id = threadIdx.x % WARP_SIZE;                               \
                                                                               \
    extern __shared__ float data_soft_max_f32[];                               \
    float *buf_iw = data_soft_max_f32;                                         \
    float *vals = buf_iw + WARP_SIZE;                                          \
                                                                               \
    float max_val = -CUDART_INF_F;                                                 \
    _Pragma("unroll") for (int col0 = 0; col0 < shape_2; col0 += block_size) { \
      const int col = col0 + threadIdx.x;                                      \
      if (col >= shape_2) {                                                    \
        break;                                                                 \
      }                                                                        \
                                                                               \
      const float val = x[col * stride_2] * scale + mask[col * mask_stride_2]; \
      vals[col] = val;                                                         \
      max_val = max(max_val, val);                                             \
    }                                                                          \
                                                                               \
    max_val = warp_reduce_max(max_val);                                        \
    if (block_size > WARP_SIZE) {                                              \
      if (warp_id == 0) {                                                      \
        buf_iw[lane_id] = -CUDART_INF_F;                                           \
      }                                                                        \
      __syncthreads();                                                         \
                                                                               \
      if (lane_id == 0) {                                                      \
        buf_iw[warp_id] = max_val;                                             \
      }                                                                        \
      __syncthreads();                                                         \
                                                                               \
      max_val = buf_iw[lane_id];                                               \
      max_val = warp_reduce_max(max_val);                                      \
    }                                                                          \
                                                                               \
    float tmp = 0.0f;                                                          \
    _Pragma("unroll") for (int col0 = 0; col0 < shape_2; col0 += block_size) { \
      const int col = col0 + threadIdx.x;                                      \
      if (col >= shape_2) {                                                    \
        break;                                                                 \
      }                                                                        \
                                                                               \
      const float val = expf(vals[col] - max_val);                             \
      tmp += val;                                                              \
      vals[col] = val;                                                         \
    }                                                                          \
                                                                               \
    tmp = warp_reduce_sum(tmp);                                                \
    if (block_size > WARP_SIZE) {                                              \
      __syncthreads();                                                         \
      if (warp_id == 0) {                                                      \
        buf_iw[lane_id] = 0.0f;                                                \
      }                                                                        \
      __syncthreads();                                                         \
                                                                               \
      if (lane_id == 0) {                                                      \
        buf_iw[warp_id] = tmp;                                                 \
      }                                                                        \
      __syncthreads();                                                         \
                                                                               \
      tmp = buf_iw[lane_id];                                                   \
      tmp = warp_reduce_sum(tmp);                                              \
    }                                                                          \
                                                                               \
    const float inv_sum = 1.0f / tmp;                                          \
                                                                               \
    _Pragma("unroll") for (int col0 = 0; col0 < shape_2; col0 += block_size) { \
      const int col = col0 + threadIdx.x;                                      \
      if (col >= shape_2) {                                                    \
        return;                                                                \
      }                                                                        \
      dst[col * out_stride_2] = vals[col] * inv_sum;                           \
    }                                                                          \
  }

#define INSTANTIATE_RMS_NORM(name, T, bname, block_size)                       \
  extern "C" __global__ void rms_norm_##bname##name(                           \
      const T *x, T *dst, const int shape_0, const int shape_1,                \
      const int shape_2, const int strides_0, const int strides_1,             \
      const int strides_2, const float eps) {                                      \
    int base_idx = (blockIdx.x % shape_2) * strides_2 +                        \
                   (blockIdx.x / shape_2) * strides_0;                         \
                                                                               \
    float tmp = 0.0f;                                                          \
                                                                               \
    for (int i = threadIdx.x; i < shape_1; i += blockDim.x) {                  \
      const float xi = x[base_idx + i * strides_1];                            \
      tmp += xi * xi;                                                          \
    }                                                                          \
                                                                               \
    tmp = warp_reduce_sum(tmp);                                                \
    if constexpr (block_size > WARP_SIZE) {                                    \
      __shared__ float s_sum[32];                                              \
      const int warp_id = threadIdx.x / WARP_SIZE;                             \
      const int lane_id = threadIdx.x % WARP_SIZE;                             \
      if (lane_id == 0) {                                                      \
        s_sum[warp_id] = tmp;                                                  \
      }                                                                        \
      __syncthreads();                                                         \
      tmp = s_sum[lane_id];                                                    \
      tmp = warp_reduce_sum(tmp);                                              \
    }                                                                          \
                                                                               \
    const float mean = tmp / shape_1;                                          \
    const float scale = rsqrtf(mean + eps);                                    \
                                                                               \
    for (int i = threadIdx.x; i < shape_1; i += blockDim.x) {                  \
      int idx = base_idx + i * strides_1;                                      \
      dst[idx] = scale * (float)x[idx];                                               \
    }                                                                          \
  }

INSTANTIATE_APPLY_ROPE(f32, float)
INSTANTIATE_APPLY_ROPE(f16, __half)

INSTANTIATE_RMS_NORM(f32, float, small_, 32)
INSTANTIATE_RMS_NORM(f32, float, , 1024)
INSTANTIATE_RMS_NORM(f16, __half, small_, 32)
INSTANTIATE_RMS_NORM(f16, __half, , 1024)

INSTANTIATE_SOFTMAX(f32, float, small_, 32)
INSTANTIATE_SOFTMAX(f32, float, , 1024)
INSTANTIATE_SOFTMAX(f16, __half, small_, 32)
INSTANTIATE_SOFTMAX(f16, __half, , 1024)

#define INSTANTIATE_SCALED_MASKED_SOFTMAX_FOR_T(name, T)                       \
  INSTANTIATE_SCALED_MASKED_SOFTMAX(name, T, 32_, 32)                          \
  INSTANTIATE_SCALED_MASKED_SOFTMAX(name, T, 64_, 64)                          \
  INSTANTIATE_SCALED_MASKED_SOFTMAX(name, T, 128_, 126)                        \
  INSTANTIATE_SCALED_MASKED_SOFTMAX(name, T, 256_, 256)                        \
  INSTANTIATE_SCALED_MASKED_SOFTMAX(name, T, 512_, 512)                        \
  INSTANTIATE_SCALED_MASKED_SOFTMAX(name, T, 1024_, 1024)                      \
  INSTANTIATE_SCALED_MASKED_SOFTMAX(name, T, 2048_, 1024)                      \
  INSTANTIATE_SCALED_MASKED_SOFTMAX(name, T, 4096_, 1024)                      \
  INSTANTIATE_SCALED_MASKED_SOFTMAX(name, T, 8192_, 1024)                      \
  INSTANTIATE_SCALED_MASKED_SOFTMAX(name, T, 16384_, 1024)                     \
  INSTANTIATE_SCALED_MASKED_SOFTMAX(name, T, 32768_, 1024)                     \
  INSTANTIATE_SCALED_MASKED_SOFTMAX(name, T, 0_, 0)

INSTANTIATE_SCALED_MASKED_SOFTMAX_FOR_T(f32, float)
INSTANTIATE_SCALED_MASKED_SOFTMAX_FOR_T(f16, __half)

INSTANTIATE_REDUCE(f32, float, small_, 32)
INSTANTIATE_REDUCE(f32, float, , 1024)
INSTANTIATE_REDUCE(f16, __half, small_, 32)
INSTANTIATE_REDUCE(f16, __half, , 1024)
