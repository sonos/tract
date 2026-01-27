#include <cuda_runtime.h>
#include <math_constants.h>
#include "common.cuh"

#define GELU_COEF_A 0.044715f
#define SQRT_2_OVER_PI 0.79788456080286535587989211986876f

template <class Op, int width>
__device__ __forceinline__ typename Op::acc_t warp_reduce(typename Op::acc_t v) {
  #pragma unroll
  for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1) {
    auto other = __shfl_xor_sync(0xffffffff, v, offset, width);
    v = Op::combine(v, other);
  }
  return v;
}

template <typename Acc>
struct MaxOp {
  using acc_t = Acc;
  __device__ __forceinline__ static acc_t identity() { return -CUDART_INF_F; }
  __device__ __forceinline__ static acc_t pre(acc_t a) { return a; }
  __device__ __forceinline__ static acc_t combine(acc_t a, acc_t b) { return a > b ? a : b; }
  __device__ __forceinline__ static acc_t norm(acc_t a, int32_t size) { return a; }
};


template <typename Acc>
struct MinOp {
  using acc_t = Acc;
  __device__ __forceinline__ static acc_t identity() { return CUDART_INF_F; }
  __device__ __forceinline__ static acc_t pre(acc_t a) { return a; }
  __device__ __forceinline__ static acc_t combine(acc_t a, acc_t b) { return a < b ? a : b; }
  __device__ __forceinline__ static acc_t norm(acc_t a, int32_t size) { return a; }
};

template <typename Acc>
struct AddOp {
  using acc_t = Acc;
  __device__ __forceinline__ static acc_t identity() { return acc_t(0); }
  __device__ __forceinline__ static acc_t pre(acc_t a) { return a; }
  __device__ __forceinline__ static acc_t combine(acc_t a, acc_t b) { return a + b; }
  __device__ __forceinline__ static acc_t norm(acc_t a, int32_t size) { return a; }
};

template <typename Acc>
struct MulOp {
  using acc_t = Acc;
  __device__ __forceinline__ static acc_t identity() { return acc_t(1); }
  __device__ __forceinline__ static acc_t pre(acc_t a) { return a; }
  __device__ __forceinline__ static acc_t combine(acc_t a, acc_t b) { return a * b; }
  __device__ __forceinline__ static acc_t norm(acc_t a, int32_t size) { return a; }
};

template <typename Acc>
struct MeanOfSquaresOp {
  using acc_t = Acc;
  __device__ __forceinline__ static acc_t identity() { return acc_t(0); }
  __device__ __forceinline__ static acc_t pre(acc_t a) { return a * a; }
  __device__ __forceinline__ static acc_t combine(acc_t a, acc_t b) { return a + b; }
  __device__ __forceinline__ static acc_t norm(acc_t a, int32_t size) { return a / (acc_t) size; }
};

template<typename T, int block_size, class Op>
__device__ void reduce(
      const T *input, T *output,
      const int32_t shape_0, const int32_t shape_1, const int32_t shape_2,
      const int32_t in_stride_0, const int32_t in_stride_1, const int32_t in_stride_2,
      const int32_t out_stride_0, const int32_t out_stride_1, const int32_t out_stride_2
    ) {
    using Acc = typename Op::acc_t;

    input += blockIdx.z * in_stride_0 + blockIdx.x * in_stride_2;              
    output += blockIdx.z * out_stride_0 + blockIdx.x * out_stride_2;           
                                                                               
    const int warp_id = threadIdx.x / WARP_SIZE;                               
    const int lane_id = threadIdx.x % WARP_SIZE;                               
                                                                               
    Acc accu = Op::identity();
    _Pragma("unroll")
    for (int i = threadIdx.x; i < shape_1; i += blockDim.x) {                                  
      accu = Op::combine(accu, Op::pre(input[i * in_stride_1]));
    }                                                                          
    accu = warp_reduce<Op, block_size>(accu);
    if (block_size > WARP_SIZE) {                                              
      __shared__ float shared[32];
      if (warp_id == 0) {                                          
        shared[lane_id] = Op::identity();                                            
      }                                                                        
      __syncthreads();                                                         
                                                                               
      if (lane_id == 0) {                                                      
        shared[warp_id] = accu;
      }                                                                        
      __syncthreads();                                                         
                                                                               
      accu = shared[lane_id];                                                
      accu = warp_reduce<Op, block_size>(accu);
    }                                                                          
                                                                               
    if (threadIdx.x == 0) {
       *output =  Op::norm(accu, shape_1);
    }                                                                          
}

  
#define INSTANTIATE_REDUCE_1(op_name, name, T, Op, bname, block_size)          \
  extern "C" __global__ void CAT5(reduce_, op_name, _, bname, name)(           \
      const T *input, T *output,                                               \
      const int32_t shape_0, const int32_t shape_1, const int32_t shape_2,     \
      const int32_t in_stride_0, const int32_t in_stride_1,                    \
      const int32_t in_stride_2,                                               \
      const int32_t out_stride_0, const int32_t out_stride_1,                  \
      const int32_t out_stride_2                                               \
      ) {                                                                      \
    reduce<T, block_size, Op<T> >                                              \
      (input, output,                                                          \
      shape_0, shape_1, shape_2,                                               \
      in_stride_0, in_stride_1, in_stride_2,                                   \
      out_stride_0, out_stride_1, out_stride_2                                 \
    );\
    }\

#define INSTANTIATE_REDUCE(name, T, bname, block_size)                         \
  INSTANTIATE_REDUCE_1(max, name, T, MaxOp, bname, block_size)                 \
  INSTANTIATE_REDUCE_1(min, name, T, MinOp, bname, block_size)                 \
  INSTANTIATE_REDUCE_1(sum, name, T, AddOp, bname, block_size)                 \
  INSTANTIATE_REDUCE_1(prod, name, T, MulOp, bname, block_size)                \
  INSTANTIATE_REDUCE_1(mean_of_squares, name, T, MeanOfSquaresOp,              \
              bname, block_size)                \

                  
extern "C" __global__ void gelu_approx_f32(const float *input, float *output,
                                           int32_t len) {
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
                                           int32_t len) {
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
                                                float *output, int32_t len) {
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
                                                __half *output, int32_t len) {
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
                                                float *output, int32_t len,
                                                float alpha) {
  int i = blockIdx.x * blockDim.x + threadIdx.x;
  if (i < len) {
    float x = input[i];
    output[i] = x * (x < 0 ? alpha : 1.0);
  }
}

extern "C" __global__ void leaky_relu_f16(const __half *input,
                                                __half *output, int32_t len,
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
      const T *input, const T *cos, const T *sin, T *output, int32_t in_shape_0,   \
      int32_t in_shape_1, int32_t in_strides_0, int32_t in_strides_1,                      \
      int32_t cos_sin_strides_0, int32_t cos_sin_strides_1, int32_t out_strides_0,         \
      int32_t out_strides_1) {                                                     \
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
      const T *input, const T *cos, const T *sin, T *output, int32_t in_shape_0,   \
      int32_t in_shape_1, int32_t in_shape_2, int32_t in_strides_0, int32_t in_strides_1,      \
      int32_t in_strides_2, int32_t cos_sin_strides_0, int32_t cos_sin_strides_1,          \
      int32_t cos_sin_strides_2, int32_t out_strides_0, int32_t out_strides_1,             \
      int32_t out_strides_2) {                                                     \
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
      const T *input, const T *cos, const T *sin, T *output, int32_t in_shape_0,   \
      int32_t in_shape_1, int32_t in_shape_2, int32_t in_shape_3, int32_t in_strides_0,        \
      int32_t in_strides_1, int32_t in_strides_2, int32_t in_strides_3,                    \
      int32_t cos_sin_strides_0, int32_t cos_sin_strides_1, int32_t cos_sin_strides_2,     \
      int32_t cos_sin_strides_3, int32_t out_strides_0, int32_t out_strides_1,             \
      int32_t out_strides_2, int32_t out_strides_3) {                                  \
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
      const T *x, T *dst, const int32_t shape_0, const int32_t shape_1,                \
      const int32_t shape_2, const int32_t stride_0, const int32_t stride_1,               \
      const int32_t stride_2) {                                                    \
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
      const T *x, const T *mask, const T scale, T *dst, const int32_t shape_0,     \
      const int32_t shape_1, const int32_t shape_2, const int32_t stride_0,                \
      const int32_t stride_1, const int32_t stride_2, const int32_t mask_stride_0,         \
      const int32_t mask_stride_1, const int32_t mask_stride_2,                        \
      const int32_t out_stride_0, const int32_t out_stride_1,                          \
      const int32_t out_stride_2) {                                                \
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
      const T *x, T *dst, const int32_t shape_0, const int32_t shape_1,                \
      const int32_t shape_2, const int32_t strides_0, const int32_t strides_1,             \
      const int32_t strides_2, const float eps) {                                      \
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
