#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

// One block handles one 128x128 recurrent matrix. Threads own output/value
// columns, which makes every state write coalesced and race-free.
extern "C" __global__ void tract_gdn_recurrent_f16(
    const __half* query,
    const __half* key,
    const __half* value,
    const float* log_decay,
    const __half* beta,
    const float* initial_state,
    __half* output,
    float* final_state,
    int heads,
    int width) {
  const int head = blockIdx.x;
  const int column = threadIdx.x;
  if (head >= heads || column >= width) return;

  extern __shared__ float shared[];
  float* q_normed = shared;
  float* k_normed = shared + width;
  float* reductions = shared + 2 * width;

  const int vector_base = head * width;
  const float q = __half2float(query[vector_base + column]);
  const float k = __half2float(key[vector_base + column]);
  reductions[column] = q * q;
  __syncthreads();
  for (int stride = width / 2; stride; stride >>= 1) {
    if (column < stride) reductions[column] += reductions[column + stride];
    __syncthreads();
  }
  const float q_inv = rsqrtf(reductions[0] + 1.0e-6f);
  reductions[column] = k * k;
  __syncthreads();
  for (int stride = width / 2; stride; stride >>= 1) {
    if (column < stride) reductions[column] += reductions[column + stride];
    __syncthreads();
  }
  const float k_inv = rsqrtf(reductions[0] + 1.0e-6f);
  q_normed[column] = q * q_inv;
  k_normed[column] = k * k_inv;
  __syncthreads();

  const float decay = expf(log_decay[head]);
  float predicted = 0.0f;
  const int matrix_base = head * width * width;
  for (int row = 0; row < width; ++row) {
    const int offset = matrix_base + row * width + column;
    const float state = initial_state[offset] * decay;
    predicted = fmaf(k_normed[row], state, predicted);
  }
  const float residual =
      (__half2float(value[vector_base + column]) - predicted) *
      __half2float(beta[head]);

  float result = 0.0f;
  for (int row = 0; row < width; ++row) {
    const int offset = matrix_base + row * width + column;
    const float state =
        fmaf(k_normed[row], residual,
             initial_state[offset] * decay);
    final_state[offset] = state;
    result = fmaf(q_normed[row], state, result);
  }
  output[vector_base + column] =
      __float2half(result * rsqrtf(static_cast<float>(width)));
}

// One thread owns one depthwise channel. Qwen3.5 decoding always appends one
// sample to a four-element causal-convolution cache.
extern "C" __global__ void tract_causal_conv1d_update_f16(
    const __half* input,
    const __half* weight,
    const __half* initial_state,
    __half* output,
    __half* final_state,
    int channels,
    int kernel_width) {
  const int channel = blockIdx.x * blockDim.x + threadIdx.x;
  if (channel >= channels) return;
  const int base = channel * kernel_width;
  float sum = 0.0f;
  for (int tap = 0; tap < kernel_width - 1; ++tap) {
    const __half sample = initial_state[base + tap + 1];
    final_state[base + tap] = sample;
    sum = fmaf(__half2float(sample), __half2float(weight[base + tap]), sum);
  }
  const __half newest = input[channel];
  final_state[base + kernel_width - 1] = newest;
  sum = fmaf(__half2float(newest),
             __half2float(weight[base + kernel_width - 1]), sum);
  output[channel] = __float2half(sum / (1.0f + expf(-sum)));
}
