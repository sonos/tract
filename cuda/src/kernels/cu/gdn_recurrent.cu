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
// Sliding-window causal conv update with fused silu, any S and batch
// (direct port of the Metal kernel of the same name). Layout: input/output
// [b, C, S], state [b, C, K], weight [C, K].
extern "C" __global__ void tract_causal_conv1d_update_f16(
    const __half* input,
    const __half* weight,
    const __half* initial_state,
    __half* output,
    __half* final_state,
    int channels,
    int kernel_width,
    int s_len,
    int batch) {
  const int gid = blockIdx.x * blockDim.x + threadIdx.x;
  const int channel = gid % channels;
  const int b = gid / channels;
  if (b >= batch) return;
  const int state_base = (b * channels + channel) * kernel_width;
  const int input_base = (b * channels + channel) * s_len;
  const int weight_base = channel * kernel_width;
  // kernel_width is small (4 for Qwen3.5), keep the window in registers.
  const int MAX_K = 8;
  float window[MAX_K];
  if (kernel_width > MAX_K) return;
  for (int tap = 0; tap < kernel_width; ++tap) {
    window[tap] = __half2float(initial_state[state_base + tap]);
  }
  for (int t = 0; t < s_len; ++t) {
    // shift left, append the new sample: window becomes full[t+1 .. t+k]
    for (int tap = 0; tap < kernel_width - 1; ++tap) {
      window[tap] = window[tap + 1];
    }
    window[kernel_width - 1] = __half2float(input[input_base + t]);
    float sum = 0.0f;
    for (int tap = 0; tap < kernel_width; ++tap) {
      sum = fmaf(window[tap], __half2float(weight[weight_base + tap]), sum);
    }
    output[input_base + t] = __float2half(sum / (1.0f + expf(-sum)));
  }
  for (int tap = 0; tap < kernel_width; ++tap) {
    final_state[state_base + tap] = __float2half(window[tap]);
  }
}
