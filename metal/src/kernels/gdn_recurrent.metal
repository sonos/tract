#include <metal_stdlib>
using namespace metal;

kernel void gdn_recurrent_f16(
    device const half *query [[buffer(0)]],
    device const half *key [[buffer(1)]],
    device const half *value [[buffer(2)]],
    device const float *log_decay [[buffer(3)]],
    device const half *beta [[buffer(4)]],
    device const float *initial_state [[buffer(5)]],
    device half *output [[buffer(6)]],
    device float *final_state [[buffer(7)]],
    constant int &heads [[buffer(8)]],
    constant int &width [[buffer(9)]],
    uint gid [[thread_position_in_grid]]) {
  const int head = gid / width;
  const int column = gid % width;
  if (head >= heads) return;
  const int vector_base = head * width;
  const int matrix_base = head * width * width;
  float q_norm = 0.0f;
  float k_norm = 0.0f;
  for (int row = 0; row < width; ++row) {
    const float q = float(query[vector_base + row]);
    const float k = float(key[vector_base + row]);
    q_norm += q * q;
    k_norm += k * k;
  }
  const float q_inv = rsqrt(q_norm + 1.0e-6f);
  const float k_inv = rsqrt(k_norm + 1.0e-6f);
  const float decay = exp(log_decay[head]);
  float predicted = 0.0f;
  for (int row = 0; row < width; ++row) {
    predicted += float(key[vector_base + row]) * k_inv
        * initial_state[matrix_base + row * width + column] * decay;
  }
  const float residual =
      (float(value[vector_base + column]) - predicted) * float(beta[head]);
  float result = 0.0f;
  for (int row = 0; row < width; ++row) {
    const int offset = matrix_base + row * width + column;
    const float next = initial_state[offset] * decay
        + float(key[vector_base + row]) * k_inv * residual;
    final_state[offset] = next;
    result += float(query[vector_base + row]) * q_inv * next;
  }
  output[vector_base + column] = half(result * rsqrt(float(width)));
}

kernel void causal_conv1d_update_f16(
    device const half *input [[buffer(0)]],
    device const half *weight [[buffer(1)]],
    device const half *initial_state [[buffer(2)]],
    device half *output [[buffer(3)]],
    device half *final_state [[buffer(4)]],
    constant int &channels [[buffer(5)]],
    constant int &kernel_width [[buffer(6)]],
    uint channel [[thread_position_in_grid]]) {
  if (channel >= uint(channels)) return;
  const int base = channel * kernel_width;
  float sum = 0.0f;
  for (int tap = 0; tap < kernel_width - 1; ++tap) {
    const half sample = initial_state[base + tap + 1];
    final_state[base + tap] = sample;
    sum += float(sample) * float(weight[base + tap]);
  }
  const half newest = input[channel];
  final_state[base + kernel_width - 1] = newest;
  sum += float(newest) * float(weight[base + kernel_width - 1]);
  output[channel] = half(sum / (1.0f + exp(-sum)));
}
