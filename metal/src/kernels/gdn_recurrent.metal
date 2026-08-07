#include <metal_stdlib>
using namespace metal;

// Gated delta rule over the whole sequence. Layout matches the CPU op:
// query/key/value/output [b, S, h, w], log_decay/beta [b, S, h],
// state [b, h, w, w]. One thread per (batch, head, column); each thread
// owns its state column exclusively, so the sequential S loop is
// race-free (later steps read this thread's own prior writes in
// final_state).
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
    constant int &s_len [[buffer(10)]],
    constant int &batch [[buffer(11)]],
    uint gid [[thread_position_in_grid]]) {
  const int column = gid % width;
  const int head = (gid / width) % heads;
  const int b = gid / (width * heads);
  if (b >= batch) return;
  const int matrix_base = (b * heads + head) * width * width;
  const float out_scale = rsqrt(float(width));
  for (int s = 0; s < s_len; ++s) {
    const int vector_base = ((b * s_len + s) * heads + head) * width;
    const int gate_ix = (b * s_len + s) * heads + head;
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
    const float decay = exp(log_decay[gate_ix]);
    device const float *state_in = (s == 0) ? initial_state : final_state;
    float predicted = 0.0f;
    for (int row = 0; row < width; ++row) {
      predicted += float(key[vector_base + row]) * k_inv
          * state_in[matrix_base + row * width + column] * decay;
    }
    const float residual =
        (float(value[vector_base + column]) - predicted)
        * float(beta[gate_ix]);
    float result = 0.0f;
    for (int row = 0; row < width; ++row) {
      const int offset = matrix_base + row * width + column;
      const float next = state_in[offset] * decay
          + float(key[vector_base + row]) * k_inv * residual;
      final_state[offset] = next;
      result += float(query[vector_base + row]) * q_inv * next;
    }
    output[vector_base + column] = half(result * out_scale);
  }
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
